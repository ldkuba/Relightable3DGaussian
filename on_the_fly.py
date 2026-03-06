import os
import sys
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from simple_knn._C import distCUDA2

from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH

from PIL import Image

from scipy.spatial import cKDTree as KDTree, cKDTree

sys.path.append('external')
from on_the_fly_nvs.utils import (
    get_lapla_norm,
)

sys.path.append("external/on_the_fly_nvs/submodules/Depth-Anything-V2")
sys.path.append("external/on_the_fly_nvs")
from depth_anything_v2.dpt import DepthAnythingV2
from on_the_fly_nvs.poses.triangulator import matches_to_points

class GaussianBatch(NamedTuple):
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    normals: torch.Tensor
    shs_dc: torch.Tensor
    shs_rest: torch.Tensor
    max_radii2D: torch.Tensor
    weights: torch.Tensor
    xyz_gradient_accum: torch.Tensor
    normal_gradient_accum: torch.Tensor
    denom: torch.Tensor

class OnTheFly:

    def __init__(self, width, height, max_sh_degree, pcd, scene_info = None, prob_scale = 1.0, bg = 0.0):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.width = width
        self.height = height
        self.prob_scale = prob_scale
        self.max_sh_degree = max_sh_degree
        self.bg = bg
        self.xyz = torch.from_numpy(pcd.points).to(self.DEVICE)
        self.scene_info = scene_info
        self.cnt_tmp = 0

        print("pcd.shape: ", self.xyz.shape)

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        # for depth map


        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitl'

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'models/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(self.DEVICE).eval()

        self.camera_processed = set()

        self.to_save_tmp = np.zeros((0, 3))
        self.colmap_to_save_tmp = np.zeros((0, 3))

    @torch.no_grad()
    def project(self, cam, points):
        points = points.to(dtype=torch.float32)
        points = torch.concatenate((points, torch.ones(points.shape[0], 1)), dim=1)
        cam_world = points @ cam.extrinsics.T
        projected = cam_world[:,:3] @ cam.intrinsics.T
        projected = projected / torch.unsqueeze(projected[:,2], dim=1)
        projected = torch.floor(projected).to(dtype=torch.int32)

        x = projected[:,0]
        y = projected[:,1]

        x = x[(x >= 0) & (x < cam.image_width)]
        y = y[(y >= 0) & (y < cam.image_height)]

        mask = torch.zeros((cam.image_width, cam.image_height), device=projected.device)
        mask[y, x] = 1.0

        return mask


    @torch.no_grad()
    def create_prob_map(self, img):
        """Creates a probability map, reference add_new_gaussians in scene_model from on the fly
        - kernel is eq. 1 in paper"""
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        prob_mask = get_lapla_norm(img, self.disc_kernel)  # eq. 1

        return prob_mask

    @torch.no_grad()
    def generate_depth_map(self, img):
        detached = img.cpu().detach().numpy()
        detached = detached.transpose(1, 2, 0)
#        print("detached.shape: ", detached.shape)
#        print(np.max(detached))
        depth = self.model.infer_image(detached * 255, input_size=self.width)

        return torch.from_numpy(depth).to(img.device)

    def generate_mask(self, img):
        return img != self.bg


    @torch.no_grad()
    def generate_gaussians(self, prob_mask, depth_mask, cam):
        # ++ means ++

        c2w = torch.linalg.inv(cam.extrinsics)
        c2w = c2w.T

        image_mask = torch.squeeze(cam.image_mask).to(device=c2w.device)

        # generate probability mask
        sample_mask = torch.rand_like(prob_mask) < prob_mask
        sample_mask = image_mask.to(torch.bool) & sample_mask

        # solely for vizual debugging, set background to mean of object
        depth_mask[~image_mask.to(torch.bool)] = torch.mean(depth_mask[image_mask.to(torch.bool)])

        # get list of (u, v) and z
        coords = torch.nonzero(sample_mask)
        z_np = depth_mask[coords[:,0], coords[:,1]].to(dtype=torch.float32)
        z_np = torch.unsqueeze(z_np, dim=0)
        z_np = z_np.T

        # get (u, v) to (x, y) / (u = y, v = x)
        coords = coords[:, [1, 0]]
        coords = coords.to(dtype=torch.float32) + 0.5

        # prepare for unproject -> (zu, zv, z)
        coords = coords * z_np
        pre_intrinsics = torch.concatenate((coords, z_np), dim=1)

        # filter out points too close to camera and possible glitches from depth estimator
#        valid_mask = torch.isfinite(pre_intrinsics[:,2]) & (pre_intrinsics[:,2] > 1e-6)
#        pre_intrinsics = pre_intrinsics[valid_mask]

        # apply inverse intrinsics
        unproj = torch.linalg.inv(cam.intrinsics).T

        cam_world = pre_intrinsics @ unproj

        # debug
        with open(os.path.expanduser("~/gaussian-splatting/pcd/snap_np.npy"), "wb") as f:
            np.save(f, cam_world.detach().cpu().numpy())

        print("cam.image_name", cam.image_name)

        # put reconstruction into world space
        homo_ones = torch.ones((cam_world.shape[0], 1))
        cam_world = torch.concatenate((cam_world, homo_ones), dim=1)

        world_points = cam_world @ c2w
        world_points = world_points[:,:3]
        assert ~ torch.isnan(cam_world).any()
        assert ~ torch.isnan(world_points).any()

        # debug
        self.to_save_tmp = np.concatenate((self.to_save_tmp, world_points.detach().cpu().numpy()))


        # ++ scales ++
        filtered_prob_mask = prob_mask[sample_mask]
        scales = 1.0 / (2 * torch.sqrt(filtered_prob_mask)).unsqueeze(1)
        f = cam.image_width / (2 * np.tan(cam.FoVx / 2))
        scales = (depth_mask[sample_mask].unsqueeze(dim=1).to(dtype=torch.float32) * scales) / f
        scales = torch.log(scales)
        scales = scales.repeat(1, 3)

        # ++ rotations ++
        rots = torch.zeros((world_points.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # ++ opacities ++
        opacities = inverse_sigmoid(0.1 * torch.ones((world_points.shape[0], 1), dtype=torch.float, device="cuda"))
        prim_axis = -cam.get_primary_axis()

        # ++ normals ++
        normals = prim_axis.unsqueeze(0).repeat(world_points.shape[0], 1)

        img = cam.original_image

        # ++ base colors ++
        base_color = img.permute(1, 2, 0)[sample_mask]

        # ++ shs ++
        shs = torch.zeros((base_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        shs[:, :3, 0] = RGB2SH(base_color)
        shs[:, 3:, 1:] = 0.0

        shs_dc = shs[:, :, 0:1].transpose(1, 2)
        shs_rest = shs[:, :, 1:].transpose(1, 2)

        # ++ max_radii2D ++
        max_radii2D = torch.zeros((world_points.shape[0]), device="cuda")

        # ++ weights ++
        weights = torch.zeros(world_points.shape[0], 1, device="cuda")

        xyz_gradient_accum = torch.zeros((world_points.shape[0], 1), device="cuda")
        normal_gradient_accum = torch.zeros((world_points.shape[0], 1), device="cuda")
        denom = torch.zeros((world_points.shape[0], 1), device="cuda")

        return GaussianBatch(
            means=world_points, scales=scales, rotations=rots, normals=normals, shs_dc=shs_dc, shs_rest=shs_rest,
            opacities=opacities, max_radii2D=max_radii2D, weights=weights, xyz_gradient_accum=xyz_gradient_accum,
            normal_gradient_accum=normal_gradient_accum, denom=denom
        )

    def check_and_set_cam(self, uid):
        if uid in self.camera_processed:
            return False
        self.camera_processed.add(uid)
        return True

    @torch.no_grad()
    def get_key_points(self, viewpoint_cam):
        id_to_xyz = self.scene_info.colmap_point_cloud
        key_points = self.scene_info.key_points.get(f'{viewpoint_cam.image_name}.png')

        id3d = id_to_xyz[0]
        id2d = key_points[0]

        _, m3d, m2d = np.intersect1d(id3d, id2d, return_indices=True)

        tmp = np.round(key_points[1][m2d, :]).astype(dtype=np.int32)
        n = 800
        coord = np.zeros((n, n), dtype=float)
        coord[tmp[:, 1], tmp[:, 0]] = 1.0

        proj = self.project(viewpoint_cam, torch.from_numpy(id_to_xyz[1][m3d, :].astype(dtype=float)).to(device="cuda")).detach().cpu().numpy()

        return key_points[1][m2d, :], id_to_xyz[1][m3d, :]

    @torch.no_grad()
    def adjust_depth_map(self, viewpoint_cam, depth_map):
        uv, xyzs = self.get_key_points(viewpoint_cam)
        uv = np.round(uv).astype(dtype=np.int32)
        u = uv[:,0]
        v = uv[:,1]

        D_rel = depth_map.detach().cpu().numpy()
        xyzs_camera_world = (np.concatenate((xyzs, np.ones((xyzs.shape[0], 1))), axis=1) @ viewpoint_cam.extrinsics.T.detach().cpu().numpy())
        self.colmap_to_save_tmp = np.concatenate((self.colmap_to_save_tmp, xyzs_camera_world[:,:3]), axis=0)

        xyzs_camera_world = xyzs_camera_world[:,:3] @ viewpoint_cam.intrinsics.T.detach().cpu().numpy()
        D_sfm = 1.0 / xyzs_camera_world[:,2]

        t_sfm = self.t(D_sfm)
        t_rel = self.t(D_rel[v, u])
        s_sfm = self.s(D_sfm, t_sfm)
        s_rel = self.s(D_rel[v, u], t_rel)

        D = (s_sfm / s_rel) * D_rel + t_sfm - t_rel * (s_sfm / s_rel)
        D = np.clip(D, 1e-6, 1e6)
        D = 1.0 / D
        D = self.adjust_depth_knn(D, uv, xyzs_camera_world[:,2])
        img_D = D.copy()
        self.save_img(img_D, f"{viewpoint_cam.image_name}_cor")
        img_dav2 = depth_map.detach().cpu().numpy()
        self.save_img(img_dav2, f"{viewpoint_cam.image_name}_dav2", inv=True)
        return torch.from_numpy(D).to(device=depth_map.device)

    def save_img(self, img_D, name, inv=False):
        img_D = ((img_D - img_D.min()) / (img_D.max() - img_D.min())) * 255
        if (inv):
            img_D = 255 - img_D
        image_D = Image.fromarray(img_D.astype(np.uint8))
        image_D.save(os.path.expanduser(f"~/gaussian-splatting/pcd/depth_maps/{name}_depth_map.png"))

    @torch.no_grad()
    def adjust_depth_knn(self, D, uv_desc, z_sfm, k=8, stride=4, p=2.0, eps=1e-6):
        H, W = D.shape

        u = uv_desc[:, 0].astype(dtype=np.int32)
        v = uv_desc[:, 1].astype(dtype=np.int32)

        z_map = D[v, u]

        delta = z_sfm - z_map

        tree = cKDTree(uv_desc)

        ys = np.arange(0, H, stride, dtype=np.int32)
        xs = np.arange(0, W, stride, dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys)
        Q = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        kk = min(k, len(uv_desc))
        dists, idx = tree.query(Q, k=kk, workers=-1)  # workers=-1 uses all cores (newer SciPy)

        # make shapes consistent for kk=1
        if kk == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        # inverse-distance weights
        w = 1.0 / (dists ** p + eps)
        Delta = (w * delta[idx]).sum(axis=1) / (w.sum(axis=1) + eps)
        Delta = Delta.reshape(len(ys), len(xs))

        # upsample back to full res (fast nearest-neighbor)
        Delta_full = np.repeat(np.repeat(Delta, stride, axis=0), stride, axis=1)[:H, :W]

        return D + Delta_full

    @torch.no_grad()
    def knn_delta_map(self, u, v, delta, H, W, k=4, p=2, eps=1e-6):
        key_uv = np.stack([u, v], axis=1)  # (x,y)
        grid_u, grid_v = np.meshgrid(np.arange(W), np.arange(H))
        pts = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)

        tree = KDTree(key_uv)
        dists, idxs = tree.query(pts, k=min(k, key_uv.shape[0]))  # (HW,k)

        # If k=1, make shapes consistent
        if idxs.ndim == 1:
            idxs = idxs[:, None]
            dists = dists[:, None]

        w = 1.0 / (dists + eps) ** p
        deltas = delta[idxs]  # (HW,k)
        delta_interp = (w * deltas).sum(1) / w.sum(1)
        return delta_interp.reshape(H, W)

    def add_gaussians(self, gaussians, viewpoint_cam):
        if len(self.scene_info.key_points.get(f'{viewpoint_cam.image_name}.png')[0]) < 1000:
            return
        prob_mask = self.create_prob_map(viewpoint_cam.original_image)
        depth_mask = self.generate_depth_map(viewpoint_cam.original_image)
        if self.scene_info is not None:
            depth_mask = self.adjust_depth_map(viewpoint_cam, depth_mask)
        gaussian_batch = self.generate_gaussians(prob_mask, depth_mask, viewpoint_cam)

        gaussians.add_on_the_fly_gaussians(gaussian_batch)

        if self.cnt_tmp >= 20:
            self.save()
            return

        self.cnt_tmp = self.cnt_tmp + 1

    def save(self):
        print("self.to_save_tmp.shape: ", self.to_save_tmp.shape)
        with open(os.path.expanduser("~/gaussian-splatting/pcd/world.npy"), "wb") as f:
            np.save(f, self.to_save_tmp)

    def t(self, D):
        return np.median(D)

    def s(self, D, tD):
        return np.mean(np.abs(D - tD))