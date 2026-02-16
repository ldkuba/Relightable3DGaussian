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

sys.path.append('external')
from on_the_fly_nvs.utils import (
    get_lapla_norm,
)

sys.path.append("external/on_the_fly_nvs/submodules/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

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
        c2wT = torch.linalg.inv(cam.extrinsics.T)

        # ++ means ++

# let's do everything in numpy, I change it later

        image_mask_np = np.squeeze(cam.image_mask.detach().cpu().numpy(), axis=0)
        sample_mask = torch.rand_like(prob_mask) < prob_mask
        sample_mask_np = sample_mask.detach().cpu().numpy()
        sample_mask_np = image_mask_np.astype(bool) & sample_mask_np
        sample_mask = torch.from_numpy(sample_mask_np).to(device=sample_mask.device)
        depth_mask_np = depth_mask.detach().cpu().numpy()
        depth_mask_np[~image_mask_np.astype(bool)] = np.mean(depth_mask_np[image_mask_np.astype(bool)])
        coords_np = np.argwhere(sample_mask_np)
        z_np = depth_mask_np[coords_np[:,0], coords_np[:,1]]
        z_np = np.expand_dims(z_np, axis=0)
        z_np = z_np.T
        coords_np = coords_np * z_np
        pre_intrinsics_np = np.concatenate((coords_np, z_np), axis=1)
        valid_mask_np = np.isfinite(pre_intrinsics_np[:,2]) & (pre_intrinsics_np[:,2] > 1e-6)
        pre_intrinsics_np = pre_intrinsics_np[valid_mask_np]

        unprojT_np = cam.intrinsics.T.detach().cpu().numpy()
        unprojT_np = np.linalg.inv(unprojT_np)

        cam_world_np = pre_intrinsics_np @ unprojT_np

        with open(os.path.expanduser("~/gaussian-splatting/pcd/snap_np.npy"), "wb") as f:
            np.save(f, cam_world_np)

        print("cam.image_name", cam.image_name)

#        sys.exit(0)

        homo_ones = np.ones((cam_world_np.shape[0], 1))
        cam_world_np = np.concatenate((cam_world_np, homo_ones), axis=1).astype(np.float32)

        cam_world = torch.from_numpy(cam_world_np).to(device=c2wT.device)
        world_points = cam_world @ c2wT
        world_points = world_points[:,:3]
        assert ~ torch.isnan(cam_world).any()
        assert ~ torch.isnan(world_points).any()

#        self.project(cam)

        print(f"lower/upper of x: {torch.min(world_points[:,0])}/{torch.max(world_points[:,0])}")
        print(f"lower/upper of y: {torch.min(world_points[:,1])}/{torch.max(world_points[:,1])}")
        print(f"lower/upper of z: {torch.min(world_points[:,2])}/{torch.max(world_points[:,2])}")

#
        self.to_save_tmp = np.concatenate((self.to_save_tmp, world_points.detach().cpu().numpy()))


        # ++ scales ++
        dist2 = torch.clamp_min(distCUDA2(world_points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

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
#        base_color = img.permute(1, 2, 0)[sample_mask]
        base_color = torch.ones(world_points.shape, dtype=torch.float32)

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
    def adjust_depth_map(self, viewpoint_cam, depth_map):
        id_to_xyz = self.scene_info.colmap_point_cloud
        key_points = self.scene_info.key_points.get(f'{viewpoint_cam.image_name}.png')

        id3d = id_to_xyz[0]
        id2d = key_points[0]

        _, m3d, m2d = np.intersect1d(id3d, id2d, return_indices=True)

        xys = key_points[1][m2d, :]
        xyzs = id_to_xyz[1][m3d, :]

        pixel = np.floor(xys).astype(np.int32)
        depth = depth_map[pixel[:,1], pixel[:,0]].detach().cpu().numpy()
        xyzs_transformed = (np.concatenate((xyzs, np.ones((xyzs.shape[0], 1))), axis=1) @ viewpoint_cam.extrinsics.T.detach().cpu().numpy())
        xyzs_transformed = xyzs_transformed[:,:3] @ viewpoint_cam.intrinsics.T.detach().cpu().numpy()
        z = xyzs_transformed[:,2]
        inv_z = 1 / z
        A = np.stack((depth, np.ones(depth.shape)), axis=1)
        a, b = np.linalg.lstsq(A, inv_z, rcond=None)[0]
        print(f"a: {a}, b: {b}")
        self.save_plot(inv_z, A[:,0], a, b)
        return 1.0 / (a * depth_map + b)

    def add_gaussians(self, gaussians, viewpoint_cam):
        prob_mask = self.create_prob_map(viewpoint_cam.original_image)
        depth_mask = self.generate_depth_map(viewpoint_cam.original_image)
        if self.scene_info is not None:
            depth_mask = self.adjust_depth_map(viewpoint_cam, depth_mask)
        gaussian_batch = self.generate_gaussians(prob_mask, depth_mask, viewpoint_cam)
        sys.exit(0)

        gaussians.add_on_the_fly_gaussians(gaussian_batch)

    def save(self):
        self.to_save_tmp[np.isinf(self.to_save_tmp)] = 0
        lo = np.percentile(self.to_save_tmp, 1, axis=0)
        hi = np.percentile(self.to_save_tmp, 99, axis=0)
        mask = np.all((hi > self.to_save_tmp) & (lo < self.to_save_tmp), axis=1)
        self.to_save_tmp = self.to_save_tmp[mask]
        print("lo: ", lo)
        print("hi: ", hi)
        print("self.to_save_tmp.shape: ", self.to_save_tmp.shape)
        print("self.to_save_tmp[:100,:]: ", self.to_save_tmp[:100,:])
        with open(os.path.expanduser("~/gaussian-splatting/pcd/world.npy"), "wb") as f:
            np.save(f, self.to_save_tmp)

    def save_plot(self, x, y, a, b):
        # https://numpy.org/doc/2.2/reference/generated/numpy.linalg.lstsq.html
        import matplotlib.pyplot as plt
        _ = plt.plot(x, y, 'o', label='Original data', markersize=10)
        _ = plt.plot(x, a * x + b, 'r', label='Fitted line')
        _ = plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.expanduser("~/gaussian-splatting/model.png"), dpi=300, bbox_inches="tight")
