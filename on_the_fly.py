import os
import sys
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from simple_knn._C import distCUDA2
from sympy.benchmarks.bench_discrete_log import data_set_1

from external.Relightable3DGaussian.arguments import OnTheFlyParams
from external.Relightable3DGaussian.scene.gaussian_model import GaussianModel
from external.Relightable3DGaussian.utils.loss_utils import gaussian
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH

from PIL import Image

from scipy.spatial import cKDTree as KDTree, cKDTree

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

#class OnTheFlyParams(ParamGroup):
#    def __init__(self, parser):
#        self.on_the_fly = False
#        self.knn_p = 2
#        self.error_threshold = 0.75
#        self.feature_threshold = 500
#        self.base_prob = 0.025
#        self.normalize_prob = False
#        self.knn_n = 8
#        self.knn_stride = 1
#        self.knn_epsilon = 1e-6
#        super().__init__(parser, "Pipeline Parameters")

class OnTheFly:

    def __init__(self, width, height, max_sh_degree, otfp: OnTheFlyParams, dataset, args, render_fn, pipe, opt, pbr_kwargs, scene_info = None, bg = 0.0):

        # set device
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # set parameters
        self.knn_p = otfp.knn_p
        self.error_threshold = otfp.error_threshold
        self.feature_threshold = otfp.feature_threshold
        self.base_prob = otfp.base_prob
        self.normalize_prob = otfp.normalize_prob
        self.knn_n = otfp.knn_n
        self.knn_stride = otfp.knn_stride
        self.knn_epsilon = otfp.knn_epsilon
        self.neighbourhood_angle = otfp.neighbourhood_angle_criteria


        self.dataset = dataset
        self.args = args
        self.pipe = pipe
        self.opt = opt
        self.pbr_kwargs = pbr_kwargs

        self.width = width
        self.height = height

        self.prob_scale = 1.0
        self.max_sh_degree = max_sh_degree
        self.bg = bg

        # filter colmap points with too high reporjection error
        error_mask = scene_info.errors < self.error_threshold
        self.xyzs = scene_info.xyzs[error_mask]
        self.p3ids = scene_info.p3ids[error_mask]


        self.scene_info = scene_info

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

        self.gaussian_batches = dict()

        self.render_fn = render_fn

        self.neighbourhood = dict()

    @torch.no_grad()
    def create_density_map(self, img):
        """Creates a probability map, reference add_new_gaussians in scene_model from on the fly
        - kernel is eq. 1 in paper"""
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        prob_mask = self.get_lapla_norm(img, self.disc_kernel)  # eq. 1

        return prob_mask

    @torch.no_grad()
    def generate_depth_map(self, img):
        detached = img.cpu().detach().numpy()
        detached = detached.transpose(1, 2, 0)
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

        # prob_mask config
        prob_mask += self.base_prob

        if self.normalize_prob:
            prob_mask = prob_mask / torch.max(prob_mask)

        # apply penalty to prob_mask
        penalty = self.create_density_map(self.render_img(cam))

        if self.normalize_prob:
            penalty = penalty / torch.max(penalty)

        penalized = prob_mask - penalty

        # generate probability mask
        sample_mask = torch.rand_like(penalized) < penalized
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

        gaussian_batch = GaussianBatch(
            means=world_points, scales=scales, rotations=rots, normals=normals, shs_dc=shs_dc, shs_rest=shs_rest,
            opacities=opacities, max_radii2D=max_radii2D, weights=weights, xyz_gradient_accum=xyz_gradient_accum,
            normal_gradient_accum=normal_gradient_accum, denom=denom)

        self.gaussian_batches[cam.image_name] = gaussian_batch

        return gaussian_batch

    def check_and_set_cam(self, uid):
        if uid in self.camera_processed:
            return False
        self.camera_processed.add(uid)
        return True

    @torch.no_grad()
    def get_key_points(self, viewpoint_cam):
        xyzs = self.xyzs
        key_points = self.scene_info.key_points.get(f'{viewpoint_cam.image_name}.png')

        id3d = self.p3ids
        id2d = key_points[0]

        _, m3d, m2d = np.intersect1d(id3d, id2d, return_indices=True)

        return key_points[1][m2d, :], xyzs[m3d, :]

    @torch.no_grad()
    def adjust_depth_map(self, viewpoint_cam, depth_map):
        uv, xyzs = self.get_key_points(viewpoint_cam)
        uv = np.round(uv).astype(dtype=np.int32)
        u = uv[:,0]
        v = uv[:,1]

        with open(os.path.expanduser(f"~/gaussian-splatting/pcd/colmaps/{viewpoint_cam.image_name}.npy"), "wb") as f:
            np.save(f, xyzs)


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
        D = self.adjust_depth_knn(D, uv, xyzs_camera_world[:,2], k=self.knn_n, stride=self.knn_stride, p=self.knn_p, eps=self.knn_epsilon)
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
    def adjust_depth_knn(self, D, uv_desc, z_sfm, k, stride, p, eps):
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
        dists, idx = tree.query(Q, k=kk, workers=-1)

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

    def add_gaussians(self, gaussians, viewpoint_cam):
        if len(self.scene_info.key_points.get(f'{viewpoint_cam.image_name}.png')[0]) < self.feature_threshold:
            return
        prob_mask = self.create_density_map(viewpoint_cam.original_image)
        depth_mask = self.generate_depth_map(viewpoint_cam.original_image)
        if self.scene_info is not None:
            depth_mask = self.adjust_depth_map(viewpoint_cam, depth_mask)
        gaussian_batch = self.generate_gaussians(prob_mask, depth_mask, viewpoint_cam)

        gaussians.add_on_the_fly_gaussians(gaussian_batch)

        self.render_img(viewpoint_cam)

    def save(self):
        print("self.to_save_tmp.shape: ", self.to_save_tmp.shape)
        with open(os.path.expanduser("~/gaussian-splatting/pcd/world.npy"), "wb") as f:
            np.save(f, self.to_save_tmp)


    def t(self, D):
        return np.median(D)

    def s(self, D, tD):
        return np.mean(np.abs(D - tD))

    """
    from on the fly
    """
    def get_lapla_norm(self, img, kernel):
        laplacian_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda", dtype=torch.float32
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
        laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
        laplacian_norm = torch.linalg.vector_norm(laplacian, ord=1, dim=1, keepdim=True)
        laplacian_norm[..., :, 0] = 0
        laplacian_norm[..., :, -1] = 0
        laplacian_norm[..., 0, :] = 0
        laplacian_norm[..., -1, :] = 0
        return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)

    @torch.no_grad()
    def render_img(self, viewpoint_cam):
        gaussian_model = GaussianModel(self.dataset.sh_degree, render_type=self.args.type)
        gaussian_model.training_setup(self.opt)

        cnt = 0
        for cam_neighbour in self.neighbourhood[viewpoint_cam.image_name]:
            if cam_neighbour in self.gaussian_batches:
                gaussian_model.add_on_the_fly_gaussians(self.gaussian_batches[cam_neighbour])
                cnt += 1

        if cnt == 0:
            return torch.zeros(viewpoint_cam.original_image.shape)

        render_pkg = self.render_fn(viewpoint_cam, gaussian_model, self.pipe, torch.tensor(self.bg, dtype=torch.float32, device="cuda"),
                               opt=self.opt, is_training=False, dict_params=self.pbr_kwargs, iteration=-1)

        return render_pkg["render"]

    def init_neighbourhood(self, train_cameras):
        for cam1 in train_cameras:
            self.neighbourhood[cam1.image_name] = []
            for cam2 in train_cameras:
                if cam1.image_name == cam2.image_name:
                    continue
                prim_axis1 = cam1.get_primary_axis().detach().cpu().numpy() / np.linalg.norm(cam1.get_primary_axis().detach().cpu().numpy())
                prim_axis2 = cam2.get_primary_axis().detach().cpu().numpy() / np.linalg.norm(cam2.get_primary_axis().detach().cpu().numpy())
                if np.dot(prim_axis1, prim_axis2) > np.cos(self.neighbourhood_angle):
                    self.neighbourhood[cam1.image_name].append(cam2.image_name)

        for key in self.neighbourhood.keys():
            print(f"{key}:")
            for val in self.neighbourhood[key]:
                print(val)


