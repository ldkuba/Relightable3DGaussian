import os
import math
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch

from simple_knn._C import distCUDA2

from DepthEstimator import DepthEstimator
from MaskSampler import MaskSampler
from arguments import OnTheFlyParams, EvalParams
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH

from PIL import Image

class GaussianBatch(NamedTuple):
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    normals: torch.Tensor
    shs_dc: torch.Tensor
    shs_rest: torch.Tensor


def save_gaussian_batch(path, gaussian_batch: GaussianBatch):
    torch.save(gaussian_batch._asdict(), path)


def load_gaussian_batch(path) -> GaussianBatch:
    data = torch.load(path, weights_only=False)
    return GaussianBatch(**data)

class OnTheFly:

    def __init__(self, width, height, max_sh_degree, otfp: OnTheFlyParams, dataset, args,
                 render_fn, pipe, opt, scene_info = None, bg = 0.0):

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
        self.dav2_target_width = otfp.dav2_target_width
        self.apply_penalty_map = otfp.apply_penalty_map

        self.feature_sigma = otfp.feature_sigma  # blur radius in pixels
        self.feature_min_coverage = otfp.feature_min_coverage
        self.feature_gate_mode = otfp.feature_gate_mode
        self.position_lr_scale_factor = otfp.position_lr_scale_factor
        self.adjust_by_median = otfp.adjust_by_median
        self.adjust_by_scipy_cKDTree = otfp.adjust_by_scipy_cKDTree
        self.adjust_by_pytorch3d_knn_points = otfp.adjust_by_pytorch3d_knn_points
        # "multiply" or "hard"

        self.dataset = dataset
        self.args = args
        self.pipe = pipe
        self.opt = opt

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
        self.key_points_torch = self._move_key_points_to_device(scene_info.key_points)

        self.camera_processed = set()

        self.to_save_tmp = np.zeros((0, 3))
        self.colmap_to_save_tmp = np.zeros((0, 3))

        self.spawned_batches: List[GaussianBatch] = []
        self.gaussian_batches: Dict[str, GaussianBatch] = {}

        self.render_fn = render_fn

        self.neighbourhood = dict()

        self.depth_estimator = DepthEstimator(otfp, scene_info, self.p3ids, self.xyzs, self.key_points_torch)
        self.mask_sampler = MaskSampler(otfp, scene_info, width, height, self.key_points_torch)


    def _move_key_points_to_device(self, key_points):
        key_points_torch = {}
        for image_name, value in key_points.items():
            if value is None:
                continue
            key_ids, key_uv = value
            key_points_torch[image_name] = (
                torch.as_tensor(key_ids, device=self.DEVICE, dtype=torch.long),
                torch.as_tensor(key_uv, device=self.DEVICE, dtype=torch.float32),
            )
        return key_points_torch

    def generate_mask(self, img):
        return img != self.bg

    @torch.no_grad()
    def generate_gaussians(self, depth_mask, prob_mask, sample_mask, image_mask, cam):
        device = depth_mask.device

        # ++ means ++
        c2w = torch.linalg.inv(cam.extrinsics)
        c2w = c2w.T

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

        # put reconstruction into world space
        homo_ones = torch.ones((cam_world.shape[0], 1), device=device, dtype=cam_world.dtype)
        cam_world = torch.concatenate((cam_world, homo_ones), dim=1)

        world_points = cam_world @ c2w
        world_points = world_points[:,:3]
        assert ~ torch.isnan(cam_world).any()
        assert ~ torch.isnan(world_points).any()

        # ++ scales ++
        filtered_prob_mask = prob_mask[sample_mask]
        scales = 1.0 / (2 * torch.sqrt(filtered_prob_mask)).unsqueeze(1)
        f = cam.image_width / (2 * np.tan(cam.FoVx / 2))
        scales = (depth_mask[sample_mask].unsqueeze(dim=1).to(dtype=torch.float32) * scales) / f
        scales = torch.log(scales)
        scales = scales.repeat(1, 3)

        # ++ rotations ++
        rots = torch.zeros((world_points.shape[0], 4), device=device, dtype=torch.float32)
        rots[:, 0] = 1

        # ++ opacities ++
        opacities = inverse_sigmoid(0.1 * torch.ones((world_points.shape[0], 1), dtype=torch.float32, device=device))
        prim_axis = -cam.get_primary_axis()

        # ++ normals ++
        normals = prim_axis.unsqueeze(0).repeat(world_points.shape[0], 1)

        img = cam.original_image

        # ++ base colors ++
        base_color = img.permute(1, 2, 0)[sample_mask]

        # ++ shs ++
        shs = torch.zeros(
            (base_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32,
            device=device,
        )
        shs[:, :3, 0] = RGB2SH(base_color)
        shs[:, 3:, 1:] = 0.0

        shs_dc = shs[:, :, 0:1].transpose(1, 2)
        shs_rest = shs[:, :, 1:].transpose(1, 2)

        gaussian_batch = GaussianBatch(
            means=world_points, scales=scales, rotations=rots, normals=normals, shs_dc=shs_dc, shs_rest=shs_rest,
            opacities=opacities)

        self.spawned_batches.append(gaussian_batch)
        if self.apply_penalty_map:
            self.gaussian_batches[cam.image_name] = gaussian_batch

        return gaussian_batch

    def check_and_set_cam(self, uid):
        if uid in self.camera_processed:
            return False
        self.camera_processed.add(uid)
        return True

    def save_img(self, img_D, name, inv=False):
        print(f"[on_the_fly] Saving depth map image {name}")
        img_D = ((img_D - img_D.min()) / (img_D.max() - img_D.min())) * 255
        if (inv):
            img_D = 255 - img_D
        image_D = Image.fromarray(img_D.astype(np.uint8))
        image_D.save(os.path.expanduser(f"~/gaussian-splatting/pcd/depth_maps/{name}_depth_map.png"))

    def merge_gaussian_batches(self) -> Optional[GaussianBatch]:
        if len(self.spawned_batches) == 0:
            return None
        batches = self.spawned_batches

        return GaussianBatch(
            means=torch.cat([batch.means for batch in batches], dim=0),
            scales=torch.cat([batch.scales for batch in batches], dim=0),
            rotations=torch.cat([batch.rotations for batch in batches], dim=0),
            opacities=torch.cat([batch.opacities for batch in batches], dim=0),
            normals=torch.cat([batch.normals for batch in batches], dim=0),
            shs_dc=torch.cat([batch.shs_dc for batch in batches], dim=0),
            shs_rest=torch.cat([batch.shs_rest for batch in batches], dim=0),
        )

    def add_gaussians(self, viewpoint_cam):
        key_points = self.key_points_torch.get(f"{viewpoint_cam.image_name}.png")
        if key_points is None or key_points[0].numel() < self.feature_threshold:
            return None

        # generate masks
        depth_mask = self.depth_estimator.generate_depth_map(viewpoint_cam)
        rendered_img = self.render_img(viewpoint_cam) if self.apply_penalty_map else None
        prob_mask, sample_mask, image_mask, _feature_coverage = self.mask_sampler.generate_sample_mask(
            viewpoint_cam=viewpoint_cam,
            rendered_img=rendered_img,
        )

        gaussian_batch = self.generate_gaussians(
            depth_mask=depth_mask,
            prob_mask=prob_mask,
            sample_mask=sample_mask,
            image_mask=image_mask,
            cam=viewpoint_cam)

        return gaussian_batch

    def save(self):
        print("[on_the_fly] save means of generated gaussians - self.to_save_tmp.shape: ", self.to_save_tmp.shape)
        with open(os.path.expanduser("~/gaussian-splatting/pcd/world.npy"), "wb") as f:
            np.save(f, self.to_save_tmp)


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
                               opt=self.opt, is_training=False, iteration=-1)

        return render_pkg["render"]

    @torch.no_grad()
    def render_single_img(self, viewpoint_cam, gaussian_batch, save_path=None):
        gaussian_model = GaussianModel(self.dataset.sh_degree, render_type=self.args.type)
        gaussian_model.training_setup(self.opt)

        gaussian_model.add_on_the_fly_gaussians(gaussian_batch)

        render_pkg = self.render_fn(
            viewpoint_cam,
            gaussian_model,
            self.pipe,
            torch.tensor(self.bg, dtype=torch.float32, device="cuda"),
            opt=self.opt,
            is_training=False,
            iteration=-1,
        )

        render = render_pkg["render"]  # expected shape: [C, H, W]

        if save_path is not None:
            img = render.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # HWC
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(save_path)

        return render

    def init_neighbourhood(self, train_cameras):
        if not self.apply_penalty_map:
            return

        if len(train_cameras) == 0:
            return

        image_names = [cam.image_name for cam in train_cameras]
        primary_axes = torch.stack(
            [cam.get_primary_axis().detach() for cam in train_cameras], dim=0
        )
        primary_axes = torch.nn.functional.normalize(primary_axes, p=2, dim=1)

        cosine_matrix = primary_axes @ primary_axes.T
        neighbour_mask = (cosine_matrix > math.cos(self.neighbourhood_angle)).cpu().numpy()

        for i, image_name in enumerate(image_names):
            self.neighbourhood[image_name] = [
                image_names[j]
                for j in range(len(image_names))
                if j != i and neighbour_mask[i, j]
            ]

    def free_gpu_mem(self):
        self.spawned_batches.clear()
        self.gaussian_batches.clear()
        del self.depth_estimator.model
        self.depth_estimator.model = None
        torch.cuda.empty_cache()

    def to_string(self):
        return (
            "OnTheFly[\n"
            f"  device={self.DEVICE}, width={self.width}, height={self.height}, "
            f"max_sh_degree={self.max_sh_degree}, bg={self.bg}, "
            f"error_threshold={self.error_threshold}, feature_threshold={self.feature_threshold}, "
            f"neighbourhood_angle={self.neighbourhood_angle}, "
            f"position_lr_scale_factor={self.position_lr_scale_factor}, "
            f"sfm_points={len(self.xyzs)}, keypoint_images={len(self.key_points_torch)}\n"
            f"  depth_estimator={self.depth_estimator.to_string()}\n"
            f"  mask_sampler={self.mask_sampler.to_string()}\n"
            "]"
        )
