import os
from typing import NamedTuple

import numpy as np
import torch

from simple_knn._C import distCUDA2
from tqdm import tqdm

from DepthEstimator import DepthEstimator
from MaskSampler import MaskSampler
from arguments import OnTheFlyParams
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.image_utils import psnr
from utils.loss_utils import ssim
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
    max_radii2D: torch.Tensor
    weights: torch.Tensor
    xyz_gradient_accum: torch.Tensor
    normal_gradient_accum: torch.Tensor
    denom: torch.Tensor

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
        self.dav2_target_width = otfp.dav2_target_width

        self.feature_sigma = otfp.feature_sigma  # blur radius in pixels
        self.feature_min_coverage = otfp.feature_min_coverage
        self.feature_gate_mode = otfp.feature_gate_mode
        # "multiply" or "hard"

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

        self.camera_processed = set()

        self.to_save_tmp = np.zeros((0, 3))
        self.colmap_to_save_tmp = np.zeros((0, 3))

        self.gaussian_batches = dict()

        self.render_fn = render_fn

        self.neighbourhood = dict()

        self.depth_estimator = DepthEstimator(otfp, scene_info, self.p3ids, self.xyzs)
        self.mask_sampler = MaskSampler(otfp, scene_info, width, height)

        self.cnt = 0

    def generate_mask(self, img):
        return img != self.bg


    @torch.no_grad()
    def generate_gaussians(self, depth_mask, prob_mask, sample_mask, image_mask, cam):
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

    def save_img(self, img_D, name, inv=False):
        img_D = ((img_D - img_D.min()) / (img_D.max() - img_D.min())) * 255
        if (inv):
            img_D = 255 - img_D
        image_D = Image.fromarray(img_D.astype(np.uint8))
        image_D.save(os.path.expanduser(f"~/gaussian-splatting/pcd/depth_maps/{name}_depth_map.png"))

    def add_gaussians(self, gaussians, viewpoint_cam):
        if len(self.scene_info.key_points.get(f'{viewpoint_cam.image_name}.png')[0]) < self.feature_threshold:
            return

        # generate masks
        depth_mask = self.depth_estimator.generate_depth_map(viewpoint_cam)
        prob_mask, sample_mask, image_mask, _feature_coverage = self.mask_sampler.generate_sample_mask(
            rendered_img=self.render_img(viewpoint_cam),
            viewpoint_cam=viewpoint_cam
        )

        gaussian_batch = self.generate_gaussians(
            depth_mask=depth_mask,
            prob_mask=prob_mask,
            sample_mask=sample_mask,
            image_mask=image_mask,
            cam=viewpoint_cam)

        gaussians.add_on_the_fly_gaussians(gaussian_batch)

    def save(self):
        print("self.to_save_tmp.shape: ", self.to_save_tmp.shape)
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
                               opt=self.opt, is_training=False, dict_params=self.pbr_kwargs, iteration=-1)

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
            dict_params=self.pbr_kwargs,
            iteration=-1,
        )

        render = render_pkg["render"]  # expected shape: [C, H, W]

        if save_path is not None:
            img = render.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # HWC
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(save_path)

        return render

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

    @torch.no_grad()
    def log_rendered_view_metrics_summary(
            self,
            global_step,
            split_name,
            cameras,
            gaussians,
            render_fn,
            pipe,
            background,
            opt,
            wandb_run=None,
            use_pbr_if_available=True,
            **pbr_kwargs,
    ):
        if cameras is None or len(cameras) == 0:
            print("duck")
            return {}

        psnr_values = []
        ssim_values = []

        for viewpoint in cameras:
            results = render_fn(
                viewpoint,
                gaussians,
                pipe,
                background,
                opt=opt,
                is_training=False,
                dict_params=pbr_kwargs,
            )

            if use_pbr_if_available and getattr(gaussians, "use_pbr", False) and "pbr" in results:
                pred = results["pbr"]
            else:
                pred = results["render"]

            pred = torch.clamp(pred, 0.0, 1.0)
            gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            psnr_val = psnr(pred, gt).mean().detach()
            ssim_val = ssim(pred, gt).mean().detach()

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

        psnr_tensor = torch.stack(psnr_values).float()
        ssim_tensor = torch.stack(ssim_values).float()

        metrics = {
            f"{split_name}/psnr_avg": psnr_tensor.mean().item(),
            f"{split_name}/psnr_max": psnr_tensor.max().item(),
            f"{split_name}/psnr_min": psnr_tensor.min().item(),
            f"{split_name}/psnr_median": psnr_tensor.median().item(),
            f"{split_name}/ssim_avg": ssim_tensor.mean().item(),
            f"{split_name}/ssim_max": ssim_tensor.max().item(),
            f"{split_name}/ssim_min": ssim_tensor.min().item(),
            f"{split_name}/ssim_median": ssim_tensor.median().item(),
        }

        if wandb_run is not None:
            print(f"Logging to wandb at step {global_step}: {metrics}")
            wandb_run.log(metrics, step=global_step)
            print("asdf3")
        print("asdf2")

        return metrics
