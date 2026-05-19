import os
import math
import json
import csv
import time
import subprocess
from typing import NamedTuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from simple_knn._C import distCUDA2

from DepthEstimator import DepthEstimator
from MaskSampler import MaskSampler
from arguments import OnTheFlyParams
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.sh_utils import RGB2SH
from lpipsPyTorch import lpips

from PIL import Image

class GaussianBatch(NamedTuple):
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    normals: torch.Tensor
    shs_dc: torch.Tensor
    shs_rest: torch.Tensor

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
        self.apply_penalty_map = otfp.apply_penalty_map

        self.feature_sigma = otfp.feature_sigma  # blur radius in pixels
        self.feature_min_coverage = otfp.feature_min_coverage
        self.feature_gate_mode = otfp.feature_gate_mode
        # "multiply" or "hard"
        self.eval_split = otfp.eval_split
        self.eval_enabled = bool(otfp.eval_enabled)
        self.eval_profile_path = otfp.eval_profile_path
        self.eval_output_dir = otfp.eval_output_dir
        self.eval_run_id = otfp.eval_run_id
        self.eval_dataset_id = otfp.eval_dataset_id
        self.eval_scene = otfp.eval_scene
        self.eval_checkpoint = otfp.eval_checkpoint
        self.eval_git_commit = otfp.eval_git_commit
        self.eval_save_pic_x_iter_override = otfp.eval_save_pic_x_iter

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
        self.key_points_torch = self._move_key_points_to_device(scene_info.key_points)

        self.camera_processed = set()

        self.to_save_tmp = np.zeros((0, 3))
        self.colmap_to_save_tmp = np.zeros((0, 3))

        self.gaussian_batches = dict()

        self.render_fn = render_fn

        self.neighbourhood = dict()

        self.depth_estimator = DepthEstimator(otfp, scene_info, self.p3ids, self.xyzs, self.key_points_torch)
        self.mask_sampler = MaskSampler(otfp, scene_info, width, height, self.key_points_torch)

        self.eval_profile = self._load_eval_profile()
        local_profile = self.eval_profile.get("local", {})
        self.eval_git_commit_value = self._default_git_commit()
        self.eval_raw_csv_path = None
        self.eval_save_pic_x_iter = int(local_profile.get("save_pic_x_iter", -1))

        if self.eval_save_pic_x_iter_override and self.eval_save_pic_x_iter_override > 0:
            self.eval_save_pic_x_iter = self.eval_save_pic_x_iter_override

        self.local_eval_enabled = self.eval_enabled and bool(local_profile.get("enabled", False))
        self.local_time_enabled = self.local_eval_enabled and bool(local_profile.get("time", False))
        wandb_profile = self.eval_profile.get("wandb", {})
        self.wandb_time_enabled = bool(wandb_profile.get("enabled", False)) and bool(wandb_profile.get("time", False))
        self.eval_run_dir = None
        self.eval_render_dir = None

        if self.local_eval_enabled:
            self._prepare_eval_output()
            self._write_eval_metadata()

        print(self.to_string())

    def _load_eval_profile(self):
        profile = {
            "wandb": {
                "enabled": False,
                "psnr_avg": False,
                "ssim_avg": False,
                "lpips_avg": False,
                "l1_avg": False,
                "time": False,
            },
            "local": {
                "enabled": False,
                "psnr_uniq": False,
                "ssim_uniq": False,
                "lpips_uniq": False,
                "l1_uniq": False,
                "time": False,
                "save_pic_x_iter": -1,
            },
        }
        if self.eval_profile_path and os.path.exists(self.eval_profile_path):
            with open(self.eval_profile_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                if isinstance(loaded.get("wandb"), dict) or isinstance(loaded.get("local"), dict):
                    if isinstance(loaded.get("wandb"), dict):
                        profile["wandb"].update(loaded["wandb"])
                    if isinstance(loaded.get("local"), dict):
                        profile["local"].update(loaded["local"])
                else:
                    # Backward compatibility with flat profiles.
                    profile["wandb"]["enabled"] = bool(loaded.get("wandb", False))
                    for metric in ["psnr", "ssim", "lpips", "l1"]:
                        profile["wandb"][f"{metric}_avg"] = bool(loaded.get(f"{metric}_avg", False))
                        profile["local"][f"{metric}_uniq"] = bool(loaded.get(f"{metric}_uniq", False))
                    profile["wandb"]["time"] = bool(loaded.get("time", False))
                    profile["local"]["time"] = bool(loaded.get("time", False))
                    profile["local"]["save_pic_x_iter"] = int(loaded.get("save_pic_x_iter", -1))

        if self.eval_enabled:
            profile["local"]["enabled"] = bool(profile["local"].get("enabled", False))
        return profile

    def _default_git_commit(self):
        if self.eval_git_commit:
            return self.eval_git_commit
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip()
        except Exception:
            return ""

    def _prepare_eval_output(self):
        run_id = self.eval_run_id or os.path.basename(self.args.model_path.rstrip("/"))
        root_dir = self.eval_output_dir or os.path.join(self.args.model_path, "eval")
        self.eval_run_dir = os.path.join(root_dir, "on_the_fly", run_id)
        self.eval_render_dir = os.path.join(self.eval_run_dir, "renders")
        metrics_dir = os.path.join(self.eval_run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        self.eval_raw_csv_path = os.path.join(metrics_dir, "raw_metrics.csv")
        if not os.path.exists(self.eval_raw_csv_path):
            with open(self.eval_raw_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_id",
                        "dataset_id",
                        "scene",
                        "split",
                        "camera_name",
                        "iteration",
                        "time_kind",
                        "psnr",
                        "ssim",
                        "lpips",
                        "l1",
                        "initial_spawn_time_ms",
                        "iteration_time_ms",
                        "num_gaussians",
                        "cuda_memory_allocated_mb",
                        "cuda_memory_reserved_mb",
                        "render_type",
                        "use_pbr",
                        "checkpoint",
                        "git_commit",
                        "render_path",
                    ]
                )

    def _write_eval_metadata(self):
        if not self.eval_run_dir:
            return
        metadata_path = os.path.join(self.eval_run_dir, "metadata.json")
        metadata = {
            "run_id": self.eval_run_id or os.path.basename(self.args.model_path.rstrip("/")),
            "dataset_id": self.eval_dataset_id,
            "scene": self.eval_scene,
            "split": self.eval_split,
            "model_path": self.args.model_path,
            "eval_profile_path": self.eval_profile_path,
            "eval_profile": self.eval_profile,
            "git_commit": self.eval_git_commit_value,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def should_log_eval(self, global_step, wandb_run=None):
        wandb_profile = self.eval_profile.get("wandb", {})
        local_profile = self.eval_profile.get("local", {})

        wandb_requested = (
            wandb_run is not None
            and bool(wandb_profile.get("enabled", False))
            and (
                bool(wandb_profile.get("psnr_avg", False))
                or bool(wandb_profile.get("ssim_avg", False))
                or bool(wandb_profile.get("lpips_avg", False))
                or bool(wandb_profile.get("l1_avg", False))
            )
        )

        local_requested = (
            self.eval_enabled
            and bool(local_profile.get("enabled", False))
            and (
                bool(local_profile.get("psnr_uniq", False))
                or bool(local_profile.get("ssim_uniq", False))
                or bool(local_profile.get("lpips_uniq", False))
                or bool(local_profile.get("l1_uniq", False))
                or int(local_profile.get("save_pic_x_iter", -1)) > 0
            )
        )
        return wandb_requested or local_requested

    def _should_save_render_for_step(self, global_step):
        if self.eval_save_pic_x_iter <= 0:
            return False
        return global_step % self.eval_save_pic_x_iter == 0

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

        gaussian_batch = GaussianBatch(
            means=world_points, scales=scales, rotations=rots, normals=normals, shs_dc=shs_dc, shs_rest=shs_rest,
            opacities=opacities)

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

    def add_gaussians(self, gaussians, viewpoint_cam):
        key_points = self.key_points_torch.get(f"{viewpoint_cam.image_name}.png")
        if key_points is None or key_points[0].numel() < self.feature_threshold:
            return

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

        gaussians.add_on_the_fly_gaussians(gaussian_batch)

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

    @torch.no_grad()
    def log_rendered_view_metrics_summary(
            self,
            global_step,
            split_name,
            cameras,
            current_viewpoint,
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
            print("[on_the_fly] Can't log rendered view metrics, since no camera views are provided.")
            return {}

        profile = self.eval_profile
        wandb_profile = profile.get("wandb", {})
        local_profile = profile.get("local", {})

        wandb_enabled = wandb_run is not None and bool(wandb_profile.get("enabled", False))
        local_enabled = self.eval_enabled and bool(local_profile.get("enabled", False))

        local_psnr_uniq = local_enabled and bool(local_profile.get("psnr_uniq", False))
        local_ssim_uniq = local_enabled and bool(local_profile.get("ssim_uniq", False))
        local_lpips_uniq = local_enabled and bool(local_profile.get("lpips_uniq", False))
        local_l1_uniq = local_enabled and bool(local_profile.get("l1_uniq", False))
        local_time = self.local_time_enabled
        local_save_pic_x_iter = int(local_profile.get("save_pic_x_iter", -1))

        wandb_psnr_avg = wandb_enabled and bool(wandb_profile.get("psnr_avg", False))
        wandb_ssim_avg = wandb_enabled and bool(wandb_profile.get("ssim_avg", False))
        wandb_lpips_avg = wandb_enabled and bool(wandb_profile.get("lpips_avg", False))
        wandb_l1_avg = wandb_enabled and bool(wandb_profile.get("l1_avg", False))

        need_psnr = local_psnr_uniq or wandb_psnr_avg
        need_ssim = local_ssim_uniq or wandb_ssim_avg
        need_lpips = local_lpips_uniq or wandb_lpips_avg
        need_l1 = local_l1_uniq or wandb_l1_avg

        if local_save_pic_x_iter > 0:
            self.eval_save_pic_x_iter = local_save_pic_x_iter
        should_save_render = local_enabled and self._should_save_render_for_step(global_step)

        if not (need_psnr or need_ssim or need_lpips or need_l1 or should_save_render):
            return {}

        psnr_values = []
        ssim_values = []
        lpips_values = []
        l1_values = []
        rows = []
        target_save_camera_name = None
        if current_viewpoint is not None:
            target_save_camera_name = getattr(current_viewpoint, "image_name", None)

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

            psnr_val = psnr(pred, gt).mean().detach() if need_psnr else None
            ssim_val = ssim(pred, gt).mean().detach() if need_ssim else None
            lpips_val = lpips(pred, gt, net_type="vgg").mean().detach() if need_lpips else None
            l1_val = F.l1_loss(pred, gt).mean().detach() if need_l1 else None

            if psnr_val is not None:
                psnr_values.append(psnr_val)
            if ssim_val is not None:
                ssim_values.append(ssim_val)
            if lpips_val is not None:
                lpips_values.append(lpips_val)
            if l1_val is not None:
                l1_values.append(l1_val)

            render_path = ""
            if (
                should_save_render
                and self.eval_render_dir is not None
                and target_save_camera_name is not None
                and viewpoint.image_name == target_save_camera_name
            ):
                iter_dir = os.path.join(self.eval_render_dir, f"iter_{global_step:06d}")
                os.makedirs(iter_dir, exist_ok=True)
                render_path = os.path.join(iter_dir, f"{viewpoint.image_name}.png")
                img = pred.detach().permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                Image.fromarray(img).save(render_path)

            if local_enabled and self.eval_raw_csv_path:
                if torch.cuda.is_available():
                    cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    cuda_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                else:
                    cuda_alloc_mb = ""
                    cuda_reserved_mb = ""
                rows.append(
                    {
                        "run_id": self.eval_run_id or os.path.basename(self.args.model_path.rstrip("/")),
                        "dataset_id": self.eval_dataset_id,
                        "scene": self.eval_scene,
                        "split": split_name or self.eval_split,
                        "camera_name": viewpoint.image_name,
                        "iteration": global_step,
                        "time_kind": "",
                        "psnr": psnr_val.item() if local_psnr_uniq and psnr_val is not None else "",
                        "ssim": ssim_val.item() if local_ssim_uniq and ssim_val is not None else "",
                        "lpips": lpips_val.item() if local_lpips_uniq and lpips_val is not None else "",
                        "l1": l1_val.item() if local_l1_uniq and l1_val is not None else "",
                        "initial_spawn_time_ms": "",
                        "iteration_time_ms": "",
                        "num_gaussians": int(gaussians.get_xyz.shape[0]) if hasattr(gaussians, "get_xyz") else "",
                        "cuda_memory_allocated_mb": cuda_alloc_mb if local_time else "",
                        "cuda_memory_reserved_mb": cuda_reserved_mb if local_time else "",
                        "render_type": self.args.type,
                        "use_pbr": bool(getattr(gaussians, "use_pbr", False)),
                        "checkpoint": self.eval_checkpoint or (self.args.checkpoint or ""),
                        "git_commit": self.eval_git_commit_value,
                        "render_path": render_path,
                    }
                )

        if rows and self.eval_raw_csv_path:
            with open(self.eval_raw_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writerows(rows)

        metrics = {}
        if psnr_values and wandb_psnr_avg:
            psnr_tensor = torch.stack(psnr_values).float()
            metrics[f"{split_name}/psnr_avg"] = psnr_tensor.mean().item()
        if ssim_values and wandb_ssim_avg:
            ssim_tensor = torch.stack(ssim_values).float()
            metrics[f"{split_name}/ssim_avg"] = ssim_tensor.mean().item()
        if lpips_values and wandb_lpips_avg:
            lpips_tensor = torch.stack(lpips_values).float()
            metrics[f"{split_name}/lpips_avg"] = lpips_tensor.mean().item()
        if l1_values and wandb_l1_avg:
            l1_tensor = torch.stack(l1_values).float()
            metrics[f"{split_name}/l1_avg"] = l1_tensor.mean().item()

        should_log_wandb = wandb_enabled
        if should_log_wandb and len(metrics) > 0:
            wandb_run.log(metrics, step=global_step)

        return metrics

    def _append_time_row(self, row):
        if not self.local_time_enabled or not self.eval_raw_csv_path:
            return
        with open(self.eval_raw_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def log_initial_spawn_timing(self, spawn_index, camera_name, duration_ms, num_gaussians, wandb_run=None):
        if self.local_time_enabled:
            if torch.cuda.is_available():
                cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            else:
                cuda_alloc_mb = ""
                cuda_reserved_mb = ""
            row = {
                "run_id": self.eval_run_id or os.path.basename(self.args.model_path.rstrip("/")),
                "dataset_id": self.eval_dataset_id,
                "scene": self.eval_scene,
                "split": self.eval_split,
                "camera_name": camera_name,
                "iteration": spawn_index,
                "time_kind": "initial_spawn",
                "psnr": "",
                "ssim": "",
                "lpips": "",
                "l1": "",
                "initial_spawn_time_ms": duration_ms,
                "iteration_time_ms": "",
                "num_gaussians": int(num_gaussians),
                "cuda_memory_allocated_mb": cuda_alloc_mb,
                "cuda_memory_reserved_mb": cuda_reserved_mb,
                "render_type": self.args.type,
                "use_pbr": "",
                "checkpoint": self.eval_checkpoint or (self.args.checkpoint or ""),
                "git_commit": self.eval_git_commit_value,
                "render_path": "",
            }
            self._append_time_row(row)

        if self.wandb_time_enabled and wandb_run is not None:
            wandb_run.log({"timing/initial_spawn_time_ms": duration_ms}, step=spawn_index)

    def log_iteration_timing(self, iteration, duration_ms, num_gaussians, wandb_run=None):
        if self.local_time_enabled:
            if torch.cuda.is_available():
                cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            else:
                cuda_alloc_mb = ""
                cuda_reserved_mb = ""
            row = {
                "run_id": self.eval_run_id or os.path.basename(self.args.model_path.rstrip("/")),
                "dataset_id": self.eval_dataset_id,
                "scene": self.eval_scene,
                "split": self.eval_split,
                "camera_name": "",
                "iteration": iteration,
                "time_kind": "train_iteration",
                "psnr": "",
                "ssim": "",
                "lpips": "",
                "l1": "",
                "initial_spawn_time_ms": "",
                "iteration_time_ms": duration_ms,
                "num_gaussians": int(num_gaussians),
                "cuda_memory_allocated_mb": cuda_alloc_mb,
                "cuda_memory_reserved_mb": cuda_reserved_mb,
                "render_type": self.args.type,
                "use_pbr": "",
                "checkpoint": self.eval_checkpoint or (self.args.checkpoint or ""),
                "git_commit": self.eval_git_commit_value,
                "render_path": "",
            }
            self._append_time_row(row)

        if self.wandb_time_enabled and wandb_run is not None:
            wandb_run.log({"timing/iteration_time_ms": duration_ms}, step=iteration)

    def free_gpu_mem(self):
        self.gaussian_batches.clear()
        del self.depth_estimator.model
        self.depth_estimator.model = None
        torch.cuda.empty_cache()

    def to_string(self):
        return (
            f"OnTheFly[device={self.DEVICE}, width={self.width}, height={self.height}, "
            f"max_sh_degree={self.max_sh_degree}, bg={self.bg}, knn_p={self.knn_p}, knn_n={self.knn_n}, "
            f"knn_stride={self.knn_stride}, knn_epsilon={self.knn_epsilon}, "
            f"error_threshold={self.error_threshold}, feature_threshold={self.feature_threshold}, "
            f"base_prob={self.base_prob}, normalize_prob={self.normalize_prob}, "
            f"apply_penalty_map={self.apply_penalty_map}, neighbourhood_angle={self.neighbourhood_angle}, "
            f"dav2_target_width={self.dav2_target_width}, feature_sigma={self.feature_sigma}, "
            f"feature_min_coverage={self.feature_min_coverage}, feature_gate_mode={self.feature_gate_mode}, "
            f"eval_enabled={self.eval_enabled}, local_eval_enabled={self.local_eval_enabled}, eval_split={self.eval_split}]"
        )
