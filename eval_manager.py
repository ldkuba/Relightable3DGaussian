import torch
import wandb
import json
import os
import time
from pathlib import Path

from arguments import EvalParams
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


class EvalManager:
    def __init__(self, ep: EvalParams, args, enabled: bool, name: str = "OnTheFly"):
        self.ep = ep
        self.args = args
        self.enabled = enabled
        self.wandb_run = None
        local_log_path = os.environ.get("R3DG_EVAL_LOG_PATH")
        self.local_log_path = Path(local_log_path) if local_log_path else None
        self.initialization_time_sec = 0.0
        self.training_time_sec = 0.0
        self._tracked_since: float | None = None
        self._initialization_logged = False
        if self.local_log_path is not None:
            self.local_log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.enabled and self.local_log_path is None:
            self.wandb_run = wandb.init(
                entity="ldkuba-tu-wien",
                project="GS-Reconstruction-Acceleration",
                name=name,
            )

    def track_time(self):
        if self._tracked_since is None:
            self._tracked_since = time.perf_counter()

    def pause_time(self):
        if self._tracked_since is None:
            return
        self.training_time_sec += time.perf_counter() - self._tracked_since
        self._tracked_since = None

    def set_initialization_time(self, initialization_time_sec: float, num_gaussians: int | None = None):
        self.initialization_time_sec = initialization_time_sec
        if self._initialization_logged:
            return
        self._initialization_logged = True
        payload = {
            "event": "initialization",
            "iteration": 0,
            "initialization_time_sec": self.initialization_time_sec,
            "training_time_sec": self.training_time_sec,
            "cumulative_time_sec": self.initialization_time_sec + self.training_time_sec,
        }
        if num_gaussians is not None:
            payload["num_gaussians"] = int(num_gaussians)
        self._write_log(payload, step=0)

    def _write_log(self, payload: dict, step: int):
        if self.local_log_path is not None:
            with self.local_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        elif self.wandb_run is not None:
            wandb.log(payload, step=step)

    def log(self, iteration, scene, gaussians, render_fn, pipe, background, opt=None, pbr_kwargs=None):
        """Render every train view, compute PSNR stats, and upload tdhem to wandb."""
        if not self.enabled and self.local_log_path is None:
            return None

        if iteration <= 10_000 and iteration % 20 != 0:
            return None
        elif iteration > 10_000 and iteration % 500 != 0:
            return None

        eval_cameras = scene.getTestCameras()
        split_name = "test"
        if not eval_cameras:
            eval_cameras = scene.getTrainCameras()
            split_name = "train"
        if not eval_cameras:
            print(f"[ITER {iteration}] EvalManager: no cameras available, skipping eval log.")
            return None

        psnr_values = []
        ssim_values = []
        lpips_values = []
        render_kwargs = dict(pbr_kwargs or {})
        render_kwargs["iteration"] = iteration

        self.pause_time()
        try:
            with torch.no_grad():
                for viewpoint in eval_cameras:
                    render_pkg = render_fn(
                        viewpoint,
                        gaussians,
                        pipe,
                        background,
                        opt=opt,
                        is_training=False,
                        dict_params=render_kwargs,
                        iteration=iteration,
                    )

                    if getattr(gaussians, "use_pbr", False) and "pbr" in render_pkg:
                        image = render_pkg["pbr"]
                    else:
                        image = render_pkg["render"]

                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)
                    psnr_values.append(psnr(image, gt_image).mean().detach().double().cpu())
                    ssim_values.append(ssim(image, gt_image).mean().detach().double().cpu())
                    lpips_values.append(lpips(image, gt_image, net_type='vgg').mean().detach().double().cpu())
        finally:
            self.track_time()

        psnr_values = torch.stack(psnr_values)
        ssim_values = torch.stack(ssim_values)
        lpips_values = torch.stack(lpips_values)
        metrics = {
            "lowest_psnr": psnr_values.min().item(),
            "highest_psnr": psnr_values.max().item(),
            "average_psnr": psnr_values.mean().item(),
            "lowest_ssim": ssim_values.min().item(),
            "highest_ssim": ssim_values.max().item(),
            "average_ssim": ssim_values.mean().item(),
            "lowest_lpips": lpips_values.min().item(),
            "highest_lpips": lpips_values.max().item(),
            "average_lpips": lpips_values.mean().item(),
            "iteration": int(iteration),
            "num_gaussians": int(gaussians.get_xyz.shape[0]),
            "training_time_sec": self.training_time_sec,
            "cumulative_time_sec": self.initialization_time_sec + self.training_time_sec,
        }

        self._write_log(metrics, step=int(iteration))
        print(
            f"[ITER {iteration}] EvalManager {split_name}: "
            f"gaussians {metrics['num_gaussians']} | "
            f"PSNR min {metrics['lowest_psnr']:.4f} "
            f"avg {metrics['average_psnr']:.4f} "
            f"max {metrics['highest_psnr']:.4f} "            
            f"| SSIM min {metrics['lowest_ssim']:.4f} "
            f"avg {metrics['average_ssim']:.4f} "
            f"max {metrics['highest_ssim']:.4f} "
            f"| LPIPS min {metrics['lowest_lpips']:.4f} "
            f"avg {metrics['average_lpips']:.4f} "
            f"max {metrics['highest_lpips']:.4f}"
        )
        return metrics

    def finish(self):
        self.pause_time()
        if self.enabled and self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
