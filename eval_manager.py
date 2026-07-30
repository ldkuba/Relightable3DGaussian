import torch
import wandb
import json
import os
from pathlib import Path

from arguments import EvalParams
from utils.image_utils import psnr


class EvalManager:
    def __init__(self, ep: EvalParams, args, enabled: bool, name: str = "OnTheFly"):
        self.ep = ep
        self.args = args
        self.enabled = enabled
        self.wandb_run = None
        local_log_path = os.environ.get("R3DG_EVAL_LOG_PATH")
        self.local_log_path = Path(local_log_path) if local_log_path else None
        if self.local_log_path is not None:
            self.local_log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.enabled and self.local_log_path is None:
            self.wandb_run = wandb.init(
                entity="ldkuba-tu-wien",
                project="GS-Reconstruction-Acceleration",
                name=name,
            )

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
        render_kwargs = dict(pbr_kwargs or {})
        render_kwargs["iteration"] = iteration

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

        psnr_values = torch.stack(psnr_values)
        metrics = {
            "lowest_psnr": psnr_values.min().item(),
            "highest_psnr": psnr_values.max().item(),
            "average_psnr": psnr_values.mean().item(),
            "iteration": int(iteration),
        }

        if self.local_log_path is not None:
            with self.local_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        elif self.wandb_run is not None:
            wandb.log(metrics, step=int(iteration))
        print(
            f"[ITER {iteration}] EvalManager {split_name} PSNR: "
            f"min {metrics['lowest_psnr']:.4f} "
            f"avg {metrics['average_psnr']:.4f} "
            f"max {metrics['highest_psnr']:.4f}"
        )
        return metrics

    def finish(self):
        if self.enabled and self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
