import torch
import wandb

from arguments import EvalParams
from utils.image_utils import psnr


class EvalManager:
    def __init__(self, ep: EvalParams, args, enabled: bool, name: str = "OnTheFly"):
        self.ep = ep
        self.args = args
        self.enabled = enabled
        self.wandb_run = None

        if self.enabled:
            self.wandb_run = wandb.init(
                entity="ldkuba-tu-wien",
                project="GS-Reconstruction-Acceleration",
                name=name,
            )

    def log(self, iteration, scene, gaussians, render_fn, pipe, background, opt=None, pbr_kwargs=None):
        """Render every train view, compute PSNR stats, and upload them to wandb."""
        if not self.enabled or self.wandb_run is None:
            return None

        if iteration % 10 != 0:
            return None

        train_cameras = scene.getTrainCameras()
        if not train_cameras:
            print(f"[ITER {iteration}] EvalManager: no train cameras available, skipping wandb eval log.")
            return None

        psnr_values = []
        render_kwargs = dict(pbr_kwargs or {})
        render_kwargs["iteration"] = iteration

        with torch.no_grad():
            for viewpoint in train_cameras:
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

        wandb.log(metrics, step=int(iteration))
        print(
            f"[ITER {iteration}] EvalManager train PSNR: "
            f"min {metrics['lowest_psnr']:.4f} "
            f"avg {metrics['average_psnr']:.4f} "
            f"max {metrics['highest_psnr']:.4f}"
        )
        return metrics

    def finish(self):
        if self.enabled and self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
