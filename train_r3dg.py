from dataclasses import dataclass
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint

from wrapt import patches
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from gui import GUI
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from scene.utils import save_render_orb, save_depth_orb, save_normal_orb, save_albedo_orb, save_roughness_orb

import wandb

import visualization.visualize_covariance as vis_cov
import visualization.visualize_everything as vis

import time

def process_gaussian_images(gaussian_renders, opacity_threshold=0.95, depth_variance_threshold=0.05, hit_mask=False, output_names=[]):

    combined_mask = torch.ones((gaussian_renders['render'].shape[1], gaussian_renders['render'].shape[2]), dtype=torch.bool, device=gaussian_renders['render'].device)

    # Filter by opacity
    if opacity_threshold > 0:
        opacity_gaussian = gaussian_renders['opacity']
        opacity_max = opacity_gaussian.max().item()
        opacity_mask = opacity_gaussian.squeeze(0) > (opacity_max * opacity_threshold)
        combined_mask = combined_mask & opacity_mask

    # Filter by depth variance
    # TODO: Detect gaps in ray instead of just filtering out high variance pixels, which can be caused by steep edges and not just gaps
    if depth_variance_threshold > 0:
        depth_variance = gaussian_renders['depth_var']
        depth_variance_mean = depth_variance.mean().item()
        depth_variance_mask = (depth_variance.squeeze(0) < depth_variance_threshold) & (depth_variance.squeeze(0) < depth_variance_mean * 10.0)
        combined_mask = combined_mask & depth_variance_mask

    if hit_mask:
        depth = gaussian_renders['depth']
        hit_mask = depth.squeeze(0) > 0
        combined_mask = combined_mask & hit_mask

    output_images = {}
    for name in output_names:
        output_images[name] = gaussian_renders[name].clone()

    return output_images, combined_mask

@dataclass
class DepthConsistencyPoints:
    view_point: torch.Tensor
    view_dir: torch.Tensor
    points: torch.Tensor
    normals: torch.Tensor
    data: torch.Tensor = None

def filter_variance(gaussian_image, patch_size, variance_threshold = 0, patch_filter=None):
    num_kernel = patch_size * patch_size
    padding = int(np.floor(patch_size / 2))

    patches_channels = []
    for c in range(gaussian_image.shape[0]):
        patches_channels.append(F.unfold(gaussian_image[c:c+1].unsqueeze(0), kernel_size=patch_size, padding=padding).view(num_kernel, 1, gaussian_image.shape[1], gaussian_image.shape[2]))

    patches = torch.cat(patches_channels, dim=1) #[num_kernel, channels, w, h]
    variance = patches.var(dim=0)

    if patch_filter is not None:
        patch_filter_mask = patch_filter(patches)
        variance[~patch_filter_mask] = 0.0
    
    variance = variance.sum(dim=0) #[w, h]

    if variance_threshold > 0:
        variance_mask = variance < variance_threshold
        return variance, variance_mask

    return variance

def training(args, dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    ## Start wandb run
    if args.wandb_name is not None:
        wandb_run = wandb.init(
            entity="ldkuba-tu-wien",
            project="GS-Reconstruction-Pipeline",
            name=args.wandb_name,
            config={
                "model_type": "r3dg_pbr" if args.is_pbr else "r3dg_init",
                "dataset": dataset._source_path,
                "iterations": opt.iterations,
                "diff-spsr": args.diff_spsr,
                "local_smoothing_strength": args.local_smoothing_strength,
                "depth_consistency_strength": args.depth_consistency_strength,
                "resolution": dataset._resolution
            },
        )
    else:
        wandb_run = None

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    gaussians.training_setup(opt)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    if args.is_pbr:
        
        # first update visibility
        gaussians.update_visibility(pipe.sample_num)
        
        pbr_kwargs['sample_num'] = pipe.sample_num
        print("Using global incident light for regularization.")
        direct_env_light = DirectLightMap(dataset.env_resolution, opt.light_init)
        
        if args.checkpoint:
            env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
            print("Trying to load global incident light from ", env_checkpoint)
            if os.path.exists(env_checkpoint):
                direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                print("Failed to load!")

            direct_env_light.training_setup(opt)
            pbr_kwargs["env_light"] = direct_env_light

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    """ Prepare Diff-SPSR if enabled """
    if args.diff_spsr:
        import sdpsr.sdpsr_approx as sdpsr
        print("Using Diff-SPSR for reconstruction")
        sdpsr_model = sdpsr.SDPSRApprox(res=(128, 128, 128), sigma_cov=0.003, sampling_density_factor=0.7, compute_laplace=False, compute_point_variance=False)

        import utils.render_sdf as render_sdf
        sdf_renderer = render_sdf.SDFRenderer(n_samples=64, n_importance=64, up_sample_steps=4, simple_upsample=True)

        # Save camera stack for debugging
        torch.save(scene.getTrainCameras(), os.path.join(dataset.model_path, "train_cams.pth"))

    """ Prepare point cache for depth consistency if enabled """
    if args.depth_consistency_strength > 0:
        print("Using depth consistency loss")
        from utils.render_sdf import generate_rays

        depth_consistency_cache: list[DepthConsistencyPoints] = []

        # Calculate scene scale
        scene_scale = torch.norm(gaussians.get_xyz.max(dim=0).values - gaussians.get_xyz.min(dim=0).values).item()
        print("Scene scale is:", scene_scale)

    """ GUI """
    windows = None
    if args.gui:
        cam = scene.getTrainCameras()[0]
        c2w = cam.c2w.detach().cpu().numpy()
        center = gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()

        render_kwargs = {"pc": gaussians, "pipe": pipe, "bg_color": background, "opt": opt, "is_training": False,
                         "dict_params": pbr_kwargs}

        windows = GUI(cam.image_height, cam.image_width, cam.FoVy,
                      c2w=c2w, center=center,
                      render_fn=render_fn, render_kwargs=render_kwargs,
                      mode='pbr')

    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    torch.autograd.set_detect_anomaly(True)

    if os.path.exists("debug_artifacts/depth_consistency/cache_points_1_iter.pt"):
        os.remove("debug_artifacts/depth_consistency/cache_points_1_iter.pt")

    for iteration in progress_bar:

        debug_images = []

        gaussians.update_learning_rate(iteration)

        if windows is not None:
            windows.render()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Every 1000 update visibility
        # if args.is_pbr and iteration % 1000 == 0:
        #     gaussians.update_visibility(pipe.sample_num)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                               opt=opt, is_training=True, dict_params=pbr_kwargs, iteration=iteration)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]

        # Diff-SPSR
        if args.diff_spsr:
            if iteration > args.diff_spsr_start_iteration:
                # SDPSR
                sdf, variance, sdf_spacing, sdf_corner, _ = sdpsr_model(gaussians.get_xyz, gaussians.get_normal)

                # Volumetric SDF rendering
                depth_sdf, normal_sdf, ray_mask, _ = sdf_renderer.render(sdf, sdf_corner, sdf_spacing, viewpoint_cam, volumetric=False, n_sample_rays=1024)
                depth_sdf = depth_sdf[ray_mask].unsqueeze(-1)
                normal_sdf = normal_sdf[ray_mask]
                normal_sdf = normal_sdf / (torch.norm(normal_sdf, dim=-1, keepdim=True) + 1e-6)

                # Rendered gaussians
                depth_gaussian, normal_gaussian = render_pkg["depth"], render_pkg["normal"]
                selected_depth_gaussian = depth_gaussian.permute(2, 1, 0)[ray_mask]
                selected_normal_gaussian = normal_gaussian.permute(2, 1, 0)[ray_mask]

                normal_dot = (normal_sdf * selected_normal_gaussian).sum(dim=-1)

                loss_depth = F.mse_loss(depth_sdf, selected_depth_gaussian, reduction='sum')
                loss_normal = F.l1_loss(normal_sdf, selected_normal_gaussian, reduction='sum') + F.l1_loss(torch.ones_like(normal_dot), normal_dot, reduction='sum')

                loss += loss_depth * opt.lambda_vol_depth_render
                loss += loss_normal * opt.lambda_vol_normal_render

        # Common things for local smoothing and depth consistency
        if (args.local_smoothing_strength > 0 and iteration > args.local_smoothing_start_iter) or \
            (args.depth_consistency_strength > 0 and iteration > args.depth_consistency_start_iter):

            # Detect areas to regularize
            outputs_gaussians, mask_gaussians = process_gaussian_images(render_pkg, depth_variance_threshold=0.01, hit_mask=True, output_names=["render", "depth", "normal"])
            rgb_gaussians = outputs_gaussians["render"]
            depth_gaussians = outputs_gaussians["depth"]
            normal_gaussians = outputs_gaussians["normal"]

            # Filter by color variance
            _, color_variance_mask = filter_variance(rgb_gaussians, args.color_variance_patch_size, args.color_variance_threshold)
            reg_combined_mask = mask_gaussians & color_variance_mask

            debug_images.append(mask_gaussians.detach().float().unsqueeze(0).repeat(3, 1, 1))
            debug_images.append(color_variance_mask.detach().float().unsqueeze(0).repeat(3, 1, 1))
            debug_images.append(reg_combined_mask.detach().float().unsqueeze(0).repeat(3, 1, 1))

        # Local flattening of mono-color regions
        if args.local_smoothing_strength > 0 and iteration > args.local_smoothing_start_iter:

            # Calculate depth gradient variance
            depth_grad = torch.cat(torch.gradient(depth_gaussians, dim=(-2, -1), edge_order=1), dim=0)
            # depth_grad_variance = filter_variance(depth_grad, args.local_smoothing_patch_size)
            depth_grad_variance = torch.ones_like(depth_gaussians).squeeze(0)

            # Filter if max-min depth in a patch is too high
            depth_patches = F.unfold(depth_gaussians.unsqueeze(0), kernel_size=args.local_smoothing_patch_size, padding=args.local_smoothing_patch_size//2).view(args.local_smoothing_patch_size**2, depth_gaussians.shape[0], depth_gaussians.shape[1], depth_gaussians.shape[2])
            depth_max = depth_patches.max(dim=0).values
            depth_min = depth_patches.min(dim=0).values
            depth_range = (depth_max - depth_min).squeeze(0)
            depth_range = depth_range / depth_range.max()
            depth_range_mask = depth_range < args.local_smoothing_depth_range_threshold
            local_smoothing_mask = reg_combined_mask & depth_range_mask

            depth_grad_variance[~local_smoothing_mask] = 0.0

            # depth_grad_variance_mean = depth_grad_variance.mean()
            # depth_grad_variance = torch.clamp(depth_grad_variance, min=0.0, max=depth_grad_variance_mean.item() * 5.0)
            local_smoothing_loss = depth_grad_variance.mean()
            loss += local_smoothing_loss * args.local_smoothing_strength

            depth_grad_variance = depth_grad_variance.detach()
            depth_grad_variance = depth_grad_variance / depth_grad_variance.max()

            debug_images.append(depth_range_mask.detach().float().unsqueeze(0).repeat(3, 1, 1))
            debug_images.append(local_smoothing_mask.detach().float().unsqueeze(0).repeat(3, 1, 1))
            debug_images.append(depth_grad_variance.unsqueeze(0).repeat(3, 1, 1))

        if args.depth_consistency_strength > 0 and iteration > args.depth_consistency_start_iter:

            depth_consistency_loss = torch.tensor(0.0, device='cuda')

            # Check if similar view exists in cache
            camera_primary_ray = viewpoint_cam.get_primary_axis().to('cuda')
            camera_primary_ray = camera_primary_ray / torch.norm(camera_primary_ray, dim=-1, keepdim=True)

            cache_hit_id = None
            for i, cache_entry in enumerate(depth_consistency_cache):
                cache_view_dir = cache_entry.view_dir
                angle_cos = torch.dot(cache_view_dir, camera_primary_ray)
                angle_deg = torch.rad2deg(torch.acos(angle_cos))

                # print(f"Got angle: {angle_deg:.2f}")
                if angle_deg < args.depth_consistency_view_angle_threshold:
                    cache_hit_id = i
                    break

            if cache_hit_id is not None:
                # Get points and normals from cache
                cache_hit = depth_consistency_cache.pop(cache_hit_id)
                cache_points = cache_hit.points
                cache_normals = cache_hit.normals

                # Construct rays from camera to points and get their new depths
                viewpoint_cam_pos = viewpoint_cam.camera_center
                rays_to_points = cache_points - viewpoint_cam_pos.unsqueeze(0)
                cache_depths = torch.sum(rays_to_points * camera_primary_ray, dim=-1)

                # Get corresponding depths from current render (interpolate 4 pixels that are closest to rach ray)
                # Get pixel coords of each ray
                cache_points_homogeneous = torch.cat([cache_points, torch.ones((cache_points.shape[0], 1), device=cache_points.device)], dim=-1)
                screen_coords = viewpoint_cam.get_proj_matrix() @ cache_points_homogeneous.transpose(0, 1)
                screen_coords = (screen_coords[:2] / screen_coords[2]).transpose(0, 1) # N x 2

                # Mask points that ended up off-screen in the new view, including leaving one pixel border in max index for interpolation
                # This almost never happens
                offscreen_mask = (screen_coords[:, 0] < 0) | (screen_coords[:, 0] >= viewpoint_cam.image_width - 1) | (screen_coords[:, 1] < 0) | (screen_coords[:, 1] >= viewpoint_cam.image_height - 1)
                screen_coords = screen_coords[~offscreen_mask]
                cache_depths = cache_depths[~offscreen_mask]

                # Interpolate new depths from pixels
                x0 = torch.floor(screen_coords[:, 0]).int()
                x1 = x0 + 1
                x_factor = screen_coords[:, 0] - x0.float()
                y0 = torch.floor(screen_coords[:, 1]).int()
                y1 = y0 + 1
                y_factor = screen_coords[:, 1] - y0.float()

                cache_depths_debug = torch.zeros_like(depth_gaussians.detach())
                cache_depths_debug[:, y0.detach(), x0.detach()] = cache_depths.detach()
                cache_depths_debug = cache_depths_debug.repeat(3, 1, 1) / cache_depths_debug.max()
                debug_images.append(cache_depths_debug)

                # Ignore samples that land in areas of high depth gradient variance (not locally flat)
                depth_gradient = torch.cat(torch.gradient(depth_gaussians, dim=(-2, -1), edge_order=1), dim=0)
                depth_gradient_variance, depth_gradient_variance_mask = filter_variance(depth_gradient, args.depth_consistency_patch_size, args.depth_consistency_depth_variance_threshold)
                depth_grad_var_mask_x0y0 = depth_gradient_variance_mask[y0, x0]

                # Discard points where one of the pixels used in the depth variance calculation is not in the reg_combined_mask
                mask_unfolded = F.unfold(reg_combined_mask.unsqueeze(0).float(), kernel_size=args.depth_consistency_patch_size, padding=args.depth_consistency_patch_size//2).view(args.depth_consistency_patch_size**2, reg_combined_mask.shape[0], reg_combined_mask.shape[1])
                mask_any = mask_unfolded.all(dim=0)
                mask_any_x0y0 = mask_any[y0, x0]
                depth_grad_var_mask_x0y0 = depth_grad_var_mask_x0y0 & mask_any_x0y0

                x0_not = x0[~depth_grad_var_mask_x0y0]
                x0 = x0[depth_grad_var_mask_x0y0]
                x1 = x1[depth_grad_var_mask_x0y0]
                x_factor = x_factor[depth_grad_var_mask_x0y0]
                y0_not = y0[~depth_grad_var_mask_x0y0]
                y0 = y0[depth_grad_var_mask_x0y0]
                y1 = y1[depth_grad_var_mask_x0y0]
                y_factor = y_factor[depth_grad_var_mask_x0y0]
                cache_depths = cache_depths[depth_grad_var_mask_x0y0]

                cache_points = cache_points[~offscreen_mask][depth_grad_var_mask_x0y0]

                # Images are [h, w]
                interpolated_depths = (1 - x_factor) * (1 - y_factor) * depth_gaussians[0, y0, x0] + \
                                      x_factor * (1 - y_factor) * depth_gaussians[0, y0, x1] + \
                                      (1 - x_factor) * y_factor * depth_gaussians[0, y1, x0] + \
                                      x_factor * y_factor * depth_gaussians[0, y1, x1]

                # Calculate depth consistency loss
                if interpolated_depths.shape[0] > 0:
                    depth_consistency_se = (interpolated_depths - cache_depths) ** 2

                    if iteration == 5000:
                        vis_cache_points = vis.VisualizationWriter()
                        vis_cache_points.add_point_cloud("cache points", cache_points, values_tensor=depth_gradient_variance[y0, x0])
                        vis_cache_points.add_point_cloud("depth loss", cache_points, values_tensor=depth_consistency_se)
                        vis_cache_points.add_point_cloud("camera", viewpoint_cam_pos.unsqueeze(0), arrows=camera_primary_ray.unsqueeze(0))
                        vis_cache_points.add_point_cloud("og_camera", cache_hit.view_point.unsqueeze(0), arrows=cache_hit.view_dir.unsqueeze(0))
                        vis_cache_points.save("debug_artifacts/depth_consistency/cache_points_1_iter.pt")

                    print("Max loss:", depth_consistency_se.max().item(), "Mean loss:", depth_consistency_se.mean().item(), "Num points:", depth_consistency_se.shape[0])

                    depth_consistency_loss = depth_consistency_se.mean()
                    loss += depth_consistency_loss * args.depth_consistency_strength

                    loss_depth_consistency_debug = torch.zeros_like(reg_combined_mask.detach()).float()
                    loss_depth_consistency_debug[y0.detach(), x0.detach()] = depth_consistency_se.detach()
                    loss_depth_consistency_debug[y0.detach(), x0.detach()] /= loss_depth_consistency_debug.max() + 1e-8
                    loss_depth_consistency_debug = loss_depth_consistency_debug.detach().unsqueeze(0).repeat(3, 1, 1)
                    loss_depth_consistency_debug[2, y0_not.detach(), x0_not.detach()] = 1.0

                    depth_grad_variance_debug = depth_gradient_variance.detach().repeat(3, 1, 1)
                    depth_grad_variance_debug = depth_grad_variance_debug / (depth_grad_variance_debug.max() + 1e-8)
                    depth_grad_variance_debug[0:2, y0_not.detach(), x0_not.detach()] = 0.0

                    depth_grad_variance_mask_debug = depth_gradient_variance_mask.detach().float().unsqueeze(0).repeat(3, 1, 1)
                    depth_grad_variance_mask_debug[0:2, y0_not.detach(), x0_not.detach()] = 0.0
                    depth_grad_variance_mask_debug[2, y0_not.detach(), x0_not.detach()] = 1.0

                    debug_images.append(depth_grad_variance_debug)
                    debug_images.append(depth_grad_variance_mask_debug)
                    debug_images.append(loss_depth_consistency_debug)

            else:
                # === Insert points from current view into cache ===
                # Filter out pixels that can't be used for depth consistency loss

                # Apply masks
                rgb_gaussians = rgb_gaussians[:, reg_combined_mask]
                depth_gaussians = depth_gaussians[:, reg_combined_mask]
                normal_gaussians = normal_gaussians[:, reg_combined_mask]

                # Channel last
                rgb_gaussians = rgb_gaussians.permute(1, 0)
                depth_gaussians = depth_gaussians.permute(1, 0)
                normal_gaussians = normal_gaussians.permute(1, 0)

                # Normalize normals
                normal_gaussians = normal_gaussians / torch.norm(normal_gaussians, dim=-1, keepdim=True)

                # Generate rays
                rays_o, rays_d = generate_rays(viewpoint_cam)
                rays_d = rays_d[reg_combined_mask.flatten()]

                # Unproject points
                ray_projections = torch.sum(rays_d * camera_primary_ray, dim=-1)
                rays_d = rays_d / ray_projections.unsqueeze(-1)
                points_gaussians = rays_o[None, :] + rays_d * depth_gaussians

                # point_vis.add_point_cloud(f"Depth Consistency Cache {iteration}", points_gaussians.detach().cpu(), torch.ones_like(points_gaussians) * (len(point_vis.data_dict) + 1))
                # if len(point_vis.data_dict) > 10:
                #     print("Exit")
                #     point_vis.save("debug_artifacts/depth_consistency/cache_points.pt")
                #     sys.exit(0)

                depth_consistency_cache.append(DepthConsistencyPoints(view_point=viewpoint_cam.camera_center, view_dir=camera_primary_ray, points=points_gaussians.detach(), normals=normal_gaussians.detach()))

        loss.backward()
        

        if wandb_run is not None:
            wandb_run.log({
                "loss_total": loss.item(),
                "psnr": tb_dict["psnr"]
            }, step=iteration)

            if args.local_smoothing_strength > 0 and iteration > args.local_smoothing_start_iter:
                pass
                wandb_run.log({
                    "loss_smoothing": local_smoothing_loss.item(),
                }, step=iteration)

            if args.depth_consistency_strength > 0 and iteration > args.depth_consistency_start_iter:
                wandb_run.log({
                    "depth_consistency_cache_size": len(depth_consistency_cache),
                    "loss_depth_consistency": depth_consistency_loss.item(),
                }, step=iteration)

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(args, viewpoint_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration, debug_images, pbr_kwargs)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            if args.is_pbr:
                pbar_dict["light_mean"] = direct_env_light.get_env.mean().item()
                pbar_dict["env"] = direct_env_light.H
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            training_report(args, tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,
                            bg_color=background, dict_params=pbr_kwargs)

            # densification
            # TODO: Use variance to influence densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 
                                                    render_pkg['weights'])
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_grad_normal_threshold = opt.densify_grad_normal_threshold if iteration > opt.normal_densify_from_iter else 99999
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                densify_grad_normal_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            # Optimizer step
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass
            
            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    if dataset.eval:
        eval_render(args, scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)

    if wandb_run is not None:
        wandb_run.finish()

    return gaussians, sdpsr_model if args.diff_spsr else None


def training_report(args, tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # BRDF
                    base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
                    roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))

                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     base_color, roughness.repeat(3, 1, 1)], dim=0), nrow=3)

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        torch.cuda.empty_cache()


def save_training_vis(args, viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, debug_images, pbr_kwargs):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=False, dict_params=pbr_kwargs)

            visualization_list = [
                render_pkg["render"],
                viewpoint_cam.original_image.cuda(),
                visualize_depth(render_pkg["depth"]),
                (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
                render_pkg["opacity"].repeat(3, 1, 1),
                render_pkg["normal"] * 0.5 + 0.5,
                render_pkg["pseudo_normal"] * 0.5 + 0.5,
            ]

            if args.is_pbr:
                
                H, W = render_pkg["pbr"].shape[1:]
                env = F.interpolate(render_pkg['env'].permute(0, 3, 1, 2), (H, 2*W))
                env_0 = env[0, :, :, :W]
                env_1 = env[0, :, :, W:]
                visualization_list.extend([
                    render_pkg["base_color"],
                    render_pkg["roughness"].repeat(3, 1, 1),
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["diffuse"],
                    # render_pkg["lights"],
                    render_pkg["specular"],
                    # render_pkg["local_lights"],
                    render_pkg["global_lights"],
                    render_pkg["pbr"],
                    rgb_to_srgb(env_0),
                    rgb_to_srgb(env_1),
                ])

            for debug_image in debug_images:
                visualization_list.append(debug_image)

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            scale = grid.shape[-2] / 800
            # scale = 1
            grid = F.interpolate(grid[None], (int(grid.shape[-2]/scale), int(grid.shape[-1]/scale)))[0]
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))

def eval_render(args, scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    if gaussians.use_pbr:
        os.makedirs(os.path.join(args.model_path, 'eval', 'base_color'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'lights'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'local'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'global'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'visibility'), exist_ok=True)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            if gaussians.use_pbr:
                image = results["pbr"]
            else:
                image = results["render"]

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            save_image(results["normal"] * 0.5 + 0.5,
                       os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            if gaussians.use_pbr:
                save_image(results["base_color"],
                           os.path.join(args.model_path, 'eval', "base_color", f"{viewpoint.image_name}.png"))
                save_image(results["roughness"],
                           os.path.join(args.model_path, 'eval', "roughness", f"{viewpoint.image_name}.png"))
                save_image(results["lights"],
                           os.path.join(args.model_path, 'eval', "lights", f"{viewpoint.image_name}.png"))
                save_image(results["local_lights"],
                           os.path.join(args.model_path, 'eval', "local", f"{viewpoint.image_name}.png"))
                save_image(results["global_lights"],
                           os.path.join(args.model_path, 'eval', "global", f"{viewpoint.image_name}.png"))
                save_image(results["visibility"],
                           os.path.join(args.model_path, 'eval', "visibility", f"{viewpoint.image_name}.png"))

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))

def main(args):
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)

    # Diff PSR args
    parser.add_argument("--diff-spsr", action='store_true', default=False, help='use diff-spsr for reconstruction')
    parser.add_argument("--diff-spsr-start-iteration", type=int, default=4000, help='iteration to start applying diff-spsr loss')

    # Common
    parser.add_argument("--color-variance-threshold", type=float, default=0.005, help='threshold for color variance to apply local smoothing')
    parser.add_argument("--color-variance-patch-size", type=int, default=11, help='patch size for calculating color variance for local smoothing')
    
    # Local smoothing args
    parser.add_argument("--local-smoothing-strength", type=float, default=0, help='local smoothing for flat, mono-color regions')
    parser.add_argument("--local-smoothing-start-iter", type=int, default=4000, help='iteration to start applying local smoothing')
    parser.add_argument("--local-smoothing-depth-range-threshold", type=float, default=0.1, help='threshold for depth range in a patch to apply local smoothing')
    parser.add_argument("--local-smoothing-patch-size", type=int, default=3, help='patch size for local smoothing')

    # Depth consistency
    parser.add_argument("--depth-consistency-strength", type=float, default=0, help='enforce depth consistency by projecting point cache')
    parser.add_argument("--depth-consistency-start-iter", type=int, default=4000, help='iteration to start applying depth consistency loss')
    parser.add_argument("--depth-consistency-patch-size", type=int, default=3, help='patch size for rejecting points for depth consistency')
    parser.add_argument("--depth-consistency-view-angle-threshold", type=float, default=30.0, help='angle threshold (in degrees) for considering points for depth consistency based on their original view direction and current view direction')
    parser.add_argument("--depth-consistency-depth-variance-threshold", type=float, default=0.1, help='threshold for depth variance in a patch to consider points for depth consistency')

    parser.add_argument("--wandb-name", type=str, default=None, help="name for wandb logging")

    args = parser.parse_args(args)
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.set_default_device('cuda')

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.is_pbr = args.type in ['neilf']

    gaussians = training(args, lp.extract(args), op.extract(args), pp.extract(args))
    print("\nTraining complete.")
    return gaussians

if __name__ == "__main__":
    main(sys.argv[1:])