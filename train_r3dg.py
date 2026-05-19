import os
import time
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint


from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, OnTheFlyParams
from gui import GUI
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from on_the_fly import OnTheFly


def training(args, dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, on_the_fly_args: OnTheFlyParams, wandb_run=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    torch.cuda.memory._record_memory_history(enabled="all",stacks="all", max_entries=100_000)

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

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

    tmp = scene.getTrainCameras()[0]
    on_the_fly_obj = OnTheFly(width=tmp.image_width, height=tmp.image_height, max_sh_degree=dataset.sh_degree,
                              otfp=on_the_fly_args, dataset=dataset, args=args, scene_info=scene.scene_info, render_fn=render_fn, pipe=pipe, pbr_kwargs=pbr_kwargs, opt=opt)
    track_runtime_timing = on_the_fly_obj.local_time_enabled or (wandb_run is not None and on_the_fly_obj.wandb_time_enabled)

    on_the_fly_obj.init_neighbourhood(scene.getTrainCameras())

    """ Prepare Diff-SPSR if enabled """
    if args.diff_spsr:
        import sdpsr.sdpsr_approx as sdpsr
        print("Using Diff-SPSR for reconstruction")
        sdpsr_model = sdpsr.SDPSRApprox(res=(128, 128, 128), sigma_cov=0.003, sampling_density_factor=0.7, compute_laplace=False, compute_point_variance=False)

        import utils.render_sdf as render_sdf
        sdf_renderer = render_sdf.SDFRenderer(n_samples=64, n_importance=64, up_sample_steps=4)

        # Save camera stack for debugging
        torch.save(scene.getTrainCameras(), os.path.join(dataset.model_path, "train_cams.pth"))

        # torch.cuda.memory._record_memory_history()

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

    initial_progb = tqdm(range(0, len(scene.getTrainCameras())), desc="Initial gaussian spawning",
                         initial=0, total=len(scene.getTrainCameras()))

    with torch.no_grad():

        if pipe.on_the_fly:
            print("##### RUN ON THE FLY GAUSSIAN SPAWNING #####")
            viewpoint_stack_on_the_fly = scene.getTrainCameras().copy()
            for iter in initial_progb:
                viewpoint_cam = viewpoint_stack_on_the_fly.pop()
                if pipe.on_the_fly and on_the_fly_obj.check_and_set_cam(viewpoint_cam.uid):
                    spawn_start_time = time.perf_counter() if track_runtime_timing else None
                    on_the_fly_obj.add_gaussians(gaussians, viewpoint_cam)
                    if track_runtime_timing:
                        spawn_elapsed_ms = (time.perf_counter() - spawn_start_time) * 1000.0
                        on_the_fly_obj.log_initial_spawn_timing(
                            spawn_index=iter + 1,
                            camera_name=viewpoint_cam.image_name,
                            duration_ms=spawn_elapsed_ms,
                            num_gaussians=gaussians.get_xyz.shape[0],
                            wandb_run=wandb_run,
                        )

    for iteration in progress_bar:
        iter_start_time = time.perf_counter() if track_runtime_timing else None
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

#        if wandb_run is not None:
#            wandb_run.log({
#                "psnr": tb_dict['psnr'],
#                "ssim": tb_dict['ssim']
#            }, step=iteration)

        if on_the_fly_obj.should_log_eval(iteration, wandb_run=wandb_run):
            split_name = on_the_fly_obj.eval_split if on_the_fly_obj.eval_split in ["train", "test"] else "train"
            eval_cameras = scene.getTestCameras() if split_name == "test" else scene.getTrainCameras()
            on_the_fly_obj.log_rendered_view_metrics_summary(
                global_step=iteration,
                split_name=split_name,
                cameras=eval_cameras,
                current_viewpoint=viewpoint_cam,
                gaussians=scene.gaussians,
                render_fn=render_fn,
                pipe=pipe,
                background=background,
                opt=opt,
                wandb_run=wandb_run,
                **pbr_kwargs,
            )

        # Diff-SPSR
        if args.diff_spsr:
                
            if iteration > 10000:
                # SDPSR
                sdf, variance, sdf_spacing, sdf_corner = sdpsr_model(gaussians.get_xyz, gaussians.get_normal)

                # Volumetric SDF rendering
                depth_sdf, normal_sdf, ray_mask = sdf_renderer.render(sdf, sdf_corner, sdf_spacing, viewpoint_cam, volumetric=False, n_sample_rays=1024)
                depth_sdf = depth_sdf[ray_mask].unsqueeze(-1)
                normal_sdf = normal_sdf[ray_mask]

                # Rendered gaussians
                depth_gaussian, normal_gaussian = render_pkg["depth"], render_pkg["normal"]
                selected_depth_gaussian = depth_gaussian.flatten(1, -1).permute(1, 0)[ray_mask]
                selected_normal_gaussian = normal_gaussian.flatten(1, -1).permute(1, 0)[ray_mask]

                normal_dot = (normal_sdf * selected_normal_gaussian).sum(dim=-1)

                loss_depth = F.mse_loss(depth_sdf, selected_depth_gaussian, reduction='sum')
                loss_normal = F.l1_loss(normal_sdf, selected_normal_gaussian, reduction='sum') + F.l1_loss(torch.ones_like(normal_dot), normal_dot, reduction='sum')

                loss += loss_depth * opt.lambda_vol_depth_render
                loss += loss_normal * opt.lambda_vol_normal_render

#        if wandb_run is not None:
#            wandb_run.log({
#                "loss_total": loss.item(),
#                "loss_depth_sdf": loss_depth.item() if args.diff_spsr and iteration > 10000 else 0,
#                "loss_normal_sdf": loss_normal.item() if args.diff_spsr and iteration > 10000 else 0,
#            }, step=iteration)

        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(args, viewpoint_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration, pbr_kwargs)
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
                            bg_color=background, dict_params=pbr_kwargs, wandb_run=wandb_run)

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

            if track_runtime_timing:
                iter_elapsed_ms = (time.perf_counter() - iter_start_time) * 1000.0
                on_the_fly_obj.log_iteration_timing(
                    iteration=iteration,
                    duration_ms=iter_elapsed_ms,
                    num_gaussians=gaussians.get_xyz.shape[0],
                    wandb_run=wandb_run,
                )
            
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

    return gaussians


def training_report(args, tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, wandb_run=None, **kwargs):
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


def save_training_vis(args, viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs):
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

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            scale = grid.shape[-2] / 800
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

def main(args, wandb_run=None):
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    otfp = OnTheFlyParams(parser)
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
    # ...other args...

    args = parser.parse_args(args)
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.set_default_device('cuda')

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.is_pbr = args.type in ['neilf']

    gaussians = training(args, lp.extract(args), op.extract(args), pp.extract(args), otfp.extract(args), wandb_run=wandb_run)
    print("\nTraining complete.")
    return gaussians

if __name__ == "__main__":
    main(sys.argv[1:])
