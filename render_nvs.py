import os
import torch
from gaussian_renderer import render_fn_dict
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.direct_light_map import DirectLightMap

from arguments import get_combined_args

class GaussianRendererWrapper:

    def __init__(self, r3dg_args):
        parser = ArgumentParser(description="Testing script parameters")
        mp = ModelParams(parser, sentinel=True)
        pp = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')
        parser.add_argument("-c", "--checkpoint", type=str, default=None)
        self.args = get_combined_args(parser, r3dg_args)

        safe_state(self.args.quiet)

        self.dataset = mp.extract(self.args)
        self.pipeline = pp.extract(self.args)

        with torch.no_grad():
            self.gaussians = GaussianModel(self.dataset.sh_degree, render_type=self.args.type)
            self.scene = Scene(self.dataset, self.gaussians, shuffle=False)
            bg_color = [1,1,1] if self.dataset.white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if self.args.checkpoint:
                print("Create Gaussians from checkpoint {}".format(self.args.checkpoint))
                iteration = self.gaussians.create_from_ckpt(self.args.checkpoint, restore_optimizer=True)
            elif self.scene.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.dataset.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.scene.loaded_iter),
                                                "point_cloud.ply"))
                iteration = self.scene.loaded_iter
            else:
                self.gaussians.create_from_pcd(self.scene.scene_info.point_cloud, self.scene.cameras_extent)
                iteration = self.scene.loaded_iter

            self.pbr_kwargs = dict()
            if iteration is not None and self.gaussians.use_pbr:
                self.gaussians.update_visibility(self.args.sample_num)
            
                self.pbr_kwargs['sample_num'] = self.args.sample_num
                print("Using global incident light for regularization.")
                direct_env_light = DirectLightMap(self.args.env_resolution)
                
                if self.args.checkpoint:
                    env_checkpoint = os.path.dirname(self.args.checkpoint) + "/env_light_" + os.path.basename(self.args.checkpoint)
                    print("Trying to load global incident light from ", env_checkpoint)
                    if os.path.exists(env_checkpoint):
                        direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                        print("Successfully loaded!")
                    else:
                        print("Failed to load!")
                    self.pbr_kwargs["env_light"] = direct_env_light

            self.render_fn = render_fn_dict[self.args.type]

    @torch.no_grad()
    def render(self, views):
        render_results = []
        for view in tqdm(views, desc="Rendering progress"):
            results = self.render_fn(view, self.gaussians, self.pipeline, self.background, dict_params=self.pbr_kwargs)
            render_results.append(results)

        return render_results

    def get_cameras(self, type='train'):
        if type == 'train':
            return self.scene.getTrainCameras()
        elif type == 'test':
            return self.scene.getTestCameras()
        elif type == 'all':
            return self.scene.getTrainCameras() + self.scene.getTestCameras()
        else:
            raise ValueError(f"Unknown camera type: {type}")

    # Create new camera from pose, based on properties of existing cameras in the scene
    def create_camera(self, camera_pose):
        pass