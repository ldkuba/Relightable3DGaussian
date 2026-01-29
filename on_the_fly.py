import sys
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH

from PIL import Image

sys.path.append('external')
from on_the_fly_nvs.utils import (
    get_lapla_norm,
)

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

class OnTheFly:

    def __init__(self, width, height, max_sh_degree, prob_scale = 1.0):
        self.width = width
        self.height = height
        self.prob_scale = prob_scale
        self.max_sh_degree = max_sh_degree

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

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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

    @torch.no_grad()
    def create_prob_map(self, img):
        """Creates a probability map, reference add_new_gaussians in scene_model from on the fly
        - kernel is eq. 1 in paper"""
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        prob_mask = get_lapla_norm(img, self.disc_kernel)  # eq. 1
#        print("init_proba.shape: ", init_proba.shape)
#        print("init_proba.max")

#        import sys
#        import numpy
#        from PIL import Image
#
#        img = Image.open(sys.argv[1]).convert('L')
#
#        im = numpy.array(img)
#        fft_mag = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(im)))
#
#        visual = numpy.log(fft_mag)
#        visual = (visual - visual.min()) / (visual.max() - visual.min())
#
#        result = Image.fromarray((visual * 255).astype(numpy.uint8))
#        result.save('out.bmp')

        return prob_mask

    def generate_depth_map(self, img):
        detached = img.cpu().detach().numpy()
        detached = detached.transpose(1, 2, 0)
#        print("detached.shape: ", detached.shape)
#        print(np.max(detached))
        depth = self.model.infer_image(detached * 255, input_size=self.width)

        return torch.from_numpy(depth).to(img.device)

    @torch.no_grad()
    def generate_gaussians(self, prob_mask, depth_mask, cam):
        c2w = torch.linalg.inv(cam.proj_matrix)

        # ++ means ++
        sample_mask = torch.rand_like(prob_mask) < prob_mask
        full_coords = [torch.arange(0, sample_mask.shape[0]), torch.arange(0, sample_mask.shape[1])]
        coords = torch.stack(torch.meshgrid(full_coords), dim=-1)[sample_mask]
        coords = torch.cat((coords, depth_mask[coords[:, 0], coords[:, 1]].unsqueeze(-1),
                            torch.ones_like(coords[:,0]).unsqueeze(-1)), dim=-1)

        c2w = c2w.transpose(0, 1)
        world_points = coords @ c2w
        world_points = world_points[:,:3]

        # ++ scales ++
        dist2 = torch.clamp_min(distCUDA2(world_points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

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

        return GaussianBatch(
            means=world_points, scales=scales, rotations=rots, normals=normals, shs_dc=shs_dc, shs_rest=shs_rest,
            opacities=opacities, max_radii2D=max_radii2D, weights=weights, xyz_gradient_accum=xyz_gradient_accum,
            normal_gradient_accum=normal_gradient_accum, denom=denom
        )

    def check_and_set_cam(self, uid):
        if uid in self.camera_processed:
            return False
        self.camera_processed.add(uid)
        return True
