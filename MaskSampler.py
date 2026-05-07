import numpy as np
import torch

import torch.nn.functional as F

class MaskSampler:

    def __init__(self, otfp, scene_info, width, height, key_points_torch):

        # set device
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.scene_info = scene_info
        self.key_points_torch = key_points_torch

        self.width = width
        self.height = height
        self.base_prob = otfp.base_prob
        self.normalize_prob = otfp.normalize_prob

        self.feature_sigma = otfp.feature_sigma  # blur radius in pixels
        self.truncate = 3.0
        self.feature_min_coverage = otfp.feature_min_coverage
        self.feature_gate_mode = otfp.feature_gate_mode
        # "multiply" or "hard"

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

        self.feature_kernel = self._make_gaussian_kernel().to(self.DEVICE)

    """
    from on the fly
    TODO cite
    """
    @torch.no_grad()
    def get_lapla_norm(self, img, kernel):
        laplacian_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda", dtype=torch.float32
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
        laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
        laplacian_norm = torch.linalg.vector_norm(laplacian, ord=1, dim=1, keepdim=True)
        laplacian_norm[..., :, 0] = 0
        laplacian_norm[..., :, -1] = 0
        laplacian_norm[..., 0, :] = 0
        laplacian_norm[..., -1, :] = 0
        return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)

    @torch.no_grad()
    def get_feature_coverage_map(self, viewpoint_cam):
        """
        Returns a blurred feature support map in [0, 1], shape [H, W].
        High values mean: this pixel is near matched keypoints / SfM support.
        """
        key_points = self.key_points_torch.get(f"{viewpoint_cam.image_name}.png")
        if key_points is None or key_points[0].numel() == 0:
            return torch.zeros((self.height, self.width), device=self.DEVICE)

        uv = torch.round(key_points[1]).long()  # shape [N, 2], assumed (u, v)

        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < self.height)
        )
        uv = uv[valid]

        if uv.numel() == 0:
            return torch.zeros((self.height, self.width), device=self.DEVICE)

        feat_map = torch.zeros((1, 1, self.height, self.width), device=self.DEVICE)
        u = uv[:, 0]
        v = uv[:, 1]

        feat_map[0, 0, v, u] = 1.0

        # Blur sparse impulses into a smooth coverage field
        pad_h = self.feature_kernel.shape[-2] // 2
        pad_w = self.feature_kernel.shape[-1] // 2
        coverage = F.conv2d(feat_map, self.feature_kernel, padding=(pad_h, pad_w))[0, 0]

        # Normalize to [0, 1]
        maxv = coverage.max()
        if maxv > 0:
            coverage = coverage / maxv

        return coverage

    def _make_gaussian_kernel(self):
        radius = int(self.truncate * self.feature_sigma + 0.5)
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-(coords ** 2) / (2 * self.feature_sigma ** 2))
        g = g / g.sum()
        kernel2d = torch.outer(g, g)
        kernel2d = kernel2d / kernel2d.sum()
        return kernel2d[None, None]  # [1,1,H,W]

    @torch.no_grad()
    def create_density_map(self, img):
        """Creates a probability map, reference add_new_gaussians in scene_model from on the fly
        - kernel is eq. 1 in paper"""
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        prob_mask = self.get_lapla_norm(img, self.disc_kernel)  # eq. 1

        return prob_mask

    @torch.no_grad()
    def generate_sample_mask(self, rendered_img, viewpoint_cam):
        original_img = viewpoint_cam.original_image
        prob_mask = self.create_density_map(original_img)
        prob_mask = torch.clamp(prob_mask, min=self.base_prob)
        if self.normalize_prob:
            prob_mask = prob_mask / torch.max(prob_mask)

        covered_prob_mask, feature_coverage = self.apply_feature_gating(prob_mask, viewpoint_cam)

        penalty = self.create_density_map(rendered_img)
        if self.normalize_prob:
            penalty = penalty / torch.max(penalty)

        penalized = covered_prob_mask - penalty
        image_mask = torch.squeeze(viewpoint_cam.image_mask).to(device=penalized.device).bool()
        sample_mask = (torch.rand_like(penalized) < penalized) & image_mask
        return prob_mask, sample_mask, image_mask, feature_coverage

    @torch.no_grad()
    def apply_feature_gating(self, prob_mask, viewpoint_cam):
        coverage = self.get_feature_coverage_map(viewpoint_cam)

        image_mask = torch.squeeze(viewpoint_cam.image_mask).to(device=prob_mask.device).bool()
        coverage = coverage * image_mask.float()

        if self.feature_gate_mode == "multiply":
            gated = prob_mask * coverage
        elif self.feature_gate_mode == "hard":
            gated = prob_mask.clone()
            gated[coverage < self.feature_min_coverage] = 0.0
        else:
            raise ValueError(f"Unknown feature_gate_mode: {self.feature_gate_mode}")

        return gated, coverage
