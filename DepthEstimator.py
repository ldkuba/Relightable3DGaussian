import os
import sys
import urllib.request
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

DA_ROOT = Path("/home/people/adeak/gaussian-splatting/external/on_the_fly_nvs/submodules/Depth-Anything-V2").resolve()
DA_METRIC = (DA_ROOT / "metric_depth").resolve()

# Remove any existing Depth-Anything paths first
sys.path = [
    p for p in sys.path
    if Path(p).resolve() not in {DA_ROOT, DA_METRIC}
]

# Put the NON-metric repo root first
sys.path.insert(0, str(DA_ROOT))

# Clear already-imported cached modules
for k in list(sys.modules):
    if k == "depth_anything_v2" or k.startswith("depth_anything_v2."):
        del sys.modules[k]

from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:

    def __init__(self, otfp, scene_info, p3ids, xyzs):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.knn_p = otfp.knn_p
        self.knn_n = otfp.knn_n
        self.knn_stride = otfp.knn_stride
        self.knn_epsilon = otfp.knn_epsilon
        self.dav2_target_width = otfp.dav2_target_width

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        encoder = "vitl"
        ckpt = f"models/depth_anything_v2_{encoder}.pth"
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

        os.makedirs("models", exist_ok=True)
        if not os.path.exists(ckpt):
            print(f"Downloading {ckpt} ...")
            urllib.request.urlretrieve(url, ckpt)

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f"models/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        self.model = self.model.to(self.DEVICE).eval()

        self.scene_info = scene_info
        self.p3ids = p3ids
        self.xyzs = xyzs

        self.adjust_by_knn = otfp.adjust_by_knn
        self.adjust_by_median = otfp.adjust_by_median

        # Depth-Anything-V2 normalization constants.
        self._dav2_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.DEVICE).view(1, 3, 1, 1)
        self._dav2_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.DEVICE).view(1, 3, 1, 1)

    def _compute_dav2_size(self, h, w):
        # Match DA-V2 Resize(keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound').
        scale = max(self.dav2_target_width / float(w), self.dav2_target_width / float(h))
        new_h = max(self.dav2_target_width, int(math.ceil((h * scale) / 14.0) * 14))
        new_w = max(self.dav2_target_width, int(math.ceil((w * scale) / 14.0) * 14))
        return new_h, new_w

    @torch.no_grad()
    def estimate_depth(self, img):
        # Keep the whole inference path on-device to avoid GPU->CPU->GPU copies.
        if img.dim() != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(img.shape)}")

        h, w = img.shape[-2:]
        new_h, new_w = self._compute_dav2_size(h, w)

        x = img.unsqueeze(0).to(device=self.DEVICE, dtype=torch.float32)
        x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
        x = (x - self._dav2_mean) / self._dav2_std

        depth = self.model(x)
        depth = F.interpolate(depth[:, None], size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.to(img.device)

    @torch.no_grad()
    def get_key_points(self, viewpoint_cam):
        xyzs = self.xyzs
        key_points = self.scene_info.key_points.get(f"{viewpoint_cam.image_name}.png")

        id3d = self.p3ids
        id2d = key_points[0]

        _, m3d, m2d = np.intersect1d(id3d, id2d, return_indices=True)
        return key_points[1][m2d, :], xyzs[m3d, :]

    def t(self, D):
        return np.median(D)

    def s(self, D, tD):
        return np.mean(np.abs(D - tD))

    @torch.no_grad()
    def adjust_depth_knn(self, D, uv_desc, z_sfm):
        if len(uv_desc) == 0:
            return D

        H, W = D.shape

        u = uv_desc[:, 0].astype(dtype=np.int32)
        v = uv_desc[:, 1].astype(dtype=np.int32)
        z_map = D[v, u]
        delta = z_sfm - z_map

        tree = cKDTree(uv_desc)

        ys = np.arange(0, H, self.knn_stride, dtype=np.int32)
        xs = np.arange(0, W, self.knn_stride, dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys)
        Q = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        kk = min(self.knn_n, len(uv_desc))
        dists, idx = tree.query(Q, k=kk, workers=-1)

        if kk == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        w = 1.0 / (dists ** self.knn_p + self.knn_epsilon)
        Delta = (w * delta[idx]).sum(axis=1) / (w.sum(axis=1) + self.knn_epsilon)
        Delta = Delta.reshape(len(ys), len(xs))
        Delta_full = np.repeat(np.repeat(Delta, self.knn_stride, axis=0), self.knn_stride, axis=1)[:H, :W]

        return D + Delta_full

    @torch.no_grad()
    def adjust_depth_median(self, depth_map, uv_desc, z_sfm):
        u = uv_desc[:, 0]
        v = uv_desc[:, 1]

        D_rel = depth_map.detach().cpu().numpy()
        D_sfm = 1.0 / z_sfm

        t_sfm = self.t(D_sfm)
        t_rel = self.t(D_rel[v, u])
        s_sfm = self.s(D_sfm, t_sfm)
        s_rel = self.s(D_rel[v, u], t_rel)

        D = (s_sfm / s_rel) * D_rel + t_sfm - t_rel * (s_sfm / s_rel)
        D = np.clip(D, 1e-6, 1e6)
        D = 1.0 / D

        return torch.from_numpy(D).to(device=depth_map.device)

    @torch.no_grad()
    def generate_depth_map(self, viewpoint_cam):
        depth_map = self.estimate_depth(viewpoint_cam.original_image)

        if self.scene_info is None:
            return depth_map

        uv, xyzs = self.get_key_points(viewpoint_cam)
        if len(uv) == 0:
            return depth_map

        uv = np.round(uv).astype(dtype=np.int32)
        xyzs_cam_space = np.concatenate((xyzs, np.ones((xyzs.shape[0], 1))), axis=1) @ viewpoint_cam.extrinsics.T.detach().cpu().numpy()


        if self.adjust_by_median:
            depth_map = self.adjust_depth_median(depth_map, uv, xyzs_cam_space[:,2])

        if self.adjust_by_knn:
            depth_map_np = self.adjust_depth_knn(
                depth_map.detach().cpu().numpy(),
                uv,
                xyzs_cam_space[:, 2],
            )
            depth_map = torch.from_numpy(depth_map_np).to(device=depth_map.device)

        return depth_map
