import os
import sys
import urllib.request
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from pytorch3d.ops import knn_points

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

    def __init__(self, otfp, scene_info, p3ids, xyzs, key_points):
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
            print(f"[DepthEstimator] Downloading {ckpt} ...")
            urllib.request.urlretrieve(url, ckpt)

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f"models/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        self.model = self.model.to(self.DEVICE).eval()

        self.scene_info = scene_info
        self.p3ids = p3ids
        self.xyzs = xyzs
        self.key_points = key_points
        self.p3ids_torch = torch.as_tensor(self.p3ids, device=self.DEVICE, dtype=torch.long)
        self.xyzs_torch = torch.as_tensor(self.xyzs, device=self.DEVICE, dtype=torch.float32)

        self.adjust_by_pytorch3d_knn_points = otfp.adjust_by_pytorch3d_knn_points
        self.adjust_by_median = otfp.adjust_by_median
        self.adjust_by_scipy_cKDTree = otfp.adjust_by_scipy_cKDTree

        # Depth-Anything-V2 normalization constants.
        self._dav2_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.DEVICE).view(1, 3, 1, 1)
        self._dav2_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.DEVICE).view(1, 3, 1, 1)
        print(self.to_string())

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
            raise ValueError(f"[DepthEstimator] Expected CHW image tensor, got shape {tuple(img.shape)}")

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
        key_points = self.key_points.get(f"{viewpoint_cam.image_name}.png")
        if key_points is None or key_points[0].numel() == 0:
            empty_uv = torch.empty((0, 2), device=self.DEVICE, dtype=torch.float32)
            empty_xyz = torch.empty((0, 3), device=self.DEVICE, dtype=torch.float32)
            return empty_uv, empty_xyz

        id2d = key_points[0].reshape(-1)
        uv = key_points[1]
        id3d = self.p3ids_torch.reshape(-1)

        id3d_sorted, sort_idx = torch.sort(id3d)
        pos = torch.searchsorted(id3d_sorted, id2d)
        in_bounds = pos < id3d_sorted.numel()
        if not torch.any(in_bounds):
            empty_uv = torch.empty((0, 2), device=self.DEVICE, dtype=torch.float32)
            empty_xyz = torch.empty((0, 3), device=self.DEVICE, dtype=torch.float32)
            return empty_uv, empty_xyz

        id2d_in = id2d[in_bounds]
        pos_in = pos[in_bounds]
        match = id3d_sorted[pos_in] == id2d_in
        if not torch.any(match):
            empty_uv = torch.empty((0, 2), device=self.DEVICE, dtype=torch.float32)
            empty_xyz = torch.empty((0, 3), device=self.DEVICE, dtype=torch.float32)
            return empty_uv, empty_xyz

        matched_uv = uv[in_bounds][match]
        matched_xyz_idx = sort_idx[pos_in[match]]
        matched_xyz = self.xyzs_torch[matched_xyz_idx]
        return matched_uv, matched_xyz

    def t(self, D):
        if not torch.is_tensor(D):
            D = torch.as_tensor(D)
        return torch.median(D)

    def s(self, D, tD):
        if not torch.is_tensor(D):
            D = torch.as_tensor(D)
        if not torch.is_tensor(tD):
            tD = torch.as_tensor(tD, device=D.device, dtype=D.dtype)
        else:
            tD = tD.to(device=D.device, dtype=D.dtype)
        return torch.mean(torch.abs(D - tD))

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
        device = depth_map.device
        dtype = depth_map.dtype

        uv_t = torch.as_tensor(uv_desc, device=device, dtype=torch.long)
        u = uv_t[:, 0]
        v = uv_t[:, 1]

        D_rel = depth_map
        z_sfm_t = torch.as_tensor(z_sfm, device=device, dtype=dtype)
        D_sfm = 1.0 / z_sfm_t

        D_rel_samples = D_rel[v, u]
        t_sfm = self.t(D_sfm)
        t_rel = self.t(D_rel_samples)
        s_sfm = self.s(D_sfm, t_sfm)
        s_rel = self.s(D_rel_samples, t_rel)

        ratio = s_sfm / s_rel
        D = ratio * D_rel + t_sfm - t_rel * ratio
        D = torch.clamp(D, 1e-6, 1e6)
        D = 1.0 / D

        return D

    @torch.no_grad()
    def adjust_depth_pytorch3d_knn(self, depth_map, uv_desc, z_sfm):
        if len(uv_desc) == 0:
            return depth_map

        device = depth_map.device
        dtype = depth_map.dtype
        H, W = depth_map.shape

        uv_desc_t = torch.as_tensor(uv_desc, device=device, dtype=torch.float32)
        z_sfm_t = torch.as_tensor(z_sfm, device=device, dtype=dtype)

        uv_idx = uv_desc_t.long()
        u = uv_idx[:, 0]
        v = uv_idx[:, 1]
        z_map = depth_map[v, u]
        delta = z_sfm_t - z_map

        ys = torch.arange(0, H, self.knn_stride, device=device, dtype=torch.float32)
        xs = torch.arange(0, W, self.knn_stride, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        Q = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)

        kk = min(self.knn_n, uv_desc_t.shape[0])
        knn = knn_points(Q[None], uv_desc_t[None], K=kk, return_sorted=False)
        dists = torch.sqrt(torch.clamp(knn.dists[0], min=0.0))
        idx = knn.idx[0]

        w = 1.0 / (dists.pow(self.knn_p) + self.knn_epsilon)
        delta_neighbors = delta[idx]
        Delta = (w * delta_neighbors).sum(dim=1) / (w.sum(dim=1) + self.knn_epsilon)
        Delta = Delta.reshape(ys.numel(), xs.numel())
        Delta_full = Delta.repeat_interleave(self.knn_stride, dim=0).repeat_interleave(self.knn_stride, dim=1)[:H, :W]

        return depth_map + Delta_full.to(dtype=dtype)

    @torch.no_grad()
    def generate_depth_map(self, viewpoint_cam):
        depth_map = self.estimate_depth(viewpoint_cam.original_image)

        if self.scene_info is None:
            return depth_map

        uv, xyzs = self.get_key_points(viewpoint_cam)
        if uv.shape[0] == 0:
            return depth_map

        uv = torch.round(uv).to(dtype=torch.int64)
        xyzs_h = torch.cat((xyzs, torch.ones((xyzs.shape[0], 1), device=xyzs.device, dtype=xyzs.dtype)), dim=1)
        xyzs_cam_space = xyzs_h @ viewpoint_cam.extrinsics.T.to(device=xyzs.device, dtype=xyzs.dtype)


        if self.adjust_by_median:
            depth_map = self.adjust_depth_median(depth_map, uv, xyzs_cam_space[:, 2])

        if self.adjust_by_scipy_cKDTree:
            depth_map_np = self.adjust_depth_knn(
                depth_map.detach().cpu().numpy(),
                uv.detach().cpu().numpy(),
                xyzs_cam_space[:, 2].detach().cpu().numpy(),
            )
            depth_map = torch.from_numpy(depth_map_np).to(device=depth_map.device)

        if self.adjust_by_pytorch3d_knn_points:
            depth_map = self.adjust_depth_pytorch3d_knn(
                depth_map,
                uv,
                xyzs_cam_space[:, 2],
            )

        return depth_map

    def to_string(self):
        return (
            f"DepthEstimator[device={self.DEVICE}, knn_p={self.knn_p}, knn_n={self.knn_n}, "
            f"knn_stride={self.knn_stride}, knn_epsilon={self.knn_epsilon}, "
            f"dav2_target_width={self.dav2_target_width}, adjust_by_median={self.adjust_by_median}, "
            f"adjust_by_scipy_cKDTree={self.adjust_by_scipy_cKDTree}, "
            f"adjust_by_pytorch3d_knn_points={self.adjust_by_pytorch3d_knn_points}]"
        )
