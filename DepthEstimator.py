import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
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

    @torch.no_grad()
    def estimate_depth(self, img):
        detached = img.cpu().detach().numpy()
        detached = detached.transpose(1, 2, 0)
        depth = self.model.infer_image(detached * 255, input_size=self.dav2_target_width)
        return torch.from_numpy(depth).to(img.device)

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
    def adjust_depth_knn(self, D, uv_desc, z_sfm, k, stride, p, eps):
        H, W = D.shape

        u = uv_desc[:, 0].astype(dtype=np.int32)
        v = uv_desc[:, 1].astype(dtype=np.int32)
        z_map = D[v, u]
        delta = z_sfm - z_map

        tree = cKDTree(uv_desc)

        ys = np.arange(0, H, stride, dtype=np.int32)
        xs = np.arange(0, W, stride, dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys)
        Q = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        kk = min(k, len(uv_desc))
        dists, idx = tree.query(Q, k=kk, workers=-1)

        if kk == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        w = 1.0 / (dists ** p + eps)
        Delta = (w * delta[idx]).sum(axis=1) / (w.sum(axis=1) + eps)
        Delta = Delta.reshape(len(ys), len(xs))
        Delta_full = np.repeat(np.repeat(Delta, stride, axis=0), stride, axis=1)[:H, :W]

        return D + Delta_full

    @torch.no_grad()
    def adjust_depth_map(self, viewpoint_cam, depth_map):
        uv, xyzs = self.get_key_points(viewpoint_cam)
        uv = np.round(uv).astype(dtype=np.int32)
        u = uv[:, 0]
        v = uv[:, 1]

        D_rel = depth_map.detach().cpu().numpy()
        xyzs_camera_world = np.concatenate((xyzs, np.ones((xyzs.shape[0], 1))), axis=1) @ viewpoint_cam.extrinsics.T.detach().cpu().numpy()
        xyzs_camera_world = xyzs_camera_world[:, :3] @ viewpoint_cam.intrinsics.T.detach().cpu().numpy()
        D_sfm = 1.0 / xyzs_camera_world[:, 2]

        t_sfm = self.t(D_sfm)
        t_rel = self.t(D_rel[v, u])
        s_sfm = self.s(D_sfm, t_sfm)
        s_rel = self.s(D_rel[v, u], t_rel)

        D = (s_sfm / s_rel) * D_rel + t_sfm - t_rel * (s_sfm / s_rel)
        D = np.clip(D, 1e-6, 1e6)
        D = 1.0 / D

        D = self.adjust_depth_knn(
            D,
            uv,
            xyzs_camera_world[:, 2],
            k=self.knn_n,
            stride=self.knn_stride,
            p=self.knn_p,
            eps=self.knn_epsilon,
        )

        return torch.from_numpy(D).to(device=depth_map.device)

    @torch.no_grad()
    def generate_depth_map(self, viewpoint_cam):
        depth_map = self.estimate_depth(viewpoint_cam.original_image)

        if self.scene_info is not None:
            depth_map = self.adjust_depth_map(viewpoint_cam, depth_map)

        return depth_map
