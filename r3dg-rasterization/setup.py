#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import platform

os.path.dirname(os.path.abspath(__file__))

debug_mode = True

extra_compile_args = {
    "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
            "-O3" if not debug_mode else "-O0",],
    "cxx": ["-O3" if not debug_mode else "-O0",]
}
extra_link_args = []

if debug_mode:
    extra_compile_args["nvcc"].append("-g")
    extra_link_args.extend(["-O0"])
    if platform.system() == "Windows":
        extra_compile_args["cxx"].append("-Zl")
        extra_link_args.extend(["-Zl"])

setup(
    name="r3dg_rasterization",
    packages=['r3dg_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="r3dg_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
