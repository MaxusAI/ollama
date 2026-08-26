#!/usr/bin/env python3
"""Generate a SHAREABLE synthetic trigger for the qwen2.5vl-3B fp16-accumulate bug.

A 56 px black/white checkerboard at 1350x1800 measures 69,120 max |activation| at
the vision tower's final block (1.06x fp16's 65,504 ceiling) and deterministically
returns '?' x31 with done_reason=null on stock ollama + stock qwen2.5vl:3b on CUDA.
Serving it with GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (or bf16) returns a normal answer.
"""
import numpy as np
from PIL import Image

WIDTH, HEIGHT, PITCH = 1350, 1800, 56
ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
cells = ((xs // PITCH) + (ys // PITCH)) % 2
pixels = np.where(cells[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
Image.fromarray(pixels).save("poison_repro_checker56_1350x1800.png")
print("wrote poison_repro_checker56_1350x1800.png")
