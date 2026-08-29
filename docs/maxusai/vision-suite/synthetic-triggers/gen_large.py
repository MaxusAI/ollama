import numpy as np
from PIL import Image
import os

def save(name, arr):
    Image.fromarray(arr).save(os.path.join(os.path.dirname(__file__), name))
    print(name)

def rgb(mask):
    return np.where(mask[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)

# 2048^2 ~ 5.3k tokens, 2560^2 ~ 8.4k, 3584^2 = 16384 tokens (the model max)
for gname, (H, W) in {"2048x2048": (2048, 2048), "2560x2560": (2560, 2560), "3584x3584": (3584, 3584)}.items():
    ys, xs = np.mgrid[0:H, 0:W]
    for p in (28, 56):
        save(f"checker_p{p}_{gname}.png", rgb(((xs // p) + (ys // p)) % 2 == 1))
    for p in (14, 56):
        save(f"stripes_v_p{p}_{gname}.png", rgb((xs // p) % 2 == 1))
    cy, cx = H / 2, W / 2
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    save(f"rings_p5_{gname}.png", rgb((r // 5) % 2 == 1))
