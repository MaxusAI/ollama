import numpy as np
from PIL import Image
import os

def save(name, arr):
    Image.fromarray(arr).save(os.path.join(os.path.dirname(__file__), name))
    print(name)

def rgb(mask):  # bool -> black/white RGB
    return np.where(mask[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)

GEOMS = {"1350x1800": (1800, 1350), "1288x616": (616, 1288), "1800x860": (860, 1800)}

for gname, (H, W) in GEOMS.items():
    ys, xs = np.mgrid[0:H, 0:W]
    for p in (14, 28, 56, 112):
        save(f"checker_p{p}_{gname}.png", rgb(((xs // p) + (ys // p)) % 2 == 1))
    save(f"checker_p56_inv_{gname}.png", rgb(((xs // 56) + (ys // 56)) % 2 == 0))
    for p in (14, 56):
        save(f"stripes_v_p{p}_{gname}.png", rgb((xs // p) % 2 == 1))
        save(f"stripes_h_p{p}_{gname}.png", rgb((ys // p) % 2 == 1))
    save(f"diag_p56_{gname}.png", rgb(((xs + ys) // 56) % 2 == 1))
    cy, cx = H / 2, W / 2
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    for p in (5, 14):
        save(f"rings_p{p}_{gname}.png", rgb((r // p) % 2 == 1))
    save(f"dots_p56_{gname}.png", rgb(((xs % 56) < 28) & ((ys % 56) < 28)))
# benign control
save("flat_gray_1350x1800.png", np.full((1800, 1350, 3), 128, dtype=np.uint8))
