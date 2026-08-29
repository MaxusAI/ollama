import numpy as np
from PIL import Image
import os

D = os.path.dirname(__file__)
def save(name, arr):
    Image.fromarray(arr.astype(np.uint8)).save(os.path.join(D, name)); print(name)
def rgb(mask):
    return np.where(mask[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)

# 1288x616 is the shape of the corpus image that kills 0.7.1; others for size axis.
GEOMS = {"1288x616": (616, 1288), "1350x1800": (1800, 1350), "2560x2560": (2560, 2560)}

for g, (H, W) in GEOMS.items():
    ys, xs = np.mgrid[0:H, 0:W]

    # A. sub-patch frequencies: many alternations INSIDE each 14px patch
    for p in (1, 2, 4, 7):
        save(f"subchk_p{p}_{g}.png", rgb(((xs // p) + (ys // p)) % 2 == 1))

    # B. off-axis gratings: patches see mixed content, not uniform blocks
    for ang in (15, 30, 45):
        t = np.deg2rad(ang)
        proj = xs * np.cos(t) + ys * np.sin(t)
        for p in (6, 20):
            save(f"rot{ang}_p{p}_{g}.png", rgb((proj // p).astype(int) % 2 == 1))

    # C. mixed spectrum: superposed gratings + radial chirp
    plaid = np.zeros((H, W))
    for f in (3, 11, 29, 61):
        plaid += np.sin(2 * np.pi * xs / f) + np.sin(2 * np.pi * ys / f)
    save(f"plaid_{g}.png", rgb(plaid > 0))
    r = np.sqrt((ys - H / 2) ** 2 + (xs - W / 2) ** 2)
    save(f"chirp_r_{g}.png", rgb(np.sin(r ** 2 / 900) > 0))
    save(f"chirp_h_{g}.png", rgb(np.sin(xs ** 2 / 2000) > 0))

    # D. deterministic noise at several block sizes
    for b in (1, 2, 7):
        rs = np.random.RandomState(0)
        small = rs.randint(0, 2, size=((H + b - 1) // b, (W + b - 1) // b))
        save(f"noise_b{b}_{g}.png", rgb(np.kron(small, np.ones((b, b)))[:H, :W] == 1))

    # E. COLOR — wave 1 was all grayscale; the patch conv sees 3 channels
    chk = ((xs // 7) + (ys // 7)) % 2 == 1
    col = np.zeros((H, W, 3))
    col[..., 0] = np.where(chk, 255, 0)                      # R alternates
    col[..., 1] = np.where(((xs // 14) % 2) == 1, 255, 0)    # G different pitch
    col[..., 2] = np.where(((ys // 3) % 2) == 1, 255, 0)     # B fine
    save(f"colorclash_{g}.png", col)
    opp = np.zeros((H, W, 3))
    opp[..., 0] = np.where(chk, 255, 0)
    opp[..., 1] = np.where(chk, 0, 255)
    opp[..., 2] = np.where(chk, 255, 0)                      # magenta/green opponent
    save(f"opponent_p7_{g}.png", opp)

    # F. dense text-like rows (high horizontal frequency, blank gutters)
    row = ((xs // 3) % 2 == 1) & ((ys % 18) < 11)
    save(f"textrows_{g}.png", rgb(row))

# benign control
save("flat_gray_1350x1800.png", np.full((1800, 1350, 3), 128))
