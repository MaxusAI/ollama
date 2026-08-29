import numpy as np
from PIL import Image
import os
D = os.path.dirname(__file__)
def save(name, a):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(D, name)); print(name)
def g3(a):  # gray 2D -> RGB
    return np.repeat(a[..., None], 3, axis=2)

# ALL geometries <= 1.0 MP so the Go engine's ~1280-token cap does NOT downscale them.
GEOMS = {"1288x616": (616, 1288), "1120x896": (896, 1120), "1400x700": (700, 1400), "896x1120": (1120, 896)}

for gname, (H, W) in GEOMS.items():
    ys, xs = np.mgrid[0:H, 0:W]
    rs = np.random.RandomState(7)

    # A. 1/f "pink" noise — natural image spectrum
    for beta in (1.0, 2.0):
        f = np.fft.fftfreq(H)[:, None] ** 2 + np.fft.fftfreq(W)[None, :] ** 2
        f[0, 0] = 1e-6
        spec = np.fft.fft2(rs.randn(H, W)) / (f ** (beta / 2))
        img = np.real(np.fft.ifft2(spec))
        img = 255 * (img - img.min()) / ((img.max() - img.min()) + 1e-9)
        save(f"pink{beta:.0f}_{gname}.png", g3(img))

    # B. localized extremes: smooth mid-tone field + few saturated blobs
    base = 128 + 40 * np.sin(2 * np.pi * xs / (W / 3)) * np.cos(2 * np.pi * ys / (H / 3))
    spots = base.copy()
    for cx, cy, s in [(W * .2, H * .3, 9), (W * .7, H * .6, 5), (W * .45, H * .8, 14)]:
        m = ((xs - cx) ** 2 + (ys - cy) ** 2) < s ** 2
        spots[m] = 255
    save(f"specular_{gname}.png", g3(spots))
    inv = base.copy()
    for cx, cy, s in [(W * .3, H * .5, 11), (W * .8, H * .2, 7)]:
        inv[((xs - cx) ** 2 + (ys - cy) ** 2) < s ** 2] = 0
    save(f"voids_{gname}.png", g3(inv))

    # C. THE POISON-PHOTO SHAPE: smooth wood-like grain + thin high-contrast cables
    grain = 120 + 35 * np.sin(2 * np.pi * ys / 23 + 2 * np.sin(2 * np.pi * xs / 400))
    grain += 12 * rs.randn(H, W)
    cables = grain.copy()
    for k in range(9):
        amp, per, off = 30 + 12 * k, 260 + 40 * k, H * (0.1 + 0.09 * k)
        cy = (off + amp * np.sin(2 * np.pi * xs / per)).astype(int)
        for t in (-1, 0, 1):
            yy = np.clip(cy + t, 0, H - 1)
            cables[yy, np.arange(W)] = 255 if k % 2 else 12
    save(f"cables_{gname}.png", g3(cables))
    save(f"cables_hicontrast_{gname}.png", g3(np.where(cables > 128, 255, 0)))

    # D. edge cluster: one busy region, smooth elsewhere (spatially sparse structure)
    field = np.full((H, W), 140.0)
    rx0, ry0 = int(W * .55), int(H * .25)
    rw, rh = int(W * .3), int(H * .45)
    sub_y, sub_x = np.mgrid[0:rh, 0:rw]
    field[ry0:ry0 + rh, rx0:rx0 + rw] = np.where(((sub_x // 2) + (sub_y // 2)) % 2 == 1, 255, 0)
    save(f"edgecluster_{gname}.png", g3(field))

    # E. localized color saturation (single-channel extremes on neutral)
    col = np.full((H, W, 3), 130.0)
    for ch, (cx, cy) in enumerate([(W * .25, H * .3), (W * .5, H * .6), (W * .75, H * .4)]):
        m = ((xs - cx) ** 2 + (ys - cy) ** 2) < (min(H, W) * .09) ** 2
        col[m] = 0; col[m, ch] = 255
    save(f"colorblobs_{gname}.png", col)

    # F. fine texture over gradient (sub-patch detail preserved at native res)
    tex = np.linspace(20, 235, W)[None, :] * np.ones((H, 1))
    tex = tex + 60 * (((xs // 2) + (ys // 2)) % 2)
    save(f"gradtex_{gname}.png", g3(tex))

save("flat_gray_1350x1800.png", np.full((1800, 1350, 3), 128))
