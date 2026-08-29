"""Generate the public triggers for scb10x/typhoon-ocr1.5-3b (upstream ollama#17687).

70 px black/white checkerboards at 1350x1800, phase-shifted so the check
boundaries land off the patch grid. Both fire 5/5 on stock ollama with no
environment variables; the dx=37 variant carries more fp16 headroom and is the
preferred reproducer.

  dx=37, dy=35  headroom 1.068  (max |ffn_down-31| = 69,940)   <- preferred
  dx=35, dy=35  headroom 1.006  (max |ffn_down-31| = 65,919)
  dx=0,  dy=0   headroom 0.919  -> HEALTHY, ships as the paired control

Phase is a sharp axis: dx=36 gives 1.028 and dx=38 gives 0.999, so the peak is
narrow. Deterministic -- no RNG, no fonts, no external assets.
"""
import numpy as np
from PIL import Image

W, H, C = 1350, 1800, 70
for name, (dx, dy) in {
    "trigger_typhoon_c70_dx37_dy35_1350x1800.png": (37, 35),   # preferred
    "trigger_typhoon_c70_halfphase_1350x1800.png": (35, 35),
    "control_typhoon_c70_phase0_1350x1800.png":    (0,  0),    # healthy control
}.items():
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (((xx + dx) // C) + ((yy + dy) // C)) % 2
    img = np.where(mask[..., None] == 0, np.array([0, 0, 0]), np.array([255, 255, 255]))
    Image.fromarray(img.astype(np.uint8)).save(name)
