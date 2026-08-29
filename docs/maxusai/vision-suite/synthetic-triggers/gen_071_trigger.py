"""Generate the public trigger for ollama 0.7.1's Go engine (qwen2.5vl:3b).

0.7.1 has no clip graph; its vision tower runs through the Go engine, whose
summation order differs from the clip path, so the checkerboards that trigger
0.30+ do nothing here. 297 generated patterns found nothing; a metered search
showed why -- synthetic families top out ~30 % short of the cliff, while
photographs do not.

Base: NASA image 20040421_exp9_02 ("control room"), PUBLIC DOMAIN (NASA), 1280x830.
Transform: contrast x1.5 about mid-grey, then an INTER_AREA downscale so the
image sits under 0.7.1's ~1,003,520 px cap and so reaches the tower unresized
by the server (1280x830 -> 1244x806).

    out = clip((in - 128) * 1.5 + 128, 0, 255)   then fit to <= 1003520 px

The UNMODIFIED base photo is healthy (HHH); only the contrast-boosted version
is degenerate (XXXXX). They ship together as a paired control that differs by
one multiplier. Gain amplifies structure the image already has -- it could not
lift any flat synthetic over the line.
"""
import numpy as np, cv2
from PIL import Image

CAP = 1003520
src = np.array(Image.open("base_nasa_20040421_exp9_02.jpg").convert("RGB")).astype(np.float32)
out = np.clip((src - 128.0) * 1.5 + 128.0, 0, 255).astype(np.uint8)
h, w = out.shape[:2]
if h * w > CAP:
    s = (CAP / (h * w)) ** 0.5
    out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
Image.fromarray(out).save("trigger_071_nasa_contrast15.png")
