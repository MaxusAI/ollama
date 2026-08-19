#!/usr/bin/env python3
"""Render the SPEC C13-C18 geometry set: one scene, many image sizes.

SPEC vision-bbox-response-contract.md section 4 pins the contract's untested
axis -- every rate in that document was scored on `scene_hd` at 1920x1080. This
emits the same scene at fourteen geometries so the anchor can be measured off
that single point.

WRITES TO visimgs/geom/ AND NOTHING ELSE. It deliberately does not touch
visimgs/*.png: those four fixtures were rendered 2026-08-07 against DejaVu at
/usr/share/fonts/truetype/dejavu/, a path that does not exist on macOS, so
re-running gen_scenes.py here would silently replace the entire scored corpus
with a different-font render. Every published rate would then be comparing
against images nobody measured.

Shapes are held as FRACTIONS of (W, H) rather than pixels, so ground truth
scales exactly with the fixture and no geometry needs its own truth measured.
The fractions are the canonical gen_scenes.py:16-23 boxes divided by 1920x1080,
and `_assert_canonical()` scales them back and compares to the integers -- drift
between the two generators fails loudly at import instead of producing a corpus
that looks fine and is not the same scene.
"""
import hashlib
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visimgs", "geom")

# The canonical scene, verbatim from gen_scenes.py:16-23. Kept as pixels here
# purely so _assert_canonical() has something to check the fractions against.
CANON_W, CANON_H = 1920, 1080
CANON_SHAPES = [
    ("rectangle", "red",    (200, 40, 40),   (140, 160, 420, 360)),
    ("ellipse",   "blue",   (40, 70, 200),   (620, 120, 900, 330)),
    ("rectangle", "green",  (40, 160, 70),   (1150, 180, 1500, 420)),
    ("ellipse",   "orange", (235, 140, 30),  (220, 600, 480, 860)),
    ("rectangle", "purple", (120, 50, 160),  (760, 640, 1040, 920)),
    ("ellipse",   "teal",   (0, 150, 150),   (1350, 620, 1720, 880)),
]
LABELS = ["ANCHOR", "BEACON", "CIPHER", "DYNAMO", "EMBER", "FALCON"]
SERIAL = "SN-4921-XK"

# Fractional form: what actually drives rendering at every geometry.
FRAC_SHAPES = [
    (kind, cname, rgb, (bb[0] / CANON_W, bb[1] / CANON_H,
                        bb[2] / CANON_W, bb[3] / CANON_H))
    for kind, cname, rgb, bb in CANON_SHAPES
]

# SPEC section 4.1. Tier 1 isolates one variable per pair; tier 2 is the frozen
# random.seed(20260818) draw standing in for "pasted into a chat window".
# hd_al32 / hd_al48 are BOTH literals -- C15 constrains which one a run selects
# (by the arch's patch_stride), not how it is computed, because a fixture is a
# PNG written before any model is loaded.
GEOMETRIES = [
    # tier 1
    ("hd",       1920, 1080, "control -- every existing rate in the SPEC"),
    ("hd_al32",  1920, 1088, "alignment twin, stride 32"),
    ("hd_al48",  1920, 1104, "alignment twin, stride 48"),
    ("sq320",     320,  320, "below floor; square kills BOTH C7 checks"),
    ("vga",       800,  600, "below floor on qwen35; asymmetric pad at 48"),
    ("portrait", 1080, 1920, "aspect only -- same pixel count as hd"),
    ("uhd",      3072, 1728, "above budget; aligned at both strides"),
    ("uhd4k",    3840, 2160, "C17 frame < input"),
    # tier 2
    ("paste1",   1668,  733, "pasted"),
    ("paste2",   2812, 2135, "pasted"),
    ("paste3",   1235, 1181, "pasted -- near-square, C7 aspect dead"),
    ("paste4",   2750, 2379, "pasted -- near-square, C7 aspect dead"),
    ("paste5",   3030, 1549, "pasted"),
    ("paste6",   3011, 2317, "pasted"),
]

# DejaVu first so a Linux run matches the scored corpus; Arial is the macOS
# fallback. The resolved path AND its sha256 go into ground truth: a geometry
# result that disagrees across hosts must be attributable to the font rather
# than argued about, and "which font" is not recoverable from a PNG later.
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]

# Labels must stay readable or the task stops being a geometry test and becomes
# a legibility test -- a model cannot emit a box for DYNAMO it cannot read. Pure
# proportional scaling puts the 20px label at 4px on sq320, so the size is
# clamped and the clamp is recorded per geometry. SPEC section 4.3 already bars
# text metrics from gating conformance across geometries; this keeps the
# *shape* task well-posed regardless.
LABEL_PX_HD, SERIAL_PX_HD, MIN_LABEL_PX = 20, 14, 11


def _assert_canonical():
    """Fractions must reproduce gen_scenes.py's integers at 1920x1080."""
    for (_, _, _, frac), (_, _, _, px) in zip(FRAC_SHAPES, CANON_SHAPES):
        back = (round(frac[0] * CANON_W), round(frac[1] * CANON_H),
                round(frac[2] * CANON_W), round(frac[3] * CANON_H))
        assert back == px, f"fraction drift: {back} != {px}"


def _fonts():
    for regular, bold in FONT_CANDIDATES:
        if os.path.exists(regular):
            digest = hashlib.sha256(open(regular, "rb").read()).hexdigest()[:16]
            return regular, (bold if os.path.exists(bold) else regular), digest
    sys.exit("no usable font found; tried:\n  " +
             "\n  ".join(r for r, _ in FONT_CANDIDATES))


def render(name, W, H, why, font_path, font_sha):
    scale = (W * H / (CANON_W * CANON_H)) ** 0.5     # area-preserving
    label_px = max(MIN_LABEL_PX, round(LABEL_PX_HD * scale))
    serial_px = max(6, round(SERIAL_PX_HD * scale))
    label_font = ImageFont.truetype(font_path, label_px)
    tiny_font = ImageFont.truetype(font_path, serial_px)

    img = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(img)
    objects = []
    for (kind, cname, rgb, frac), label in zip(FRAC_SHAPES, LABELS):
        bb = (round(frac[0] * W), round(frac[1] * H),
              round(frac[2] * W), round(frac[3] * H))
        (d.rectangle if kind == "rectangle" else d.ellipse)(bb, fill=rgb)
        # Label sits above the shape at HD; at small geometries the clamped
        # font would otherwise overlap the shape it names, so the offset
        # tracks the font actually used rather than a scaled constant.
        d.text((bb[0], max(0, bb[1] - label_px - 8)), label,
               font=label_font, fill=(20, 20, 20))
        objects.append({"label": label, "kind": kind, "color": cname,
                        "bbox": list(bb)})
    d.text((max(0, W - round(150 * scale)), max(0, H - round(30 * scale))),
           SERIAL, font=tiny_font, fill=(90, 90, 90))
    img.save(os.path.join(OUT, f"scene_{name}.png"))

    return {
        "objects": objects, "serial": SERIAL, "size": [W, H],
        "aspect": round(W / H, 4), "why": why,
        "label_px": label_px, "label_px_clamped": label_px != round(LABEL_PX_HD * scale),
        "serial_px": serial_px,
        # Not scored across geometries (SPEC 4.3) -- recorded so a reader can
        # see WHY, rather than discovering it from a puzzling zero.
        "serial_legible": serial_px >= 9,
        "font": os.path.basename(font_path), "font_sha256_16": font_sha,
    }


def main():
    _assert_canonical()
    os.makedirs(OUT, exist_ok=True)
    font_path, _bold, font_sha = _fonts()
    gt = {}
    for name, W, H, why in GEOMETRIES:
        gt[name] = render(name, W, H, why, font_path, font_sha)
    with open(os.path.join(OUT, "ground_truth.json"), "w") as fh:
        json.dump(gt, fh, indent=1)
    print(f"font: {font_path} (sha256:{font_sha})")
    for name, W, H, _ in GEOMETRIES:
        g = gt[name]
        flag = "  <- label clamped" if g["label_px_clamped"] else ""
        print(f"  scene_{name}.png  {W}x{H}  aspect {g['aspect']:.3f}  "
              f"label {g['label_px']}px{flag}")


if __name__ == "__main__":
    main()
