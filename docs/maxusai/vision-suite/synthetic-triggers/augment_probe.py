"""Augment a base image and measure how close blk31.ffn_down gets to the fp16 cliff.

Objective (continuous, defined even when nothing overflows):

    ratio_c = max_t |x[t, c]|  /  thr_c        thr_c = 65504 / max_row |W[:, c]|

x is the INPUT to v.blk.31.ffn_down (the SwiGLU output), W its f16 weight.
ratio >= 1.0 means at least one single product in that channel's dot-product
already exceeds the fp16 range, i.e. the accumulator is guaranteed to blow.

Reported per augmentation: max ratio, how many channels sit above 0.5/0.8/
0.9/1.0, the raw peak, the down_proj output peak (comparable to the numbers
in screen_activation_peak.py) and the token count.

CPU work (augmentation + processor patchify) runs in DataLoader workers so the
GPU stays fed; the model and the hook live in the main process only.

Results carry NUMBERS ONLY -- no image data, no model output text. The base
image may be private; nothing derived from its content is written here.
"""
import os, json, sys
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import cv2, albumentations as A

BASE    = os.environ["BASE_IMAGE"]
OUT     = os.environ.get("OUT_JSONL", "/out/aug.jsonl")
NPZ     = os.environ["BLK31_NPZ"]
MODEL   = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
WORKERS = int(os.environ.get("WORKERS", "8"))
MAXPX   = int(os.environ.get("MAX_PIXELS", str(2048 * 2048)))
TAG     = os.environ.get("BASE_TAG", "base")
FP16_MAX = 65504.0

# ---------------------------------------------------------------- augmentations
# Deterministic parametric ops (numpy) for the axes we want swept precisely;
# albumentations for the non-trivial photometric ones.
def _u8(a): return np.clip(a, 0, 255).astype(np.uint8)

def contrast(alpha):                      # pivot on mid-grey
    return lambda im: _u8((im.astype(np.float32) - 128.0) * alpha + 128.0)
def brightness(beta):
    return lambda im: _u8(im.astype(np.float32) + beta)
def gamma(g):
    lut = _u8(((np.arange(256) / 255.0) ** g) * 255.0)
    return lambda im: lut[im]
def invert():
    return lambda im: 255 - im
def solarize(t):
    return lambda im: np.where(im >= t, 255 - im, im).astype(np.uint8)
def posterize(bits):
    m = 256 - (1 << (8 - bits))
    return lambda im: (im & m).astype(np.uint8)
def saturate(s):
    def f(im):
        h = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
        h[..., 1] = np.clip(h[..., 1] * s, 0, 255)
        return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return f
def noise(sigma, seed):
    def f(im):
        rng = np.random.default_rng(seed)
        return _u8(im.astype(np.float32) + rng.normal(0, sigma, im.shape))
    return f
def scale(k):
    def f(im):
        h, w = im.shape[:2]
        return cv2.resize(im, (max(28, int(w * k)), max(28, int(h * k))),
                          interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return f
def alb(t):
    return lambda im: t(image=im)["image"]

AUGS = [("identity", lambda im: im)]
AUGS += [(f"contrast_a{a:.2f}",   contrast(a))   for a in (1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)]
AUGS += [(f"bright_b{b:+d}",      brightness(b)) for b in (-60, -40, -20, 10, 25, 40, 60, 80)]
AUGS += [(f"gamma_g{g:.2f}",      gamma(g))      for g in (0.4, 0.6, 0.8, 1.25, 1.6, 2.2)]
AUGS += [("invert", invert())]
AUGS += [(f"solarize_t{t}",       solarize(t))   for t in (96, 128, 160, 192, 224)]
AUGS += [(f"posterize_b{b}",      posterize(b))  for b in (6, 5, 4, 3, 2)]
AUGS += [(f"saturate_s{s:.2f}",   saturate(s))   for s in (0.0, 0.5, 1.5, 2.0, 3.0)]
AUGS += [(f"noise_sig{s}_r{r}",   noise(s, 1000 + 17 * r + s)) for s in (1, 2, 4, 8, 16, 32) for r in range(3)]
AUGS += [(f"scale_k{k:.2f}",      scale(k))      for k in (0.5, 0.71, 1.25, 1.5, 2.0)]
AUGS += [
    ("alb_clahe",      alb(A.CLAHE(clip_limit=(4.0, 4.0), p=1))),
    ("alb_equalize",   alb(A.Equalize(p=1))),
    ("alb_sharpen",    alb(A.Sharpen(alpha=(0.5, 0.5), lightness=(1.0, 1.0), p=1))),
    ("alb_tonecurve",  alb(A.RandomToneCurve(scale=0.5, p=1))),
    ("alb_autocontr",  alb(A.AutoContrast(p=1))),
]
# combinations of the two axes the user flagged as most promising
AUGS += [(f"contrast_a{a:.2f}+bright_b{b:+d}",
          (lambda a=a, b=b: (lambda im: _u8((im.astype(np.float32) - 128.0) * a + 128.0 + b)))())
         for a in (1.5, 2.0, 2.5) for b in (25, 50)]
AUGS += [(f"invert+contrast_a{a:.2f}",
          (lambda a=a: (lambda im: _u8((255.0 - im.astype(np.float32) - 128.0) * a + 128.0)))())
         for a in (1.5, 2.0, 2.5)]

# ---------------------------------------------------------------- data pipeline
class AugSet(Dataset):
    def __init__(self):
        self.src = np.array(Image.open(BASE).convert("RGB"))
        self.proc = None
    def __len__(self): return len(AUGS)
    def __getitem__(self, i):
        if self.proc is None:   # build once per worker
            self.proc = AutoProcessor.from_pretrained(MODEL, min_pixels=3136, max_pixels=MAXPX)
        name, fn = AUGS[i]
        img = Image.fromarray(fn(self.src))
        out = self.proc.image_processor(images=[img], return_tensors="pt")
        return name, out["pixel_values"], out["image_grid_thw"], img.size

def collate(b): return b[0]

def main():
    W = np.load(NPZ)["v_blk_31_ffn_down_weight"].astype(np.float32)   # [1280, 3420]
    thr = FP16_MAX / np.maximum(np.abs(W).max(axis=0), 1e-9)          # per input channel
    thr_t = torch.from_numpy(thr).cuda()
    absW = torch.from_numpy(np.abs(W)).cuda()                         # [1280, 3420]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    vis = model.model.visual if hasattr(model.model, "visual") else model.visual
    cap = {}
    vis.blocks[31].mlp.down_proj.register_forward_hook(
        lambda m, i, o: cap.update(x=i[0].detach(), y=o.detach()))

    dl = DataLoader(AugSet(), batch_size=1, num_workers=WORKERS, collate_fn=collate)
    rows = []
    with open(OUT, "w") as fh:
        for name, pv, thw, size in dl:
            cap.clear()
            with torch.no_grad():
                vis(pv.to("cuda:0", torch.bfloat16), grid_thw=thw.to("cuda:0"))
            x = cap["x"].float()                            # [tokens, 3420]
            a = x.abs().amax(dim=0)                         # per-channel max over tokens
            ratio = a / thr_t
            # Worst-case partial sum for every output element: no accumulation
            # order can exceed this, so l1 < 65504 => provably safe, and
            # l1 >> 65504 => the fp16 accumulator has little headroom.
            l1 = x.abs() @ absW.T                           # [tokens, 1280]
            y = cap["y"].float().abs()
            r = dict(base=TAG, aug=name, wh=list(size),
                     tokens=int(thw[0][1] * thw[0][2]),
                     l1_max=round(float(l1.max()), 1),
                     l1_hr=round(float(l1.max()) / FP16_MAX, 4),
                     n_l1_over=int((l1 > FP16_MAX).sum()),
                     frac_l1_over=round(float((l1 > FP16_MAX).float().mean()), 8),
                     max_ratio=round(float(ratio.max()), 5),
                     n_ge_100=int((ratio >= 1.00).sum()),
                     peak_in=round(float(a.max()), 1),
                     peak_out=round(float(y.max()), 1),
                     n_out_gt_50k=int((y > 50000).sum()))
            fh.write(json.dumps(r) + "\n"); fh.flush()
            rows.append(r)
            print(f"{name:<32} l1x={r['l1_hr']:>7.3f}  n_over={r['n_l1_over']:>8,}  "
                  f"peak_out={r['peak_out']:>9,.0f}  tok={r['tokens']:>6}", flush=True)

    b = next(x for x in rows if x["aug"] == "identity")
    print(f"\n=== ranked by worst-case partial sum vs fp16 max "
          f"(identity: l1x={b['l1_hr']:.3f}, n_over={b['n_l1_over']:,}, tok={b['tokens']}) ===")
    print(f"{'l1x':>8} {'vs base':>9} {'n_over':>10} {'per-1e6':>9} {'peak_out':>10} {'tok':>7}  augmentation")
    for r in sorted(rows, key=lambda x: -x["l1_hr"])[:30]:
        print(f"{r['l1_hr']:>8.3f} {r['l1_hr']/b['l1_hr']:>8.2f}x {r['n_l1_over']:>10,} "
              f"{r['frac_l1_over']*1e6:>9.1f} {r['peak_out']:>10,.0f} {r['tokens']:>7}  {r['aug']}")

if __name__ == "__main__":
    main()
