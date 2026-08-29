"""Do the trigger's activations land on the channels the fp16 audit flagged?

Hooks the INPUT of the two matmuls fp16_audit.py flagged unsafe:
  v.blk.31.ffn_down  (thr 31,054; 29/3420 channels unsafe)
  v.blk.17.ffn_up    (thr 37,431;  2/1280 channels unsafe)
and reports per-channel max|x| across all image tokens vs those thresholds.
"""
import os, sys, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
WORK = os.environ.get("PROBE_DIR", "")  # holds blk31_weights.npz and the decoded probe images
if not WORK:
    raise SystemExit("PROBE_DIR is not set; it must hold blk31_weights.npz and the probe PNGs.")
FP16_MAX = 65504.0
W = np.load(os.path.join(WORK, "blk31_weights.npz"))
thr31 = FP16_MAX / np.maximum(np.abs(W["v_blk_31_ffn_down_weight"].astype(np.float32)).max(axis=0), 1e-9)

proc = AutoProcessor.from_pretrained(MODEL, min_pixels=3136, max_pixels=1003520)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()

vis = model.model.visual if hasattr(model.model, "visual") else model.visual
blocks = vis.blocks
grab = {}
def mk(tag):
    def hook(mod, inp, out): grab[tag] = inp[0].detach().float().abs().amax(dim=0).cpu().numpy()
    return hook
blocks[31].mlp.down_proj.register_forward_hook(mk("b31_down_in"))
blocks[17].mlp.up_proj.register_forward_hook(mk("b17_up_in"))

def run(path, label):
    grab.clear()
    img = Image.open(path).convert("RGB")
    msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe"}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        vis(inputs["pixel_values"].to(torch.bfloat16), grid_thw=inputs["image_grid_thw"])
    a = grab["b31_down_in"]
    over = a > thr31
    print(f"{label:<26} tokens_px={tuple(inputs['image_grid_thw'][0].tolist())}")
    print(f"   b31.ffn_down input: max|x| {a.max():>10,.0f}   channels exceeding their own threshold: "
          f"{int(over.sum()):>3} / {len(a)}")
    if over.any():
        idx = np.where(over)[0][:6]
        print(f"      e.g. ch {list(idx)} -> |x| {[f'{a[i]:,.0f}' for i in idx]} vs thr {[f'{thr31[i]:,.0f}' for i in idx]}")
    print(f"   b17.ffn_up   input: max|x| {grab['b17_up_in'].max():>10,.0f}")
    return int(over.sum())

CACHE = os.environ.get("CORPUS_IMAGE_DIR", "")  # required; no default (private corpus)
if not CACHE:
    raise SystemExit("CORPUS_IMAGE_DIR is not set. These probes read a private image\ncorpus that is deliberately not committed; point it at your own directory.")
res = {}
res["04431b0d GO-decode  [X]"] = run(os.path.join(WORK, "decodecmp/go_decode.png"), "04431b0d GO-decode  [X]")
res["04431b0d libjpeg    [H]"] = run(os.path.join(WORK, "encodetest/A_lossless_png.png"), "04431b0d libjpeg    [H]")
for m, lbl in (("39823be1","39823be1 same-geom [H]"), ("74cee2fe","74cee2fe same-geom [H]"), ("11c11aa8","11c11aa8 same-geom [H]")):
    p = [f for f in os.listdir(CACHE) if f.startswith(m)]
    if p: res[lbl] = run(os.path.join(CACHE, p[0]), lbl)
print("\n=== summary: channels over their fp16 product threshold ===")
for k, v in res.items(): print(f"  {k:<26} {v}")
