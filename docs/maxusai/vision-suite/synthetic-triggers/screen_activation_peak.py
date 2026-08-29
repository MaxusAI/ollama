"""Rank candidate images by block-31 peak activation (the overflow proxy)."""
import os, sys, glob, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
SP = os.path.dirname(os.path.abspath(__file__))
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=3136, max_pixels=1003520)  # 0.7.1's cap
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
vis = model.model.visual if hasattr(model.model, "visual") else model.visual
cap = {}
vis.blocks[31].mlp.down_proj.register_forward_hook(lambda m,i,o: cap.__setitem__("y", o.detach()))
def peak(path):
    img = Image.open(path).convert("RGB")
    msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"d"}]}]
    t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[img], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        vis(inp["pixel_values"].to(torch.bfloat16), grid_thw=inp["image_grid_thw"])
    return float(cap["y"].float().abs().max()), int(inp["image_grid_thw"][0][1]*inp["image_grid_thw"][0][2])
cands = []
for pat in ("w6/*.png",):
    cands += sorted(glob.glob(os.path.join(SP, pat)))
cands = [c for c in cands if "flat_gray" not in c]
rows = []
for c in cands:
    try: p, n = peak(c); rows.append((p, n, os.path.basename(c)))
    except Exception as e: print("skip", os.path.basename(c), e)
rows.sort(reverse=True)
print(f"\nfp16 max = 65,504.  Reference: 04431b0d(Go)=60,672 [X], 39823be1=63,744 [H on 0.7.1]\n")
print(f"{'peak |y| blk31':>15} {'%fp16':>7} {'patches':>8}  candidate")
for p, n, name in rows[:18]:
    print(f"{p:>15,.0f} {100*p/65504:>6.1f}% {n:>8}  {name}")
