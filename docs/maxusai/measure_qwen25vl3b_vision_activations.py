"""Why is a poison image poison? Measure per-stage max |activation| in the
Qwen2.5-VL-3B vision tower (bf16 compute = fp32-like range) for the two known
0.32.x/0.7.1 trigger images vs a known-good image. fp16's max finite value is
65,504 -- any stage whose magnitudes approach or exceed that in a partial sum
explains the CUDA fp16-accumulate garbage.
"""
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = "/mnt/4TB_SN850X_RAID1_BTRFS/opt/github/Qwen/Qwen2.5-VL-3B-Instruct"
assert os.path.isdir(MODEL_PATH), MODEL_PATH
CACHE = ("/mnt/4TB_SN850X_RAID1_BTRFS/opt/github/SyncTechAU/data/experiments/"
         "00017.8/image_cache")
IMAGES = {
    "good":        CACHE + "/003a01a50fac895a4693c2d0d914a0f7_3136_802816_28_v2.png",
    "poison-032x": CACHE + "/02c9d7e1563a7c6089f688ddff8ad590_3136_802816_28_v2.png",
    "poison-071":  CACHE + "/04431b0d166a35c73231afb5855ff836_3136_802816_28_v2.png",
}
FP16_MAX = 65504.0

processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=3136, max_pixels=802816)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
    attn_implementation="sdpa")
model.eval()

stage_max = {}


def make_hook(name):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            value = tensor.abs().max().item()
            stage_max[name] = max(stage_max.get(name, 0.0), value)
    return hook


hooks = []
for name, module in model.visual.named_modules():
    if name and name.count(".") <= 2:
        hooks.append(module.register_forward_hook(make_hook(name)))

report = {}
for label, path in IMAGES.items():
    stage_max.clear()
    image = Image.open(path).convert("RGB")
    batch = processor(images=[image], text="<|vision_start|><|image_pad|><|vision_end|>",
                      return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        model.visual(batch["pixel_values"].to(torch.bfloat16),
                     grid_thw=batch["image_grid_thw"])
    report[label] = dict(stage_max)

good = report["good"]
print(f"{'stage':<38} {'good':>12} {'poison-032x':>12} {'poison-071':>12}")
interesting = sorted(
    report["poison-032x"],
    key=lambda k: max(report[v].get(k, 0) for v in report), reverse=True)[:14]
for key in interesting:
    flags = ""
    for label in ("poison-032x", "poison-071"):
        if report[label].get(key, 0) > FP16_MAX:
            flags += f"  <-- {label} EXCEEDS fp16 max"
    print(f"{key:<38} {good.get(key, 0):>12,.0f} "
          f"{report['poison-032x'].get(key, 0):>12,.0f} "
          f"{report['poison-071'].get(key, 0):>12,.0f}{flags}")

for label in IMAGES:
    peak_stage = max(report[label], key=report[label].get)
    peak = report[label][peak_stage]
    print(f"\n{label}: peak |activation| = {peak:,.0f} at '{peak_stage}' "
          f"({peak / FP16_MAX:.2f}x fp16 max)")

for hook in hooks:
    hook.remove()
