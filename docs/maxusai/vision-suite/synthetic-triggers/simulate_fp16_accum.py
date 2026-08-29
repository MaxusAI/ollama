"""Simulate fp16 split-K accumulation of blk31 ffn_down with REAL activations.

Ground truth on 0.7.1:  04431b0d-Go = X ; 04431b0d-libjpeg = H ; 39823be1 = H ; 11c11aa8 = H
If a tiling reproduces exactly that pattern, we have a predictive objective
for synthesis.  If none does, peak-driven synthesis has no validated target.
"""
import os, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
SP=os.path.dirname(os.path.abspath(__file__)); FP16=65504.0
proc=AutoProcessor.from_pretrained(MODEL,min_pixels=3136,max_pixels=1003520)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,torch_dtype=torch.bfloat16,device_map="cuda:0",attn_implementation="sdpa").eval()
vis=model.model.visual if hasattr(model.model,"visual") else model.visual
cap={}
vis.blocks[31].mlp.down_proj.register_forward_hook(lambda m,i,o: cap.__setitem__("x",i[0].detach()))
W=vis.blocks[31].mlp.down_proj.weight.detach().half()          # (1280, 3420)

def acts(path):
    img=Image.open(path).convert("RGB")
    msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"d"}]}]
    t=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    inp=proc(text=[t],images=[img],return_tensors="pt").to("cuda:0")
    with torch.no_grad(): vis(inp["pixel_values"].to(torch.bfloat16),grid_thw=inp["image_grid_thw"])
    return cap["x"].half()

def simulate(x, T):
    """split-K: sum each K-tile in fp16, then accumulate tile results in fp16."""
    N,K=x.shape; M=W.shape[0]; nt=K//T
    run=torch.zeros(N,M,device="cuda",dtype=torch.float16)
    worst=0.0; over=0
    for t in range(nt):
        s=slice(t*T,(t+1)*T)
        part=(x[:,s].float() @ W[:,s].T.float()).half()        # tile sum
        run=(run+part)                                          # fp16 running accumulate
        w=run.abs().max().item()
        worst=max(worst,w if w==w else float("inf"))
        over+=int((~torch.isfinite(run)).sum())
        if over: break
    return worst, over

CACHE = os.environ.get("CORPUS_IMAGE_DIR", "")  # required; no default (private corpus)
if not CACHE:
    raise SystemExit("CORPUS_IMAGE_DIR is not set. These probes read a private image\ncorpus that is deliberately not committed; point it at your own directory.")
imgs=[("04431b0d GO  [X]", f"{SP}/decodecmp/go_decode.png"),
      ("04431b0d lib [H]", f"{SP}/encodetest/A_lossless_png.png")]
for m in ("39823be1","11c11aa8"):
    p=[f for f in os.listdir(CACHE) if f.startswith(m)]
    if p: imgs.append((f"{m}   [H]", os.path.join(CACHE,p[0])))
print(f"fp16 max {FP16:,.0f}\n")
print(f"{'image':<20}" + "".join(f"{'K='+str(T):>18}" for T in (32,128,512,3420)))
for label,path in imgs:
    x=acts(path); row=""
    for T in (32,128,512,3420):
        w,o=simulate(x,T)
        row+=f"{w:>12,.0f}{'  INF' if o else '     '}"
    print(f"{label:<20}{row}")
    del x; torch.cuda.empty_cache()
