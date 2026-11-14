#!/usr/bin/env python3
"""Post-deploy smoke on production :11497 (v0.40.13-dynres): one text and one image request on gemma4:e4b, then unload it."""
import base64, json, time, urllib.request
H = "http://127.0.0.1:11497"; M = "gemma4:e4b-it-q4_K_M"
IMG = "/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch/wt-v0413tag/docs/maxusai/vision-suite/visimgs/scene_hd.png"
def post(path, body, timeout=1500):
    t = time.time()
    r = urllib.request.Request(H + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        d = json.load(f)
    d["_wall_s"] = round(time.time() - t, 1)
    return d
opts = {"num_ctx": 8192, "temperature": 0, "num_predict": 300}
def show(name, d):
    print(f"{name}: wall={d['_wall_s']}s done={d.get('done_reason')} prompt={d.get('prompt_eval_count')} eval={d.get('eval_count')} text={d.get('message',{}).get('content','')[:300]!r}", flush=True)
try:
    show("text", post("/api/chat", {"model": M, "stream": False, "think": False, "options": opts,
        "messages": [{"role": "user", "content": "Reply with the single word: ready."}]}))
    print("ps:", json.load(urllib.request.urlopen(H + "/api/ps", timeout=10)), flush=True)
    img = base64.b64encode(open(IMG, "rb").read()).decode()
    show("image", post("/api/chat", {"model": M, "stream": False, "think": False, "options": opts,
        "messages": [{"role": "user", "content": "List the shapes and colours in this image in one sentence.", "images": [img]}]}))
finally:
    post("/api/generate", {"model": M, "keep_alive": 0}, timeout=120)
    print("after unload ps:", json.load(urllib.request.urlopen(H + "/api/ps", timeout=10)).get("models"), flush=True)
