#!/usr/bin/env python3
"""Does image_max_tokens still move the MLX gemma4 prompt on the v0.33.3 fold?

D1-A kept the fork's gemma4 MLX vision because upstream's PrepareMedia has no
per-request budget. Preflight's pinned_image_token_budget covers the
llama-server path only, so that decision had no end-to-end probe. This is it:
one image, the same request, image_max_tokens swept over gemma4's documented
soft-token ladder, prompt_eval_count read back. A working seam steps the count
down; a silently-ignored knob returns the same count every time.

    probe-mlx-budget.py <host> <label> [model] [image]
"""
import base64, json, sys, time, urllib.request, urllib.error

host, label = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "gemma4:12b-nvfp4"
img = sys.argv[4] if len(sys.argv) > 4 else "/opt/github/MaxusAI/ollama/docs/maxusai/vision-suite/preflight/ladderimgs/2048x1152.png"
b64 = base64.b64encode(open(img, "rb").read()).decode()

def ask(budget):
    opts = {"num_predict": 1, "num_ctx": 8192, "temperature": 0}
    if budget is not None:
        opts["image_max_tokens"] = budget
    payload = {"model": model, "prompt": "Describe this image in one word.",
               "images": [b64], "stream": False, "think": False, "options": opts}
    req = urllib.request.Request(host + "/api/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        return r.get("prompt_eval_count"), None, time.time() - t0
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:120]}", time.time() - t0
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}", time.time() - t0

print(f"[{label}] model={model} image={img.split('/')[-1]}", flush=True)
n, err, dt = ask(None)   # warm-up + the default-budget reading
print(f"  {'default':>8}: prompt_eval_count={n} {err or ''} ({dt:.1f}s, includes cold start)", flush=True)
rows = [("default", n)]
for b in (1120, 560, 280, 140, 70):
    n, err, dt = ask(b)
    print(f"  {b:>8}: prompt_eval_count={n} {err or ''} ({dt:.1f}s)", flush=True)
    rows.append((str(b), n))
vals = [v for _, v in rows if isinstance(v, int)]
distinct = len(set(vals))
print(f"RESULT [{label}] readings={rows} distinct={distinct} "
      f"verdict={'KNOB LIVE' if distinct > 1 else 'KNOB INERT (all equal)'}", flush=True)
