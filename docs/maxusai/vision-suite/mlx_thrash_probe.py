#!/usr/bin/env python3
"""Reproduce MLX's CUDA graph-cache thrashing fuse (ml-explore/mlx #4326) from a cold start.

    python3 mlx_thrash_probe.py <host> <label> <n> [model]

Sends n text-only /api/generate requests whose PREFILL LENGTHS are all distinct
("Count: 0 1 2 ... i"), num_predict=1, num_ctx=8192. Each new length is a new
CUDA-graph key; MLX's LRUCache keeps a LIFETIME miss counter and throws once it
exceeds 2 * MLX_CUDA_GRAPH_CACHE_SIZE (default 400). Ollama turns that throw into
a runner abort, and the deferred prefix-cache close() then panics on the poisoned
encoder with cudaGraphAddDependencies -- which is the error that actually gets
logged. See ../mlx-thrash-check-masks-as-cudagraph.md.

Reports per request: ok / 500-thrash / 500-cudaGraph / other, and a RESULT line
with first_fail_at. Measured 2026-08-22/23 (gemma4:12b-nvfp4, cold container per run):

  MLX_CUDA_GRAPH_CACHE_SIZE=8                 -> 120/120 fail from request 1 (new AND old image)
  defaults, n=1000                            -> request 708 fails as 500-cudaGraph, thrash text absent
  MLX_ENABLE_CACHE_THRASHING_CHECK=0          -> 120/120 ok under the cap-8 conditions that failed 120/120

Text-only on purpose: the mechanism is shape-count-driven, not modality-driven, and
a text driver reproduces in minutes what the vision suite took hours to hit.
"""
import json, sys, time, urllib.request, urllib.error

host, label, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = sys.argv[4] if len(sys.argv) > 4 else "gemma4:12b-nvfp4"
first = None; fails = 0; oks = 0; streak = []
t_start = time.time()
for i in range(1, n + 1):
    prompt = "Count: " + " ".join(str(k) for k in range(i))      # i distinct -> unique token length
    pl = {"model": model, "prompt": prompt, "stream": False, "think": False,
          "options": {"num_predict": 1, "num_ctx": 8192, "temperature": 0}}
    req = urllib.request.Request(host + "/api/generate", data=json.dumps(pl).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        b = json.loads(urllib.request.urlopen(req, timeout=1800).read()); oks += 1; st = "ok"; pe = b.get("prompt_eval_count")
    except urllib.error.HTTPError as e:
        body = e.read().decode(); pe = None
        st = "500-cudaGraph" if "cudaGraph" in body else ("500-thrash" if "thrashing" in body else f"{e.code}:{body[:50]}")
        fails += 1
        if first is None: first = i
    except Exception as e:
        st = f"err:{str(e)[:40]}"; pe = None; fails += 1
        if first is None: first = i
    streak.append(st)
    if i <= 3 or st != "ok" or i % 50 == 0:
        print(f"  [{label}] req{i:4d} prompt_eval={pe} {st} ({time.time()-t0:.1f}s)", flush=True)
print(f"RESULT [{label}] n={n} ok={oks} fail={fails} first_fail_at={first} "
      f"tail={''.join('.' if s == 'ok' else 'X' for s in streak[-40:])} wall={time.time()-t_start:.0f}s", flush=True)
