#!/usr/bin/env python3
"""Matched decode-throughput pairs between two serving stacks on one host.

Built for the `mlx-cuda` vs `cuda` question the README used to answer from an
operational impression. Platform names are the preflight ones (`preflight.py
--platform`): `cuda` is the llama.cpp/llama-server path, `mlx-cuda` is the MLX
runtime on the same GPU. Nothing here touches `mlx-metal`, and no number it
produces may be carried across to that surface -- the quantizations differ, so
the two platforms are not measuring comparable things.

    python3 bench_engine_throughput.py --host http://127.0.0.1:11517 \
        --out runs/mlx-cuda-vs-cuda.json

WHAT THIS MEASURES, AND WHAT IT CANNOT
--------------------------------------
Engine and quantization move together: nvfp4 is the `mlx-cuda` arm, q4_K_M the
`cuda` arm, because those are the artefacts that exist. A ratio measured here
describes THE TWO STACKS AS SHIPPED, not the engine in isolation. The README
already states this confound for output quality; it applies to throughput
identically, and the meta block records it so a downstream reader cannot lose it.

Decode only. `prompt_eval_duration` is not trustworthy on the MLX arm -- measured
2026-08-30, the same arm reported 369 tok/s on one request and 3,348,513 tok/s on
the next, while the `cuda` arm reported coherent figures throughout. Prefill
rates are still recorded, but do not quote them without re-establishing that.

METHOD, AND WHY EACH PIECE IS LOAD-BEARING
------------------------------------------
One server, one container, one version. SPEC H11 makes `server_version` the
comparability boundary; running both arms through a single process guarantees it
rather than asserting it.

  * num_ctx is PINNED. The MLX runner derives a default from free VRAM (262144
    measured) which costs ~25x decode speed. An unpinned MLX arm measures the
    default, not the stack, and every ratio built on it is garbage.
  * think=false. These are reasoning models; a think-on arm spends its budget
    inside the reasoning block and eval_count stops describing answer throughput.
  * num_predict forces every arm to `done_reason: "length"`, so eval_count is a
    constant across arms and decode rate is the only free variable. The harness
    records done_reason; if an arm shows "stop", its rate is over fewer tokens
    and is not directly comparable.
  * One model resident (OLLAMA_MAX_LOADED_MODELS=1), so no arm is measured while
    sharing the GPU with the previous model's weights.
  * Arms INTERLEAVED, headline pair repeated last as an ORDER CONTROL. Without
    it a host that drifts over the run would encode run order as an engine
    difference, and nothing in the data would reveal it.

STATIONARITY IS EVIDENCE, NOT AN ASSUMPTION
-------------------------------------------
The `mlx-cuda` arm is often NOT stationary, and the profile is per-model. All
measured 2026-08-30 on 0.33.2-dynres-5-g2b95b4a:

    gemma4:31b-nvfp4       cold 1.42 -> full speed on request 2, flat after
    gemma4:26b-nvfp4       cold 41.99 -> climbs over 3 requests -> plateau
    qwen3.8:27b-nvfp4      cold 2.31 -> plateau ~25 -> humps to 38.6 at
                           request 8 -> decays to 34 by request 15
    qwen3.6:35b-a3b-nvfp4  settled instantly at n=5 in one run; needed 12
                           requests to reach the same value in the next

The `cuda` arm was flat first time on all ten arms measured (drift within
+/-1.5%). So this whole apparatus exists for one of the two stacks.

Consequences, each of which cost a wasted run to learn:

  * DO NOT trust a discard count. "Discard 3 and you are safe" is false; one arm
    needed 12. `--discard` sets the floor, it does not establish steady state.
  * DO NOT use first-to-last drift. It is noise-dominated -- it reported -8.6%
    on an arm with no trend at all. This harness reports the FIRST-HALF vs
    SECOND-HALF median shift instead, which is what distinguishes a trend from
    scatter.
  * DO NOT read a flat drift figure at small n as "settled". The same arm gave
    +0.5% at n=5 and +23.2% at n=12.
  * The estimator is the TAIL-HALF MEDIAN, reported with the half-shift beside
    it. An arm whose half-shift is large has no single throughput to quote, and
    the honest report says so rather than printing its median.

A short window undersampled 3 of 4 `mlx-cuda` arms by 22-33% and never once
undersampled `cuda`. Errors that all point one way are worse than noisy ones:
they read as confirmation of whatever prior you brought.

Results are written after EVERY request. A kill at minute 40 of a 40-minute run
must not discard 40 minutes of finished measurements.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

# (label, mlx-cuda model, cuda model). Same base model -- verify with /api/show
# that block_count, embedding_length and parameter count agree before adding a
# pair, and record whether it is dense or MoE (expert_count in the GGUF
# metadata). A mismatched pair measures two models, not two stacks.
PAIRS = [
    ("gemma4-31b",      "gemma4:31b-nvfp4",      "gemma4:31b-it-q4_K_M"),
    ("qwen3.8-27b",     "qwen3.8:27b-nvfp4",     "qwen3.8:27b-q4_K_M"),
    ("gemma4-26b",      "gemma4:26b-nvfp4",      "gemma4:26b-a4b-it-q4_K_M"),
    ("qwen3.6-35b-a3b", "qwen3.6:35b-a3b-nvfp4", "qwen3.6:35b-a3b-q4_K_M"),
]

# Long enough that every arm runs into num_predict rather than a stop token.
PROMPT = (
    "Explain in technical detail how a modern GPU executes a large matrix "
    "multiplication: memory hierarchy, tiling, warp scheduling, and where the "
    "arithmetic intensity roofline binds. Be thorough and specific."
)

# A fresh container pays a one-time MLX JIT before its first request: 437.6 s
# and 443.9 s measured on two independent starts of 0.33.2. Spend it on a small
# model, record it, exclude it. It is not a hang -- size timeouts accordingly.
WARMUP_MODEL = "qwen3.5:0.8b-mlx"

MLX, GGML = "mlx-cuda", "cuda"


def post(host, path, payload, timeout):
    req = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate(host, model, num_ctx, num_predict, timeout):
    return post(host, "/api/generate", {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": num_ctx,          # PINNED -- see module docstring
            "num_predict": num_predict,
            "temperature": 0,
            "seed": 42,
        },
    }, timeout)


def rates(resp):
    """Decode and prefill rates from ollama's own timing fields (ns)."""
    ev, evd = resp.get("eval_count"), resp.get("eval_duration")
    pe, ped = resp.get("prompt_eval_count"), resp.get("prompt_eval_duration")
    return {
        "eval_count": ev,
        "prompt_eval_count": pe,
        "decode_tok_s": round(ev / (evd / 1e9), 2) if ev and evd else None,
        # Not trustworthy on mlx-cuda -- see module docstring.
        "prefill_tok_s": round(pe / (ped / 1e9), 2) if pe and ped else None,
        "load_s": round(resp.get("load_duration", 0) / 1e9, 2),
        "total_s": round(resp.get("total_duration", 0) / 1e9, 2),
        "done_reason": resp.get("done_reason"),
    }


def summarize(samples):
    """Tail-half median, with the half-shift that says whether to trust it.

    Reporting a median without the shift is what makes a moving quantity look
    like a plateau. `stationary` is False when the two halves differ by more
    than 5%, and an arm flagged False has no single throughput to quote.
    """
    d = [s["decode_tok_s"] for s in samples if s["decode_tok_s"]]
    if not d:
        return {}
    h = len(d) // 2
    out = {
        "decode_median": round(statistics.median(d), 2),
        "decode_min": round(min(d), 2),
        "decode_max": round(max(d), 2),
        "spread_pct": round((max(d) - min(d)) / statistics.median(d) * 100, 1),
        "n": len(d),
    }
    if h:
        first, second = statistics.median(d[:h]), statistics.median(d[h:])
        out["decode_tail_median"] = round(second, 2)
        out["half_shift_pct"] = round((second - first) / first * 100, 1)
        out["stationary"] = abs(out["half_shift_pct"]) < 5.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=12)
    # A floor, not a guarantee of steady state -- see module docstring.
    ap.add_argument("--discard", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--warmup-timeout", type=int, default=1200)
    a = ap.parse_args()

    ver = json.load(urllib.request.urlopen(a.host + "/api/version", timeout=30))
    state = {
        "meta": {
            "purpose": f"matched {MLX} vs {GGML} decode throughput, same host, "
                       "same server process",
            "server_version": ver["version"],
            "host": a.host,
            "num_ctx_pinned": a.num_ctx,
            "num_predict": a.num_predict,
            "repeats": a.repeats,
            "discard": a.discard,
            "think": False,
            "estimator": "tail-half median; half_shift_pct reports whether the "
                         "arm is stationary enough for a single figure to mean "
                         "anything",
            "confound": "engine and quantization move together (nvfp4 on "
                        f"{MLX} vs q4_K_M on {GGML}); this measures the two "
                        "stacks as shipped, not the engine in isolation",
            "not_applicable_to": "mlx-metal -- different quantization, "
                                 "different hardware, measured separately",
        },
        "warmup": None,
        "runs": [],
    }

    def save():
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, a.out)

    save()
    print(f"server_version={ver['version']} num_ctx={a.num_ctx} "
          f"num_predict={a.num_predict} discard={a.discard} "
          f"repeats={a.repeats}", flush=True)

    print(f"[warmup] {WARMUP_MODEL} (up to {a.warmup_timeout}s cold JIT)",
          flush=True)
    t0 = time.time()
    try:
        r = generate(a.host, WARMUP_MODEL, a.num_ctx, 32, a.warmup_timeout)
        state["warmup"] = {"model": WARMUP_MODEL, "ok": True,
                           "wall_s": round(time.time() - t0, 1), **rates(r)}
    except Exception as e:
        state["warmup"] = {"model": WARMUP_MODEL, "ok": False,
                           "wall_s": round(time.time() - t0, 1),
                           "error": f"{type(e).__name__}: {e}"}
    save()
    print(f"[warmup] done in {state['warmup']['wall_s']}s "
          f"ok={state['warmup']['ok']}", flush=True)

    schedule = []
    for label, mlx, gguf in PAIRS:
        schedule.append((label, MLX, mlx, "first"))
        schedule.append((label, GGML, gguf, "first"))
    label, mlx, gguf = PAIRS[0]
    schedule.append((label, MLX, mlx, "order-control"))
    schedule.append((label, GGML, gguf, "order-control"))

    for label, engine, model, slot in schedule:
        print(f"[{label}/{engine}/{slot}] {model}", flush=True)
        rec = {"pair": label, "engine": engine, "model": model, "slot": slot,
               "warm_discarded": [], "samples": [], "error": None}
        state["runs"].append(rec)
        save()
        try:
            for _ in range(a.discard):
                w = generate(a.host, model, a.num_ctx, a.num_predict, a.timeout)
                rec["warm_discarded"].append(rates(w))
                save()
            for i in range(a.repeats):
                r = generate(a.host, model, a.num_ctx, a.num_predict,
                             a.timeout)
                rec["samples"].append(rates(r))
                save()          # after EVERY sample, not at the end
                print(f"    n{i+1} decode="
                      f"{rec['samples'][-1]['decode_tok_s']} tok/s", flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            save()
            print(f"    ERROR {rec['error']}", flush=True)
            continue

        rec.update(summarize(rec["samples"]))
        save()
        if "decode_tail_median" in rec:
            note = "" if rec["stationary"] else "  <-- NOT stationary; no " \
                                                "single figure to quote"
            print(f"    tail median {rec['decode_tail_median']} tok/s "
                  f"(all-n median {rec['decode_median']}, "
                  f"spread {rec['spread_pct']}%, "
                  f"half-shift {rec['half_shift_pct']:+}%){note}", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
