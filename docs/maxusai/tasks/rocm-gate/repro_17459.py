#!/usr/bin/env python3
"""#17459 — gemma4 emits repeated <unused49> when think=false.

    repro_17459.py <label> <image> <prod|reporter> [KEY=VALUE ...]

Reporter's request verbatim (model tag mapped to the local build of the same
model), with the maintainer's question answered directly: every arm is run both
cold (model unloaded first, load_duration recorded as proof) and warm (the model
already resident from the opposite think mode).
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

import gatelib as g

MODEL = "gemma4:31b-it-q4_K_M"          # reported as gemma4:31b; digest 6316f0629137
PROMPT = "Explain DHCP in one paragraph."
REPS = 3
UNUSED = re.compile(r"<unused\d+>")


def chat(think):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "stream": True, "think": think, "options": {"num_predict": 128}}
    req = urllib.request.Request(g.HOST + "/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rec = {"think": think, "frames": 0, "think_frames": 0, "unused_frames": 0, "done": False,
           "content": "", "thinking": "", "http_status": None, "error": None, "final": None}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            rec["http_status"] = r.status
            for line in r:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "error" in d:
                    rec["error"] = d["error"]
                    break
                m = d.get("message") or {}
                c, th = m.get("content", ""), m.get("thinking", "")
                if c:
                    rec["frames"] += 1
                if th:
                    rec["think_frames"] += 1
                # A <unused49> flood can arrive in either stream: think=true spends
                # the whole 128-token cap inside the thinking block, so content is
                # empty and a content-only check would score a flood there as clean.
                if UNUSED.search(c) or UNUSED.search(th):
                    rec["unused_frames"] += 1
                rec["content"] += c
                rec["thinking"] += th
                if d.get("done"):
                    rec["done"] = True
                    rec["final"] = {k: d.get(k) for k in (
                        "done_reason", "load_duration", "prompt_eval_count",
                        "eval_count", "eval_duration", "total_duration")}
    except urllib.error.HTTPError as e:
        # 0.34.x aborts a token-repeat run with HTTP 500 instead of a truncated 200
        rec["http_status"] = e.code
        rec["error"] = e.read().decode(errors="replace")[:500]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["unused_tokens"] = len(UNUSED.findall(rec["content"])) + len(UNUSED.findall(rec["thinking"]))
    rec["longest_repeat_run"] = max(longest_run(rec["content"]), longest_run(rec["thinking"]))
    rec["degenerate"] = bool(rec["unused_tokens"] or not rec["done"] or rec["error"]
                             or rec["longest_repeat_run"] >= 20)
    ld = (rec["final"] or {}).get("load_duration") or 0
    rec["load_s"] = round(ld / 1e9, 2)
    return rec


def longest_run(text):
    """Longest run of one repeated token-ish unit, the degeneration signature."""
    toks = re.findall(r"<[^>]{1,20}>|\w+|[^\w\s]", text)
    best = cur = 0
    for i, t in enumerate(toks):
        cur = cur + 1 if i and t == toks[i - 1] else 1
        best = max(best, cur)
    return best


def arm(name, think, cold):
    if cold:
        g.unload(MODEL)
    r = chat(think)
    r["arm"], r["cold_requested"] = name, cold
    flag = "DEGENERATE" if r["degenerate"] else "ok"
    print(f"  {name:22s} think={str(think):5s} load={r['load_s']:6.2f}s "
          f"frames={r['frames']:3d}+{r['think_frames']:3d}th unused={r['unused_tokens']:3d} run={r['longest_repeat_run']:3d} "
          f"done={r['done']} http={r['http_status']} {flag}"
          + (f"  err={r['error'][:80]}" if r["error"] else ""), flush=True)
    return r


def main():
    label, image, profile = sys.argv[1:4]
    env = dict(g.PROD_ENV if profile == "prod" else {})
    env.update(kv.split("=", 1) for kv in sys.argv[4:])
    meta = g.start(image, env, f"17459_{label}")
    rows = []
    try:
        digest = next(m["digest"] for m in g.api("/api/tags")["models"] if m["name"] == MODEL)
        meta["model"], meta["model_digest"] = MODEL, digest[:12]
        for rep in range(1, REPS + 1):
            for name, think, cold in (("cold_think_false", False, True),
                                      ("warm_think_true", True, False),
                                      ("cold_think_true", True, True),
                                      ("warm_think_false", False, False)):
                r = arm(name, think, cold)
                r["rep"] = rep
                rows.append(r)
    finally:
        log = g.stop(f"17459_{label}")
        meta["server_log"] = log
        meta["offload_evidence"] = g.offload_evidence(log)
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        g.save(f"result_17459_{label}.json", {"meta": meta, "rows": rows})
    bad = [r for r in rows if r["degenerate"]]
    print(f"== {label}: {len(bad)}/{len(rows)} degenerate "
          f"(think=false: {sum(1 for r in bad if not r['think'])}/{sum(1 for r in rows if not r['think'])})",
          flush=True)


if __name__ == "__main__":
    main()
