#!/usr/bin/env python3
"""Measure /api/generate against /api/chat on the same prompt and image.

    python3 endpoint_compare.py <host> <model> [probe ...]

preflight/probes.py is pinned to /api/generate because every expectation in
expectations.toml was calibrated there, and the suite now defaults to /api/chat
because that is the endpoint upstream keeps current. Those two facts are only
safe together if the difference between the endpoints is KNOWN. Otherwise a
preflight PASS and a campaign result are describing different code paths and
nobody can say by how much.

This runs identical requests both ways through the one shared client
(client.generate with endpoint_override) and reports the deltas. It changes no
baseline; it measures the gap so the pinning can be justified rather than
assumed.

What to expect, and what would be alarming:

  * prompt_eval_count should MATCH. ollama applies the model's template on
    /api/generate as well (it is only raw when raw:true), so a single-turn
    request tokenizes the same both ways -- measured 1511/1511 on
    gemma4:31b-it-q4_K_M. A nonzero delta means one endpoint is templating
    differently and the baselines are not portable.
  * eval_count and the ANSWER should be near-identical at temperature 0. A large
    divergence means the endpoints are not feeding the model equivalent input --
    that is a finding, not noise.
  * a think-on cell returning reasoning on one endpoint and none on the other
    means the think plumbing differs between them, which would invalidate every
    think-mode comparison made on the other endpoint.
"""
import json
import os
import sys

import client

DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(DIR, "visimgs")

# Deliberately the real probes, not a toy prompt: the delta that matters is the
# one on the requests the campaigns actually issue.
import vision_suite as vs

DEFAULT_PROBES = ["bbox_contract", "scene_single"]


def run(host, model, probe):
    entry = next((t for t in vs.tests if t[0] == probe), None)
    if not entry:
        return {"probe": probe, "error": f"unknown probe {probe!r}"}
    _, prompt, images, _ = entry[:4]
    b64 = [client.b64_file(os.path.join(IMG, i)) for i in images]
    out = {"probe": probe}
    for ep in ("generate", "chat"):
        try:
            r = client.generate(host, model, prompt, b64, endpoint_override=ep)
            out[ep] = {
                "prompt_eval_count": r.get("prompt_eval_count"),
                "eval_count": r.get("eval_count"),
                "answer_chars": len(r.get("response", "") or ""),
                "thinking_chars": len(r.get("thinking", "") or ""),
                "answer": (r.get("response", "") or "")[:4000],
            }
        except Exception as e:
            out[ep] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    host, model = sys.argv[1], sys.argv[2]
    probes = sys.argv[3:] or DEFAULT_PROBES
    think = "on" if client.think_on() else "off"

    rows = [run(host, model, p) for p in probes]
    print(f"\n**{model}** — think={think}, temperature per sampling.py\n")
    print("| probe | metric | /api/generate | /api/chat | delta |")
    print("|---|---|---|---|---|")
    for r in rows:
        g, c = r.get("generate", {}), r.get("chat", {})
        if "error" in g or "error" in c:
            print(f"| {r['probe']} | ERROR | {g.get('error','—')} | {c.get('error','—')} | — |")
            continue
        for k in ("prompt_eval_count", "eval_count", "answer_chars", "thinking_chars"):
            gv, cv = g.get(k), c.get(k)
            d = (cv - gv) if isinstance(gv, int) and isinstance(cv, int) else None
            mark = ""
            # The prompt delta is expected; flag it only if it is ZERO, because
            # that would mean chat is not applying a template at all.
            if k == "prompt_eval_count" and d not in (0, None):
                mark = " ⚠ endpoints tokenize differently"
            # The headline defect this tool was built to catch: reasoning
            # returned by one endpoint and silently dropped by the other, while
            # eval_count shows the tokens were generated and paid for either way.
            if k == "thinking_chars" and d is not None and (gv or 0) == 0 and (cv or 0) > 0:
                mark = " ⚠ /api/generate DROPPED the reasoning"
            # Output divergence at temperature 0 is the alarming case.
            if k == "answer_chars" and d is not None and gv and abs(d) > 0.1 * gv:
                mark = " ⚠ >10% output divergence"
            print(f"| {r['probe']} | {k} | {gv} | {cv} | "
                  f"{'' if d is None else f'{d:+d}'}{mark} |")
        same = g.get("answer") == c.get("answer")
        print(f"| {r['probe']} | answer identical | {'—'} | {'—'} | "
              f"{'yes' if same else '**no**'} |")

    out = os.path.join(DIR, f"endpoint_compare_{model.replace(':','_').replace('.','_')}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nraw -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
