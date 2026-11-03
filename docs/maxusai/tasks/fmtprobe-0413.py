#!/usr/bin/env python3
"""nemotron3 format probe for #501 (CUDA side of the ROCm host's bisect): the suite's own bboxm requests, greedy,
think off, num_ctx 8192, each arm once with format "json" and once with none, in a fixed order on a fresh server.
usage: probe.py <host> <out.json> [model]"""
import base64
import json
import os
import sys

VS = "/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch/wt-sync0413/docs/maxusai/vision-suite"
sys.path.insert(0, VS)
import client  # noqa: E402
import vision_suite  # noqa: E402

host, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "nemotron3:33b-q4_K_M"
vision_suite.MODEL = model
arms = {name: prompt for name, prompt, _imgs, _score in vision_suite.COMPOSED_ARMS}
img = base64.b64encode(open(os.path.join(VS, "visimgs", "scene_hd.png"), "rb").read()).decode()
order = ["bboxm_pin_anc_named", "bboxm_free_anc_named", "bboxm_free_anc_pos", "bboxm_free_noanc_named"]
res = []
for fmt in ("json", None):
    for name in order:
        p = arms[name]
        d = client.generate(host, model, p() if callable(p) else p, [img], num_ctx=8192, fmt=fmt, think=False)
        r = {"arm": name, "format": fmt, "eval_count": d.get("eval_count"), "done_reason": d.get("done_reason"),
             "prompt_eval_count": d.get("prompt_eval_count"), "response": d.get("response", "")}
        res.append(r)
        txt = r["response"]
        nested = '"bbox_2d": {' in txt or '"bbox_2d":{' in txt
        print(f"{name:26s} fmt={str(fmt):5s} tok={r['eval_count']} done={r['done_reason']} "
              f"nested_bbox_2d={nested} head={txt[:80]!r}", flush=True)
json.dump(res, open(out, "w"), indent=1)
