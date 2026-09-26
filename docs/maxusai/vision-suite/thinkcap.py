#!/usr/bin/env python3
"""thinkcap.py HOST MODEL TEST NUM_CTX OUT.json -- one suite cell's FULL response, thinking kept, cold.

The suite scores a cell but does not keep its thinking; a loop investigation needs the text for the
repetition profile and for the first divergence between two arms. The request is built by the suite's own
gen() -> client.generate() (one payload builder, SPEC H9), with the runner's environment for think-on
(ENDPOINT=chat, THINK=on, num_predict = num_ctx - 8192). Cold: every model is evicted first, so each capture
starts from the same cache state. The output is the server's response (without `context`) plus a `capture`
block naming the host, model, test and num_ctx, which kvloop_read.py reads.

Written for the v0.34.4 fold's loop probe (#375); used by kvloop.sh (tasks/kv-precision-think-loops.md)."""
import json
import os
import sys

host, model, test, num_ctx, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
os.environ.setdefault("ENDPOINT", "chat")
os.environ["THINK"] = "on"
os.environ["NUM_CTX"] = str(num_ctx)
os.environ["NUM_PREDICT"] = str(num_ctx - 8192)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402
import vision_suite as vs  # noqa: E402

vs.HOST, vs.MODEL = host, model
entry = next(t for t in vs.tests if t[0] == test)
prompt = entry[1]() if callable(entry[1]) else entry[1]
images = [vs.b64(i) for i in entry[2]]
client.evict_all(host)
r = vs.gen(prompt, images, num_predict=num_ctx - 8192, num_ctx=num_ctx)
r = {k: v for k, v in r.items() if k != "context"}
r["capture"] = {"host": host, "model": model, "test": test, "num_ctx": num_ctx}
with open(out, "w") as fh:
    json.dump(r, fh, ensure_ascii=False)
print(test, r.get("done_reason"), r.get("eval_count"), len(r.get("thinking") or ""), len(r.get("response") or ""))
