#!/usr/bin/env python3
"""promptcap.py HOST MODEL TEST NUM_CTX VARIANT OUT.json -- thinkcap.py with a prompt variant.

The maintainer's question (2026-09-26): can the prompt, or production's sampling, stop the "real pixel" loops? The
two prompts that ask for absolute pixel coordinates are the only qwen3.6 bbox cells that never finish; the same
scene in the model's own normalized frame finishes (adv_norm1 in 1,395 tokens against adv_real, which never does).

VARIANT
  orig    the suite's prompt, unchanged
  size    replaces the one instruction the model cannot satisfy ("If you resized the image internally, give the
          size YOU used ..."; the resize is invisible to it) with the image's real pixel size, as a client that
          holds the image can state it
  commit  replaces the same instruction with one to commit to a single size estimate and not revisit it
Both variants drop the same sentence, so they differ only in whether the model is given the size.
Sampling is sampling.py's: greedy for think-on by default (the suite's policy), or the model card's with
THINK_TEMPERATURE=1, which is what production sends. Cold, like thinkcap.py: every model is evicted first. The output
adds a capture block naming the variant and the sampling actually applied."""
import json
import os
import re
import sys

host, model, test, num_ctx, variant, out = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5],
                                            sys.argv[6])
os.environ.setdefault("ENDPOINT", "chat")
os.environ["THINK"] = "on"
os.environ["NUM_CTX"] = str(num_ctx)
os.environ["NUM_PREDICT"] = str(num_ctx - 8192)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402
import sampling  # noqa: E402
import vision_suite as vs  # noqa: E402

RESIZE = re.compile(r"If\s+you\s+resized\s+the\s+image\s+internally,\s+give\s+the\s+size\s+YOU\s+used,\s+not\s+the"
                    r"\s+size\s+you\s+were\s+sent\.")


def transform(prompt, variant):
    if variant == "orig":
        return prompt
    w, h = vs.GT["scene_hd"]["size"]
    if variant == "size":
        new, n = RESIZE.subn(f"The image is {w}x{h} pixels (width x height); give pixel coordinates in that frame.",
                             prompt, 1)
    elif variant == "commit":
        new, n = RESIZE.subn("If you do not know the image's pixel size, choose your best estimate once, give it "
                             "as ref_size, and do not revisit it.", prompt, 1)
    else:
        sys.exit(f"unknown variant {variant!r}")
    if n != 1:
        sys.exit(f"variant {variant!r} did not apply to {test}'s prompt")
    return new


if __name__ == "__main__":
    vs.HOST, vs.MODEL = host, model
    entry = next(t for t in vs.tests if t[0] == test)
    prompt = transform(entry[1]() if callable(entry[1]) else entry[1], variant)
    if os.environ.get("DRY"):
        print(prompt)
        sys.exit(0)
    images = [vs.b64(i) for i in entry[2]]
    client.evict_all(host)
    r = vs.gen(prompt, images, num_predict=num_ctx - 8192, num_ctx=num_ctx)
    r = {k: v for k, v in r.items() if k != "context"}
    r["capture"] = {"host": host, "model": model, "test": test, "num_ctx": num_ctx, "variant": variant,
                    "sampling": sampling.sampling_for(model, True, warn=False)}
    with open(out, "w") as fh:
        json.dump(r, fh, ensure_ascii=False)
    print(test, variant, r.get("done_reason"), r.get("eval_count"), len(r.get("thinking") or ""),
          len(r.get("response") or ""))
