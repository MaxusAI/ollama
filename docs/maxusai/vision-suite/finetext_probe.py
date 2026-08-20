#!/usr/bin/env python3
"""Dense fine-text probe: does the gemma4 1120-token budget beat upstream's
default on genuinely small text? Generates a 1568x1568 compliance page with
reference codes at descending font sizes, asks for exact transcription,
scores per-size recall.

Usage: finetext_probe.py <host> <tag> <model>
Env: THINK=on|false, ENDPOINT=chat|generate (default chat), NUM_CTX, NUM_PREDICT, HTTP_TIMEOUT
"""
import json
import re, os, sys, base64, random, urllib.request

from sampling import sampling_for, provenance
import client

DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(DIR, "visimgs", "finetext.png")
GT = os.path.join(DIR, "visimgs", "finetext_gt.json")
# FONT_PATH overrides for hosts without the Debian DejaVu path (macOS:
# point it at e.g. matplotlib's bundled DejaVuSansMono.ttf).
FONT = os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

SIZES = [22, 16, 12, 9, 7]  # px per bucket, 4 codes each
CHARS = "ACDEFHJKMNPRTUVWXY"  # unambiguous set


def make_code(rng):
    return "%s%s%s-%d%d%d%d-%s%s%d%d" % tuple(
        [rng.choice(CHARS) for _ in range(3)] + [rng.randrange(10) for _ in range(4)]
        + [rng.choice(CHARS) for _ in range(2)] + [rng.randrange(10) for _ in range(2)])


def generate():
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(42)
    img = Image.new("RGB", (1568, 1568), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 40), "COMPLIANCE REGISTER — SECTION 7", fill="black",
           font=ImageFont.truetype(FONT, 34))
    d.text((60, 100), "Each entry below must be transcribed exactly for audit.",
           fill="black", font=ImageFont.truetype(FONT, 18))
    gt = {}
    y = 170
    for size in SIZES:
        f = ImageFont.truetype(FONT, size)
        d.text((60, y), f"[{size}px tier]", fill="gray", font=ImageFont.truetype(FONT, 14))
        y += 26
        codes = []
        for i in range(4):
            c = make_code(rng)
            codes.append(c)
            d.text((90 + (i % 2) * 700, y + (i // 2) * (size + 10)),
                   f"entry {c} status ACTIVE", fill="black", font=f)
        gt[str(size)] = codes
        y += 2 * (size + 10) + 28
    img.save(IMG)
    json.dump(gt, open(GT, "w"), indent=1)
    return gt


# Module-level so vision_suite.py's `finetext` test uses the SAME prompt and
# scorer rather than a copy. A drifted copy would silently make scores from the
# two entry points non-comparable, which is the whole reason this probe ships
# committed assets instead of regenerating them.
PROMPT = ("Transcribe EVERY reference code on this page exactly as printed. "
          "Codes look like ABC-1234-DE56 and appear at several text sizes, "
          "including very small ones; read carefully down to the smallest. "
          "Respond with a SINGLE JSON object, no prose: "
          '{"codes": [<string>, ...]} listing every code you can read.')

# The generation allowance this probe needs. 20 codes plus JSON scaffolding does
# not fit vision_suite.py's 2200-token default, and an exhausted allowance looks
# like a vision failure rather than a truncation (the num_predict trap in the
# preflight skill).
NUM_PREDICT = 4000
NUM_CTX = 32768


def score_codes(body):
    """Transcription body -> per-tier recall. Returns only the scored fields;
    callers add their own tag//timing metadata."""
    gt = json.load(open(GT)) if os.path.exists(GT) else generate()
    s = {"json_valid": False}
    found = []
    # Fence tolerance: engines that do not enforce format:"json" (the MLX
    # runner before x/structured, ADR 0009) fence the JSON. No-op on
    # grammar-constrained output; recorded so the engine is identifiable.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fenced:
        body = fenced.group(1)
        s["fenced"] = True
    try:
        found = [str(x).strip().upper() for x in json.loads(body).get("codes", [])]
        s["json_valid"] = True
    except Exception:
        pass
    for size, codes in sorted(gt.items(), key=lambda kv: -int(kv[0])):
        s[f"recall_{size}px"] = sum(1 for c in codes if c in found)
    # Every model observed so far returns all 20 regardless of what it can
    # actually resolve, so a full total_found with zeroed small tiers means
    # fabricated codes, not omitted ones. Worth keeping visible.
    s["total_found"] = len(found)
    return s


def run(host, tag, model):
    prompt = PROMPT
    img_b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    num_ctx = int(os.environ.get("NUM_CTX", str(NUM_CTX)))
    num_predict = int(os.environ.get("NUM_PREDICT", str(NUM_PREDICT)))
    # ONE request path for the whole suite (SPEC H1 / ADR 0028). This probe used
    # to build its own payload, and it had already drifted from vision_suite's:
    # different endpoint default, no `thinking` normalisation, no persistence,
    # and no translation of the context-overflow 400. All of that now comes from
    # client.generate() for free, and a fix there reaches both callers.
    # Bound from the shared helper rather than re-reading THINK, so this file
    # cannot disagree with the request that was actually sent. The rewire dropped
    # this binding while leaving provenance(model, think) below, so every run
    # raised NameError AFTER the inference completed -- the model was paid for and
    # the result discarded, and under run_engine_compare.sh's `set -eu` that
    # aborted the whole campaign.
    think = client.think_on()
    r = client.generate(host, model, PROMPT, [img_b64],
                        num_predict=num_predict, num_ctx=num_ctx)
    body = r.get("response", "")
    chars = client.persist(tag, "finetext", r)
    s = {"tag": tag}
    s.update(score_codes(body))
    s["host"] = r.get("_host")
    s["server_version"] = r.get("_server_version")
    s["prompt_eval_count"] = r.get("prompt_eval_count")
    s["eval_count"] = r.get("eval_count")
    # Server's stop verdict, same contract as vision_suite (SPEC H4b):
    # recorded only when present, so old blocks and connection-closed finals
    # stay distinguishable by absence and fall back to the arithmetic.
    if r.get("done_reason"):
        s["done_reason"] = r["done_reason"]
    # Free, tokenizer-free halves of the split; token_split.py turns these into
    # exact thinking/answer/control token counts afterwards.
    s.update(chars)
    # The request window these numbers were achieved under. Recall that comes in
    # low may be the model or may be a capped generation; without num_ctx and
    # num_predict the two are indistinguishable after the fact. Note this probe
    # defaults higher than vision_suite.py (32768/4000 vs 16384/2200), so a run
    # that does not set both explicitly measures the two harnesses at different
    # windows — run_engine_compare.sh sets them.
    s["num_ctx"] = num_ctx
    s["num_predict"] = num_predict
    # ADR 0005 asks runs to record their runtime configuration. Without this a
    # capped cell cannot be attributed to a sampling mode after the fact.
    s.update(provenance(model, think))
    print(f"--- finetext [{tag}] ---")
    print(json.dumps(s, indent=1))
    json.dump(s, open(os.path.join(DIR, f"ft_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "gen":
        generate(); print("finetext.png + gt written")
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
