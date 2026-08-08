#!/usr/bin/env python3
"""Token-budget measurement against the ollama-rocm-nemotron test container.
Method: docs/maxusai/vision-token-budget-measurements.md — /api/generate,
num_predict:1, prompt_eval_count minus the text prefix as it is tokenised
*in an image-bearing request* (see BASELINE below)."""
import json, sys, base64, os, urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11435"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "nemotron3:33b-q4_K_M"
IMGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testimgs")
SIZES = ["320x240", "640x480", "896x896", "1568x1568", "1920x1080",
         "2048x1664", "3000x2000", "3200x32"]

# One prompt for the baseline AND every image probe. Two separate traps here,
# both measured 2026-08-08 on the :11437 canary (0.32.5-dynres-4987dd49):
#
# 1. MISMATCHED PROMPTS. This script used to baseline with "Hi" and probe with
#    "Describe briefly.", so the *text* length difference landed in every row.
#    On nemotron3:33b-q8 "Hi" tokenises to 18 and "Describe briefly." to 21.
#
# 2. THE TEXT PREFIX IS NOT THE TEXT-ONLY COUNT. Attaching an image can change
#    how the template renders the surrounding text. Same prompt, same model:
#    text-only = 21, but the prefix inside an image-bearing request = 20. So
#    even a matched-prompt text-only baseline over-subtracts by 1 on
#    nemotron_h_omni. (gemma4:31b measured 19 both ways — the offset is
#    arch-specific, so it must be derived, not assumed.)
#
# Net effect on nemotron: the old code reported grid+4, a matched text-only
# baseline reports grid+1, and the documented convention is grid+2.
PROBE_PROMPT = "Describe briefly."


def gen(payload, timeout=900):
    req = urllib.request.Request(HOST + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def load(name):
    path = f"{IMGDIR}/{name}.png"
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run `python3 gen_geoms.py` first")
    return base64.b64encode(open(path, "rb").read()).decode()


def count(images, **opts):
    return gen({"model": MODEL, "prompt": PROBE_PROMPT, "images": images,
                "stream": False, "options": dict(num_predict=1, **opts)})["prompt_eval_count"]


# Baseline calibration. The text prefix cancels in a two-image difference, so it
# can be recovered without trusting any text-only probe and without assuming a
# grid. With prefix P and per-image costs cA/cB:
#     count(A) + count(B) - count(A, B) = (P + cA) + (P + cB) - (P + cA + cB) = P
a, b = SIZES[0], SIZES[1]
try:
    one_a, one_b = count([load(a)]), count([load(b)])
    both = count([load(a), load(b)])
except Exception as e:
    sys.exit(f"baseline calibration failed on {a}/{b}: {e}")
BASELINE = one_a + one_b - both

textonly = gen({"model": MODEL, "prompt": PROBE_PROMPT, "stream": False,
                "options": {"num_predict": 1}})["prompt_eval_count"]
print(f"text prefix in an image request: {BASELINE}  (calibrated on {a} + {b})")
print(f"text-only count, same prompt:    {textonly}"
      + ("" if textonly == BASELINE else
         f"  <-- differs by {textonly - BASELINE}; subtracting this would skew every row"))

results = {}
for name in SIZES:
    try:
        pec = one_a if name == a else one_b if name == b else count([load(name)])
        delta = pec - BASELINE
        results[name] = delta
        print(f"{name:>10}: prompt_eval_count={pec:>5}  visual+markers={delta}")
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"{name:>10}: ERROR {e}")

# knob check: image_max_tokens=1024 on a large image. This runs on a *different*
# runner (Runner option -> reload), but BASELINE still applies: the prefix was
# re-calibrated under the knob on 2026-08-08 and came back 20, same as default.
pec = count([load("1920x1080")], image_max_tokens=1024)
print(f"knob 1920x1080 @ image_max_tokens=1024: prompt_eval_count={pec}  visual+markers={pec - BASELINE}")

# coherence smoke: one short caption
r = gen({"model": MODEL, "prompt": "What color is this image? Answer in one short sentence.",
         "images": [load("1920x1080")], "stream": False, "options": {"num_predict": 60}})
print("coherence sample:", json.dumps(r["response"]))
