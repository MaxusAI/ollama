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
Both variants drop the same sentence, so they differ only in whether the model is given the size. They rewrite the
two prompts that were studied: bbox_contract_real_1img ("... give the size YOU used, not the size you were sent."),
and multi_3img_anchored's calibration box, which is about image 1 ("If you resized image 1 internally, use the size
YOU used."). multi_3img is the same prompt without that calibration paragraph, and bbox_contract_adv_real asks for
pixels without the sentence; on those the variants refuse. Three more prompts ask for the size YOU used in other
words, and the variants refuse them as well: bbox_contract and bbox_contract_multi ("If you resized the image
internally, give the size YOU used, not the original."), and bbox_contract_reasoning ("... give the size YOU used.")
(the gfx1151 host's review of MaxusAI/ollama#425). The size stated is image 1's.
  nodistract  drops _BBOX_PLACEMENT_HEAD's "Only the FIRST image contains the shapes to report; the others are
          distractors and must be ignored.", and changes nothing else. The single-image arms built on that head
          (bbox_contract_anchored_1img, bbox_contract_box2d_1img, bbox_contract_positional_1img) send one image, so
          the sentence is false there. vision_suite.py keeps the FRAME arm off the head for that reason. The CUDA
          host's review of MaxusAI/ollama#420 named it as a candidate trap (2026-09-29).
  truedistract  the control that keeps the position: replaces that sentence with a true one, "The FIRST image is
          the only image attached; it contains the shapes to report."
  neutral  the control that edits elsewhere: drops a different, true sentence from the same head, "The coordinate
          space is 1000x1000 whatever the image's shape is.", which restates the norm-1000 definition before it
          (the CUDA host's review of MaxusAI/ollama#421). If a greedy trajectory also finishes under a neutral edit,
          a nodistract finish does not single out the sentence.
All three refuse unless the test sends exactly one image, and unless its prompt carries the distractor sentence, so
they run only on the single-image arms built on the head. The three-image arms that carry the sentence do send
distractors, so it is true there: bbox_contract_anchored, _pinned and _perobject are built on the head, and
bbox_contract_multi, _adv_real and _adv_norm1 carry their own copy. The four single-image bboxm_pin_* prompts carry
the coordinate-space sentence (through _PIN_PARA) but not the distractor sentence, so `neutral` would control nothing
there, and it refuses them too (the CUDA host's review of MaxusAI/ollama#422).
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

RESIZE = re.compile(r"If\s+you\s+resized\s+(?P<img>the\s+image|image\s+1)\s+internally,\s+"
                    r"(?:give\s+the\s+size\s+YOU\s+used,\s+not\s+the\s+size\s+you\s+were\s+sent"
                    r"|use\s+the\s+size\s+YOU\s+used)\.")
DISTRACT = re.compile(r"Only\s+the\s+FIRST\s+image\s+contains\s+the\s+shapes\s+to\s+report;\s+the\s+others\s+are\s+"
                      r"distractors\s+and\s+must\s+be\s+ignored\.\s*")
TRUE_ONE = "The FIRST image is the only image attached; it contains the shapes to report.\n\n"
NEUTRAL = re.compile(r"\s+The\s+coordinate\s+space\s+is\s+1000x1000\s+whatever\s+the\s+image's\s+shape\s+is\.")
SINGLE_IMAGE_VARIANTS = ("nodistract", "truedistract", "neutral")


def transform(prompt, variant, first_image="scene_hd.png"):
    if variant == "orig":
        return prompt
    if variant in SINGLE_IMAGE_VARIANTS:
        # All three are controls for the distractor sentence, so a prompt without it has nothing to control.
        if not DISTRACT.search(prompt):
            sys.exit(f"variant {variant!r} applies only to prompts carrying the distractor sentence; {test}'s does not")
        pattern, repl = {"nodistract": (DISTRACT, ""), "truedistract": (DISTRACT, TRUE_ONE),
                         "neutral": (NEUTRAL, "")}[variant]
        new, n = pattern.subn(repl, prompt, 1)
        if n != 1:
            sys.exit(f"variant {variant!r} did not apply to {test}'s prompt")
        return new
    if variant not in ("size", "commit"):
        sys.exit(f"unknown variant {variant!r}")
    # Refuse before the size lookup: a prompt without the sentence has nothing to replace, whatever its image
    # (finetext's has no ground truth). Both prompts with the sentence send scene_hd.png, which has one; a new prompt
    # whose image has none is refused too, rather than raising.
    if not RESIZE.search(prompt):
        sys.exit(f"variant {variant!r} did not apply to {test}'s prompt")
    gt = vs.GT.get(os.path.splitext(first_image)[0])
    if gt is None:
        sys.exit(f"variant {variant!r} needs the pixel size of {first_image}, which has no ground truth")
    w, h = gt["size"]

    def single(m):  # the single-image contract prompts' wording, or multi_3img_anchored's image 1
        return m.group("img").startswith("the")

    if variant == "size":
        def repl(m):
            return f"{'The image' if single(m) else 'Image 1'} is {w}x{h} pixels (width x height); give pixel " \
                   f"coordinates in that frame."
    else:
        def repl(m):
            if single(m):
                return ("If you do not know the image's pixel size, choose your best estimate once, give it as "
                        "ref_size, and do not revisit it.")
            return "If you do not know image 1's pixel size, choose your best estimate once and do not revisit it."
    return RESIZE.sub(repl, prompt, 1)


if __name__ == "__main__":
    vs.HOST, vs.MODEL = host, model
    entry = next((t for t in vs.tests if t[0] == test), None)
    if entry is None:
        sys.exit(f"unknown test {test!r}")
    if variant in SINGLE_IMAGE_VARIANTS and len(entry[2]) != 1:
        sys.exit(f"variant {variant!r} applies only to single-image tests; {test} sends {len(entry[2])} images")
    prompt = transform(entry[1]() if callable(entry[1]) else entry[1], variant, entry[2][0])
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
