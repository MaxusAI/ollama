#!/usr/bin/env python3
"""Emit the EXACT request payload the campaign sends, for copy-paste use.

    python3 emit_request.py <model> <think: false|on> [arm]

Prints the JSON body client.generate() would POST for the given model, think
mode and arm (default bboxm_pin_anc_named — the trustable arm), with the
base64 image elided. Window, sampling, think field, format and endpoint form
are CAPTURED from client.py's own payload construction rather than
re-derived, so this tool cannot drift from the wire format (SPEC H9: one
payload builder). The recommendations doc pastes this tool's output verbatim
(H7) instead of hand-assembling request examples — hand-assembled fragments
are exactly what made the first draft unreconstructable.

think=on emits the 16384-rung request; a done_reason of "length" in the
response means escalate the window (the CONTEXT ladder), not that the model
failed.
"""
import io
import json
import sys
import urllib.error
import urllib.request

import client


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model, think_s = sys.argv[1], sys.argv[2]
    arm = sys.argv[3] if len(sys.argv) > 3 else "bboxm_pin_anc_named"
    think = think_s == "on"

    import vision_suite as vs
    vs.MODEL = model  # composed-arm prompts pick the model's dialect from this
    entry = next((t for t in vs.tests if t[0] == arm), None)
    if entry is None:
        sys.exit(f"unknown arm {arm!r}; have: {', '.join(t[0] for t in vs.tests)}")
    prompt = entry[1]() if callable(entry[1]) else entry[1]
    images = [f"<base64 of visimgs/{i}>" for i in entry[2]]

    captured = {}

    def capture(req, timeout=None):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data)
        # A deterministic 400 makes client.py fail FAST: it never retries a
        # 4xx (H10), whereas a generic exception here costs three transport
        # retries and 50 seconds of backoff before surfacing.
        raise urllib.error.HTTPError(req.full_url, 400, "captured", {},
                                     io.BytesIO(b"{}"))

    orig, urllib.request.urlopen = urllib.request.urlopen, capture
    try:
        nc = 16384
        client.generate("http://HOST:11497", model, prompt, images,
                        num_predict=(nc - 8192) if think else 2200,
                        num_ctx=nc, think=think)
    except Exception:
        pass
    finally:
        urllib.request.urlopen = orig
    if "payload" not in captured:
        sys.exit("payload was not captured")

    print(f"POST {captured['url']}")
    print(json.dumps(captured["payload"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
