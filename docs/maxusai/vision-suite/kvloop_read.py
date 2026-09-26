#!/usr/bin/env python3
"""kvloop_read.py CAPTURE.json... -- read out thinkcap.py captures for a loop investigation.

For each capture: how it ended (done_reason, tokens), the thinking's repetition profile, and the answer scored
by the suite's own scorer for that test. A loop shows as a second half with few distinct lines and one line
repeated many times; a finished capture has done_reason "stop" and an answer.

The arm label is the KV type and flash-attention tag at the start of the file name (kvloop.sh writes
<kv>-fa<on|off>_<model>_<test>_<num_ctx>.json). The test comes from the capture block thinkcap.py writes, or, for older captures, from the
longest suite test name found in the file name.

tasks/kv-precision-think-loops.md, ADR 0043."""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_suite as vs  # noqa: E402

SCORERS = {t[0]: t[3] for t in vs.tests if len(t) > 3 and callable(t[3])}
# KV type names can contain underscores (q8_0), so the arm label is matched, not split.
ARM = re.compile(r"^((?:q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl|bf16|f16|f32)(?:-fa(?:on|off))?)_")
WINDOW, NOVEL = 100, 0.05   # a loop: a 100-line window where under 5% of lines are new to the thinking


def test_of(path, r):
    cap = r.get("capture") or {}
    if cap.get("test") in SCORERS:
        return cap["test"]
    base = os.path.basename(path)
    hits = [name for name in SCORERS if f"_{name}_" in base]
    return max(hits, key=len) if hits else None


def profile(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return "no thinking"
    half = lines[len(lines) // 2:]
    top, n = collections.Counter(lines).most_common(1)[0]
    return (f"{len(set(lines))}/{len(lines)} lines distinct, second half {len(set(half))}/{len(half)}, "
            f"most repeated x{n}: {top[:60]!r}")


def loop_onset(text, tokens):
    """Where the thinking turns into a loop: the first WINDOW-line window in which under NOVEL of the lines are
    new, i.e. have not appeared earlier in the thinking. Unlike a distinct-lines window, this finds cycles of any
    period. Reported as a line number and as a token estimate, scaled by the capture's characters per token."""
    raw = [l for l in (text or "").splitlines() if l.strip()]
    seen, new = set(), []
    for l in raw:
        k = l.strip()
        new.append(k not in seen)
        seen.add(k)
    for i in range(0, max(0, len(raw) - WINDOW + 1), 10):
        if sum(new[i:i + WINDOW]) < NOVEL * WINDOW:
            chars = sum(len(l) + 1 for l in raw[:i])
            total = sum(len(l) + 1 for l in raw) or 1
            est = f", about token {int(tokens * chars / total):,}" if tokens else ""
            return f"loop from line {i:,} of {len(raw):,}{est}"
    return "no loop found"


def main(paths):
    for path in paths:
        name = os.path.basename(path)[:-5] if path.endswith(".json") else os.path.basename(path)
        try:
            r = json.load(open(path))
        except Exception as exc:  # a capture still being written, or a failed one
            print(f"{name}: unreadable ({exc})")
            continue
        test = test_of(path, r)
        m = ARM.match(name)
        arm = m.group(1) if m else name.split("_")[0]
        print(f"{name}\n  arm={arm}  done={r.get('done_reason')}  tokens={r.get('eval_count')}  "
              f"thinking={len(r.get('thinking') or '')} chars  answer={len(r.get('response') or '')} chars")
        print(f"  thinking: {profile(r.get('thinking'))}")
        print(f"  onset: {loop_onset(r.get('thinking'), r.get('eval_count'))}")
        if test is None:
            print("  score: unknown test")
            continue
        s = SCORERS[test](r.get("response") or "")
        keys = [k for k in ("json_valid", "labels_found", "hits_declared", "iou_declared", "hits_anchor",
                            "hits_bestfit", "bestfit_dialect", "contract_followed") if k in s]
        print(f"  score ({test}): " + " ".join(f"{k}={s[k]}" for k in keys))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
