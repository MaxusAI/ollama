#!/usr/bin/env python3
"""Recover usable structure from a model response that JSON alone cannot.

SPEC C12 already says a malformed BOX is dropped individually rather than
invalidating a response. This is the same principle one level up: a response
whose CONTENT is right and whose SHAPE is wrong should be recovered and marked,
not scored as a failure.

Every step here was written against a real observed failure, not a hypothetical:

  fence            engines that do not enforce format:"json" wrap the object in
                   ```json ... ```. Already handled by vision_suite._loads; kept
                   here so one function covers every case.

  embedded_key     THE ONE THAT MATTERS, and the reason a generic brace-repair
                   would not have helped. qwen3.6:35b-a3b-q8_0 think-on returned
                   VALID json whose `answers` object had been serialised into a
                   STRING inside the images array:

                     {"images": [ {...}, {...}, {...},
                                  "answers\\": {\\"q1\\": 2, ...}}}```json{  " ]}

                   json.loads succeeds, `answers` is not a key, and the cell
                   scored 0/3 on q1/q2/q4 with chart 5/5 -- the chart came from
                   the well-formed part. Content was correct: q1 2 and q2 Q4/128,
                   identical to the same model's passing run on the other arm.
                   Scoring that as a model failure is simply wrong.

  truncation       generation stopped mid-object. Close what is open, in reverse
                   order, and drop a trailing incomplete key.

  largest_object   last resort: the biggest balanced {...} span that parses.

`method` is always reported. A response salvaged at `fence` is close to clean; one
salvaged at `largest_object` is barely a response at all, and a rate that mixes
the two without saying so is not a measurement. Steps 3 and 4 in particular are
warning signs, not successes -- SPEC C8's "reject rather than convert" still
applies to the coordinate space; this only recovers the envelope around it.
"""
import json
import re

# Ordered: cheapest and most faithful first. The name of the step that worked is
# the salvage method reported on the cell.
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _try(text):
    try:
        return json.loads(text), True
    except Exception:
        return None, False


def _strip_fence(text):
    m = FENCE_RE.search(text)
    return m.group(1) if m else None


def _embedded_key(obj, key):
    """Find `"<key>": {...}` that got serialised into a string value.

    Returns the decoded object, or None. Walks strings anywhere in the structure
    because the misplacement is a generation artefact -- it lands wherever the
    model lost the thread, which in the observed case was the last element of an
    unrelated array."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and key in node:
            # The string carries `answers": {...}` -- note the LEADING quote is
            # already consumed by the enclosing string, so rebuild it.
            m = re.search(rf'{re.escape(key)}"?\s*:\s*(\{{.*)', node, re.S)
            if not m:
                return
            frag = m.group(1)
            # Trim to the last balanced brace; the tail is usually fence garbage.
            depth = 0
            for i, ch in enumerate(frag):
                depth += (ch == "{") - (ch == "}")
                if depth == 0:
                    o, ok = _try(frag[:i + 1])
                    if ok:
                        found.append(o)
                    return

    walk(obj)
    return found[0] if found else None


def _close_truncated(text):
    """Close unterminated string/array/object, in reverse order of opening."""
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack and not in_str:
        return None
    out = text + ('"' if in_str else "")
    # A trailing incomplete key/value would fail to parse; drop back to the last
    # complete element before closing.
    for cut in (len(out), out.rfind(","), out.rfind("}"), out.rfind("]")):
        if cut <= 0:
            continue
        cand = out[:cut] + "".join("}" if c == "{" else "]" for c in reversed(stack))
        o, ok = _try(cand)
        if ok:
            return cand
    return None


def _largest_object(text):
    best = None
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                o, ok = _try(text[start:i + 1])
                if ok and (best is None or i + 1 - start > best[1]):
                    best = (o, i + 1 - start)
                break
    return best[0] if best else None


def loads(text, require_key=None):
    """Parse a response. Returns (obj, method).

    method is None when the text parsed cleanly and, if require_key was given,
    already carried it. Otherwise it names the step that recovered the object.
    Returns (None, None) when nothing parsed.

    require_key exists because valid JSON is not the same as usable JSON: the
    observed q8_0 failure parsed perfectly and simply did not contain `answers`.
    A caller that needs a key should say so, or salvage cannot know it is needed.
    """
    obj, ok = _try(text)
    if ok:
        if require_key is None or (isinstance(obj, dict) and require_key in obj):
            return obj, None
        rec = _embedded_key(obj, require_key)
        if rec is not None:
            if isinstance(obj, dict):
                obj[require_key] = rec
                return obj, "embedded_key"
            return {require_key: rec}, "embedded_key"
        return obj, None          # parsed, key genuinely absent — not a salvage

    inner = _strip_fence(text)
    if inner is not None:
        obj, ok = _try(inner)
        if ok:
            return obj, "fence"
        text = inner

    fixed = _close_truncated(text)
    if fixed is not None:
        obj, ok = _try(fixed)
        if ok:
            return obj, "truncation"

    obj = _largest_object(text)
    if obj is not None:
        return obj, "largest_object"
    return None, None
