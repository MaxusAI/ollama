import json, base64, sys, urllib.request
S = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, S)
from vision_suite import SCENE_PROMPT, score_scene, b64
import client
HOST = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:11435"
def gen(extra, fmt, think):
    # Shared request path (SPEC H1 / ADR 0028). Pinned to /api/generate and to
    # temperature 0 with apply_sampling=False so the numbers stay comparable to
    # the ones already published: this probe exists to isolate the grammar and
    # think variables, so it must not inherit per-model sampling or the suite's
    # chat default and change two things at once.
    return client.generate(HOST, "nemotron3:33b-q4_K_M",
                           SCENE_PROMPT.format(w=1920, h=1080),
                           [b64("scene_hd.png")],
                           num_predict=3000, num_ctx=16384,
                           fmt="json" if fmt else None,
                           think=think, send_think=True, apply_sampling=False,
                           extra_opts={"temperature": 0},
                           endpoint_override="generate", timeout=1800)


mode = sys.argv[1]
fmt, think = {"nogrammar": (False, False), "thinkon": (True, True)}[mode]
r = gen(mode, fmt, think)
text = r.get("response", "")
open(f"{S}/resp_variant_{mode}.json", "w").write(text)
# strip code fences if present for scoring
t = text.strip()
if t.startswith("```"):
    t = t.split("```")[1]
    t = t[4:] if t.startswith("json") else t
sc = score_scene(t)
sc["eval_count"] = r.get("eval_count"); sc["thinking_len"] = len(r.get("thinking") or "")
print(mode, json.dumps(sc))
