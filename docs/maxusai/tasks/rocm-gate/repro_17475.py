#!/usr/bin/env python3
"""#17475 — cross-request content leakage on a shared slot.

    repro_17475.py <label> <image> <prod|reporter> <V1,V2,V3> [KEY=VALUE ...]

The reporter's three protocols, run exactly as described, on synthetic documents
only (gen_forms.py). A victim output is contaminated if it contains any 6-character
window of the donor-only marker; the victim document shares none, by construction
and by assertion.
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import gatelib as g

MODEL = "qwen3-vl:30b-a3b-thinking"           # the reporter's model
# Reporter's noise models by default. GATE_TEXT_NOISE / GATE_VLM_NOISE swap in
# smaller ones: at 17-22 GB each, the reporter's models cost 60-75 s per noise
# request here (evict + load while the victim loop saturates the iGPU), so the
# loop cycled 1-3 times per V3 and the multi-model SWAPPING that V3 exists to
# test barely happened. Measured across six payloads.
TEXT_NOISE = os.environ.get("GATE_TEXT_NOISE", "gemma4:26b-a4b-it-q4_K_M")
VLM_NOISE = os.environ.get("GATE_VLM_NOISE", "qwen3.6:35b-a3b-q4_k_m")
OPTS = {"temperature": 0, "num_ctx": 8192, "num_predict": 6144}
# The report aborts the donor at 8 s, which on the reporter's hardware landed
# mid-generation of a num_predict=6144 request. This host answers the SAME request
# in ~3.9 s, so an 8 s timeout never fires and V2/V3 silently degrade into V1 with
# concurrency — measured, first run. The mechanism under test is a client hanging up
# WHILE the slot generates, so the window is shortened rather than the prompt changed.
ABORT_S = float(os.environ.get("GATE_ABORT_S", "2.5"))
PROMPT = ("Extract the following fields from this vehicle insurance claim form and "
          "return them as a JSON object with exactly these keys: policy_number, "
          "insured_name, vehicle, vin, incident_date, claim_amount. Return only the JSON.")

TRUTH = json.load(open(os.path.join(g.OUT, "forms_truth.json")))
WINDOWS = TRUTH["marker_windows"]
IMG = {k: base64.b64encode(open(os.path.join(g.OUT, f"{k}.png"), "rb").read()).decode()
       for k in ("donor", "victim")}


def generate(body, timeout):
    req = urllib.request.Request(g.HOST + "/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    rec = {"aborted": False, "error": None, "http_status": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["http_status"] = r.status
            d = json.loads(r.read())
        rec.update(response=d.get("response", ""), thinking=d.get("thinking", ""),
                   done_reason=d.get("done_reason"), eval_count=d.get("eval_count"),
                   prompt_eval_count=d.get("prompt_eval_count"))
    except urllib.error.HTTPError as e:   # before OSError: HTTPError subclasses it
        rec["http_status"], rec["error"] = e.code, e.read().decode(errors="replace")[:400]
    except (TimeoutError, OSError) as e:
        if "timed out" in str(e):
            rec["aborted"] = True          # the client hung up: this IS the abort
        else:
            rec["error"] = f"{type(e).__name__}: {e}"
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["t_start"] = time.strftime("%H:%M:%S", time.localtime(t0))
    return rec


def extract(which, abort=False):
    body = {"model": MODEL, "prompt": PROMPT, "images": [IMG[which]], "stream": False,
            "think": True, "options": OPTS}
    r = generate(body, ABORT_S if abort else 1200)
    r["doc"] = which
    text = (r.get("response") or "") + "\n" + (r.get("thinking") or "")
    r["marker_hits"] = sorted(w for w in WINDOWS if w in text)
    r["full_marker"] = TRUTH["marker"] in text
    r["own_vin_found"] = TRUTH[which]["vin"] in text
    return r


def show(tag, r):
    s = ("ABORTED" if r["aborted"] else
         f"done={r.get('done_reason')} eval={r.get('eval_count')} own_vin={r['own_vin_found']}")
    hit = f"  <<< MARKER {r['marker_hits'][:3]}" if (r["doc"] == "victim" and r["marker_hits"]) else ""
    print(f"    {tag:16s} {r['doc']:6s} {r['wall_s']:6.1f}s {s}"
          + (f" err={r['error'][:70]}" if r["error"] else "") + hit, flush=True)


def v1(n=6):
    rows = []
    for i in range(n):
        d = extract("donor"); show(f"V1.{i+1} donor", d)
        v = extract("victim"); show(f"V1.{i+1} victim", v)
        rows += [d, v]
    return rows


def v2(n=6):
    rows = []
    for i in range(n):
        d = extract("donor", abort=True); show(f"V2.{i+1} donor", d)
        v = extract("victim"); show(f"V2.{i+1} victim", v)
        rows += [d, v]
    return rows


def v3(n=int(os.environ.get("GATE_V3_VICTIMS", "8")), min_noise=6, max_victims=40, max_s=1200):
    """The reporter's V3, with the victim count adapted to this host's speed.

    Their V3 is the only protocol that contaminated, and its stated ingredient is
    continuous multi-model swapping during the victim extracts. Their victims took
    minutes each, so the noise loop cycled many times. Here a victim answers in
    ~4 s while one noise request costs ~74 s -- 0.32.x sees only 27 GiB, so the
    noise model cannot be resident beside qwen3-vl and every cycle is an evict and
    reload. Measured on the first attempt: 8 victims and exactly ONE noise
    request, i.e. no swapping at all during the window.

    So victims continue until the noise loop has cycled min_noise times (three
    alternations) as well as reaching n, and the contamination denominator is
    however many victims that took. Bounded so a slow payload cannot run forever.
    """
    stop = threading.Event()
    rows, lock = [], threading.Lock()
    noise_done = [0]

    def keep(r):
        with lock:
            rows.append(r)

    def donor_loop():
        i = 0
        while not stop.is_set():
            r = extract("donor", abort=(i % 2 == 0)); r["thread"] = "donor"
            show(f"V3 donor#{i+1}", r); keep(r); i += 1

    def noise_loop():
        i = 0
        while not stop.is_set():
            if i % 2 == 0:
                body = {"model": TEXT_NOISE, "stream": False, "think": False,
                        "prompt": "Summarise the history of the printing press in three sentences.",
                        "options": {"num_predict": 256}}
            else:
                body = {"model": VLM_NOISE, "stream": False, "think": False, "images": [IMG["donor"]],
                        "prompt": "Classify this document as invoice, form, letter or other. One word.",
                        "options": {"num_predict": 256}}
            r = generate(body, 900)
            r.update(thread="noise", doc=body["model"], marker_hits=[], own_vin_found=None)
            print(f"    V3 noise#{i+1}   {body['model'][:22]:22s} {r['wall_s']:6.1f}s"
                  + (f" err={r['error'][:60]}" if r["error"] else ""), flush=True)
            keep(r); i += 1
            with lock:
                noise_done[0] = i

    ts = [threading.Thread(target=donor_loop, daemon=True),
          threading.Thread(target=noise_loop, daemon=True)]
    for t in ts:
        t.start()
    time.sleep(2)          # let the donor get into the slot first, as in the report
    t0, i = time.time(), 0
    while True:
        v = extract("victim"); v["thread"] = "victim"; i += 1
        show(f"V3 victim#{i}", v); keep(v)
        with lock:
            nd = noise_done[0]
        if i >= n and nd >= min_noise:
            break
        if i >= max_victims or time.time() - t0 > max_s:
            print(f"    V3 stop: {i} victims, {nd} noise cycles (bound reached)", flush=True)
            break
    stop.set()
    for t in ts:
        t.join(timeout=1500)
    return rows


def main():
    label, image, profile, protos = sys.argv[1:5]
    env = dict(g.PROD_ENV if profile == "prod" else g.REPORTER_ENV)
    env.update(kv.split("=", 1) for kv in sys.argv[5:])
    meta = g.start(image, env, f"17475_{label}")
    meta.update(model=MODEL, options=OPTS, abort_s=ABORT_S,
                abort_note=("8 s in the report; shortened here because this host completes "
                            "the same request in ~3.9 s and an 8 s client timeout never fires"),
                prompt=PROMPT,
                images={k: {"file": TRUTH[k]["file"], "format": "PNG", "size": TRUTH[k]["size"]}
                        for k in ("donor", "victim")},
                marker=TRUTH["marker"])
    out = {}
    try:
        tags = {m["name"]: m["digest"][:12] for m in g.api("/api/tags")["models"]}
        meta["digests"] = {m: tags.get(m) for m in (MODEL, TEXT_NOISE, VLM_NOISE)}
        for p in protos.split(","):
            print(f"  -- {p}", flush=True)
            out[p] = {"V1": v1, "V2": v2, "V3": v3}[p]()
            vic = [r for r in out[p] if r["doc"] == "victim"]
            bad = [r for r in vic if r["marker_hits"]]
            ab = [r for r in out[p] if r.get("aborted")]
            print(f"  == {p}: {len(bad)}/{len(vic)} victims contaminated; "
                  f"{len(ab)} donor aborts actually fired", flush=True)
            g.save(f"result_17475_{label}.json", {"meta": meta, "protocols": out})
    finally:
        log = g.stop(f"17475_{label}")
        meta["server_log"] = log
        meta["offload_evidence"] = g.offload_evidence(log)
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        g.save(f"result_17475_{label}.json", {"meta": meta, "protocols": out})
    for p, rows in out.items():
        vic = [r for r in rows if r["doc"] == "victim"]
        print(f"{label} {p}: {sum(1 for r in vic if r['marker_hits'])}/{len(vic)} contaminated", flush=True)


if __name__ == "__main__":
    main()
