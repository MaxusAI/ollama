#!/usr/bin/env python3
"""Direct-I/O effect on repeated model load.

The noise thread loads a different model on every request, so its wall time is
dominated by load rather than generation. Compared here only across runs that
used the SAME small-model noise set (*fastnoise2) and the same 900 s V3 window,
so the only variable is --load-mode dio.

A median would mislead: without dio the first load of a model is quick and a
later one stalls for minutes, so every wall time is listed rather than summarised.
900.1 s means the request had not returned when the protocol window closed.
"""
import glob, json, os, re

print("| run | payload | dio | completed cycles | every noise request, s |")
print("|---|---|---|---|---|")
for p in sorted(glob.glob("result_17475_*fastnoise2.json")):
    d = json.load(open(p)); lab = re.sub(r"^result_17475_|\.json$", "", os.path.basename(p))
    dio = "**on**" if "--load-mode dio" in open(d["meta"]["server_log"], errors="replace").read() else "off"
    pay = next(k for k in ("b9888", "b10091", "b10864") if lab.startswith(k))
    w = [r["wall_s"] for r in sum(d["protocols"].values(), [])
         if (r.get("thread") or r.get("doc")) not in ("donor", "victim")]
    done = [x for x in w if x < 900]
    cell = (", ".join(f"{x:g}" for x in w) if len(w) <= 8
            else f"{min(done):g}–{max(done):g} over {len(w)} requests")
    print(f"| `{lab}` | {pay} | {dio} | **{len(done)}** | {cell} |")
