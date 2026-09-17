#!/usr/bin/env python3
"""Generate the #17459 results table from the saved run JSONs.

ADR 0012 rule 8: generated from the result files and the servers' own logs,
never transcribed. Columns:
  runner      -c/-np actually passed, and whether --load-mode dio was present.
              Identical within a profile, so the layer counts are comparable.
  VRAM seen   the scheduler's own msg="gpu memory" available=, which is the
              figure Ollama places layers against.
  layers      llama.cpp's "offloaded N/61 layers to GPU".
"""
import glob, json, os, re

rows = []
for p in sorted(glob.glob("result_17459_*.json")):
    d = json.load(open(p))
    label = re.sub(r"^result_17459_|\.json$", "", os.path.basename(p))
    meta, rs = d["meta"], d["rows"]
    pay = next(k for k in ("b9888", "b10091", "b10864") if label.startswith(k))
    log = open(meta["server_log"], errors="replace").read()

    cmd = re.search(r"--model \S+ .*", log)
    cmd = cmd.group(0) if cmd else ""
    ctx = re.search(r"-c (\d+) -np (\d+)", cmd)
    runner = f"`-c {ctx.group(1)} -np {ctx.group(2)}`" if ctx else "—"
    runner += ", dio" if "--load-mode dio" in cmd else ""

    lays = sorted({int(m) for m in re.findall(r"offloaded (\d+)/61 layers", log)})
    lay = f"{lays[0]}–{lays[-1]}/61" if len(lays) > 1 else (f"{lays[0]}/61" if lays else "—")
    av = sorted({float(m) for m in re.findall(r'msg="gpu memory".*?available="([\d.]+) GiB', log)})
    vram = f"{av[0]:.1f}" + (f"–{av[-1]:.1f}" if av[-1] != av[0] else "") + " GiB" if av else "—"

    unused = sum(r["unused_tokens"] for r in rs)
    failed = [r for r in rs if r["error"] or not r["done"]]
    cold = [r["load_s"] for r in rs if r["cold_requested"]]
    warm = [r["load_s"] for r in rs if not r["cold_requested"]]
    fmt = lambda v: "—" if not v else (f"{min(v):.2f}" if min(v) == max(v) else f"{min(v):.2f}–{max(v):.2f}")
    rows.append((pay, label, runner, len(rs), unused,
                 "—" if failed else fmt(cold), "—" if failed else fmt(warm),
                 lay, vram, f"**{len(failed)}/{len(rs)} failed to load**" if failed else "clean"))

print("| payload | run | runner | requests | `<unused>` tokens | cold load s | warm load s | layers on GPU | VRAM seen | result |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| {} | `{}` | {} | {} | **{}** | {} | {} | {} | {} | {} |".format(*r))
print()
print(f"Total: {sum(r[3] for r in rows)} requests, {sum(r[4] for r in rows)} `<unused>` tokens.")
