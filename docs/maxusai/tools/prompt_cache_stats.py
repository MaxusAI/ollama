#!/usr/bin/env python3
"""What llama-server's host-RAM prompt cache did, read from ollama's own log.

Usage:
    journalctl -u ollama -u systemd-journald -b -o short-iso-precise | prompt_cache_stats.py parse \
        --manifests <models dir>/manifests [--window LABEL=START/END ...] > stats.json
    prompt_cache_stats.py render stats.json
    prompt_cache_stats.py ab ON.json OFF.json [ON.json OFF.json ...]

llama.cpp's server keeps a prompt cache in host RAM, 8192 MiB unless `--cache-ram` says
otherwise. When a request cannot reuse the slot's prompt, the server copies the slot's
whole state into that cache, looks for a cached prompt that matches the new request
better, and restores it if it finds one. At ollama's `--log-verbosity 4` it logs every
step, and times each update itself ("prompt cache update took ... ms"). `parse` reads
those lines and nothing else, so its numbers are llama.cpp's, not a client's estimate.

A request is attributed to the model of the `llama-server` launch it ran under (the
`--model` blob, named through the store's manifests) and to the first `--window` whose
START/END (ISO 8601, UTC) contains it. Requests outside every window are counted under
"other". The output names models and windows, never a host: the journal's host field is
read past and dropped.

journald rate-limits a unit (10,000 lines per 30 s by default), and at verbosity 4 a fast arm
with the cache on logs every cached prompt on every request. Its "Suppressed N messages from
ollama.service" notices are read too: a row they fall in carries `journal_lines_dropped`, and
`render` names it, because its counts are then lower bounds.

`render` prints the tables from that JSON alone (SPEC vision-harness-reuse H7: a
published table is a generator's output, pasted verbatim).

This tool sends no request. The requests that take turns between contexts, the case the cache
exists for, come from `vision-suite/prompt_cache_probe.py`, through `client.generate()`
(SPEC H9).

`ab` compares extbench arms run the same two ways: pairs of `ext_<tag>_<bench>.json`, cache on
then off, over the same rows. It prints the mean client-side seconds per request, leaving out
each arm's first request because that one loads the model, and how many outputs differ as text.
"""
import argparse
import gzip
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone

LINE = re.compile(r"^(\S+) \S+ [^:]+: (.*)$")
SUPPRESSED = re.compile(r"Suppressed (\d+) messages from ollama")
MODEL_BLOB = re.compile(r"--model \S*sha256-([0-9a-f]{64})")


def manifest_models(root):
    """{blob hex: model name} from an ollama store's manifests directory."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            try:
                with open(path) as f:
                    layers = json.load(f).get("layers", [])
            except (ValueError, OSError):
                continue
            rel = os.path.relpath(path, root).split(os.sep)
            model = f"{rel[-2]}:{rel[-1]}" if len(rel) >= 2 else rel[-1]
            for layer in layers:
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    out.setdefault(layer["digest"].split(":", 1)[1], model)
    return out


def parse_window(spec):
    label, _, span = spec.partition("=")
    start, _, end = span.partition("/")
    t = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if s else None  # noqa: E731
    return label, t(start), t(end)


def percentile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def parse(lines, models, windows):
    rows, model = {}, None

    def row(t):
        label = next((w for w, a, b in windows if a <= t and (b is None or t <= b)), "other")
        return rows.setdefault((model, label), {
            "requests": 0, "updates_ms": [], "restores": 0, "slot_reuse": 0,
            "state_mib": [], "prompt_len": [], "compute_ms": [], "dropped": 0})

    for raw in lines:
        m = LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        t = datetime.fromisoformat(m[1]).astimezone(timezone.utc)
        msg = m[2]
        if "starting llama-server" in msg:
            b = MODEL_BLOB.search(msg)
            model = models.get(b[1], "blob " + b[1][:12]) if b else "unknown"
            continue
        if model is None:
            continue
        dropped = SUPPRESSED.search(msg)
        if dropped:
            row(t)["dropped"] += int(dropped[1])
        elif "prompt cache update took" in msg:
            row(t)["updates_ms"].append(float(re.search(r"took ([0-9.]+) ms", msg)[1]))
        elif "found better prompt" in msg:
            row(t)["restores"] += 1
        elif "selected slot by LCP similarity" in msg:
            row(t)["slot_reuse"] += 1
        elif "saving prompt with length" in msg:
            s = re.search(r"length (\d+), total state size = ([0-9.]+) MiB", msg)
            r = row(t)
            r["prompt_len"].append(int(s[1]))
            r["state_mib"].append(float(s[2]))
        elif "print_timing" in msg and "total time" in msg:
            r = row(t)
            r["requests"] += 1
            r["compute_ms"].append(float(re.search(r"total time =\s+([0-9.]+) ms", msg)[1]))

    out = []
    for (mdl, label), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        u = r["updates_ms"]
        rec = {"model": mdl, "window": label, "requests": r["requests"], "cache_updates": len(u),
               "restores": r["restores"], "slot_reuse": r["slot_reuse"]}
        if r["dropped"]:
            rec["journal_lines_dropped"] = r["dropped"]
        if r["compute_ms"]:
            rec["compute_ms_median"] = round(statistics.median(r["compute_ms"]))
        if u:
            rec.update(update_ms_median=round(statistics.median(u)), update_ms_p90=round(percentile(u, 0.9)),
                       update_s_total=round(sum(u) / 1000))
        if r["state_mib"]:
            rec.update(state_mib_median=round(statistics.median(r["state_mib"]), 1),
                       prompt_len_median=round(statistics.median(r["prompt_len"])),
                       mib_per_token=round(statistics.median(
                           s / n for s, n in zip(r["state_mib"], r["prompt_len"]) if n), 3))
        out.append(rec)
    every = [x for r in rows.values() for x in r["updates_ms"]]
    totals = {"cache_updates": len(every), "restores": sum(r["restores"] for r in rows.values()),
              "journal_lines_dropped": sum(r["dropped"] for r in rows.values())}
    if every:
        totals.update(update_ms_mean=round(statistics.mean(every)),
                      update_ms_p10=round(percentile(every, 0.1)),
                      update_ms_median=round(statistics.median(every)),
                      update_ms_p90=round(percentile(every, 0.9)),
                      update_ms_max=round(max(every)),
                      update_h_total=round(sum(every) / 3.6e6, 2))
    return {"windows": [{"label": w, "start": a.isoformat() if a else None, "end": b.isoformat() if b else None}
                        for w, a, b in windows],
            "rows": out, "totals": totals}


def render(d):
    print("| model | window | requests | cache updates | restores | slot reuse | state, median | MiB per token "
          "| update, median | update, p90 | compute, median |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d["rows"]:
        if r["window"] == "other":
            continue
        g = lambda k, fmt="{}": fmt.format(r[k]) if k in r else "—"  # noqa: E731
        print(f"| `{r['model']}` | {r['window']} | {r['requests']} | {r['cache_updates']} | {r['restores']} "
              f"| {r['slot_reuse']} | {g('state_mib_median', '{} MiB')} | {g('mib_per_token')} "
              f"| {g('update_ms_median', '{} ms')} | {g('update_ms_p90', '{} ms')} | {g('compute_ms_median', '{} ms')} |")
    t = d["totals"]
    print()
    print("| the whole log | cache updates | restores | mean | p10 | median | p90 | max | total |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| all | {t['cache_updates']} | {t['restores']} | {t.get('update_ms_mean', '—')} ms "
          f"| {t.get('update_ms_p10', '—')} ms | {t.get('update_ms_median', '—')} ms | {t.get('update_ms_p90', '—')} ms "
          f"| {t.get('update_ms_max', '—')} ms | {t.get('update_h_total', '—')} h |")
    short = [r for r in d["rows"] if r.get("journal_lines_dropped") and r["window"] != "other"]
    if short:
        print("\njournald dropped lines in " + "; ".join(
            f"`{r['model']}` {r['window']} ({r['journal_lines_dropped']} lines)" for r in short)
              + ": those rows' counts are lower bounds.")
    if "meta" in d:
        print(f"\n{d['meta']}")


def ab(paths):
    """One row per (cache on, cache off) pair of extbench score files over the same rows."""
    if not paths or len(paths) % 2:
        raise ValueError("ab takes pairs: a cache-on score file, then its cache-off twin")
    print("| model | rows | requests | cache on, s per request | cache off, s per request | change "
          "| outputs that differ |")
    print("|---|---|---|---|---|---|---|")
    for on_path, off_path in zip(paths[::2], paths[1::2]):
        runs = []
        for p in (on_path, off_path):
            with open(p) as f:
                runs.append(json.load(f))
        (on, off), keys = runs, ("model", "benchmark", "dataset", "split", "offset", "requested")
        if [on["summary"].get(k) for k in keys] != [off["summary"].get(k) for k in keys]:
            raise ValueError(f"{on_path} and {off_path} are not the same arm")
        a, b = ({r["i"]: r for r in d["records"] if "error" not in r} for d in (on, off))
        first = on["summary"]["offset"]
        shared = sorted(i for i in set(a) & set(b) if i != first)
        m_on = statistics.mean(a[i]["secs"] for i in shared)
        m_off = statistics.mean(b[i]["secs"] for i in shared)
        differ = sum(1 for i in shared + ([first] if first in a and first in b else [])
                     if a[i]["pred"] != b[i]["pred"])
        s = on["summary"]
        print(f"| `{s['model']}` | {s['benchmark']} {s['offset']}..{s['offset'] + s['requested']} "
              f"| {len(shared)} | {m_on:.3f} | {m_off:.3f} | {(m_off / m_on - 1) * 100:+.1f} % | {differ} |")
    print("\nextbench records each request to 0.1 s, so a difference under about 0.05 s per request is "
          "below this table's resolution; the journal times each cache update to the millisecond.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--journal", default="-", help="journalctl -o short-iso-precise output, .gz or plain (default stdin)")
    p.add_argument("--manifests", required=True, help="the ollama store's manifests directory")
    p.add_argument("--window", action="append", default=[], help="LABEL=START/END, ISO 8601 UTC; END may be empty")
    p.add_argument("--meta", help="a line that render prints under the tables")
    r = sub.add_parser("render")
    r.add_argument("stats")
    b = sub.add_parser("ab")
    b.add_argument("pairs", nargs="+", help="cache-on score file, then its cache-off twin, repeated")
    args = ap.parse_args()
    if args.cmd == "render":
        with open(args.stats) as f:
            render(json.load(f))
        return
    if args.cmd == "ab":
        try:
            ab(args.pairs)
        except ValueError as e:
            ap.error(str(e))
        return
    src = sys.stdin if args.journal == "-" else (
        gzip.open(args.journal, "rt", errors="replace") if args.journal.endswith(".gz")
        else open(args.journal, errors="replace"))
    with src:
        d = parse(src, manifest_models(args.manifests), [parse_window(w) for w in args.window])
    if args.meta:
        d["meta"] = args.meta
    json.dump(d, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
