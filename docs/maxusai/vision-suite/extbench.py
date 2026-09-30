#!/usr/bin/env python3
"""External-benchmark slices against an ollama endpoint, scored locally.

Usage: extbench.py <host> <tag> [model] [benchmark]
e.g.   extbench.py http://127.0.0.1:11435 canon qwen3.6:35b-a3b-q4_k_m ocrbench

Benchmarks (rows pulled from the HF datasets-server REST API — no `datasets`
install, no HF token, public datasets only):

  ocrbench      echo840/OCRBench test        contains-match  (lmms-eval semantics)
  countbenchqa  vikhyatk/CountBenchQA test   integer match
  chartqa       lmms-lab-encoder/ChartQA test relaxed accuracy (+-5% numeric)
  refcoco       lmms-lab-encoder/RefCOCO val  dialect-aware bbox IoU

Env: LIMIT (default 50), OFFSET (0), REFRESH_ROWS=1 (re-fetch the cached row
slice), THINK=on|false (false), ENDPOINT=generate|chat
(generate), NUM_PREDICT, NUM_CTX (16384), TIMEOUT (900), SLEEP (0 — seconds between
requests, to yield the GPU on a shared host).

Writes ext_<tag>_<bench>.json (per-item records + summary) beside the script and
caches images and the row slice under extimgs/<bench>/.

Every image is fetched before the first request. The rows' image links are signed and
expire an hour after the rows were fetched, and the row cache outlives them: on
2026-09-30 both arms of a 1000-item OCRBench run stopped on a 403 at item 682, an hour
in, and wrote nothing. An expired link is refreshed from its /rows page, which must still
hold the same items; an image that still cannot be fetched is recorded as an error, as a
failed request is. A running arm checkpoints to ext_<tag>_<bench>.partial.json every 50
items and when it is stopped (Ctrl-C, SIGTERM); finishing removes the checkpoint.

The refcoco scorer reuses the dialect logic of vision_suite.py: it searches
pixel / norm-1000 / norm-0-1 spaces and xyxy / yxyx orders per item and keeps the
best, so qwen3.6 (bbox_2d, xyxy, norm-1000), gemma4 (box_2d, yxyx, norm-1000) and
nemotron3 (self-chosen key, pixel under reasoning) are all scored fairly. External
harnesses do not do this: lmms-eval's refcoco_bbox_rec demands normalized 0-1 xyxy
and scores everything else ~0, VLMEvalKit auto-detects the scale but not the order.
"""
import base64, json, os, re, signal, sys, time, urllib.parse, urllib.request

import client

DIR = os.path.dirname(os.path.abspath(__file__))
HOST = TAG = MODEL = BENCH = None

BENCHES = {
    "ocrbench":     dict(dataset="echo840/OCRBench",            config="default", split="test"),
    "countbenchqa": dict(dataset="vikhyatk/CountBenchQA",       config="default", split="test"),
    "chartqa":      dict(dataset="lmms-lab-encoder/ChartQA",    config="default", split="test"),
    "refcoco":      dict(dataset="lmms-lab-encoder/RefCOCO",    config="default", split="val"),
}

# Per-benchmark prompt suffixes. Kept close to the lmms-eval task YAMLs so scores are
# comparable to published harness numbers; refcoco asks for the model's own JSON
# dialect instead of forcing one convention (see module docstring).
SUFFIX = {
    "ocrbench": "\nAnswer the question using a single word or phrase.",
    "countbenchqa": "\nAnswer with a single number.",
    "chartqa": "\nAnswer the question using a single word or phrase.",
}

BBOX_KEYS = ("bbox", "bbox_2d", "box_2d", "box")


def http_json(url, timeout=60, attempts=4):
    """GET with retries: a transient DNS or gateway blip must not end a 20-minute arm."""
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "maxusai-extbench/1"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception as e:                                        # noqa: BLE001
            last = e
            if i + 1 < attempts:
                time.sleep(2 ** i)
    raise last


def rows_cache_path(bench, offset, limit):
    return os.path.join(DIR, "extimgs", bench, f"rows_{offset}_{limit}.json")


def rows_page(bench, offset, length):
    spec = BENCHES[bench]
    q = urllib.parse.urlencode(dict(dataset=spec["dataset"], config=spec["config"],
                                    split=spec["split"], offset=offset, length=length))
    page = http_json("https://datasets-server.huggingface.co/rows?" + q, timeout=120)
    return [r["row"] for r in page.get("rows", [])]


def save_rows(path, rows):
    """Write a slice to the row cache atomically, without the fields main() adds (_w, _h, _expr)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], f)
    os.replace(tmp, path)


def fetch_rows(bench, offset, limit):
    """HF datasets-server /rows, paged at its 100-row maximum, cached on disk.

    A slice is a fixed set of items, so it is fetched once and reused: re-fetching per
    arm re-asks a remote service for an answer that must not change between arms, and
    makes every arm depend on that service still resolving. Set REFRESH_ROWS=1 to
    re-fetch (and to notice a dataset that moved under a pinned offset).

    The rows' image links are signed and expire -- an hour after the fetch, measured
    2026-09-30 -- so a cached slice outlives its links. ensure_image() refreshes them.
    """
    path = rows_cache_path(bench, offset, limit)
    if os.path.exists(path) and os.environ.get("REFRESH_ROWS", "") != "1":
        with open(path) as f:
            return json.load(f)
    out = []
    while len(out) < limit:
        rows = rows_page(bench, offset + len(out), min(100, limit - len(out)))
        if not rows:
            break
        out.extend(rows)
    if len(out) == limit:                    # only a complete slice is worth caching
        save_rows(path, out)
    return out


# ---------------------------------------------------------------- images

# How close to its Expires time a signed link counts as expired: a download started a
# minute before expiry may not finish before it.
LINK_EXPIRY_MARGIN_S = 60


def image_src(row):
    return row["image"]["src"] if isinstance(row["image"], dict) else row["image"]


def link_expiry(src):
    """The Expires time (unix seconds) of a signed datasets-server link; None if unsigned."""
    m = re.search(r"[?&]Expires=(\d+)", src or "")
    return int(m.group(1)) if m else None


def link_expired(src, now=None):
    expiry = link_expiry(src)
    now = time.time() if now is None else now
    return expiry is not None and expiry - LINK_EXPIRY_MARGIN_S <= now


def image_path(bench, idx, src):
    ext = ".png" if ".png" in src.lower().split("?")[0] else ".jpg"
    return os.path.join(DIR, "extimgs", bench, f"{idx:05d}{ext}")


def download(src, path):
    """Fetch src to path atomically. Writing straight to path left a truncated file behind
    when a download failed part-way, and the next arm read it as a cached image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        req = urllib.request.Request(src, headers={"User-Agent": "maxusai-extbench/1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def cache_image(bench, idx, src):
    path = image_path(bench, idx, src)
    if not os.path.exists(path):
        download(src, path)
    return path


class SliceMoved(Exception):
    """A re-fetched /rows page no longer holds the items the cached slice holds."""


def same_item(a, b):
    """Two rows are the same item when everything but the image link matches."""
    def strip(row):
        return {k: v for k, v in row.items() if k != "image" and not k.startswith("_")}
    return strip(a) == strip(b)


def refresh_links(bench, offset, rows, i):
    """Re-fetch the /rows page holding item i for fresh image links, updating rows in place.

    Only the links may change. A row that differs in anything else means the dataset moved
    under the pinned offset, and taking it would score a different item under the same
    index, so that raises SliceMoved and leaves rows alone. The fresh links go back into
    the row cache, so the next arm starts from them rather than from expired ones.
    """
    start = i - i % 100                          # fetch_rows pages from the slice's start
    n = min(100, len(rows) - start)
    fresh = rows_page(bench, offset + start, n)
    if len(fresh) != n or not all(same_item(a, b) for a, b in zip(rows[start:start + n], fresh)):
        raise SliceMoved(f"rows {offset + start}..{offset + start + n} changed under the pinned "
                         f"offset; REFRESH_ROWS=1 re-fetches the slice")
    for j, row in enumerate(fresh):
        rows[start + j]["image"] = row["image"]
    path = rows_cache_path(bench, offset, len(rows))
    if os.path.exists(path):
        save_rows(path, rows)


def ensure_image(bench, offset, rows, i):
    """Item i's image on local disk: (path, None), or (None, why) when it cannot be had.

    A link past its Expires time is refreshed before use, and a failed download is retried
    once on a refreshed link. What still fails is returned, not raised: one missing image
    is one error record, where it used to end the arm and lose every result before it
    (2026-09-30: both OCRBench arms of a 1000-item slice stopped at item 682 on a 403).
    """
    src = image_src(rows[i])
    path = image_path(bench, offset + i, src)
    if os.path.exists(path):
        return path, None
    try:
        refreshed = link_expired(src)
        if refreshed:
            refresh_links(bench, offset, rows, i)
        try:
            return cache_image(bench, offset + i, image_src(rows[i])), None
        except OSError:                          # HTTPError and URLError are OSErrors
            if refreshed:
                raise
        refresh_links(bench, offset, rows, i)
        return cache_image(bench, offset + i, image_src(rows[i])), None
    except Exception as e:                                        # noqa: BLE001
        return None, f"image not fetched: {e}"


def prefetch_images(bench, offset, rows):
    """Every image of the slice, fetched before the first request, so no arm depends on a
    link that expires while it runs. Returns {item index: why} for the ones that failed."""
    failed = {}
    for i in range(len(rows)):
        _, err = ensure_image(bench, offset, rows, i)
        if err:
            failed[offset + i] = err
    return failed


def gen(prompt, image_path, fmt=None):
    """External-benchmark request, through the shared client (SPEC H1 / ADR 0028).

    Pinned to temperature 0 with apply_sampling=False and to whichever endpoint
    ENDPOINT names, defaulting to /api/generate as this tool always has: the
    published external-benchmark slices were measured that way, and a benchmark
    that silently changed sampling or endpoint would be scoring a different
    configuration under the same name.
    """
    num_ctx = int(os.environ.get("NUM_CTX", "16384"))
    num_predict = int(os.environ.get("NUM_PREDICT", "512"))
    return client.generate(
        HOST, MODEL, prompt, [client.b64_file(image_path)],
        num_predict=num_predict, num_ctx=num_ctx, fmt=fmt,
        apply_sampling=False, extra_opts={"temperature": 0},
        endpoint_override=os.environ.get("ENDPOINT", "generate"),
        timeout=int(os.environ.get("TIMEOUT", "900")))


# ---------------------------------------------------------------- scorers

def norm_text(s):
    return re.sub(r"\s+", " ", str(s).lower().strip().replace("\n", " "))


def score_ocrbench(pred, row):
    """lmms-eval ocrbench_process_results: gold answer contained in the prediction.

    Handwritten-Mathematical-Expression-Recognition items are whitespace-stripped on
    both sides upstream; everything else is a plain normalized substring test.
    """
    golds = row.get("answer") or []
    if isinstance(golds, str):
        golds = [golds]
    p = norm_text(pred)
    if row.get("question_type") == "Handwritten Mathematical Expression Recognition":
        p_ns = re.sub(r"\s", "", p)
        return any(re.sub(r"\s", "", norm_text(g)) in p_ns for g in golds)
    return any(norm_text(g) in p for g in golds)


def score_count(pred, row):
    m = re.search(r"-?\d+", str(pred).replace(",", ""))
    if not m:
        return False
    try:
        return int(m.group()) == int(str(row["number"]).strip())
    except (ValueError, KeyError):
        return False


def score_chartqa(pred, row):
    """Relaxed accuracy: +-5% for numeric golds, normalized exact match otherwise."""
    gold = row.get("answer")
    if isinstance(gold, list):
        gold = gold[0] if gold else ""
    g = norm_text(gold)
    p = norm_text(pred)
    try:
        gv = float(re.sub(r"[%$,]", "", g))
    except ValueError:
        return p == g or g in p.split()
    m = re.findall(r"-?\d+\.?\d*", p.replace(",", "").replace("%", "").replace("$", ""))
    if not m:
        return False
    for cand in m:
        try:
            pv = float(cand)
        except ValueError:
            continue
        if gv == 0:
            if pv == 0:
                return True
        elif abs(pv - gv) / abs(gv) <= 0.05:
            return True
    return False


def parse_boxes(text):
    """Every plausible 4-number box in the response, with its JSON key when it had one."""
    out = []
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in BBOX_KEYS and isinstance(v, list) and len(v) == 4:
                    try:
                        out.append((k, [float(x) for x in v]))
                    except (TypeError, ValueError):
                        pass
                else:
                    walk(v)
        elif isinstance(o, list):
            if len(o) == 4 and all(isinstance(x, (int, float)) for x in o):
                out.append((None, [float(x) for x in o]))
            for v in o:
                walk(v)

    if obj is not None:
        walk(obj)
    if not out:
        m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,"
                      r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", str(text))
        if m:
            out.append((None, [float(g) for g in m.groups()]))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def decode_box(box, space, order, W, H):
    if order == "yxyx":
        box = [box[1], box[0], box[3], box[2]]
    if space == "norm1000":
        fx, fy = W / 1000.0, H / 1000.0
    elif space == "norm01":
        fx, fy = float(W), float(H)
    else:
        fx = fy = 1.0
    x1, y1, x2, y2 = box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def score_refcoco(pred, row):
    """Dialect-aware IoU: best over {pixel, norm-1000, norm-0-1} x {xyxy, yxyx}.

    Ground truth is COCO [x, y, w, h] in pixels. Returns the best IoU, the winning
    dialect, and the JSON key the model chose (so a run doubles as a dialect probe).
    """
    gt = row.get("bbox")
    if isinstance(gt, str):
        gt = json.loads(gt)
    gtb = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
    W, H = row["_w"], row["_h"]
    best = (0.0, None, None)
    for key, box in parse_boxes(pred):
        for space in ("pixel", "norm1000", "norm01"):
            for order in ("xyxy", "yxyx"):
                v = iou(decode_box(box, space, order, W, H), gtb)
                if v > best[0]:
                    best = (v, f"{space}/{order}", key)
    return best


def image_size(path):
    """PNG/JPEG dimensions without Pillow (the suite's images come as either)."""
    with open(path, "rb") as f:
        head = f.read(2)
        if head == b"\x89P":
            f.seek(16)
            w, h = int.from_bytes(f.read(4), "big"), int.from_bytes(f.read(4), "big")
            return w, h
        f.seek(2)
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6",
                          b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                f.read(3)
                h = int.from_bytes(f.read(2), "big")
                w = int.from_bytes(f.read(2), "big")
                return w, h
            seg = int.from_bytes(f.read(2), "big")
            if seg < 2:
                raise ValueError(f"cannot read JPEG size: {path}")
            f.seek(seg - 2, 1)


def build_prompt(row):
    if BENCH == "refcoco":
        expr = row.get("answer")
        if isinstance(expr, str):
            try:
                expr = json.loads(expr.replace("'", '"'))
            except ValueError:
                expr = [expr]
        expr = expr[0] if isinstance(expr, list) and expr else str(expr)
        row["_expr"] = expr
        return ('Locate the region described as: "%s".\n'
                'Reply with JSON only: a single object with one key "bbox_2d" whose '
                'value is the bounding box of that region as four numbers. Use whatever '
                'coordinate convention you were trained on; do not add any other text.'
                % expr)
    return row["question"] + SUFFIX.get(BENCH, "")


def main():
    global HOST, TAG, MODEL, BENCH
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    HOST = sys.argv[1].rstrip("/")
    TAG = sys.argv[2]
    MODEL = sys.argv[3] if len(sys.argv) > 3 else "nemotron3:33b-q4_K_M"
    BENCH = sys.argv[4] if len(sys.argv) > 4 else "ocrbench"
    if BENCH not in BENCHES:
        sys.exit(f"unknown benchmark {BENCH!r}; pick one of {', '.join(BENCHES)}")

    limit = int(os.environ.get("LIMIT", "50"))
    offset = int(os.environ.get("OFFSET", "0"))
    sleep_s = float(os.environ.get("SLEEP", "0"))
    think = os.environ.get("THINK", "false")

    print(f"# {BENCH}: {BENCHES[BENCH]['dataset']} [{BENCHES[BENCH]['split']}] "
          f"rows {offset}..{offset + limit}")
    rows = fetch_rows(BENCH, offset, limit)
    print(f"# fetched {len(rows)} rows; model={MODEL} think={think} "
          f"endpoint={os.environ.get('ENDPOINT', 'generate')}")
    missing = prefetch_images(BENCH, offset, rows)
    print(f"# images: {len(rows) - len(missing)} of {len(rows)} on disk"
          + (f"; {len(missing)} could not be fetched and are recorded as errors" if missing else ""))

    records, correct, ious, dialects, empty = [], 0, [], {}, 0
    # H11: collected as SETS, not scalars. A container restart mid-run
    # (the vision suite has had them) would otherwise let one build
    # vouch for rows another build served.
    hosts, builds = set(), set()
    out = os.path.join(DIR, f"ext_{TAG}_{BENCH}.json")
    # Not ext_<tag>_<bench>.json, which summarize_extbench.py reads: a stopped arm must not
    # be summarized as a finished one.
    partial = os.path.join(DIR, f"ext_{TAG}_{BENCH}.partial.json")

    def summary(**extra):
        return dict(build_summary(offset, limit, think, records, correct, empty, hosts, builds,
                                  ious, dialects), **extra)

    def checkpoint():
        write_json(partial, {"summary": summary(complete=False, stopped_at=offset + len(records)),
                             "records": records})

    try:
        previous = signal.signal(signal.SIGTERM, stop_on_sigterm)
    except ValueError:                           # not the main thread: leave SIGTERM alone
        previous = None
    finished = False
    try:
        for i, row in enumerate(rows):
            if i and i % CHECKPOINT_EVERY == 0:
                checkpoint()
            path, err = ensure_image(BENCH, offset, rows, i)
            if err:
                records.append({"i": offset + i, "error": err})
                print(f"{offset + i:>4}  ERROR {err}")
                continue
            if BENCH == "refcoco":
                row["_w"], row["_h"] = image_size(path)
            prompt = build_prompt(row)
            t0 = time.time()
            try:
                r = gen(prompt, path, fmt="json" if BENCH == "refcoco" else None)
                pred = r.get("response", "")
            except Exception as e:                                    # noqa: BLE001
                records.append({"i": offset + i, "error": str(e)})
                print(f"{offset + i:>4}  ERROR {e}")
                continue
            dt = round(time.time() - t0, 1)
            hosts.add(r.get("_host"))
            builds.add(r.get("_server_version"))
            if not pred.strip():
                empty += 1

            rec = {"i": offset + i, "prompt": prompt, "pred": pred, "secs": dt,
                   "prompt_eval_count": r.get("prompt_eval_count"),
                   "eval_count": r.get("eval_count")}
            if BENCH == "refcoco":
                v, dialect, key = score_refcoco(pred, row)
                ious.append(v)
                ok = v >= 0.5
                if dialect:
                    dialects[f"{key or '-'} {dialect}"] = dialects.get(f"{key or '-'} {dialect}", 0) + 1
                rec.update(expr=row.get("_expr"), gt=row.get("bbox"), iou=round(v, 3),
                           dialect=dialect, key=key, ok=ok)
            else:
                ok = {"ocrbench": score_ocrbench, "countbenchqa": score_count,
                      "chartqa": score_chartqa}[BENCH](pred, row)
                rec.update(gold=row.get("answer") if BENCH != "countbenchqa" else row.get("number"),
                           ok=bool(ok))
            correct += bool(ok)
            records.append(rec)
            flag = "ok " if ok else "MISS"
            extra = f" iou={rec['iou']} {rec.get('dialect')}" if BENCH == "refcoco" else ""
            print(f"{offset + i:>4}  {flag}{extra}  {dt:>5}s  {norm_text(pred)[:70]!r}")
            if sleep_s:
                time.sleep(sleep_s)
        finished = True
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        if not finished:
            checkpoint()
            print(f"stopped before item {offset + len(records)}; the {len(records)} items "
                  f"before it are in {partial}", file=sys.stderr)

    result = summary()
    write_json(out, {"summary": result, "records": records})
    if os.path.exists(partial):
        os.remove(partial)
    print("\n" + json.dumps(result, indent=1))
    print(f"wrote {out}")


# Items between checkpoints of a running arm to ext_<tag>_<bench>.partial.json.
CHECKPOINT_EVERY = 50


def build_summary(offset, limit, think, records, correct, empty, hosts, builds, ious, dialects):
    n = len([r for r in records if "error" not in r])
    summary = {
        "tag": TAG, "model": MODEL, "benchmark": BENCH,
        "dataset": BENCHES[BENCH]["dataset"], "split": BENCHES[BENCH]["split"],
        "offset": offset, "requested": limit, "scored": n,
        "errors": len(records) - n, "empty_responses": empty,
        # BOTH: the raw env string and the flag actually sent. gen() gates on
        # `!= "on"`, so THINK=true records "true" while sending think=False --
        # a think-mode summary for a run that never enabled thinking. Recording
        # only one of these cannot distinguish those.
        "think_env": think, "think_on": think == "on",
        "endpoint": os.environ.get("ENDPOINT", "generate"),
        # H11: where it ran and which build served it. client.generate()
        # already stamps both on every response; this only persists them.
        "host": sorted(h for h in hosts if h) or None,
        "server_version": sorted(b for b in builds if b) or None,
        "correct": correct, "accuracy": round(correct / n, 4) if n else None,
    }
    if BENCH == "refcoco":
        summary["mean_iou"] = round(sum(ious) / len(ious), 4) if ious else None
        summary["acc@0.5"] = summary.pop("accuracy")
        summary["dialects"] = dialects
    return summary


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def stop_on_sigterm(signum, frame):
    """SIGTERM (systemctl stop, timeout) as an exception, so main() writes what it measured."""
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    main()
