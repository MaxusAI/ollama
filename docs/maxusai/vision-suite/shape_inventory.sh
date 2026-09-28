#!/bin/sh
# Capture every GEMM that reaches rocBLAS, per model, with an image in the
# prompt. Produces the TSV that summarize_gemm_shapes.py reads.
#
# ROCBLAS_LAYER=2 makes rocBLAS log each call as a rocblas-bench command line.
# The prompt MUST carry an image: an earlier probe used a text-only prompt, saw
# zero calls, and concluded rocBLAS was not in the hot path. The vision tower
# had simply never run.
#
# Usage: ENDPOINT=http://127.0.0.1:11434 ./shape_inventory.sh out.tsv
# The server must already be running with ROCBLAS_LAYER=2 in its environment;
# this script does not start one, because how you launch a server is yours.
set -u
OUT=${1:-gemm_shapes.tsv}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:11434}
CONTAINER=${CONTAINER:-ollama}
IMAGE=${IMAGE:-visimgs/scene_hd.png}
MODELS=${MODELS:-"gemma4:31b-it-q4_K_M qwen3.8:27b-q4_K_M nemotron3:33b-q4_K_M"}

: > "$OUT"
for m in $MODELS; do
  python3 - "$m" "$ENDPOINT" "$IMAGE" <<'PY'
import base64, json, sys, urllib.request
m, endpoint, img = sys.argv[1], sys.argv[2], sys.argv[3]
b = json.dumps({"model": m, "prompt": "Describe this image.",
 "images": [base64.b64encode(open(img, "rb").read()).decode()],
 "stream": False, "think": False,
 "options": {"temperature": 0, "num_predict": 600, "num_ctx": 16384}}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request(
 endpoint + "/api/generate", b, {"Content-Type": "application/json"}), timeout=900))
print("  %-26s prompt_eval=%s done=%s" % (m, r.get("prompt_eval_count"), r.get("done_reason")))
PY
  docker logs "$CONTAINER" 2>&1 | grep 'rocblas-bench' \
    | python3 -c '
import re, sys
model = sys.argv[1]
for ln in sys.stdin:
    sh = re.search(r"-m (\d+) -n (\d+) -k (\d+)", ln)
    if not sh:
        continue
    dt = re.search(r"--a_type ([a-z0-9_]+)", ln) or re.search(r"-r ([a-z0-9_]+)", ln)
    ta = "T" if "--transposeA T" in ln else "N"
    tb = "T" if "--transposeB T" in ln else "N"
    print("\t".join([model, dt.group(1) if dt else "?", ta + tb, *sh.groups()]))
' "$m" >> "$OUT"
done
echo "wrote $OUT ($(wc -l < "$OUT") rows)"
