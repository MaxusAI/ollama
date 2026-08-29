#!/bin/bash
# n=5 repeats, forced fp16 (harsher than any real config), on the models the gate EXCLUDES.
SP="${PROBE_DIR:?set PROBE_DIR to a working directory holding the probe images}"
PORT=11592; N=5
docker rm -f repeats >/dev/null 2>&1
docker run -d --rm --name repeats --gpus '"device=0"' -e OLLAMA_MAX_LOADED_MODELS=1 \
  -e OLLAMA_MODELS=/models/models -e GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 \
  -v docker_ollama_data:/models:ro -p 127.0.0.1:$PORT:11434 maxusai/ollama:sync-0.33.0 >/dev/null
for i in $(seq 1 90); do curl -sf -m 2 http://127.0.0.1:$PORT/api/version >/dev/null && break; sleep 1; done
trap 'docker rm -f repeats >/dev/null 2>&1' EXIT
ask () { # model, image, budget
  local b64; b64=$(base64 -w0 "$SP/xengine/$2.png")
  local opt=""; [ -n "$3" ] && opt=",\"image_max_tokens\":$3,\"image_min_tokens\":$3"
  printf '{"model":"%s","stream":false,"messages":[{"role":"user","content":"Describe this image in one sentence.","images":["%s"]}],"options":{"num_ctx":8192,"temperature":0.0,"num_predict":24%s}}' "$1" "$b64" "$opt" \
    | curl -s -m 1200 http://127.0.0.1:$PORT/api/chat --data-binary @- | python3 "$SP/grade.py" | cut -c1
}
echo "### forced fp16, n=$N, models the gate EXCLUDES"
printf '%-30s %-22s %s\n' "model" "image" "runs"
for spec in "nemotron3:33b-q4_K_M|3328" "qwen3.6:35b-a3b-q4_K_M|" "qwen2.5vl:7b-q4_K_M|" \
            "qwen3-vl:2b-thinking-q4_K_M|" "qwen3-vl:4b-thinking-q4_K_M|" "qwen3.5:2b-q4_K_M|" "qwen3.5:4b-q4_K_M|"; do
  m=${spec%|*}; b=${spec#*|}
  for img in T_synthetic_checker56 T_ead2a6c7 T_02c9d7e1; do
    r=""; for i in $(seq 1 $N); do r="$r$(ask "$m" "$img" "$b")"; done
    printf '%-30s %-22s %s\n' "${m##*/}" "$img" "$r"
  done
done
echo "### POSITIVE CONTROL — the model we know fails, same config"
for img in T_synthetic_checker56 T_ead2a6c7; do
  r=""; for i in $(seq 1 $N); do r="$r$(ask "qwen2.5vl:3b-q4_K_M" "$img" "")"; done
  printf '%-30s %-22s %s\n' "qwen2.5vl:3b" "$img" "$r"
done
