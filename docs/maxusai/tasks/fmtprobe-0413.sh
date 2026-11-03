#!/bin/bash
# CUDA side of the nemotron3 format bisect (#501): the same Go (the v0.40.5 fold's) on the b11669 payload
# (sync-0.40.5-fold) and on b11963 (fmtprobe-b11963-go0405, the v0.40.13 fold image with that Go binary swapped in),
# fresh container each, GPU0 only, production's env. Queued after the sm_120 gate, which runs after the canary.
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; D=$S/fmtprobe-0413; PORT=11534; CUR=fmtprobe-0413
. $S/gpu-fit-0350.sh
until grep -qE '### SM120-GATE rc=' $S/verify-0413/gate-sm120.log 2>/dev/null; do sleep 120; done
echo "### FMTPROBE start $(date '+%F %T') after: $(grep '### SM120-GATE' $S/verify-0413/gate-sm120.log)"
trap 'docker rm -f $CUR >/dev/null 2>&1' EXIT
for arm in "b11669 maxusai/ollama:sync-0.40.5-fold" "b11963 maxusai/ollama:fmtprobe-b11963-go0405"; do
  set -- $arm; tag=$1; img=$2
  wait_fit "$(need_for nemotron3:33b-q4_K_M)" "fmtprobe $tag"
  docker rm -f $CUR >/dev/null 2>&1
  docker run -d --name $CUR --gpus '"device=0"' -v ollama_store_8tb:/root/.ollama -v $S/jitcache:/jitcache -e CUDA_CACHE_PATH=/jitcache/nv \
    -e OLLAMA_KV_CACHE_TYPE=f16 -e OLLAMA_FORMAT_TWO_PASS=1 -e OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0 -e OLLAMA_NOPRUNE=true \
    -e OLLAMA_MAX_LOADED_MODELS=1 -e OLLAMA_HOST=0.0.0.0:11434 -p 127.0.0.1:$PORT:11434 "$img" >/dev/null
  for i in $(seq 1 90); do curl -sf --max-time 3 http://127.0.0.1:$PORT/api/version >/dev/null && break; sleep 2; done
  echo "### $tag $img serving $(curl -s http://127.0.0.1:$PORT/api/version) $(date '+%T')"
  python3 $S/wt-sync0413/docs/maxusai/tasks/fmtprobe-0413.py http://127.0.0.1:$PORT $D/resp-$tag.json 2>&1
  echo "### $tag launch: $(docker logs $CUR 2>&1 | grep -oE 'commit [0-9a-f]{9}|-b [0-9]+ -ub [0-9]+' | sort -u | tr '\n' ' ')"
  docker logs $CUR > $D/runner-$tag.log 2>&1; docker rm -f $CUR >/dev/null
done
echo "### FMTPROBE DONE $(date '+%F %T')"
