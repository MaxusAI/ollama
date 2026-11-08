#!/bin/bash
# The maintainer's word of 2026-11-04: "build the tagged v0.40.13 image and deploy it to CUDA". Production :11497 goes
# from #498's untagged fold image (0.40.3-dynres-12-g8761533) to a build of tag v0.40.13-dynres (85d636818, #501's
# merge): native inputs byte-identical to sync-0.40.13-fold, which passed gates 3-5 on CUDA (cuda-0-40-13 PASS=20,
# mlx-cuda-0-40-13 PASS=11).
#   1 swap-v0413.sh (the maintainer's "switch to the Go-only swap"): the tag's Go binary onto sync-0.40.13-fold
#     -> maxusai/ollama:v0.40.13-dynres, stamp 0.40.13-dynres-0-g85d6368, payload hash-identical to the gated image
#   2 deploy-v0402-point.sh: mirror the live container, wait for an idle production, stop OLD (kept), start NEW,
#     health-check, roll back on failure
#   3 production's own preflight on both platforms against the tag's committed expectations (the 0-40-13 profiles
#     admit the release stamp), then unload, KV allocations, two-pass, cudaMemcpy errors and OOMs
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; R=/mnt/8TB_SN850X_RAID1_BTRFS/preflight-runs
T=$S/wt-v0413tag; IMG=maxusai/ollama:v0.40.13-dynres; WANT=0.40.13-dynres-0-g85d6368; OLD=ollama-0.40.3-dynres-12-g8761533
NEW=ollama-$WANT; H=http://127.0.0.1:11497; RUN=prod-g${WANT##*-g}
ts() { date '+%F %T'; }
echo "### CHAIN-V0413 start $(ts)"
bash $S/swap-v0413.sh > $S/swap-v0413.log 2>&1; rc=$?
echo "### swap rc=$rc $(ts): $(grep -E '^### ' $S/swap-v0413.log | tr '\n' ' ')"
[ $rc -eq 0 ] || { echo "### CHAIN-V0413 DONE (swap failed) $(ts)"; exit $rc; }
v=$(docker run --rm --entrypoint /bin/ollama "$IMG" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+-dynres[^ ]*' | tail -1)
ls=$(docker run --rm --entrypoint sh "$IMG" -c '/usr/lib/ollama/llama-server --version 2>&1 | grep -oE "commit [0-9a-f]+"')
echo "### image stamps $v, llama-server $ls"
[ "$v" = "$WANT" ] || { echo "### image stamps '$v', not $WANT; deploy not run"; exit 2; }

OLD=$OLD IMG=$IMG WANT=$WANT bash $S/deploy-v0402-point.sh
rc=$?
echo "### deploy rc=$rc $(ts)"
[ $rc -eq 0 ] || { echo "### CHAIN-V0413 DONE (deploy rc=$rc) $(ts)"; exit $rc; }

. $S/gpu-fit-0350.sh
wait_fit "$(need_for nemotron3:33b-q8)" "post-deploy preflight"
python3 "$T/docs/maxusai/tools/host_profile.py" --out "$R/hp-cuda-v0413-$RUN" --label "$(head -1 ~/.config/collab/identity)" \
  --fs models=/mnt/8TB_SN850X_RAID1_BTRFS/opt/ollama/.ollama/models --ollama $H > "$R/hp-cuda-v0413-$RUN.log" 2>&1 \
  && echo "### host profile captured" || echo "### host profile capture FAILED"
prc=0
for plat in cuda mlx-cuda; do
  ( cd $T/docs/maxusai/vision-suite/preflight && ./preflight.py --host $H --platform $plat --image-tag $IMG --container $NEW \
      --out "$R/preflight-$plat-v0413-$RUN.json" > "$R/preflight-$plat-v0413-$RUN.log" 2>&1 )
  r=$?; [ $r -ne 0 ] && prc=$r
  echo "### post-deploy preflight $plat rc=$r $(ts): $(grep -E 'profile:' "$R/preflight-$plat-v0413-$RUN.log" | head -1) $(grep -E 'PASS=|FAIL=' "$R/preflight-$plat-v0413-$RUN.log" | tail -1)"
done
for m in $(curl -s --max-time 5 $H/api/ps | python3 -c 'import json,sys; print(" ".join(m["name"] for m in json.load(sys.stdin).get("models",[])))'); do
  curl -s --max-time 60 $H/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null; echo "### unloaded $m"; done
echo "### KV allocations: $(docker logs $NEW 2>&1 | grep -oE 'K \([a-z0-9_]+\)' | sort | uniq -c | tr '\n' ' ')"
echo "### two-pass in config: $(docker logs $NEW 2>&1 | grep -oE 'OLLAMA_FORMAT_TWO_PASS:[a-z]+' | head -1)"
echo "### cudaMemcpy errors: $(docker logs $NEW 2>&1 | grep -c 'cudaMemcpy'); OOMs: $(docker logs $NEW 2>&1 | grep -ciE 'out of memory')"
echo "### CHAIN-V0413 DONE deploy rc=0 preflight rc=$prc $(ts)"
echo "rollback: docker rm -f $NEW && docker start $OLD"
