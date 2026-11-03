#!/bin/bash
# The Nemotron 3.5 pulls, through upstream's stock v0.40.5 client: the fold's interim stamp (0.40.3-dynres-N) is refused
# by both version gates, which are pull-time only (the config's `requires`, and the registry's User-Agent 412).
# No GPU, pruning off, a spare port (never 11434), into production's store; torn down by the trap.
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; CUR=ollama-pull-stock-0405; PORT=11587
trap 'docker logs $CUR > $S/pull-0405-stock-server.log 2>&1; docker rm -f $CUR >/dev/null 2>&1; echo "### stock puller removed $(date +%T)"' EXIT
docker rm -f $CUR >/dev/null 2>&1
docker run -d --name $CUR -v ollama_store_8tb:/root/.ollama -e OLLAMA_NOPRUNE=true -e OLLAMA_HOST=0.0.0.0:11434 \
  -p 127.0.0.1:$PORT:11434 ollama/ollama:0.40.5 >/dev/null || exit 2
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/api/version >/dev/null && break; sleep 2; done
echo "### stock puller up: $(curl -s http://127.0.0.1:$PORT/api/version) $(date '+%F %T')"
for m in nemotron-3.5-bolt:74b-a9b nemotron-3.5-super:120b-a12b nemotron-3.5-bolt:latest nemotron-3.5-bolt:74b-a9b-iq2 nemotron-3.5-super:latest nemotron-3.5-super:120b; do
  echo "### PULL $m $(date '+%F %T')"
  curl -sN --max-time 14400 http://127.0.0.1:$PORT/api/pull -d "{\"model\":\"$m\"}" | python3 -u $S/pullprog.py
done
echo "### store free: $(df -h --output=avail /mnt/8TB_SN850X_RAID1_BTRFS | tail -1)"
echo "### PULL-0405-STOCK DONE $(date '+%F %T')"
