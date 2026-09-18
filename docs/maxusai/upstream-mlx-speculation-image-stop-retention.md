# Draft upstream issue: MLX runner retains memory per stop-terminated image request under speculative decoding

> **PARKED 2026-09-18** by standing rule — the fork does not file upstream reports until it can propose a fix, and
> this one has a reproducer, an exclusion list and a mitigation but no fix: the holder is on MLX's side of the Go
> boundary. Draft is complete for filing against **ollama/ollama**; the second part is for **ml-explore/mlx**. Do not
> file without an explicit go-ahead.

Measured on the fork's v0.34.1 build (upstream v0.34.1 + `ec3cc2307`), CUDA host (RTX PRO 6000 class, 98 GB), MLX
`d9add9d1`, MLX-C `ebc88f10`, `qwen3.6:35b-a3b-nvfp4` and `qwen3.8:27b-nvfp4` (both ship MTP heads and draft). Runner
logs and analysers are on the host: `preflight-runs/leakrepro-*-runner.log`, `claude-scratch/leak-repro*.{sh,py}`.

---

**Title (ollama/ollama):** mlxrunner: a request with an image that ends by stop under speculative decoding retains
memory the runner never releases (~0.1–0.3 GiB per request on hybrid/recurrent models, unbounded)

## Summary

On the MLX runner, a chat request that (1) carries an image, (2) ends with `done_reason: stop` (EOS or grammar
completion, as any short answer does) and (3) runs with speculative decoding on (MTP drafting, the default for models
that ship an MTP head) leaves 0.1–0.3 GiB of MLX-tracked memory behind, every such request, with no bound. Requests that
run to `num_predict`, text-only requests, and requests with drafting disabled do not. The grammar does not matter. On
the recurrent-state models this is `held` (the runner's own teardown line) growing request after request outside the
prefix trie's accounting; the trie's 8 GiB `maxPagedOutBytes` budget (ollama#17924) does not bound it because the
memory is not the trie's.

This is likely the non-trie component behind ollama#17875 and ollama#18131 (growth past the trie budget on Metal under
agent workloads: short, stop-terminated tool answers).

## Minimal reproduction

```sh
# any qwen3.5/3.6/3.8 MLX model that ships an MTP head; the 0.8b-mlx package does not and cannot reproduce it
IMG=$(base64 -w0 some_1024x576.png)
for i in $(seq 1 15); do
  curl -s localhost:11434/api/chat -d "{\"model\":\"qwen3.6:35b-a3b-nvfp4\",\"stream\":false,\"think\":false,
    \"messages\":[{\"role\":\"user\",\"content\":\"Request $i. Find every labelled shape; return JSON.\",\"images\":[\"$IMG\"]}],
    \"format\":{\"type\":\"object\",\"properties\":{\"objects\":{\"type\":\"array\"}},\"required\":[\"objects\"]},
    \"options\":{\"temperature\":0,\"seed\":1,\"num_predict\":1500}}" >/dev/null
done
# then read the runner's `msg=memory … held=` line per request (OLLAMA_DEBUG=2 adds the trie's accounting line)
```

With an image that contains nothing to find, every answer is `{"objects":[]}` (5–9 tokens, `stop`) and `held − trie`
grows on every request: on 35b-a3b **14 of 14** steps (+4.5 GiB over 15 requests), +0.42 GiB per request on average;
27b the same shape. Drop `format` — unchanged. Add `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`-style "no drafting" — flat.
Make the answers long (an image with 40 labelled shapes, so generation runs to `num_predict`) — 0 of 38 requests grow,
and the one request in such a run that happened to stop early is the one that grew (+422 MiB).

## What was excluded, each by an instrumented build rather than by reading

A Go-only swap of the runner with a live-array registry (every `mlx.Array` tracked from `New` to `free`, grouped by
owning scope), per-bucket byte accounting at teardown (weights, each target cache slot's buffers by kind, the drafter's
caches, MLX active and pool) and creation/close counters by schedule owner:

- every Go-owned array: zero root-held arrays; the live set (count, scopes, bytes by shape) is identical with and
  without drafting while MLX's active figure differs by GiB;
- unevaluated graphs: evaluating every live array at teardown and clearing the pool releases 0 B;
- the prefix cache's prefill (media-fold) captures: disabling the schedule entirely leaves the growth unchanged;
- speculation's per-round snapshots: created = closed on every request (`commitSpeculation` drains them);
- the drafter's media rows (cloned / detached / discarded balance) and its caches (flat by size);
- MLX-C handles: every `mlx_vector_array_new` / closure site has its free;
- MLX compile cache (`MLX_DISABLE_COMPILE=1`) and CUDA graph cache size (`MLX_CUDA_GRAPH_CACHE_SIZE` 400 / 20 / 50
  over 24 requests: +7.9 / +5.8 / +7.9 GiB, within run-to-run variance);
- synchronous evaluation of the round's drafts (two runs, unchanged);
- the pool-release cadence: `ec3cc2307` (v0.34.2) removed its share and is included in the build measured.

What remains is counted by MLX's allocator (`active_memory`) and referenced by something on the C++ side of the
boundary — an `array` MLX itself holds — created only by the combination above. We could not name it from the Go side.

## Evidence quality

Runs are n = 2 or more per arm where a verdict is drawn (one 8-request arm that looked flat at graph cache 20 was
n = 1 and was contradicted at 24 requests; it is listed above with its correction). The depth controller ramps
differently on every cold load, so drafting volume per request varies; the retention does not track it — it tracks
whether the request ended by stop. Device memory (nvidia-smi on the runner) grows about twice MLX's counter over the
same requests; the difference is a separate, CUDA-internal component that follows input-shape variety and is present
without drafting (see below).

---

**For ml-explore/mlx (CUDA backend), a separate note:** with no drafting at all, 12 image requests whose prompt length
and image size vary grew the runner's **device** memory by +6.6 to +6.8 GiB while MLX's `active_memory` grew +2.1 GiB
(all of it the prefix trie): ~4.5 GiB of CUDA-internal allocations per 12 distinct shapes, invisible to the allocator's
accounting and therefore to any memory limit built on it. Bounding `MLX_CUDA_GRAPH_CACHE_SIZE` to 20 shrank that
component (+2.2 instead of +3.6 over 8 requests) at a prefill-latency cost, consistent with instantiated CUDA graph
execs per distinct op sequence. The plateau over 60 shapes at cache 400 and 50 is measured in the fork's record.
Suggestion: expose the graph cache's device footprint (or bound it in bytes), so `get_active_memory` users can price it.
