# ADR 0043: production runs an f16 KV cache on every platform

- **Status:** accepted 2026-09-26 (the maintainer: the production KV cache settings are changed everywhere to
  f16). Supersedes part of decision 2 of [ADR 0005](0005-per-model-kv-cache-type.md): an instance no longer keeps
  `q8_0` as its server-wide default. ADR 0005's per-model `kv_cache_type` option and its suite guardrail stand.
- **Date:** 2026-09-26
- **Deciders:** MaxusAI fork maintainers

## Context

ADR 0005 (2026-08-02) traced qwen3.6's think-mode runaway on grounding prompts to `OLLAMA_KV_CACHE_TYPE=q8_0`.
It let an instance keep `q8_0` as its default, provided that reasoning models got f16 at the model level. On
2026-08-03 the gfx1151 host went further and recreated production with f16. Three things have happened since:

- **The default drifted back, and nothing noticed for seven weeks.** The gfx1151 compose file still said
  `q8_0`, so the 2026-08-08 cutover through compose restored it, and later promotions copied the running
  container's arguments. It was found and fixed on 2026-09-26 ([amd-upgrade-gate.md](../amd-upgrade-gate.md),
  the 2026-09-26 decision, #386). No model had a per-model `kv_cache_type`, so ADR 0005's safety net never applied. A
  server-wide default reaches every model that has no override, including the reasoning models it was meant
  to spare.
- **The other hosts never ran it.** On #375 (2026-09-26), the CUDA host reported that neither its production
  nor any gate run sets `OLLAMA_KV_CACHE_TYPE`: every KV cache in its log is `K (f16)`. The Metal host reported
  that its production sets none either. The fleet's results were already f16 everywhere except on gfx1151.
- **`q8_0` changes think-on results.** The v0.34.4 fold's think-on protocol ran on gfx1151 in production's
  `q8_0` environment, and qwen3.6 left two grounding cases unfinished at 131072 tokens. Captured cold, the KV type
  moves the trajectory within the first few hundred characters. With f16, the loops that remain start later: one
  qwen3.6 case from about token 14,650 to about 23,000. The case that finishes does so 5,500 tokens sooner. No cold
  case turns from a loop into a finish on f16 alone
  ([kv-precision-think-loops.md](../tasks/kv-precision-think-loops.md)). ADR 0005's measured inflation, up to
  5.8 times the thinking on a grounding prompt, is the same effect.

What `q8_0` saves is memory: about 3 GB against 6 GB per model at 32K context (ADR 0005). The gfx1151 host has
96 GiB of VRAM.

## Decision

1. **Production on every platform runs an f16 KV cache.** Deploy sources set `OLLAMA_KV_CACHE_TYPE=f16`
   explicitly: compose files, promotion scripts and `docker run` records. That way a copied argument list
   cannot carry an older value forward. A host whose deploy source cannot hold the setting leaves it unset,
   which is f16 in llama-server.
2. **A quantized KV cache is opt-in per model or per request**, through ADR 0005's `kv_cache_type` (for example
   `q8_0/f16`). It is never a server-wide production default.
3. **Test, bench and gate servers run production's KV type**, f16, unless the test is about the KV type. ADR
   0005's rule 3 stands: every result records the KV type it ran with.
4. **Every promotion confirms the KV type on the new container.** Check the environment with `docker inspect`,
   and check the runner's `--cache-type-k` and `--cache-type-v` flags on its first load.

## Consequences

- The KV cache takes twice the memory of `q8_0`. That is still small beside the weights at the context sizes
  production serves.
- ADR 0005's decision 2 no longer lets an instance keep `q8_0` as its default.
- Think-on results measured under `q8_0` carry that caveat. On gfx1151 that is everything since 2026-08-08,
  including the v0.34.4 fold's think-on protocol. The fold's qwen3.6 pair is re-run with f16.
- Each host measures whether KV precision and the attention path decide the loops that remain, in
  [kv-precision-think-loops.md](../tasks/kv-precision-think-loops.md).
