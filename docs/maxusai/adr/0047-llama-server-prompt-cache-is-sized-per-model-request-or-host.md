# ADR 0047: llama-server's prompt cache is sized per model, per request or per host

Date: 2026-10-03 · Status: proposed

## Context

ollama starts `llama-server` with no `--cache-ram`, upstream and in this fork, so llama.cpp's
8192 MiB host-RAM prompt cache is on for every GGUF model. With one slot (`-np 1`), every request
that cannot reuse the slot's prompt first copies the slot's whole state off the GPU into that
cache.

Measured on an H100 ([llama-server-prompt-cache.md](../llama-server-prompt-cache.md)):
- **On image requests it is a cost with nothing back.** The save is about 890 MiB and 0.56–0.82 s
  a request for `gemma4:31b`. The cache restored 0.6 % of the time and never shortened an image's
  prefill. Turned off, `gemma4:31b` answered 19–22 % faster and `qwen3.8:27b-q4_K_M` 4.5–10.6 %
  faster, with the same output on all 3,800 paired requests.
- **It pays back only on text conversations that take turns, on small-state models:** 24–32 % of
  each return to a conversation for `qwen3.8:27b-q4_K_M` and `nemotron3:33b-q8`. It costs
  `gemma4:31b` 42 % even there.

The right setting depends on the model and on the traffic. One host-wide value cannot express
that.

## Decision

1. **`prompt_cache_ram` is a runner option**, in MiB: `0` turns the cache off and `-1` removes the
   limit. It is set by a Modelfile `PARAMETER` or the request's `options`, like `num_ctx` and
   `kv_cache_type` ([ADR 0005](0005-per-model-kv-cache-type.md)). The request wins over the model.
2. **`OLLAMA_LLAMA_SERVER_CACHE_RAM` is the host's default** for a load that names none.
3. **With neither, ollama passes no `--cache-ram`**, so llama.cpp's default applies, as upstream.
   This ADR does not change the default.
4. **The scheduler compares the value a launch would pass**, not the value a client sent. An
   invalid option falls back to the host's default with a warning, so a Modelfile typo cannot make
   a model unloadable (as `kv_cache_type` does).

## Consequences

- **One host can serve image-heavy models with the cache off and chat models with it on.**
- **A different value relaunches the runner,** which costs a model load (4–17 s for the models
  measured). Clients that alternate values reload on every switch, so set it per model or per
  client, not per request.
- **A request that names the host's default does not relaunch the runner** it already has.
- **An MLX runner ignores the option** and is never relaunched for it.
- **The change is Go-only,** like `kv_cache_type`, and needs no compat patch at a fold. Upstream
  has neither the option nor the variable.
