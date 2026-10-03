# ADR 0047: llama-server's prompt cache is sized per model or per request

Date: 2026-10-03 · Status: accepted (shipped with the `prompt_cache_ram` option). Decision 2, a
host-wide variable, was withdrawn the same day: see the first addendum. The vision harness sends
`0`: see the second.

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
2. ~~**`OLLAMA_LLAMA_SERVER_CACHE_RAM` is the host's default** for a load that names none.~~
   Withdrawn: see the addendum.
3. **Without the option, ollama passes no `--cache-ram`**, so llama.cpp's default applies, as
   upstream: the cache is on. This ADR does not change the default.
4. **The scheduler compares the value a launch would pass**, not the value a client sent. An
   invalid option is ignored with a warning, so llama.cpp's default applies and a Modelfile typo
   cannot make a model unloadable (as `kv_cache_type` does).

## Consequences

- **One host can serve image-heavy models with the cache off and chat models with it on.**
- **A different value relaunches the runner,** which costs a model load (4–17 s for the models
  measured). Clients that alternate values reload on every switch, so set it per model or per
  client, not per request.
- **An invalid value launches the same runner as no value,** so it does not relaunch one.
- **An MLX runner ignores the option** and is never relaunched for it.
- **The change is Go-only,** like `kv_cache_type`, and needs no compat patch at a fold. Upstream
  has no such option.

## Addendum 2026-10-03: no host variable

The first version (#440) also read `OLLAMA_LLAMA_SERVER_CACHE_RAM` as the host's default for a
load that named no `prompt_cache_ram`. The maintainer withdrew it the same day, before any host set
it:
- **Two settings for one flag meant a precedence order** to read every load against: request, then
  model, then host.
- **A host-wide value is what the context above says cannot fit every model.**

A model that should run without the cache sets `PARAMETER prompt_cache_ram 0` in its Modelfile,
and a client sends the option. Unset, nothing changes. The A/B in
[llama-server-prompt-cache.md](../llama-server-prompt-cache.md) ran on a build that still read the
variable, which set the same `--cache-ram` flag.

## Addendum 2026-10-03: the harness turns it off

The vision harness sends `prompt_cache_ram: 0` on every request, from its one request path
(`vision-suite/client.py`, `prompt_cache_ram()`). `PROMPT_CACHE_RAM` overrides it with a size,
`-1`, or `server` to send none. Decision 3 stands: ollama's default is unchanged, for every other
client too.
- **The harness's traffic is the case measured above:** almost every request brings a new image,
  so the cache costs it up to a fifth of a request and returns nothing.
- **One value serves every tool.** `vision_suite.py` and `finetext_probe.py` share a campaign
  cell, and a value that differed between them would relaunch the runner inside it.
- **Three tools send none,** because each sends exactly the payload it measures with:
  `measure.py`, `token_split.py` and `prompt_cache_probe.py`.
- **It is recorded:** `req_prompt_cache_ram` in a score block, `prompt_cache_ram` in an extbench
  summary. `summarize_extbench.py --timing` warns when arms differ. Seconds per item measured
  before this date ran with the cache on.
- **A build without the option** drops it with a warning in its log and keeps the cache on. The
  record says what was asked, and `server_version` says which build answered.
- **It is the one exception to SPEC H4,** which wants a new knob inert by default. The SPEC
  records it under H4, with the recording that keeps what H4 protects.
