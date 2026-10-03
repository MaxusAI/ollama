# llama-server's prompt cache on vision requests

llama.cpp's server keeps a prompt cache in host RAM, 8192 MiB unless `--cache-ram` says
otherwise. ollama starts `llama-server` without `--cache-ram`, upstream and this fork alike
(`llm/llama_server.go`), so every GGUF model it serves has that cache. On an H100 (sm_90) it
cost every vision request measured here, and paid back only when text conversations take turns
on a model whose state is small.

`prompt_cache_ram` sizes it in MiB, as `--cache-ram` does: `0` turns the cache off and `-1`
removes the limit. A model sets it with a Modelfile `PARAMETER`, and a client in the request's
`options`, like `num_ctx`; a different value relaunches the model's runner. Without it,
llama.cpp's default applies and the cache is on, so nothing changes until a model or a request
sets it ([ADR 0047](adr/0047-llama-server-prompt-cache-is-sized-per-model-or-request.md)). The
vision harness sets it: since 2026-10-03 its requests carry `prompt_cache_ram: 0`
([below](#the-harness-turns-it-off)).

## What the cache does

llama.cpp b11081, `tools/server/server-context.cpp` and `server-task.cpp`:

- **ollama runs one slot** (`-np 1`). A request first looks for a slot whose prompt shares more
  than 10 % of its own prefix. A slot that does is reused in place, and only the tokens after
  the shared prefix are evaluated. The cache plays no part in that.
- **Otherwise the server updates the cache before it starts the request.** It also does when the
  slot would lose more than half of its prompt.
  - `prompt_save` allocates a zero-filled buffer the size of the slot's whole state, and copies
    the state into it from the GPU.
  - It then looks for a cached prompt that matches the new request better, and restores it if
    it finds one.
  - It evicts the oldest entries to stay under the limit.
- **The state grows with the prompt and the model.** On the H100: about 0.76–0.79 MiB per token for
  `gemma4:31b` at every quantisation (the KV cache is f16 either way), 0.19–0.20 for
  `qwen3.8:27b-q4_K_M` and 0.11–0.16 for `nemotron3:33b-q8`. 8192 MiB holds about nine
  `gemma4:31b` vision prompts.
- **llama.cpp times each update itself,** and at ollama's `--log-verbosity 4` logs it with every
  save and restore. `docs/maxusai/tools/prompt_cache_stats.py` reads those lines; no number in
  the first table below is a client's estimate.

## Two days of benchmark arms

`prompt_cache_stats.py render`, verbatim. The journal is ollama's from 2026-09-30 10:40 to
2026-10-02 05:00 UTC: every GGUF arm of the OCRBench, CountBenchQA, ChartQA and RefCOCO queues on
this machine.

| model | window | requests | cache updates | restores | slot reuse | state, median | MiB per token | update, median | update, p90 | compute, median |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | chartqa rows 0..2500 | 2500 | 1612 | 0 | 888 | 888.4 MiB | 0.785 | 797 ms | 828 ms | 1645 ms |
| `nemotron3:33b-q8` | chartqa rows 0..2500 | 2501 | 1613 | 0 | 888 | 50.4 MiB | 0.106 | 31 ms | 33 ms | 154 ms |
| `qwen3.8:27b-q4_K_M` | chartqa rows 0..2500 | 2500 | 1612 | 0 | 888 | 218.3 MiB | 0.199 | 130 ms | 135 ms | 749 ms |
| `gemma4:31b-it-q4_K_M` | countbenchqa rows 0..491 | 491 | 491 | 0 | 0 | 887.8 MiB | 0.791 | 564 ms | 573 ms | 1662 ms |
| `nemotron3:33b-q8` | countbenchqa rows 0..491 | 487 | 488 | 0 | 0 | 49.4 MiB | 0.16 | 31 ms | 54 ms | 120 ms |
| `qwen3.8:27b-q4_K_M` | countbenchqa rows 0..491 | 491 | 491 | 0 | 0 | 217.5 MiB | 0.2 | 171 ms | 225 ms | 732 ms |
| `gemma4:31b-it-bf16` | ocrbench rows 0..1000 | 1000 | 963 | 2 | 37 | 888.1 MiB | 0.788 | 563 ms | 578 ms | 1362 ms |
| `gemma4:31b-it-q8_0` | ocrbench rows 0..1000 | 1000 | 963 | 2 | 37 | 888.1 MiB | 0.788 | 573 ms | 594 ms | 1768 ms |
| `nemotron3:33b-q8` | ocrbench rows 0..1000 | 1000 | 963 | 33 | 37 | 49.6 MiB | 0.149 | 44 ms | 55 ms | 161 ms |
| `qwen3.8:27b-q4_K_M` | ocrbench rows 0..1000 | 1000 | 963 | 25 | 37 | 218.7 MiB | 0.198 | 131 ms | 188 ms | 770 ms |
| `gemma4:31b-it-q4_K_M` | refcoco rows 0..1000 | 1000 | 392 | 0 | 608 | 892.2 MiB | 0.756 | 818 ms | 849 ms | 952 ms |
| `nemotron3:33b-q8` | refcoco rows 0..1000 | 1000 | 393 | 0 | 607 | 49.9 MiB | 0.13 | 30 ms | 35 ms | 245 ms |
| `qwen3.8:27b-q4_K_M` | refcoco rows 0..1000 | 1000 | 392 | 0 | 608 | 222.2 MiB | 0.192 | 131 ms | 134 ms | 1079 ms |
| `gemma4:31b-it-q4_K_M` | refcoco rows 1000..8811 | 7093 | 5391 | 0 | 1702 | 892.1 MiB | 0.757 | 574 ms | 817 ms | 1880 ms |

| the whole log | cache updates | restores | mean | p10 | median | p90 | max | total |
|---|---|---|---|---|---|---|---|---|
| all | 18549 | 115 | 420 ms | 31 ms | 564 ms | 808 ms | 2090 ms | 2.17 h |

H100 SXM5 (sm_90), 0.34.4-dynres-0-gb43ee8e, llama.cpp b11081, llama-server -np 1, --cache-ram unset (8192 MiB): ollama's journal from 2026-09-30 10:40 to 2026-10-02 05:00 UTC

**The cost follows the state.**
- **`gemma4:31b`:** an update takes 0.56–0.82 s, against a median of 0.95–1.9 s of compute.
- **`qwen3.8:27b-q4_K_M`:** 0.13–0.17 s, against 0.7–1.1 s.
- **`nemotron3:33b-q8`:** 0.03–0.04 s, against 0.12–0.25 s.

**The cache restored almost nothing.** 18,549 updates restored a prompt 115 times (0.6 %), never in
CountBenchQA, ChartQA or RefCOCO. On OCRBench, which repeats a document image now and then, the two
small-state models restored 25 and 33 times in 1,000 requests, and each `gemma4:31b` arm twice: a
cache that holds nine of its prompts rarely still holds the one that comes back.

**Two rows are short.** This table was extracted before the tool read journald's rate-limit notices.
`nemotron3:33b-q8`'s CountBenchQA row counts 487 of its arm's 491 requests. The RefCOCO
`1000..8811` row stops where the journal did, at 7,093 of the arm's 7,811.

## The A/B, 2026-10-02 and 03

One binary served every run: production's commit (`b43ee8e`) plus this change, built the way
production was, versioned `0.34.4-dynres-0-gb43ee8e-pcache`. Each run had the server to itself
with one model loaded, after the same unload. That build read a host variable,
`OLLAMA_LLAMA_SERVER_CACHE_RAM`, which ADR 0047's addendum has since withdrawn. The cache-on runs
left it unset, and the cache-off runs set it to `0`: the same `--cache-ram` flag that
`prompt_cache_ram` sets. `llama-server`'s own lines confirm each launch:
- **cache on:** no `--cache-ram` on the command line, and "prompt cache is enabled, size limit:
  8192 MiB";
- **cache off:** `--cache-ram 0`, and "prompt cache is disabled".

### Every image new, mostly repeated, and now and then

`prompt_cache_stats.py ab`, verbatim. RefCOCO rows 4000–4300 never repeat an image from one row
to the next; rows 0–300 repeat it about 60 % of the time; OCRBench repeats a document image now
and then, not always adjacently. extbench ran each arm as it runs a benchmark: think off, greedy,
`num_ctx` 8192, `SLEEP=0`.

| model | rows | requests | cache on, s per request | cache off, s per request | change | outputs that differ |
|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | refcoco 4000..4300 | 299 | 2.527 | 1.964 | -22.3 % | 0 |
| `qwen3.8:27b-q4_K_M` | refcoco 4000..4300 | 299 | 1.267 | 1.140 | -10.0 % | 0 |
| `nemotron3:33b-q8` | refcoco 4000..4300 | 299 | 0.302 | 0.299 | -0.9 % | 0 |
| `gemma4:31b-it-q4_K_M` | refcoco 0..300 | 299 | 1.712 | 1.380 | -19.4 % | 0 |
| `qwen3.8:27b-q4_K_M` | refcoco 0..300 | 299 | 1.199 | 1.145 | -4.5 % | 0 |
| `nemotron3:33b-q8` | refcoco 0..300 | 299 | 0.301 | 0.299 | -0.6 % | 0 |
| `qwen3.8:27b-q4_K_M` | ocrbench 0..1000 | 999 | 1.191 | 1.065 | -10.6 % | 0 |
| `nemotron3:33b-q8` | ocrbench 0..1000 | 999 | 0.334 | 0.283 | -15.3 % | 0 |

host: http://127.0.0.1:11434 · build: 0.34.4-dynres-0-gb43ee8e-pcache

extbench records each request to 0.1 s, so a difference under about 0.05 s per request is below this table's resolution; the journal times each cache update to the millisecond.

- **The cache changed no answer.** Every output of the 3,800 paired requests is the same string with
  the cache on and off.
- **Turning it off saved `gemma4:31b` a fifth of every request**: 22 % when every image is new, 19 %
  when most repeat. A repeated image reuses the slot either way, so what the cache costs there is
  the update for each new one.
- **`qwen3.8:27b-q4_K_M` saved 4.5–10.6 %.** `nemotron3:33b-q8`'s update, 30–45 ms, is under extbench's
  0.1 s resolution on RefCOCO; on OCRBench, where it restored 32 times, the cache still cost it 15 %.

### Contexts that take turns

`prompt_cache_probe.py compare`, verbatim. Two conversations take turns, growing to about
3,700–4,100 prompt tokens by the last of six turns each; or two images take turns, one question
each. Every request after the first in each context could resume a state the cache saved two
requests earlier: this is the case the cache is for.

| run | model | workload | contexts × turns | first visit: prefill, request | every return: prefill, request | prompt tokens, last turn |
|---|---|---|---|---|---|---|
| cache on | `gemma4:31b-it-q4_K_M` | conversations | 2 × 6 | 2.03 s, 6.48 s | 1.66 s, 3.21 s | 3660 |
| cache off | `gemma4:31b-it-q4_K_M` | conversations | 2 × 6 | 2.03 s, 6.07 s | 2.05 s, 2.26 s | 3660 |
| cache on | `qwen3.8:27b-q4_K_M` | conversations | 2 × 6 | 1.50 s, 4.82 s | 0.55 s, 1.30 s | 3907 |
| cache off | `qwen3.8:27b-q4_K_M` | conversations | 2 × 6 | 1.50 s, 4.70 s | 1.55 s, 1.90 s | 3907 |
| cache on | `nemotron3:33b-q8` | conversations | 2 × 6 | 0.44 s, 4.87 s | 0.14 s, 0.39 s | 4066 |
| cache off | `nemotron3:33b-q8` | conversations | 2 × 6 | 0.44 s, 4.89 s | 0.37 s, 0.51 s | 4066 |
| cache on | `gemma4:31b-it-q4_K_M` | images | 2 × 6 | 1.68 s, 6.17 s | 1.61 s, 2.53 s | 1097 |
| cache off | `gemma4:31b-it-q4_K_M` | images | 2 × 6 | 1.69 s, 5.90 s | 1.61 s, 1.84 s | 1097 |
| cache on | `qwen3.8:27b-q4_K_M` | images | 2 × 6 | 0.77 s, 3.99 s | 0.70 s, 1.34 s | 1082 |
| cache off | `qwen3.8:27b-q4_K_M` | images | 2 × 6 | 0.77 s, 3.97 s | 0.71 s, 1.18 s | 1082 |
| cache on | `nemotron3:33b-q8` | images | 2 × 6 | 0.23 s, 4.71 s | 0.11 s, 0.32 s | 291 |
| cache off | `nemotron3:33b-q8` | images | 2 × 6 | 0.23 s, 4.71 s | 0.11 s, 0.28 s | 291 |

host: http://127.0.0.1:11434 · build: 0.34.4-dynres-0-gb43ee8e-pcache

**Conversations: the cache pays back on the two small-state models, and costs `gemma4:31b`.**
- **`qwen3.8:27b-q4_K_M` returns to a conversation in 1.30 s with the cache and 1.90 s without;
  `nemotron3:33b-q8` in 0.39 s against 0.51 s.** A restore leaves about a third of the prefill to
  run: 0.55 s of 1.55 s, and 0.14 s of 0.37 s.
- **`gemma4:31b` returns in 3.21 s with it and 2.26 s without.** Its restore leaves most of the
  prefill (1.66 s of 2.05 s), and saving and restoring a conversation's state, about 1,080 MiB,
  costs more than the rest saves.

**Images: no restore shortened a prefill.** An image that comes back evaluates as long with the
cache as without (1.61 s either way for `gemma4:31b`), so the cache only adds its update: 27 % of
`gemma4:31b`'s request, 12 % of the others'.

### The cache's own record of the A/B

<details>
<summary><code>prompt_cache_stats.py render</code> over both A/B runs, verbatim</summary>

A run's window is logged to the second, so a neighbouring run's last request can land in it: those
are the one- and three-request rows.

| model | window | requests | cache updates | restores | slot reuse | state, median | MiB per token | update, median | update, p90 | compute, median |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | pc-off-images-gemma4-31b-it-q4_K_M | 12 | 0 | 0 | 0 | — | — | — | — | 1768 ms |
| `nemotron3:33b-q8` | pc-off-images-nemotron3-33b-q8 | 13 | 0 | 0 | 0 | — | — | — | — | 297 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-images-qwen3.8-27b-q4_K_M | 12 | 0 | 0 | 0 | — | — | — | — | 1293 ms |
| `nemotron3:33b-q8` | pc-off-ocrbench0-nemotron3-33b-q8 | 995 | 0 | 0 | 37 | — | — | — | — | 160 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-ocrbench0-nemotron3-33b-q8 | 1 | 0 | 0 | 0 | — | — | — | — | 1035 ms |
| `nemotron3:33b-q8` | pc-off-ocrbench0-qwen3.8-27b-q4_K_M | 1 | 0 | 0 | 0 | — | — | — | — | 237 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-ocrbench0-qwen3.8-27b-q4_K_M | 999 | 0 | 0 | 37 | — | — | — | — | 761 ms |
| `gemma4:31b-it-q4_K_M` | pc-off-refcoco0-gemma4-31b-it-q4_K_M | 300 | 0 | 0 | 177 | — | — | — | — | 946 ms |
| `nemotron3:33b-q8` | pc-off-refcoco0-nemotron3-33b-q8 | 297 | 0 | 0 | 174 | — | — | — | — | 246 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-refcoco0-qwen3.8-27b-q4_K_M | 300 | 0 | 0 | 177 | — | — | — | — | 1081 ms |
| `gemma4:31b-it-q4_K_M` | pc-off-refcoco4000-gemma4-31b-it-q4_K_M | 299 | 0 | 0 | 0 | — | — | — | — | 1901 ms |
| `nemotron3:33b-q8` | pc-off-refcoco4000-nemotron3-33b-q8 | 299 | 0 | 0 | 0 | — | — | — | — | 245 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-refcoco4000-nemotron3-33b-q8 | 1 | 0 | 0 | 0 | — | — | — | — | 1296 ms |
| `gemma4:31b-it-q4_K_M` | pc-off-refcoco4000-qwen3.8-27b-q4_K_M | 1 | 0 | 0 | 0 | — | — | — | — | 1914 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-refcoco4000-qwen3.8-27b-q4_K_M | 299 | 0 | 0 | 0 | — | — | — | — | 1083 ms |
| `gemma4:31b-it-q4_K_M` | pc-off-text-gemma4-31b-it-q4_K_M | 11 | 0 | 0 | 0 | — | — | — | — | 2190 ms |
| `nemotron3:33b-q8` | pc-off-text-nemotron3-33b-q8 | 10 | 0 | 0 | 0 | — | — | — | — | 469 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-text-nemotron3-33b-q8 | 1 | 0 | 0 | 0 | — | — | — | — | 1296 ms |
| `gemma4:31b-it-q4_K_M` | pc-off-text-qwen3.8-27b-q4_K_M | 1 | 0 | 0 | 0 | — | — | — | — | 1914 ms |
| `qwen3.8:27b-q4_K_M` | pc-off-text-qwen3.8-27b-q4_K_M | 11 | 0 | 0 | 0 | — | — | — | — | 1849 ms |
| `gemma4:31b-it-q4_K_M` | pc-on-images-gemma4-31b-it-q4_K_M | 12 | 12 | 10 | 0 | 887.1 MiB | 0.796 | 680 ms | 692 ms | 1739 ms |
| `nemotron3:33b-q8` | pc-on-images-nemotron3-33b-q8 | 9 | 9 | 7 | 0 | 49.6 MiB | 0.149 | 37 ms | 38 ms | 299 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-images-qwen3.8-27b-q4_K_M | 11 | 12 | 10 | 0 | 219.0 MiB | 0.197 | 160 ms | 163 ms | 1286 ms |
| `nemotron3:33b-q8` | pc-on-ocrbench0-nemotron3-33b-q8 | 850 | 813 | 32 | 37 | 49.6 MiB | 0.149 | 45 ms | 55 ms | 164 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-ocrbench0-nemotron3-33b-q8 | 1 | 0 | 0 | 0 | — | — | — | — | 1036 ms |
| `nemotron3:33b-q8` | pc-on-ocrbench0-qwen3.8-27b-q4_K_M | 3 | 3 | 3 | 0 | 49.4 MiB | 0.165 | 37 ms | 43 ms | 239 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-ocrbench0-qwen3.8-27b-q4_K_M | 999 | 963 | 25 | 37 | 218.7 MiB | 0.198 | 131 ms | 162 ms | 762 ms |
| `gemma4:31b-it-q4_K_M` | pc-on-refcoco0-gemma4-31b-it-q4_K_M | 300 | 123 | 0 | 177 | 892.3 MiB | 0.756 | 820 ms | 836 ms | 954 ms |
| `nemotron3:33b-q8` | pc-on-refcoco0-nemotron3-33b-q8 | 301 | 123 | 0 | 177 | 49.9 MiB | 0.13 | 30 ms | 31 ms | 246 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-refcoco0-qwen3.8-27b-q4_K_M | 300 | 123 | 0 | 177 | 222.2 MiB | 0.191 | 132 ms | 136 ms | 1083 ms |
| `gemma4:31b-it-q4_K_M` | pc-on-refcoco4000-gemma4-31b-it-q4_K_M | 299 | 300 | 0 | 0 | 892.2 MiB | 0.756 | 572 ms | 585 ms | 1897 ms |
| `nemotron3:33b-q8` | pc-on-refcoco4000-nemotron3-33b-q8 | 299 | 300 | 0 | 0 | 49.9 MiB | 0.129 | 30 ms | 31 ms | 246 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-refcoco4000-nemotron3-33b-q8 | 1 | 0 | 0 | 0 | — | — | — | — | 1295 ms |
| `gemma4:31b-it-q4_K_M` | pc-on-refcoco4000-qwen3.8-27b-q4_K_M | 1 | 0 | 0 | 0 | — | — | — | — | 1921 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-refcoco4000-qwen3.8-27b-q4_K_M | 299 | 300 | 0 | 0 | 221.7 MiB | 0.192 | 132 ms | 134 ms | 1081 ms |
| `gemma4:31b-it-q4_K_M` | pc-on-text-gemma4-31b-it-q4_K_M | 11 | 12 | 8 | 0 | 1079.5 MiB | 0.302 | 1214 ms | 1561 ms | 1720 ms |
| `nemotron3:33b-q8` | pc-on-text-nemotron3-33b-q8 | 11 | 12 | 10 | 0 | 71.0 MiB | 0.018 | 97 ms | 126 ms | 244 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-text-nemotron3-33b-q8 | 1 | 0 | 0 | 0 | — | — | — | — | 1297 ms |
| `gemma4:31b-it-q4_K_M` | pc-on-text-qwen3.8-27b-q4_K_M | 1 | 0 | 0 | 0 | — | — | — | — | 1917 ms |
| `qwen3.8:27b-q4_K_M` | pc-on-text-qwen3.8-27b-q4_K_M | 12 | 12 | 10 | 0 | 389.3 MiB | 0.102 | 385 ms | 485 ms | 858 ms |

| the whole log | cache updates | restores | mean | p10 | median | p90 | max | total |
|---|---|---|---|---|---|---|---|---|
| all | 3266 | 229 | 182 ms | 30 ms | 130 ms | 574 ms | 2128 ms | 0.17 h |

journald dropped lines in `nemotron3:33b-q8` pc-on-ocrbench0-nemotron3-33b-q8 (29138 lines): those rows' counts are lower bounds.

H100 SXM5 (sm_90), 0.34.4-dynres-0-gb43ee8e-pcache (production plus OLLAMA_LLAMA_SERVER_CACHE_RAM), llama.cpp b11081, -np 1: pc-on-* with the variable unset, pc-off-* with 0; extbench runs 2026-10-02, probe runs 2026-10-03

</details>

## Per model or per request

`prompt_cache_ram` is a runner option of the same kind as `num_ctx` and `kv_cache_type`
([ADR 0005](adr/0005-per-model-kv-cache-type.md)).
- **Precedence:** the request's option, then the model's `PARAMETER`, then llama.cpp's 8192
  MiB.
- **A different value relaunches the runner,** as a different `num_ctx` does, at the cost of a
  model load: 4–17 s for the models here. Clients that alternate values reload on every switch,
  so set it per model or per client.
- **The scheduler compares the value a launch would pass,** so an invalid value, which a launch
  ignores, does not relaunch a runner started without one.
- **An invalid value** (below `-1`) is ignored with a warning, so llama.cpp's default applies
  and a Modelfile typo cannot make a model unloadable.
- **An MLX runner ignores it,** and is never relaunched for it.

Checked end to end on the H100 on 2026-10-03, one `nemotron3:33b-q8` request at a time. The
journal shows one `llama-server` launch per change and none for a repeat:
1. **no option:** launched with no `--cache-ram`, and "prompt cache is enabled, size limit: 8192
   MiB";
2. **`0`:** relaunched with `--cache-ram 0`, and "prompt cache is disabled";
3. **`0` again:** no launch;
4. **no option:** relaunched as in 1;
5. **`2048`:** relaunched with `--cache-ram 2048`, and "size limit: 2048 MiB";
6. **`-5`:** a warning, and relaunched with no `--cache-ram`, as in 1.

## What to do with it

- **A host that serves vision requests** (extraction, OCR, these benchmarks) loses up to a fifth
  of its request time to the cache and gets nothing back. Give its models
  `PARAMETER prompt_cache_ram 0` in their Modelfiles, or have its clients send
  `prompt_cache_ram: 0`.
- **A host that serves both** can give its image-heavy models `PARAMETER prompt_cache_ram 0` in a
  Modelfile, and leave its chat models on the default.
- **A host whose traffic is text conversations that take turns** gains from it on small-state
  models, a quarter to a third of each return, and loses 42 % on `gemma4:31b`. Leave it unset
  there unless it serves `gemma4:31b`.
- **The default stays llama.cpp's** in this change. A different default is a decision for the fork,
  with this document as its evidence.
- **Benchmarks:** the vision suite's throughput columns come from llama.cpp's own prompt and eval
  timings, which leave the update out, so they do not move. extbench's seconds per item include
  it, so arms measured with the cache on and off do not compare on time. The harness now turns
  it off ([below](#the-harness-turns-it-off)).
- **Logging:** at verbosity 4 every update logs every cached prompt. On this host journald
  rate-limited `ollama.service` (163,000 lines dropped, all in two fast `nemotron3` arms with the
  cache on), and the default-sized journal rotated out more than a day of logs within two days.
  Read a run's journal soon after it ends.

## The harness turns it off

Since 2026-10-03 the vision harness's one request path, `client.generate()`, sends
`prompt_cache_ram: 0` on every request (`vision-suite/client.py`, `prompt_cache_ram()`). Every
tool that measures through it runs with the cache off: `vision_suite.py`, `finetext_probe.py`,
`extbench.py`, `thinkcap.py`, `variants.py` and `endpoint_compare.py`.
- **`PROMPT_CACHE_RAM` overrides it:** a size in MiB, `-1`, or `server` to send none. One value
  serves every tool, because a value that changed between `vision_suite.py` and
  `finetext_probe.py` would relaunch the runner inside a campaign cell. A value the server would
  ignore, such as `-5`, stops the run before its first request.
- **Three tools send none,** because each sends exactly the payload it measures with:
  `measure.py`, `token_split.py`, and `prompt_cache_probe.py`, which takes `--prompt-cache-ram`.
- **It is recorded:** `req_prompt_cache_ram` in a score block, `prompt_cache_ram` in an extbench
  summary. `summarize_extbench.py` names it in the footer, and `--timing` warns when the arms
  sent different values. A file from before 2026-10-03 sent none, so it counts as the server's
  default: the cache on.
- **A build without the option** (before #440) drops it with a warning in its log and keeps the
  cache on. The record says what was asked, and `server_version` says which build answered.

Checked on the H100 on 2026-10-03
([host profile](vision-suite/bench-runs/prompt-cache/host-profile_h100-sm90-2026-10-03.json)),
against `0.34.4-dynres-0-gb43ee8e-pcache3`: production's commit, `b43ee8e3`, plus the Go diff of
#440 and #441 (`git diff 4a4ce558e 02da2ebe7 -- api llm server`), built with production's flags
and served beside production on its own port:
1. **extbench,** RefCOCO rows 4000–4049 on `gemma4:31b-it-q4_K_M`, with the A/B's settings. By
   default llama-server launched with `--cache-ram 0` and logged "prompt cache is disabled".
   With `PROMPT_CACHE_RAM=server` it launched with no `--cache-ram`, and the cache updated 50
   times in 50 requests. All 50 answers matched both arms of the A/B, whose 300 rows from the
   same offset measured 2.527 and 1.964 s (−22.3 %):

| model | rows | requests | cache on, s per request | cache off, s per request | change | outputs that differ |
|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | refcoco 4000..4050 | 49 | 2.618 | 2.039 | -22.1 % | 0 |

host: http://127.0.0.1:11535 · build: 0.34.4-dynres-0-gb43ee8e-pcache3

extbench records each request to 0.1 s, so a difference under about 0.05 s per request is below this table's resolution; the journal times each cache update to the millisecond.

2. **One campaign cell,** `vision_suite.py` then `finetext_probe.py` (think off, `/api/chat`):
   two launches, both with `--cache-ram 0`, the second for the finetext test's larger `num_ctx`.
   `finetext_probe.py` reused that runner, no request updated the cache, and all 28 blocks
   recorded `req_prompt_cache_ram: 0`.

## Reproducing

The data is committed under `vision-suite/bench-runs/prompt-cache/`, with a host profile for
each day's runs (SPEC H26): `host-profile_h100-sm90-2026-10-02.json` for the A/B and the probe,
`host-profile_h100-sm90-2026-10-03.json` for the harness check. Every table above renders from it
alone:

```bash
cd docs/maxusai
D=vision-suite/bench-runs/prompt-cache
M="gemma4-31b-it-q4_K_M qwen3.8-27b-q4_K_M nemotron3-33b-q8"
python3 tools/prompt_cache_stats.py render $D/stats_h100-2026-09-30.json
python3 tools/prompt_cache_stats.py ab $(for w in refcoco4000 refcoco0; do for m in $M; do echo $D/ext_pc-{on,off}-$w-${m}_refcoco.json; done; done) $(for m in qwen3.8-27b-q4_K_M nemotron3-33b-q8; do echo $D/ext_pc-{on,off}-ocrbench0-${m}_ocrbench.json; done)
python3 vision-suite/prompt_cache_probe.py compare $(for k in text images; do for m in $M; do echo $D/probe_pc-{on,off}-$k-$m.json; done; done)
python3 tools/prompt_cache_stats.py render $D/stats_ab-h100-2026-10-02.json
python3 tools/prompt_cache_stats.py ab $D/ext_pc3-{server,off}-refcoco4000-gemma4-31b-it-q4_K_M_refcoco.json
```

To measure a host again, run each arm twice on one build, once with the cache on and once with
it off. An extbench arm sends `prompt_cache_ram: 0` unless told otherwise, so its cache-on run
takes `PROMPT_CACHE_RAM=server`, which sends no option. A Modelfile `PARAMETER` cannot turn the
cache on for it, because the request's `0` wins. `prompt_cache_probe.py run` sends nothing
without `--prompt-cache-ram`, and takes `--prompt-cache-ram 0` for the cache-off run. Then read
the journal with `prompt_cache_stats.py parse`, one `--window` per run.
