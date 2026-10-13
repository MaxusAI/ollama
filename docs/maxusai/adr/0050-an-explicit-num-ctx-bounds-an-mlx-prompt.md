# ADR 0050: An explicit num_ctx bounds an MLX prompt as it bounds a GGUF one

- **Status:** proposed (2026-10-13). Code, unit tests and the live check on Metal
  (`0.40.2-dynres-1-g7032203`) on `fix/mlx-num-ctx-enforced`.
- **Date:** 2026-10-13
- **Deciders:** the maintainer
- **Upstream:** ollama/ollama#18125 (issue), ollama/ollama#18261 (closed unmerged; revised in
  `d944eba1`, re-filed as #18285). Upstream treats MLX ignoring a configured context as
  intentional, so this is fork-only.

## Context

On the MLX runner a `num_ctx` reached three places and none of them was the runner:

- admission priced the KV cache at it (ADR 0034), refusing an explicit rung that does not fit;
- the scheduler reloaded the runner when it changed (fork PR #279);
- `/api/ps` reported it, through the client's `reportedContextLength`.

The subprocess was started as `ollama runner --model <m> --port <p>` and served
`MaxContextLength()` — 262144 on qwen3.x — whatever was configured. `/api/chat` and
`/api/generate` turned truncation off for MLX, so a 116k-token prompt under `num_ctx 65536`
was prefilled in full (#18125: `prompt_eval_count` 116126). The KV cache grew past the rung
admission had priced, and on a 128 GB M5 Max that ended in a Metal watchdog kernel panic. The
operator saw a confirmation that the limit was in place.

What the llama.cpp path does with the same request, in this tree:

| step | GGUF (`llm/llama_server.go`, `server/prompt.go`) |
|---|---|
| server, `/api/chat` and chat-like `/api/generate` | `chatPromptFrom` drops the oldest non-system messages until the rendered prompt fits `num_ctx` (always keeps the last message); `truncate:false` turns this off |
| runner, both endpoints, raw prompts too | `completionPromptForRequest`: a prompt of `num_ctx` tokens or more keeps the first `num_keep` (+BOS) tokens and the tail, cut to `contextShiftPromptLimit` (about half the window, so generation has room) |
| refused, 400 | `shift:false` (or a model without context shift) — "the prompt is longer than the context length…" |
| not truncated on the Go side | `truncate:false`; a prompt carrying media (passed to llama-server whole) |
| automatic `num_ctx` | enforced the same way — it is llama-server's `n_ctx` |

## Decision

**1. An explicit `num_ctx` becomes the MLX runner's window.** "Explicit" is what
`usesAutomaticNumCtx` already means: a request `options.num_ctx`, a Modelfile `num_ctx`, or
`OLLAMA_CONTEXT_LENGTH`. The client passes it as `--ctx-size`; the runner serves
`min(num_ctx, model maximum)` (`servedContextLength`, the analogue of llama.cpp capping `n_ctx`
at the trained context) and reports that in `/v1/status`, which is what `/api/ps` shows. The
KV cache cannot grow past it: `Prepare` holds the prompt under it and caps `num_predict` at the
remainder, as it already did for the model maximum.

**2. An over-long prompt is truncated exactly as on GGUF, and refused where GGUF would not
truncate.** The server drops messages with the same `chatPromptFrom` loop; the runner cuts
tokens with the same `num_keep` + BOS rule and the same limit
(`llm.ContextShiftPromptLimit`, a wrapper over the upstream function, so the arithmetic has one
source). Where that is not possible the request fails with HTTP 400 naming both numbers —
`input length (N tokens) exceeds the context length (L tokens) set by num_ctx` — followed by
the reason: `truncate` off, `shift` off, or a prompt with images (MLX cannot cut through a
media span; GGUF does not cut one either). It is never served past the window. When the explicit
`num_ctx` is above the model's maximum, the window is that maximum and the message says so
instead — `exceeds the model's maximum context length (L tokens); num_ctx M is larger than the
model's maximum` — so a client does not read it as a rung a larger `num_ctx` would cure.

**3. The automatic VRAM-tier rung stays advisory.** No `--ctx-size`, no truncation; the runner
serves up to the model maximum and refuses beyond it, as upstream does. `/api/ps` keeps
reporting the rung. This matches upstream's revised branch (`d944eba1`).

**4. An explicit `num_ctx` arriving at a runner loaded on the automatic rung reloads it**, even
when the numbers are equal: that runner was started without the limit. The reverse does not
reload — an enforcing runner serves the automatic rung of the same size — so alternating
clients cannot churn it.

The cut and the refusal follow the runner, not the request (`promptTruncation` asks the
runner whether it enforces). So an automatic request that lands on a runner an explicit
`num_ctx` of the same size loaded is held to that window: its old turns are dropped, its
prompt is cut, or it is refused, exactly as under an explicit value. On a runner loaded on the
automatic rung the same request would be served up to the model maximum. This is accepted:

- That runner cannot serve past its window. Its KV cache and admission's price stop there, so
  following the request instead could only turn the cut into a refusal, or reload the runner,
  which is the churn this decision avoids.
- What the request gets is what llama-server gives every request, automatic or not.
- The two rungs must be equal for this to happen. A different rung reloads anyway.

## Options considered

- **Refuse instead of truncating** (upstream's MLX semantics since `a16f966`, and what
  #18261 does for an explicit value). Simpler and honest, but every GGUF client that relies on
  history being dropped would get a 400 on MLX for the same request. The fork's principle is
  parity with llama.cpp, so refusal is the fallback, not the rule.
- **Truncate only on the server** (drop messages, no runner-side cut). Leaves `/api/generate`
  raw prompts and a single over-long message unbounded — the runner would still need a hard
  window, and then it has to either cut or refuse; it now does what llama-server does.
- **Make the automatic rung hard too.** That is full parity — GGUF enforces its automatic
  `n_ctx` — and was rejected for now:
  - Every unpinned MLX client changes behaviour at once, including production's, while an
    explicit value is opt-in and is what #18125 asks for.
  - The automatic rung is not a stable number on MLX: admission clamps it to what fits VRAM,
    down to a 2048 floor (ADR 0034). Enforced, a busy GPU would silently truncate a
    conversation to 2048 tokens — a cut nobody asked for and cannot see.
  - Upstream's direction is to serve the trained context on MLX. Keeping the automatic path
    identical to upstream keeps folds cheap.

  The cost is that an automatic rung still does not bound the KV cache on MLX. If that proves
  to matter, the next step is to enforce only an *unclamped* automatic rung, measured first.

## Consequences

| | before | after |
|---|---|---|
| MLX, explicit `num_ctx`, prompt within it | served | served |
| MLX, explicit, `/api/chat` over it | prefilled to the model maximum | oldest messages dropped, then token-cut as GGUF |
| MLX, explicit, `/api/generate` (raw or not) over it | prefilled to the model maximum | token-cut as GGUF |
| MLX, explicit, over it with `truncate:false`, `shift:false` or images | prefilled to the model maximum | 400 naming the prompt length and the limit |
| MLX, explicit, `/api/ps` | the rung (not enforced) | the rung, enforced (capped at the model maximum) |
| MLX, automatic | served to the model maximum, 400 past it | unchanged |
| MLX, automatic, on a runner an explicit `num_ctx` of the same size loaded | served to the model maximum, 400 past it | held to that window, as explicit (decision 4) |
| GGUF | — | unchanged |

- Admission's price for an explicit rung (ADR 0034) is now a bound, not an assumption. The cache
  still grows in 256-token blocks, so the last block may be allocated up to 255 tokens past the
  window; speculative drafting appends its draft beyond the prompt as before.
- `/v1/score`'s `max_tokens` is bounded by the enforced window (it reads the same field).
- "Fits" means the prompt plus the generation. `Prepare` caps `num_predict` at the window
  minus the prompt, so a prompt that fits but leaves less room than `num_predict` asks for
  ends with `done_reason: length` at the window. That holds for pass two of the ADR 0004
  two-pass request as well, which gets what pass one left.
- The vision suite's client (`docs/maxusai/vision-suite/client.py`, `context_error`) reads
  the MLX refusal as a `num_ctx too small` error, as it reads llama-server's
  `exceed_context_size` body, so an MLX cell whose prompt is over the rung climbs the ladder.
  The model-maximum refusal is not matched: no bigger rung cures it.
- Remaining differences from GGUF:
  - Generation that reaches the window stops with `done_reason: length` on MLX, where
    llama-server context-shifts and continues.
  - The two tokenizers count the same text differently, so the cut lands at a different word.
  - **The message dropping charges an MLX image nothing.** `imageTokenCosts` sizes an image
    from the model family or a projector, and an MLX config has neither (`model_family ""`,
    no projector layer). On GGUF a 1920x1080 image costs qwen35moe 2,042 tokens and gemma4
    1,102. So an image chat over an explicit `num_ctx` that GGUF answers with 200, its old
    turns dropped, gets a 400 on MLX: the server keeps the turns because it counts only their
    text, and the runner refuses the prompt because it carries images. Nothing is served past
    the window and nothing is cut silently. A text-only chat is not affected.
  - The ADR 0004 continuation check (`routes.go`, before pass two) counts the text of the
    continuation only. That is the same on both runners, so it is not an MLX difference; with
    images, a pass two that the check lets through can still be refused by either runner,
    after pass one has streamed.

  GGUF's image function cannot stand in for MLX. The two runners size images differently:
  qwen3.5 on llama.cpp holds an image between 1,024 and 4,096 tokens, while the MLX qwen3.5
  path keeps the reference pixel bounds, about 64 to 16,384 tokens. A 448x448 image costs
  1,026 tokens on GGUF and 198 on MLX, and a 4096x4096 image costs 4,098 against 16,386.
  A correct count has to come from the MLX runner, which owns each arch's preprocessing:
  - an endpoint that runs the model's `PrepareMediaWithBudget` on each image, with the
    request's image budget, and returns the length of its expansion;
  - the server calling it once per request, before `chatPromptFrom` and
    `truncateNativeChatMessages`, in place of `imageTokenCosts` on MLX;
  - tests that the count equals the expansion `Prepare` builds, per MLX vision arch.

  That decodes and resizes every image twice per request, which needs measuring on Metal
  first. It is left for a follow-up. The suite is not affected: its prompts are a single
  turn, so there is no old turn to drop.
- Not covered: MLX embeddings still bound input by the model maximum, not `num_ctx`.

## Verification

Unit tests (no GPU): `mlxrunner/numctx_test.go` (command line, served window, the cut against
`llm.ContextShiftPromptLimit`, every refusal, the wire fields, what `Ping` reports) and
`server/routes_mlx_numctx_test.go` (the truncation decision, `/api/chat` dropping messages only
under an explicit `num_ctx`, `/api/generate` asking the runner to cut, the reload rule). They
were written first and failed against stubs carrying the old behaviour, except two that pin
behaviour which does not change (a prompt that fits; what `Ping` reports).

Added after review, pinning behaviour this ADR states rather than changing it: an automatic
chat on an enforcing runner has its old turn dropped (decision 4), and
`TestImageTokenCostsMLXChargesNothing` holds the image-cost difference above, so a fix flips
it on purpose. `docs/maxusai/vision-suite/test_client.py` `TestContextError` covers the suite
reading the MLX refusal; three of its seven cases failed before `context_error` learned the
MLX wording.

**Live check** (2026-10-13, Apple M5 Max 128 GB, build `0.40.2-dynres-1-g7032203` on :11437
against an APFS clone of production's store, powermode 2). qwen3.6:35b-a3b in the same build on
both runners: `nvfp4` on MLX, `q4_K_M` on llama.cpp. Prompts of ~31k tokens, and ~116k for
#18125's case. Each row is one request; `/api/ps` is the context length it reported after it.

| case | MLX | GGUF |
|---|---|---|
| no `num_ctx`, `/api/generate` | 200, 31014 prompt tokens, `/api/ps` 262144 | 200, 31014, 262144 |
| explicit 262144 (equal to the automatic rung) | 200, 31014; one runner start | 200, 31014; one runner start |
| no `num_ctx` again | 200, 31014; no start | 200, 31014; no start |
| explicit 4096, `/api/generate` | 200, **2050**, `length`; `/api/ps` 4096 | 200, **2050**, `length`; 4096 |
| explicit 4096, `/api/chat` whose old turn alone is over the window | 200, 38: the old turn dropped | 200, 38 |
| explicit 4096, `truncate:false` | 400 | 400 |
| explicit 4096, `shift:false` | 400 | 400 |
| explicit 4096, an image plus ~31k tokens | 400 | 400 |
| explicit 65536, ~116k-token `/api/generate` (#18125) | 200, **32770**, `length`, no crash | 200, 32770, `stop` |

The status codes and cuts agree on every case. The error wording differs: llama.cpp passes
llama-server's own message through, while MLX names both token counts. The 116k case ends `length`
on MLX and `stop` on llama.cpp, the context shift noted above.

**No regression on prompts that fit.** The think-off vision suite (27 tests and the finetext probe, 28
answers; `num_ctx` 16384, `num_predict` 2200, greedy) was compared answer by answer against the release build's cells on
the same store:
- qwen3.6:35b-a3b-nvfp4 (MLX): 28/28 answers byte-identical.
- gemma4:31b-it-q4_K_M (llama.cpp, `NUM_BATCH=2048`): 28/28.
- gemma4:31b-nvfp4 (MLX, drafted): 25/28.
- `prompt_eval_count`: equal on every test of all three models.

The three gemma4 answers that differed moved a box coordinate by 1–3 units. That was not this
change: re-run the same day, the release build gave the same new bytes, and this build repeated
them twice. Between 2026-10-10 and 03:27 on 2026-10-13, four builds had agreed on the old bytes, with the
same weights and prompts.
The day-to-day shift on a drafted MLX model is still unexplained.
