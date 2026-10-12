# ADR 0050: An explicit num_ctx bounds an MLX prompt as it bounds a GGUF one

- **Status:** proposed (2026-10-13). Code and unit tests on `fix/mlx-num-ctx-enforced`;
  **the live check on a GPU is pending**.
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
media span; GGUF does not cut one either). It is never served past the window.

**3. The automatic VRAM-tier rung stays advisory.** No `--ctx-size`, no truncation; the runner
serves up to the model maximum and refuses beyond it, as upstream does. `/api/ps` keeps
reporting the rung. This matches upstream's revised branch (`d944eba1`).

**4. An explicit `num_ctx` arriving at a runner loaded on the automatic rung reloads it**, even
when the numbers are equal: that runner was started without the limit. The reverse does not
reload — an enforcing runner serves the automatic rung of the same size — so alternating
clients cannot churn it.

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
| GGUF | — | unchanged |

- Admission's price for an explicit rung (ADR 0034) is now a bound, not an assumption. The cache
  still grows in 256-token blocks, so the last block may be allocated up to 255 tokens past the
  window; speculative drafting appends its draft beyond the prompt as before.
- `/v1/score`'s `max_tokens` is bounded by the enforced window (it reads the same field).
- Remaining differences from GGUF: generation that reaches the window stops with
  `done_reason: length` on MLX, where llama-server context-shifts and continues; the two
  tokenizers count the same text differently, so the cut lands at a different word.
- Not covered: MLX embeddings still bound input by the model maximum, not `num_ctx`.

## Verification

Unit tests (no GPU): `mlxrunner/numctx_test.go` (command line, served window, the cut against
`llm.ContextShiftPromptLimit`, every refusal, the wire fields, what `Ping` reports) and
`server/routes_mlx_numctx_test.go` (the truncation decision, `/api/chat` dropping messages only
under an explicit `num_ctx`, `/api/generate` asking the runner to cut, the reload rule). They
were written first and failed against stubs carrying the old behaviour, except two that pin
behaviour which does not change (a prompt that fits; what `Ping` reports).

**Pending:** the live check — `/v1/status` and `/api/ps` agreeing on the explicit window, the
cut's `prompt_eval_count` against the GGUF build of the same model, the refusals, and the
automatic rung unchanged.
