# Extraction blueprint: the ollama calling layer as a consumable package

**Date:** 2026-08-29. **Inputs:** four independent max-effort reviews (wire
contract both sides of the API; ladder + lifecycle; capture + provenance;
coupling audit), run against main @ `95715dad`; blocker claims re-verified
by hand against the code before this synthesis. **Scope decision:** the
consumable surface is the **Python layer over the HTTP API**. The Go side is
MaxusAI/ollama-internal; it appears here only as the wire contract the
client observes, with fork-vs-stock deltas called out for anyone pointing
the package at stock ollama.

**The thesis survived review.** The calling discipline in
`docs/maxusai/vision-suite/` — one payload builder, provenance stamped
inside it, `done_reason`-first capped detection, evict-and-verify lifecycle,
window escalation with a hard ceiling — is genuinely extractable, and three
modules are liftable **today, unchanged**: `salvage.py`, `sampling.py`,
`summarize_engine_compare.py` (stdlib-only, no corpus, no fork assumption).
What blocks a clean package is not the mechanisms but their *enforcement*:
the best mechanism in each area has 1–3 unsanctioned rivals, and several
load-bearing facts live only in stdout or in filenames.

## 1. Proposed package

Working name `ollamadrive` (final name open). Python ≥3.11, stdlib-only
core (the suite's own constraint, worth keeping). Layers, innermost first:

| layer | contents | source | state |
|---|---|---|---|
| `client` | payload builder, transport + retry, provenance stamps, persist | `client.py` | shim (fixes below) |
| `verdict` | `was_capped`, `done_reason` taxonomy, `ctx_for`, tag helpers | `summarize_engine_compare.py` | lift |
| `capture` | typed, versioned per-request record; capture/score split | new (schema exists de facto) | build |
| `ladder` | the window-escalation engine as a library | port of `run_engine_compare.sh` | port |
| `lifecycle` | evict/unload + restart-hook protocol + contention gate | `client.py` + shell | shim + port |
| `sampling` | policy table + provenance | `sampling.py` | lift |
| `salvage` | JSON recovery with named methods | `salvage.py` | lift |
| `forkext` | fork-only options behind an explicit opt-in + capability check | scattered env reads | build |

**Stays behind in vision-suite:** every scorer, the corpus (`visimgs/`,
ground truths, prompts), the summarizer *tables*, preflight's assertions.
They become the package's first consumer. `preflight/probes.py`'s duplicate
request path migrates onto `client` (it contributes back the one mechanism
`client.py` lacks: `queue_wait = wall − total_duration`).

**Design rules carried from the review, non-negotiable in the API:**

1. **One reachable implementation per invariant.** `was_capped` today has
   four implementations (shell heredoc without `done_reason`; a stale
   arithmetic copy in `summarize_matrix.py`; none at all in
   `summarize_reps.py` / `summarize_geometry.py`, which pool capped cells
   into published means). In the package, raw `eval_count`/`json_valid`
   access on a record goes through the verdict layer.
2. **No ambient environment in the core.** `client.generate()` reads 10 env
   vars at call time; `sampling.py` seven more. The package takes a config
   object; an `envconfig` adapter reproduces today's behaviour for the
   suite. The omission sentinels (`num_ctx=False`, empty images list omits
   the key, `fmt=None` omits `format`) and the tri-state `send_think` are
   load-bearing and keep their exact semantics.
3. **Identity and provenance live in the record, not the filename or
   stdout.** Blocks today carry no `model`, `endpoint`, `think`, `tag`,
   powermode, or cold-start mechanism; tags are documented as
   non-unmangleable yet two summarizers regex filenames. The capture record
   gets: `schema_version`, model, endpoint, think mode, tag, sampling +
   source, host, server build, `_retries` (captured today, dropped by every
   writer), cold-start mechanism, powermode (nullable), requested **and
   served** window, and the existing fingerprints (`prompt_sha`,
   `images_sha`, exact bytes).
4. **Verify the served state, don't assume it.** `/api/ps` already returns
   `context_length` and `digest`; the client records requested-vs-served
   window and the model digest after load. This is what turns the ladder
   from "correct if the cold restart worked" into "checked".
5. **Fork extensions are explicit.** `image_min_tokens`, `image_max_tokens`,
   `kv_cache_type` are fork wire options that **stock ollama silently
   drops** (logged server-side only). `forkext` sends them only when the
   caller opted in, and verifies the server build (version pattern per
   ADR 0032; optionally the served effect) before trusting them.

## 2. The wire contract a consumer must know (condensed)

Full map in the review appendix; the load-bearing subset:

- **Use `/api/chat`.** `/api/generate` has no native-chat response path:
  for models routed to the native template (`chatModeForModel`), thinking is
  never populated (the gemma4 "drops reasoning" finding is structural in the
  handler, arch-specific only via routing), and grammar `format` is applied
  from token zero instead of deferred past thinking. `/api/generate` can
  also 500 after a completed generation while building its `context` field.
- **Omitting `think` means think ON** for thinking-capable models — the
  server defaults it to true. "Send no field" is not a neutral baseline.
- **Never send `format: null` or `""` to chat** on this fork: the chat
  handler tests `req.Format != nil` (generate uses `formatConstrains`), so
  an explicit null triggers the full two-pass structured-outputs flow to
  apply a format that constrains nothing.
- **`done_reason` taxonomy:** `"stop"` | `"length"` | *absent*. Absent means
  connection-closed (or a pre-2026-08-20 file). Metrics are `omitempty`:
  zero values vanish. A degenerate generation (30 repeated tokens) returns
  HTTP 200 with `done:false` and **no metrics** — check `done`.
- **Fork metric semantics:** on `format` + think two-pass requests, this
  fork sums pass-one metrics into the final response (`eval_count` includes
  reasoning; can legitimately exceed `num_predict`); stock ollama does not.
  Same JSON shape, different meaning — undetectable from the response.
- **`prompt_eval_count` is cache-inclusive; `prompt_eval_duration` is not.**
  `prefill_tps` derived from their ratio inflates on any warm-cache prompt
  prefix. Published `prefill_tps` means are a lower-bound-free metric and
  need a warm/cold annotation (follow-up M3 below).
- **Window truth:** `num_ctx` is clamped to the model's trained context and
  floored (4; 2048 for vision) with no response-side indication; on the MLX
  runner all runner options bind at **first load** and later changes are
  ignored without a reload (`sched.go` exempts MLX from the options-reload
  check). Cold start per cell is what makes a ladder honest; verify it.
- **Context-overflow detection** (the actionable 400) fires only where
  llama-server's raw body passes through; ollama's own worded 400 and the
  MLX overflow text fall through to a bare error. The package's error
  taxonomy needs all three shapes.

## 3. Fix-first list (in-repo, before extraction)

**P0 — correctness of what we publish today**

1. Shell capped test → the sanctioned verdict: `run_engine_compare.sh`'s
   heredoc ignores `done_reason` (measured false-escalate already on disk in
   `cudafull1`; false-converge reachable whenever `num_ctx` binds first —
   that class currently ends a cell with **no** NOT-CONVERGED marker).
   Import `was_capped`; add it to `summarize_reps`, `summarize_geometry`,
   `summarize_matrix`, and preflight's `check_quality`.
2. Stamp identity + missing provenance into blocks (rule 3 above) and give
   `NOT CONVERGED` a machine-readable marker so ceiling cells stop
   re-climbing the whole ladder on every resume.
3. Fix the REFUSING gate's variable mismatch (it hardcodes 16384 instead of
   `CTX_START_THINKON`, so raising the start rung defeats the guard).
4. `implied_scale` / `iou_at_implied_scale` are dialect-blind — a perfect
   norm-1000 cell in our own `mlx0330nv` data reads `implied_scale 0.721`,
   `iou_at_implied_scale 0.078` against `iou_declared 0.956`. Fix the
   dialect handling or stop emitting the fields.
5. `finetext_probe.py`: unguarded `generate` aborts whole campaigns; no
   resume; `ft_` schema lacks fingerprints/durations. Bring to parity.
6. Timeout policy: `HTTP_TIMEOUT` (1800 s) neither scales with the rung nor
   is classified as deterministic — a top-rung GGUF cell cannot fit and then
   burns 3 retries. Derive the budget from `num_predict` and a floor tok/s;
   never retry a timeout.
7. Server-side (fork): chat handler adopts `formatConstrains` (the
   `format: null` two-pass trigger).

**P1 — robustness**

8. Restart verification in the driver/serve helper: capture the child PID,
   compare `/api/version` (+ `/api/ps` digest) before/after; a surviving old
   process currently passes the readiness poll and silently runs the
   campaign warm — which on MLX also disables the ladder (options bind at
   first load).
9. Eviction failure is invisible (`|| true`, exit 0 on STILL RESIDENT, early
   return when nothing unloaded). Propagate.
10. Per-arm flush in `vision_suite.py` (today one kill at arm 20/27 loses
    every completed arm's scores).
11. `server_version` cache: TTL/invalidate on any transport failure; record
    the `/api/ps` digest with it.
12. Migrate `preflight/probes.py` onto `client` (second payload builder;
    keep its `queue_wait` mechanism).
13. Ladder hygiene: validate/sort `CTX_LADDER`; `serve-apple-mlx.sh` should
    reject `OLLAMA_MAX_LOADED_MODELS=0`; `resp_*.json` holds raw text —
    rename or wrap.

**P2 — measurement follow-ups (each is an experiment or a doc, not a code
fix)**

- **M1:** re-run `endpoint_compare` with `fmt=None`. The 2026-08-19
  learnings entry read chat-vs-generate `eval_count` 1281 vs 1277 as
  "generated either way, returned by one"; mechanism review shows native
  `/api/generate` also skips grammar deferral, so the two cells plausibly
  measured *different workloads*. If the re-measurement refutes the entry,
  amend it per the log's REFUTED convention — measurement first, no silent
  rewrite.
- **M2:** audit whether `send_think=False` calibration probes (published
  token budgets) are affected by the server's think-on default for capable
  models.
- **M3:** annotate `prefill_tps` in ADR 0012 / the campaign template with
  the cache-inclusive caveat; decide whether to record a cache-hit signal.

## 4. What is already clean (preserve verbatim)

The full strengths inventory is in the four reviews; the short list that
shapes the API: the omission discipline and tri-state `send_think` in the
payload builder (each encodes a real incident); provenance stamped inside
the one request path so no probe can forget it; exact-bytes fingerprints;
`done_reason`-outranks-arithmetic with documented fallbacks; persist-before-
score; `arm_done` treating capped as unfinished; evict-then-verify via
`/api/ps`; mixed-window/mixed-provenance detection at render time; the
refusal gates that error instead of warning; `test_client.py` as the
executable contract; `emit_request.py`'s capture-not-rederive pattern; and
`OLLAMA_DEBUG_LOG_REQUESTS` server-side as the independent ground truth to
verify the client against.

## 5. Sequencing

1. Land P0 in vision-suite (the suite's own campaigns benefit immediately;
   every fix is testable with the existing offline fixtures).
2. Extract `salvage` / `sampling` / `verdict` unchanged; build `capture`
   with `schema_version: 1` and a converter for existing score files.
3. Shim `client` onto the config object; port `ladder`/`lifecycle` from
   shell with the P1 verifications built in; migrate preflight's probes.
4. vision-suite and preflight become consumers; their tests
   (`test_client.py`, `test_summarizers.py`, `test_verdicts.py`,
   `test_rescore.py`) come along as the package's contract suite.
5. Only then publish for other projects, with §2 as the consumer-facing
   contract doc and `forkext` clearly fenced.
