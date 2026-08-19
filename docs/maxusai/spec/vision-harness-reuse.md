# SPEC: vision harness reuse

MaxusAI-fork specification. Status: **implemented** — `run_engine_compare.sh`
carries `REPEATS` / `TAG_PREFIX` / `ONLY_TESTS`, and `summarize_reps.py`
imports its helpers from `summarize_engine_compare.py`. Written 2026-08-17.

Normative rules for adding to `docs/maxusai/vision-suite/`. The decision and its
evidence are [ADR 0028](../adr/0028-one-runner-one-set-of-helpers.md); report
shapes are [ADR 0012](../adr/0012-benchmark-report-templates.md).

**Scope.** These bind anything that **produces a measurement or renders a
result**. Throwaway inspection in a session scratchpad is exempt — nobody
publishes it.

## 1. Running

**H1 — Iterating models is `run_engine_compare.sh`'s job, and only its.** No
other script may loop over models or think modes. It is the only thing that
climbs the `num_ctx` ladder per cell, derives `num_predict` for think-on as
`num_ctx - CTX_PROMPT_RESERVE`, restarts the server per **cell**, and stamps the
power mode.

**H2 — An arm is a set of environment knobs, not a file.** Repeats, subsets,
tag namespaces and sampling overrides all run through H1's entry point:

```sh
TEMPERATURE=0.01 REPEATS=3 TAG_PREFIX=lt \
  ONLY_TESTS=bbox_contract,bbox_contract_anchored \
  MODELS="gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_K_M" \
  RESTART_CMD='sh docs/maxusai/vision-suite/serve-apple-mlx.sh' THINK_MODES='false on' \
  ./run_engine_compare.sh http://127.0.0.1:11436
```

**H3 — A missing capability is a patch to the runner.** Adding a knob is one
review that every future arm inherits. Forking the loop is a private copy that
inherits nothing and silently loses the ladder.

**H4 — Any new knob MUST be inert by default.** With it unset, tags, budgets and
behaviour must be byte-identical to before, so existing campaigns and every
summarizer keep working. Verify with `sh -x` on both paths before committing.

**H4a — The CONTEXT ladder always runs for think-on. The rung is a result, not
plumbing.** `CTX_MAX` MUST leave at least one rung above the think-on start rung.
`run_engine_compare.sh` refuses (exit 2) otherwise; `ALLOW_NO_LADDER=1` is the
deliberate opt-out and obliges the write-up to say a fixed window was used.

This is the **context ladder** — `CTX_LADDER`, the `num_ctx` rungs — and not the
**token ladder** of `expectations.toml`, which is image cost across five fixed
geometries. The two are unrelated axes and the word is overloaded; see
README.md §Terminology. Nothing in H4a touches the token ladder.

The window a model needs to *finish* a thinking response is a throughput fact
about that model — KV size drives decode speed, which is why two cells measured
at different rungs are not comparable on tok/s or req/h, and why the scores
carry `num_ctx` at all. Pinning the ceiling to the start rung does not make an
arm cheaper; it makes that number unobtainable, because the cell caps and the
required window is never discovered.

**The start rung is not a safe default; for some models it is the boundary.**
`num_predict` is derived as `num_ctx - CTX_PROMPT_RESERVE`, so the 16384 start
rung yields exactly 8192 generated tokens. Measured 2026-08-18,
`nemotron3:33b-q4_K_M` think-on on `multi_3img` needs **8385 / 10226 / 4127**
tokens to terminate — straddling that 8192. At the start rung it caps in about
half its runs, returning `done_reason=length` with zero characters of answer and
24–27k characters of unclosed thinking; given a rung that derives a real budget
it terminates 3/3 and answers every question correctly. The context itself was
never short — `prompt_eval_count` 6203, and a 131072 window sat at 12% occupancy.
A fixed start rung would have published this model as broken.

**req/h is computed only from a cell that terminated.** `done_reason=stop`, at
the rung the cell converged at. A capped cell's `eval_count` IS the cap, so any
tok/s or req/h derived from it measures `CTX_PROMPT_RESERVE` and the rung, not
the model — and it moves when the ladder escalates, which is what makes it look
like a real number. `was_capped()` marks these; summarizers render `capped`
rather than a rate, and a write-up must not quote one.

The refusal is deliberate in place of a warning: downstream, a capped cell and
a cell that genuinely converged at the start rung both write a `num_ctx` into
the scores and every summarizer renders them identically. A warning scrolls
past in a run that takes hours; a wrong throughput number gets published.

> Guard added after exactly this: a fixed `CTX_MAX=16384` was passed to a
> think-on arm at `temperature 0` — the sampling regime where non-termination is
> the *expected* failure — so escalation was impossible and every capped cell
> would have been recorded as a dead cell at a window nobody chose to measure.
> The stated reason was comparability with the 2026-08-17 low-temperature arm,
> which had no ladder **because it was a hand-written loop** (H1, ADR 0028).
> Matching a bespoke arm's fixed window reproduces its defect inside the
> sanctioned runner, which is the one place that defect was supposed to be
> impossible. Comparability against an arm that could not measure the rung is
> not a reason to also not measure it.

## 2. Reporting

**H5 — Shared helpers are imported, never redefined.** `engine_for`,
`was_capped`, `ctx_for`, `tag_for`, `resolve_tag`, `load` and `fmt_bool` live in
`summarize_engine_compare.py`. A summarizer needing any of them imports it.

> Not hypothetical: the first draft of `summarize_reps.py` redefined the
> capped test as `eval_count == num_predict` where `was_capped` uses `>=`, so it
> would have counted a cell that overran its cap as a scored result.

**H6 — Tag strings are produced and inverted by `tag_for`.** Tags mangle both
`:` and `.` to `_`, so they cannot be un-mangled by splitting. Never parse a tag
by hand.

**H7 — Tables are emitted by a generator and pasted verbatim**, including into
documents, chat replies and PR descriptions. Reformatting is what dropped the
`num_ctx` column and let a mid-ladder read publish `nemotron3:33b-bf16` think-on
as scene IoU 0.000 when the settled value was 0.872 at 32768. A markdown table
pasted without a code fence is both verbatim and rendered.

## 3. Before writing anything

**H8 — Check the inventory first.** `vision-suite/README.md` §Files lists every
script and what it does. Read it before adding a script or a helper. Three
separate incidents in one week — six duplicate runners, four duplicate helpers,
one hand-typed table — all had the same cause: writing something that already
existed.

**H9 — There is ONE request path, and it is `client.generate()`.** No probe,
runner or one-off builds an ollama payload of its own. This is H5 applied to the
hardest-to-get-right part of the harness: a scorer that drifts produces a visibly
wrong number, while a request that drifts produces a plausible one measured under
conditions nobody recorded.

It is not hypothetical. Five files had grown their own request code, and by the
time they were consolidated they had already diverged on: the endpoint default,
whether `thinking` was normalised out of the chat envelope, whether the response
or the reasoning was persisted at all, and whether a context-overflow 400 was
translated into an actionable message or surfaced as a bare `HTTP Error 400`.
`finetext_probe.py` had all four defects and nobody noticed, because its scores
looked ordinary.

A probe needing behaviour `client.generate()` lacks MUST grow that function —
with an explicit, named knob, defaulting to the existing behaviour (H4). The
knobs that exist are `endpoint_override`, `apply_sampling`, `use_env_opts`,
`send_think` and `num_ctx=False`, and each exists because one caller's published
numbers depend on a payload detail: a calibrated probe must be able to send
exactly what it was calibrated with. **Consolidation must not normalise a
calibrated payload** — collapsing `send_think` into a single boolean silently
turned an experimental think-on arm into "whatever the server defaults to".

Payload behaviour MUST be covered by `test_client.py`. Both defects above shipped
green through the scorer and summarizer suites, which assert nothing about what
goes on the wire.

## 4. Conformance

| requirement | enforced by |
|---|---|
| H1, H2 | `run_engine_compare.sh` is the only script in `vision-suite/` that iterates `$MODELS`; a second one is the defect |
| H3, H4 | `REPEATS` / `TAG_PREFIX` / `ONLY_TESTS` are inert when unset — verified with `sh -x` on both paths |
| H4a | `run_engine_compare.sh` exits 2 when `CTX_MAX` leaves no CONTEXT-ladder rung above the think-on start; think-off is unaffected and `ALLOW_NO_LADDER=1` overrides |
| H5, H6 | `summarize_reps.py` imports `ctx_for`, `engine_for`, `load`, `tag_for`, `was_capped` and inverts `tag_for` for display |
| H7 | ADR 0012 rules 1 and 8 |
| H9 | `client.py` is the only module that builds a payload; `test_client.py` (20 tests) asserts the wire format, including the tri-state `send_think` and the `num_ctx=False` sentinel that a naive `== False` would have collapsed |
| H8 | **Nothing enforces this.** It is a reading habit, and it is the one that would have prevented all three incidents |
