# SPEC: vision harness reuse

MaxusAI-fork specification. Status: **implemented** — `run_engine_compare.sh`
carries `REPEATS` / `TAG_PREFIX` / `ONLY_TESTS`, and `summarize_lowtemp.py`
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
  RESTART_CMD='sh docs/maxusai/vision-suite/serve-mlx.sh' THINK_MODES='false on' \
  ./run_engine_compare.sh http://127.0.0.1:11436
```

**H3 — A missing capability is a patch to the runner.** Adding a knob is one
review that every future arm inherits. Forking the loop is a private copy that
inherits nothing and silently loses the ladder.

**H4 — Any new knob MUST be inert by default.** With it unset, tags, budgets and
behaviour must be byte-identical to before, so existing campaigns and every
summarizer keep working. Verify with `sh -x` on both paths before committing.

## 2. Reporting

**H5 — Shared helpers are imported, never redefined.** `engine_for`,
`was_capped`, `ctx_for`, `tag_for`, `resolve_tag`, `load` and `fmt_bool` live in
`summarize_engine_compare.py`. A summarizer needing any of them imports it.

> Not hypothetical: the first draft of `summarize_lowtemp.py` redefined the
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

## 4. Conformance

| requirement | enforced by |
|---|---|
| H1, H2 | `run_engine_compare.sh` is the only script in `vision-suite/` that iterates `$MODELS`; a second one is the defect |
| H3, H4 | `REPEATS` / `TAG_PREFIX` / `ONLY_TESTS` are inert when unset — verified with `sh -x` on both paths |
| H5, H6 | `summarize_lowtemp.py` imports `ctx_for`, `engine_for`, `load`, `tag_for`, `was_capped` and inverts `tag_for` for display |
| H7 | ADR 0012 rules 1 and 8 |
| H8 | **Nothing enforces this.** It is a reading habit, and it is the one that would have prevented all three incidents |
