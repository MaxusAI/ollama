# SPEC: vision harness reuse

MaxusAI-fork specification. Status: **implemented** — `run_engine_compare.sh`
carries `REPEATS` / `TAG_PREFIX` / `ONLY_TESTS`, and `summarize_reps.py`
imports its helpers from `summarize_engine_compare.py`. Written 2026-08-17;
H15–H18 added 2026-09-19 from the 0.34 fold's kernel investigation
([ADR 0037](../adr/0037-keep-the-mlx-3912-kernel-fix.md)); H20 and H21 added
2026-09-20 from gating the M5 tensor path
([m5-neural-accelerators.md](../m5-neural-accelerators.md)); H22 added 2026-09-20
from the ROCm 10.0.0 throughput comparison on gfx1151
([rocm-10-throughput-2026-09-20.md](../rocm-10-throughput-2026-09-20.md)).

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

**H4b — Resume skips only FINISHED arms; a capped block always re-runs.**
`vision_suite.py`'s idempotent resume ("skip what is already scored") defines
scored via `arm_done`: present, no `error`, and not `was_capped` (H5 — the one
definition). This is not a detail of the resume feature; it is what makes
H4a's per-cell escalation work at all. The runner re-invokes the suite at each
higher rung against the same tag, so if presence alone counted as done, every
arm would be skipped and the ladder would no-op — silently, because the
capped blocks already in the file carry a plausible `num_ctx`.

Not hypothetical: resume landed 2026-08-20 00:38 (the idempotent-runs commit)
treating any error-free block as done, and the cudafull1 campaign ran that
night. Its think-on cells froze at (16384, 8192) — the runner escalated,
vision_suite answered "ALL ARMS ALREADY SCORED", and the next capped-check
passed because the stale 8192 eval_counts sat below the new rung's cap —
while g4full1, run hours earlier without resume, climbed to 131072. Fixed the
same day. The corollary for delta re-runs: re-running a campaign tag with the
fixed resume re-runs exactly the capped arms and nothing else, which is the
sanctioned way to finish an interrupted or frozen ladder without paying for
the finished cells again.

**The cap verdict is the server's `done_reason` where recorded; the token
arithmetic is the fallback.** Since 2026-08-20 every score block copies
`done_reason` from the response, and `was_capped` prefers it: `"length"` is
capped and `"stop"` is finished, whatever `eval_count` says. The arithmetic
misreads two real cases — a model that emits its final token exactly at the
cap (`eval == num_predict`, `done_reason "stop"`), and a thinking
continuation that overshoots the recorded cap (eval 8290 against 8192,
measured 2026-08-20 on qwen3.6 `bbox_contract_reasoning`). Blocks without the
field keep the `eval_count >= num_predict` fallback, so historical files
never change verdict by upgrading the summarizers.

The value inventory, from this repo's server code (the serving build is this
fork): the runner emits `stop` or `length` (`llm/llama_server.go`, from
llama.cpp's `StopType == "limit"`); `server/routes.go` synthesizes `length`
when thinking fills the window before a format-constrained continuation, and
`load`/`unload` for bare load and eviction requests, which are never scored;
`DoneReasonConnectionClosed` maps to `""` (`llm/server.go String()`) and the
API field is `omitempty`, so a dropped connection arrives with NO
`done_reason` at all — absence is not evidence of age, only of "no verdict",
which is exactly what the fallback handles.

## 2. Reporting

**H5 — Shared helpers are imported, never redefined.** `engine_for`,
`was_capped`, `ctx_for`, `tag_for`, `resolve_tag`, `load`, `save`, `fmt_bool`,
and the ladder decisions `capped_arms` / `ceiling_standing` /
`mark_not_converged` live in `summarize_engine_compare.py`. A summarizer —
or the driver, via its argv subcommands — needing any of them imports or
invokes it.

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

The same rule covers **request examples**: `emit_request.py` captures the
payload from `client.py`'s own construction (H9) for any model, think mode
and arm, and documentation pastes its output. Measured 2026-08-20: the
recommendations doc's first draft hand-assembled its request fragments, the
maintainer could not reconstruct a working request from them, and the
fragments were WRONG about the recommended shape — they showed
`box_2d`/`bbox_2d` dialect arrays where the trustable `pin_anc_named` arm
actually uses named coordinate fields. A hand-typed example asserts a wire
format nothing enforces; an emitted one cannot drift.

**H13 — Report footers derive provenance from the score files, and a MIXED
footer blocks publication.** T1 and T2 print host(s) and build(s) collected
from the H11 fields of every file they render. A file that loaded but carries
no `host`/`server_version` contributes an explicit "pre-H11 run (not
recorded)" entry — it must never vanish from the set, because a footer built
only from the rows that DO record provenance vouches for the rows that don't.
More than one distinct host, build, or recording state renders **⚠ MIXED —
rows/columns are not one campaign**, and ADR 0012 convention 10 makes that
non-publishable: re-run the odd columns under one campaign tag or split the
table.

Not hypothetical: the 2026-08-20 five-model CUDA head-to-head mixed pre-H11
`g4full1` gemma columns into `cudafull1`'s table and the footer showed one
clean host — with gemma4:26b-a4b think-on measured at `num_ctx` 131072 against
16384 everywhere else. Only the per-cell `(num_ctx)` brackets (ADR 0012 rule 6)
exposed it.

**Capped cells render `capped` in every template, and `capped` is a to-do, not
a result.** `was_capped` is the one test (H5) and T2 applies it per block via
`cap_or`. Before it did, qwen3.6:35b-a3b think-on multi published as
`❌ ❌ ❌ 0/5 (16384)` and nemotron3 fine-text as `0/0/0/0/0` — grounding-failure
renderings of cells that had merely hit `num_predict` 8192 and were never
escalated. The owed number is the final-rung score (ADR 0012 rule 8): a
campaign with `capped` cells below `CTX_MAX` has not finished running.

"Every template" means every CELL, not every table: T1 guarded only its
scene-derived throughput cells until later the same day, and its first
prefixed-campaign render printed the capped qwen3.6 think-on multi ceiling
cell as `❌ q1_right, q2_right, q4_bbox_hit` — the identical defect class,
found by rendering, fixed with the per-block `q()`/`multi_cell` guards.

**H14 — Reasoning-token counts come only from `token_split.py`'s gate-proven
split.** `eval_count` conflates thinking with answer. The split tokenizes the
persisted thinking/answer text with the server's own vocab and must prove
itself against `eval_count` — control residue (`eval - thinking - answer`)
non-negative and ≤ 64 across the sample — before `--write` stamps
`thinking_tokens` / `answer_tokens` / `control_tokens` into the score blocks.
Never estimate from characters: a 62-token response measured 21% control
tokens, so proportions lie. T1's `Think tok` column renders the stamped count
or `—`; a model that cannot pass the gate gets no number rather than a wrong
one. The gate can indict the METRIC, not just the tokenizer: for nemotron3
think-on it refused the server's own counts (`--server` mode, which has no
vocab to mismatch) at 54/54 cells because `eval_count` itself does not cover
the deferred-thinking generation — a live-reproduced server bug
(tasks/nemotron-thinkon-evalcount-undercount.md), found precisely because
the gate refuses to write a split that the identity
`eval = think + answer + control` cannot support.

A few outliers in a large clean sample indict the DATA, not the vocab — a
wrong tokenizer skews every cell. With ≥ 20 clean cells and ≤ 10% bad, the
split names and skips the irreconcilable cells instead of refusing
everything, and never writes a split for them. The measured cause was a
harness flaw, and its rule is worth stating on its own: **two producers must
never persist under one `(tag, probe)` name.** The suite's folded finetext
test and `finetext_probe.py` both wrote `(tag, "finetext")`, the probe
overwriting the suite's text — and think-on sampling is non-greedy (per the
model card, `sampling.py`), so the two generations differed: control −114
(gemma4:26b-a4b) and +444 (qwen3.8) on exactly the think-on finetext cells,
every other cell in [0, 29]. Fixed 2026-08-20: the probe persists as
`finetext_probe`. Later proven character-exact for the qwen3.8 cell: the
on-disk text matches the probe block's recorded 731 thinking chars (the
scores block recorded 1,563), and it reconciles against the probe's own
`eval_count` at control 3 — the split machinery was never wrong, only the
identity. The two generations also scored differently (recall_9px 1/4
against 2/4), which is the measured instance behind ADR 0012 conv. 4's
think-on non-reproducibility note.

**H15 — One capture of a quality cell is one draw, not a value.** A score or
size tier that differs between two builds is not attributed to anything until
its RATE is measured on each: at least ten reps per build, same binary, window
and power mode, reported as `k/N`. `n = 1` stays right for a campaign's
survey of quality ([ADR 0012](../adr/0012-benchmark-report-templates.md)); it
is not enough to conclude that a build moved a cell.

> Measured 2026-09-18: `gemma4:31b-nvfp4`'s 9 px tier read 3 on every 0.34.0
> capture of one finetext arm, and the move was published as a clean split by
> build. Ten reps put it at 2/10 reading 4, and the other arm already held a 4.
> At greedy, temperature 0.

**H16 — Both finetext arms count, and a tally that reads one is half the
evidence.** Every campaign cell carries two INDEPENDENT fine-text generations:
`vision_suite.py`'s folded arm (`scores_<tag>.json["finetext"]`) and
`finetext_probe.py` (`ft_<tag>.json`). Any count over captures reads both, and
reports disagreement between them as a result rather than resolving it.

The rendered table does NOT show both: `summarize_engine_compare.py` takes
`scores.get("finetext")` and falls back to `ft_<tag>.json` only when the suite
arm is absent, so where both exist the second draw is never rendered. H14's
note recorded the arms disagreeing at think-on; 2026-09-18 extends it to
think-off at temperature 0. On `0.34.0` one `31b-nvfp4` cell's two arms had
byte-identical provenance — same `prompt_sha`, `images_sha`, window,
`eval_count=263` and `answer_chars=351` — and differed by one character,
`RNK-0391-DW18` against `RMK-0391-DW18`, scoring 4 and 3. In the 2026-09-18
campaign 2 of 7 cells' arms disagree on a tier. A count of "twelve runs split
cleanly by build" was published from the `ft_` arm alone and was wrong.

**H18 — A cross-build accuracy claim needs a paired test on many items, and a
tier move needs the unquantized arm.** Two builds compared on a benchmark are
compared item by item on the SAME items with an exact McNemar test on the
discordant pairs — never by diffing two headline accuracies: at 200 items near 85 % the
standard error of the difference is about 3.4 points, so a real gap needs to be
roughly seven points before two headline numbers can show it. A unanimous split cannot reach
significance below six discordant pairs, so a 200-item slice routinely cannot
answer the question at all. Before a fine-text or recall tier is read as
evidence about a kernel, pin or quantization, the checkpoint with no
quantization in the path (`*-mlx-bf16`) is measured on the same binary and
window.

> Measured 2026-09-18/19: a 200-item OCRBench slice gave 4–0 to the fixed
> kernel; all 1000 items gave 9–7, p = 0.80. The slice was also
> unrepresentative (per-chunk accuracy 0.875 … 0.665), so its headline 87.5 %
> was not the checkpoint's 83.5 %.

## 3. Before writing anything


**H19 — A slice names its rows and its categories.** External sets are ordered
by task, so a `LIMIT`/`OFFSET` window is a stratum, not a sample: OCRBench rows
0–200 are four of its ten categories — regular, irregular, artistic and
handwriting recognition — and contain no VQA, key-information extraction,
digit strings or handwritten maths. A slice result names its rows and its
categories, and is never called by the benchmark's name alone or set beside a
published score for the whole set — H17's sibling problem, one level up: there a
tag did not identify the weights, here a benchmark's name does not identify the
questions. `summarize_extbench.py --categories` prints the split from the cached
rows.

**H20 — A check that could not measure says so; only one that measured says
no.** A guard exists to be believed, and one false alarm on a healthy host costs
more than the guard returns — the next red gets read as noise. So the three
states stay apart in the code rather than in the operator's head: PASS is a
measurement that matched, FAIL is a measurement that did not, and SKIP is *I
could not tell*, with the reason in its summary. A check that reads an artefact
names the artefact and the route it took to it, and refuses outright when that
route would answer about a different machine, binary or process than the one
under test. Where the question does not arise at all the skip says "does not
apply", which `release_matrix.effective()` reads as neutral rather than as
coverage owed. And a check nothing renders is a check nobody reads: a new check
joins a `release_matrix.GROUPS` column in the same change.

> Measured 2026-09-19/20, gating the M5 tensor path — ten defects in one guard,
> caught before any of them shipped, six of them by an adversarial review whose
> own headline diagnosis was itself wrong (it tested `ps -wwE` with a platform
> binary, the one class macOS hides). A hand-written copy of ggml's dummy
> `matmul2d` reported `has_tensor:false` on a host where the accelerators were
> demonstrably working (#338, now extracted from the pinned llama.cpp). An early
> `metal_tensor_payload` would have run `strings` on the *harness's* filesystem
> against a path printed from inside a container. `nax_probe` read
> `GGML_METAL_TENSOR_DISABLE` from the operator's shell instead of the server's.
> A native run without `--log-cmd` came back ERROR rather than skip. All four put
> a red mark on a healthy build; the third, read the other way, would also have
> stayed green on a server that really had lost the accelerators.

**H21 — A log window is enforced per line, or it is not enforced.** A
`--log-cmd` template that cannot substitute `{since}` — `cat <serve log>`, the
only form that works on the native macOS path — returns the WHOLE file, and the
`since` the caller passed is inert. Anything that reads such a log filters each
line on its own timestamp, and treats a line whose timestamp will not parse as
outside the window: cannot-confirm must never read as confirmed. Where a log is
shared by several servers, the line must also be tied to the one under test.

> Twice, in the same repository, from the same cause. 2026-08-30: a month-old
> engine-init line satisfied a five-second window and the MLX pin check returned
> PASS in a run that loaded no model at all. 2026-09-20: the tensor gate's
> payload half chose among **522** stale `starting llama-server` lines — zero of
> them in its actual window — and labelled the pick "launched by this run". The
> fix the first incident shipped was in `mlx_build`'s docstring, three functions
> away from the second, and was read before the second was written.

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

**H10 — Cold start and model residency are the harness's job, not the host's.**
Before a model is loaded, every OTHER resident model is evicted and the harness
**waits** for the memory to come back; then the incoming model is cold-started.
Decision and rationale in
[ADR 0031](../adr/0031-model-residency-is-managed-client-side-on-remote-hosts.md).

`RESTART_CMD` restarts the serving process and needs local control. Against a
host we do not own there is none, and the runner used to skip the step silently —
so remote campaigns measured warm loads while local ones measured cold, with
nothing in the scores to tell them apart. `keep_alive: 0` is the per-model
equivalent and is what `preflight/probes.py` has always used.

Two details are load-bearing:

- **Evict the OUTGOING model, not the incoming one.** The incoming model is
  usually not resident. `OLLAMA_MAX_LOADED_MODELS=1` is the server-side answer
  and is mandatory for a locally served sweep
  ([apple-silicon-build](apple-silicon-build.md)), but it is fixed when the
  server starts and is therefore unavailable exactly when it matters most.
- **Wait for it.** `keep_alive: 0` returns when the request is accepted, not when
  the weights are gone. Without the poll a sweep can hold two large models at
  once — and the resulting OOM lands on whichever model was next, reading as
  "that model is too big" rather than "the previous one was never freed".

`evict_all()` is the same mechanism with a different intent — it hands the host
back rather than making room, for use after a campaign or before yielding a
shared machine. Both return what they could NOT evict rather than raising: a
model held by another client is not this harness's to kill.

**Transport failures MUST be retried, and deterministic rejections MUST NOT be.**
5s / 15s / 30s. A dropped connection or a 502/503/504 says something about the
moment; a context-overflow 400 says the request itself is wrong and will fail
identically forever. Retries MUST be announced and recorded on the cell
(`_retries`) — a cell whose server restarted mid-generation is not
timing-comparable with a clean one.

An eviction is **not** a process restart: server caches and other models survive.
A `load_duration` measured after eviction MUST NOT be quoted against one measured
after `RESTART_CMD`.

**H11 — Every cell records WHERE it ran and WHICH build served it.** `host` and
`server_version` are written into every score block, unconditionally, from
`client.generate()`. Not optional and not env-gated: a score whose host and build
are unknown is not comparable with anything.

This is [ADR 0012](../adr/0012-benchmark-report-templates.md) rule 6 applied to
the two settings that matter most. Rule 6 puts `num_ctx` in the cell because a
number whose meaning depends on an unrecorded setting is not a measurement. Host
and build are worse: host changes throughput by ~4× (Apple Metal ~21 tok/s vs
CUDA ~93 on the same model and cell), and **build changes behaviour** — gemma4
returns no reasoning at all on `/api/generate` for one build and returns it
normally on another.

One build can carry two `server_version` strings when the fork re-tags an
already-built source: main @ `51718870` shipped first as
`0.32.14-dynres-112-g5171887` and was re-stamped as
`0.33.0-dynres-0-g5171887` (same Go tree, same payload `9d77fa172`).
Equivalences are recorded in
[ADR 0032](../adr/0032-fork-version-identity-tags-each-upstream-fold.md) —
and only pairs recorded there may be treated as one build by H13's MIXED
rule; any unlisted string pair stays two builds.

Before this existed, cross-host coverage could only be reconstructed from
tag-name prefixes and the memory of whoever launched the run. That is not
evidence: nothing in the data prevented an Apple cell being pooled with a CUDA
one, and nothing would have revealed it afterwards. A coverage audit run on
2026-08-20 found the geometry corpus was two hosts quoted as one number.

Historical scores predate the field. A cell with no `host` is pre-2026-08-20 and
its provenance is whatever its campaign document says — which is why the campaign
docs state the host in their header.

**H12 — Every cell records WHAT WORKLOAD it ran: `prompt_sha`, `images_sha`, and
for composed arms `prompt_parts`.** Exact bytes, no normalisation — whitespace is
part of the prompt. Written by `client.generate()`, so no arm can forget it.

A score is only comparable with another score of the *same workload*, and nothing
recorded what the workload was. Editing a prompt silently made every prior number
incomparable with no way to detect it afterwards — the reason `scene_single` had
to be left untouched and new arms added beside it, a discipline that was until
now enforced only by remembering to. Fixtures are gitignored and regenerated, so
a re-rendered `visimgs/` could change what was measured invisibly; `images_sha`
catches that.

This is the version-locking an established benchmark suite has and this harness
lacked. 3DMark cites a scene version with every score and refuses to compare
across versions; the same guarantee here is a hash comparison.

**Component hashes are the reason to compose prompts from named parts rather than
editing one blob.** A whole-prompt hash says *that* the workload changed;
`prompt_parts` says *which section*. Measured: flipping the `pin` factor moves
`convention` and `schema` and leaves `coords`, `anchor` and `dialect` byte-identical,
so a diff is attributable to the factor under test rather than to an accidental
edit elsewhere.

Two consequences worth stating, because both are cases where the hash reveals
something previously hidden:

- The positional arm's prompt **differs by model family** (gemma4 is offered
  `box_2d`, qwen and nemotron `bbox_2d`), so those cells carry different
  `prompt_sha` values. They are deliberately not the same workload, and pooling
  them is a cross-workload comparison that the data now makes visible.
- A cell with no `prompt_sha` predates 2026-08-20. Its workload is whatever its
  campaign document says, which is why campaign docs quote the arm and the date.

**H17 — A checkpoint is its manifest digest, not its tag and not its config
digest.** Registry tags are mutable: the library's current `gemma4:31b-nvfp4`
carries a bf16 vision tower, while the checkpoint every 0.34 Metal number was
measured on carries a 4-bit one — 194 vision layers differ (the MLX-CUDA
session's registry audit, #312), and a pull replaces one with the other under an
unchanged name.

The **config** digest cannot tell them apart. It is the architecture's
`config.json`, identical across quantizations of one model: locally
`31b-nvfp4` and `31b-mlx-bf16` both carry `b72c5344…`. What identifies a
checkpoint is the sha256 of its **manifest**, which changes when any layer does
(`31b-nvfp4` `637cc0ff…`, `31b-mxfp8` `1434769c…`, `31b-mlx-bf16`
`fb3f25b3…`). A campaign document records the manifest digest of every
checkpoint it measured, and a checkpoint that must stay reproducible is
preserved under a tag naming its composition before any pull — the copy's
manifest digest must equal the original's (`gemma4:31b-nvfp4-tower-nvfp4` is
`637cc0ff…`).

Two checkpoints share a component only when their FULL layer sets match by
digest. One matching tensor proves one tensor: `31b-mxfp8` and `31b-mlx-bf16`
share a byte-identical vision `down_proj` blob (digest `4b80a6f9…`), and the
claim that they share a bf16 tower was published from that tensor alone — 162 of
the 356 vision layers are 8-bit in `31b-mxfp8`. `quant_dims.py` lists a
checkpoint's quantized weights; diff the manifests by `name` and `digest`
before asserting what two checkpoints have in common.

**H22 — A throughput number names its cache class.** `prefill_tps` is not a
compute rate. On a KV-cache hit the server still reports the whole
`prompt_eval_count` but a collapsed `prompt_eval_duration`, so the quotient
jumps five- to thirty-fold while the encoder does almost nothing. Within one
campaign the metric is therefore bimodal, and the two modes are separated by a
wide empty gap rather than a gradient — so a median over mixed blocks reports
the arm's *cache-hit rate* under the name "prefill throughput". Each block is
classified `cold` or `cache`, the two populations are reported separately, and a
block whose class DIFFERS between the arms is excluded from the paired delta and
listed by name: its ratio divides an encode by a cache lookup. `gen_tps` has no
such failure mode and is reported over every warm block. The first block after a
restart is dropped in both arms — `cold_start` marks it, and it carries
model-load and clock-ramp cost no later block pays.

A throughput comparison also states its noise floor, measured on the same host
by the same generator from a pair that should differ in nothing. Without it a
few percent is unreadable: it is either the result or the instrument.

> Measured 2026-09-20, comparing ROCm 10.0.0 against 7.2.4 on gfx1151 with the
> llama.cpp payload held at `391fac164`. The noise floor came from two
> direct-I/O A/B pairs — same image, same ROCm, a knob with no measured score
> effect — and is ±0.4% on the median of paired per-test ratios, which is what
> made a −3% reading a result rather than a shrug.
>
> The cache rule was written because its absence had already produced two wrong
> statements in one session. `gemma4:31b` prefill read 168 tok/s on 0.34.1 and
> 1144 tok/s on 0.34.2 over the SAME 27 blocks, and that 7x was published here
> as a compute win attributed to ADR 0036. It is not compute: cold-prefill is
> 167 → 172 tok/s, and what changed is that 5 of 27 blocks hit cache on 0.34.1
> against 20 of 27 on 0.34.2. The same dilution then flattened the ROCm result —
> `gemma4:31b` reads +0.3% over all blocks and **−5.5%** over the blocks that
> actually encoded, and `gemma4:26b-a4b` reads −2.3% against **−10.4%**. Split
> by class, ROCm 10.0.0 regresses prefill on all five models; unsplit, it
> appeared to regress on three and be flat on gemma4.


**H23 — A knob with a prerequisite is measured in three arms, not two.** When
turning a setting off requires changing a second setting, a two-arm A/B
attributes both changes to the one being studied. Measure the prerequisite
alone as its own arm and compare against *that*. Flash attention could not be
disabled on this fork without also moving `OLLAMA_KV_CACHE_TYPE` from `q8_0` to
`f16` — the runner refuses with `quantized V cache requires flash_attn to be
enabled`. The three arms measured the cache change at 1.0% (noise) and flash
attention at 51%; a two-arm test would have reported 52% for flash attention
and been wrong by the whole of the difference. If the prerequisite cannot be
isolated, the result names both changes or it names neither.

**H24 — A percentage-of-peak says where the peak came from, and a value above
100% means the denominator is wrong.** A ratio against a theoretical ceiling is
only as trustworthy as the ceiling, and ceilings are easy to get wrong in
silence. Derive it from the hardware's own report — `rocminfo` "Compute Unit" —
not from an API whose units vary by architecture:
`hipGetDeviceProperties().multiProcessorCount` returns **WGPs** on RDNA, each
holding 2 CUs, so it reported 20 for gfx1151 against a true 40 and halved the
ceiling. `gemm_ceiling_bench.cpp` printed "165% of peak" before this was
caught; the impossible number is the only reason it was caught at all. Any
efficiency figure over 100% is a bug in the denominator, never a fast kernel.
## 4. Conformance

| requirement | enforced by |
|---|---|
| H1, H2 | `run_engine_compare.sh` is the only script in `vision-suite/` that iterates `$MODELS`; a second one is the defect |
| H3, H4 | `REPEATS` / `TAG_PREFIX` / `ONLY_TESTS` are inert when unset — verified with `sh -x` on both paths |
| H4a | `run_engine_compare.sh` exits 2 when `CTX_MAX` leaves no CONTEXT-ladder rung above the think-on start; think-off is unaffected and `ALLOW_NO_LADDER=1` overrides |
| H4b | `arm_done` in `vision_suite.py` importing `was_capped` (H5); `test_summarizers.py::TestResumeNeverSkipsCapped` asserts capped, error, and missing blocks all re-run; `::TestWasCappedPrefersDoneReason` asserts the `done_reason` verdict outranks the arithmetic and absence falls back to it |
| H5, H6 | `summarize_reps.py`, `summarize_geometry.py`, `summarize_matrix.py` and `preflight/checks.py` all import `was_capped` (`test_summarizers.py::TestCappedDiscipline`, `test_verdicts.py::TestQualityCappedExcluded`); the ladder decisions are `capped_arms` / `ceiling_standing` / `mark_not_converged` in the same module, driven by `run_engine_compare.sh` via argv subcommands (`::TestCappedArms`, `::TestLadderCeilingMarker`) |
| H7 | ADR 0012 rules 1 and 8; request examples via `emit_request.py` (payload captured from `client.py`, never re-derived) |
| H7, H11, H13 (external benchmarks) | `extbench.py` runs render through `summarize_extbench.py`, never by hand; `extbench.py` persists the `host` / `server_version` that `client.generate()` already stamps on every response, collected as SETS so a mid-run container restart cannot let one build vouch for another's rows; `test_summarizers.py::TestExtbenchSummary` asserts the unrecorded-file, mixed-campaign and clean-footer cases, and that the paired test stays EXACT (at 3-vs-1 discordant the uncorrected chi-square reads p≈0.317 against the true 0.625) |
| H13 (footers) | `test_summarizers.py::TestProvenanceFooter` — clean / all-pre-H11 / mixed-recording / two-host cases against the rendered footer |
| H13 (capped rendering) | `cap_or` in `summarize_head_to_head.py` importing `was_capped` (H5); `test_summarizers.py::TestT2CappedCells` asserts a capped scene hides score and latency but keeps tok/s; `q()`/`multi_cell` in `summarize_engine_compare.py` guard every T1 quality cell — `::TestT1CappedQualityCells` |
| H14 | `token_split.py`'s acceptance gate (`--write` refuses without it, and skips-by-name irreconcilable cells); `test_summarizers.py::TestT1ThinkTokColumn` asserts stamped-count-or-dash, never an estimate; `finetext_probe.py` persists as `finetext_probe` (one producer per persist name) |
| H9 | `client.py` is the only module that builds a payload; `test_client.py` asserts the wire format (the test count grows with the contract — trust the suite, not this row), including the tri-state `send_think` and the `num_ctx=False` sentinel that a naive `== False` would have collapsed |
| H12 | `prompt_sha` / `images_sha` / `prompt_parts` on every score block, written by `client.generate()`; absence marks a pre-2026-08-20 cell |
| H11 | `host` / `server_version` on every score block, written unconditionally by `client.generate()`; absence marks a pre-2026-08-20 cell |
| H10 | `client.RETRY_BACKOFF` = 5/15/30s with `_retries` recorded per cell; `test_client.py::TestTransportRetry` asserts a 400 calls `urlopen` exactly once while a 503 retries. `client.evict_others()` polls `/api/ps` until the eviction is observable and returns what it could not evict; `run_engine_compare.sh` calls it before each model when `RESTART_CMD` is absent, `COLD_START=0` opts out |
| H15 | **Nothing enforces this.** The 2026-09-18 replication (`bench-runs/finetext-9px-31b-quant-reps-2026-09-18.json`) is the worked example of reporting a rate |
| H16 | **Not enforced, and the renderer works against it:** `summarize_engine_compare.py` renders the suite arm and shows the probe arm only as a fallback. A render that reported both arms, or flagged their disagreement, would enforce it |
| H17 | `store_audit.py` diffs a store's manifests against the registry and names what moved, which tensor group and the size ratio; `--digests` prints the digest to cite (`test_store_audit.py`). It found the case that motivates the rule: `gemma4:31b-nvfp4` re-published with a bf16 vision tower, 194 layers different under an unchanged tag and config blob (ADR 0038). `quant_dims.py` (#316) still lists a checkpoint's quantized weights and their K. Score blocks do not yet carry the digest — that is the remaining gap |
| H18 | **Nothing enforces this.** `extbench.py` produces per-item records that a paired test can use; `bench-runs/ocrbench-v1-1000-gemma4-31b-nvfp4-0340-vs-0332.json` carries the paired items and the exact McNemar result |
| H8 | **Nothing enforces this.** It is a reading habit, and it is the one that would have prevented all three incidents |
| H19 | `summarize_extbench.py --categories` prints a slice's question-type split from the cached row file (`test_summarizers.py`); `extbench.py` caches that file per `(offset, limit)` so the rows a run answered are recoverable |
| H20 | `preflight/checks.py` keeps SKIP, FAIL and ERROR distinct per check and names the route in the summary; `test_verdicts.py::TestMetalTensorGate` asserts the toolchain-missing, remote-host, containerised-payload, missing-`--log-cmd` and does-not-apply skips, and `::TestReleaseMatrixTensorColumn` asserts the column renders, goes red, and reads N/A where there is no Metal |
| H21 | `probes.mlx_build` and `probes.launched_runner_paths` both filter on each line's own slog timestamp and drop unparseable ones; `probes.parse_metal_tensor_discovery` anchors on the port under test; `test_verdicts.py::TestTensorProbeRoutes` asserts the stale-line, untimestamped and newest-wins cases, `::TestMetalTensorGate` the wrong-server anchor |
| H22 | `summarize_tps.py` classifies every block and refuses to divide across classes; `test_summarizers.py::TestThroughputCacheClass` asserts the cache label, the excluded-and-named mismatch (`n/c`, never a 10x), the dropped `cold_start` block, that `gen_tps` still counts every warm block, and that two hosts render MIXED while two builds on one host do not. **The noise floor is not enforced** — it is a measurement the operator must re-take per host |
