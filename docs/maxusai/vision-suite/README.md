# Vision benchmark suite

Reproducible ground-truth benchmarks behind the measured tables in
[nemotron-test-image.md](../nemotron-test-image.md) and the amendments in
[vision-token-budget-measurements.md](../vision-token-budget-measurements.md).

## Files

- `preflight/` — **the pre-deploy regression gate.** Everything else here reports;
  this asserts. One entry point takes an image tag and a platform, resolves a
  profile from `(platform, version string)`, runs the applicable checks, and exits
  non-zero with an expected-vs-actual diff. Expected values live in
  `preflight/expectations.toml` and are versioned with the payload they describe,
  because a compat patch legitimately changes them (004 made gemma4 flat, 005 moved
  nemotron's pinned cost 3390 → 3270). Start at
  [preflight/README.md](preflight/README.md); the operator procedure is the
  `ollama-preflight` skill.
- `gen_scenes.py` — deterministically renders the three test images into `visimgs/`
  plus `ground_truth.json`: a 1920×1080 labeled-shapes scene (20px labels, 14px corner
  serial), a 1568×1568 fake invoice (22px line items, 17px fine print), a 1280×960 bar
  chart (19px values). Needs Pillow + DejaVu fonts
  (`/usr/share/fonts/truetype/dejavu/`; on hosts without that path — macOS — set
  `FONT_PATH`, e.g. to matplotlib's bundled `DejaVuSansMono.ttf`; applies to
  `finetext_probe.py gen` too). Regenerate any time; edit sizes/content to
  extend coverage — scoring reads `ground_truth.json`, not hardcoded values.
- `vision_suite.py <host> <tag> [model] [test]` — runs four long-prompt JSON
  extractions (single scene w/ pixel bboxes, single invoice, 3-image cross-analysis, and
  the dense fine-text probe, whose prompt and scorer are imported from `finetext_probe.py`
  rather than copied) and scores objectively (label recall, color accuracy, qty/price
  exactness, bbox center-hits, cross-image answers, per-size code recall). All four land in
  `scores_<tag>.json`, so the fine-text tiers are there without running the standalone
  probe. Env: `THINK=on|false` (default `false` — must be the literal `on`),
  `NUM_PREDICT` (default 2200; **no fixed think-on floor** — the once-documented ≥4000 was
  measured on 2026-08-09 and does not hold, every empty cell in that run sat at exactly
  eval=4000. Think-on budgets are derived from the `num_ctx` rung by the runners as
  `num_ctx - CTX_PROMPT_RESERVE`),
  `IMAGE_MIN_TOKENS` / `IMAGE_MAX_TOKENS` (fork-only per-request vision budget,
  arch-gated to gemma4 and nemotron_h_omni; unset = build default. Recorded in the
  scores as `req_image_*_tokens` so a control run is identifiable after the fact),
  `ENDPOINT=generate|chat` (default `generate` — `/api/chat` is what OpenWebUI and
  ChatOllama use, and it has carried the upstream think+format two-pass fix since
  v0.12.4, so think-on cells differ by endpoint on builds without the generate-side
  fix). Writes `resp_<tag>_<test>.json` + `scores_<tag>.json` beside the script.
- `gen_geoms.py [outdir]` — renders the eight geometries `measure.py` reads into `testimgs/`
  (deterministic noise + gridlines + corner markers, so the payload size is realistic and
  letterboxing is visible). Run it before `measure.py`; needs Pillow.
- `measure.py <host> [model]` — the token-budget protocol: `prompt_eval_count` with
  `num_predict:1` minus the calibrated text prefix, over 8 geometries + the
  `image_max_tokens` knob check. Flat 256 on nemotron = unpatched payload. It *reports*
  rather than asserts — for a pass/fail gate on the same formulas with no GPU and no server,
  use `go test ./llm/ -run TestImageTokensForSize`. Note the `image_max_tokens` probe is a
  Runner option, so it forces a full model reload.

  > **Method corrected 2026-08-08 — pre-correction output was wrong.** The script used to
  > baseline with the prompt `"Hi"` and probe with `"Describe briefly."`, so the text-length
  > difference landed in every row. It now uses one prompt throughout *and* derives the
  > baseline from a two-image difference
  > (`count(A) + count(B) − count(A,B)` cancels the prefix), because the text-only count is
  > itself the wrong subtrahend on some arches — see the two traps documented at the top of
  > the script. Verified on the :11437 canary: all eight geometries now reproduce
  > `TestImageTokensForSize` exactly, including the ceiling-exact 2048×1664 → 3,330.
  > Any figure taken from this script before 2026-08-08 is suspect; the one published set
  > is corrected in [nemotron-test-image.md](../nemotron-test-image.md#results).
- `extbench.py <host> <tag> [model] [benchmark]` — slices of four external benchmarks
  (`ocrbench`, `countbenchqa`, `chartqa`, `refcoco`) pulled from the HF datasets-server REST
  API (stdlib only, no `datasets`, no HF token) and scored locally: contains-match, integer
  match, relaxed accuracy, and dialect-aware bbox IoU respectively. Env: `LIMIT` (50),
  `OFFSET`, `SLEEP` (yield the GPU between requests), plus the same `THINK` / `ENDPOINT` /
  `NUM_PREDICT` / `NUM_CTX` knobs as `vision_suite.py`. Writes `ext_<tag>_<bench>.json`. The
  `refcoco` mode reports the winning coordinate dialect and JSON key per item, so it doubles
  as a dialect probe. See [../vision-benchmark-survey.md](../vision-benchmark-survey.md) for
  why the external harnesses' own grounding scorers cannot be trusted with our models.
- `run_grid.sh <host> <tag-prefix>` — model × think-mode grid against one host, with an
  optional restart hook between runs (see below). Budgets are **per think-mode** and set by
  the runner: think-off `num_predict` 4000, think-on `num_ctx - CTX_PROMPT_RESERVE` (8192 at
  the 16384 start rung). Do **not** export `NUM_PREDICT` or `NUM_CTX` unless you mean to pin
  *both* modes to one value — use `NUM_PREDICT_THINKON` / `NUM_CTX_THINKON` to move think-on
  alone. Unlike `run_engine_compare.sh` it does not auto-escalate; it reports a capped cell
  and the rung to retry at.
- `BINARIES.md` — **the benchmark binary archive manifest.** Every measurement in
  `docs/maxusai/` is attributable to one server binary and preflight gates on the
  version string it reports, so when a payload pin moves the old build has to
  stay reachable. Binaries live on the host at `~/.ollama/binaries/`; this file
  carries their identity, checksums, what each is the provenance for, and the
  rebuild recipe.
- `build-macos.sh` — **macOS/Metal only. Builds the native fork binary with the version
  stamp preflight gates on.** The binary is the provenance for every measurement
  here, and assembling the ldflags by hand from `spec/apple-silicon-build.md`
  got it wrong once (a bare `0.32.14` matches no profile). Derives the base
  version from the newest reachable upstream tag rather than hardcoding it, warns
  on a dirty tree because the sha would then be a lie, and takes `CLEAN_DEPS=1`
  to clear the vendored llama.cpp checkout when `LLAMA_CPP_VERSION` moves — the
  stash/unstash cycle fails otherwise.
- `serve-apple-mlx.sh` — **the RESTART_CMD hook, Apple Silicon + MLX store only.**
  Note "MLX" alone does not imply Apple: the fork also ships an `mlx_cuda_v13`
  payload for Linux/CUDA ([why it is unloadable](../upstream-mlx-cuda-payload-unloadable.md)).
  For CUDA/ROCm restart the container instead — `run_grid.sh` shows the docker form. Sets `OLLAMA_MAX_LOADED_MODELS=1`,
  without which a sweep holds every model it has served resident; measured
  106 GB used and 53.9 GB swap on a 128 GB host before this existed.
- `summarize_contract_matrix.py --think <mode> [--log <runner log>] <model…>` —
  the bbox-contract matrix plus per-model power-mode provenance. Capped cells
  render as `cap` rather than a false `❌` (ADR 0012 rule 8), and the `num_ctx`
  column shows the ladder rung each row reached.
- `summarize_lowtemp.py <tag-prefix>` — repeated ARMs as
  `(score, num_ctx, num_predict)` per model.
- `run_engine_compare.sh <host>` — **engine-parity campaign** (MLX safetensors vs
  llama-server GGUF): cold server per model via `RESTART_CMD`, then the three-suite
  run and the fine-text probe per model. `summarize_engine_compare.py <model…>`
  renders the two comparison tables from the per-tag `scores_*/ft_*` files —
  the format of [vision-campaign-2026-08-08-mlx.md](../vision-campaign-2026-08-08-mlx.md);
  keep it stable so runs diff cleanly.
- `run_compare.sh <tag-prefix>` — **stock vs fork, with a budget-matched control arm.**
  Use this rather than eyeballing two separate runs: a bare stock-vs-fork comparison
  moves two variables at once. See "Comparing against stock" below.
- `variants.py <nogrammar|thinkon> [host]` — scene-test probes that isolate
  the `format:"json"` grammar constraint and reasoning mode as variables. Mode comes
  **first**, host second; the model is hardcoded to `nemotron3:33b-q4_K_M`.

## Scoring note: markdown-fence tolerance (2026-08-08)

All scorers (`vision_suite.py`, `finetext_probe.py`) strip one markdown code
fence before `json.loads` and record `fenced: true` when they did. Engines that
enforce `format:"json"` (llama-server grammars) never produce fences, so this
is a no-op there; the MLX runner did not enforce format until x/structured
(ADR 0009) and answered with well-formed but fenced JSON. `json_valid` means
"parsed after fence tolerance" — check `fenced` when comparing engines.

## Method (match this or numbers aren't comparable)

- `temperature 0`, `format:"json"`, `num_ctx 16384`. **Let the runner set the budgets** —
  they are per think-mode and exporting `NUM_PREDICT` pins both. A single 4000 across both
  modes, which this line used to prescribe, caps think-on inside its own reasoning block and
  scores the truncation as a vision failure ([ADR 0022](../adr/0022-thinking-is-off-for-vision-work.md)).
- **Cold server per model run** when payloads under test have cross-request leakage
  (upstream #17475 reproduced on b10091): restart the serving container/process
  between runs — `run_grid.sh` does this via `RESTART_CMD`.
- **Always run both think modes.** `think:true` + `format:"json"` yields an *empty*
  `response` for nemotron3 and qwen3.6 **on stock builds** (thinking ends without a
  JSON body, well under the token budget); gemma4 handles both. Report empty cells as
  data.

  > **Updated 2026-08-07 — this is FIXED on the fork; do not expect empty cells from a
  > fork build.** Measured on `nemotron3:33b-q4_K_M`, all three tests, both a native
  > Metal build and the CPU container:
  >
  > | build | `json_valid` | `eval_count` |
  > |---|---|---|
  > | stock 0.32.6 | **False** ×3 | 562 / 485 / 833 |
  > | fork (Metal) | **True** ×3 | 5233 / 10110 / 7668 |
  > | fork (CPU container) | **True** ×3 | 5134 / 7370 / 4889 |
  >
  > Stock still generates tokens — it thinks and then emits no JSON. The fork thinks
  > and then emits valid JSON. See
  > [generate-think-format-empty-response.md](../generate-think-format-empty-response.md),
  > [ADR 0002](../adr/0002-deferred-format-constraining.md) and
  > [ADR 0004](../adr/0004-routes-layer-think-format-double-request.md).
  >
  > **Budget accordingly.** A fork think-on cell does real work where stock returns
  > almost immediately, so it is far slower — not a hang. Same run: stock 21 s for all
  > three tests, fork on Metal ~7 min, fork on the CPU container ~39 min. Raise
  > `HTTP_TIMEOUT` for CPU think-on runs.
  >
  > **Amended 2026-08-13 — a think-on cell can still come back empty, and it is the
  > harness's own fault.** The suite generates at `temperature: 0` with no
  > `presence_penalty` (hardcoded in `vision_suite.py`, `finetext_probe.py` and
  > `preflight/probes.py`). That is off-policy for these models: Qwen's card
  > recommends `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` for
  > thinking mode and names `presence_penalty` as the anti-repetition lever. With it
  > disabled, reasoning can fail to terminate — every token lands in `thinking`,
  > `eval_count` hits `num_predict`, and `response` is empty.
  >
  > Same prompt, same budget (`num_predict=24000`), measured on b10353:
  >
  > | model | `temperature: 0` (suite) | card-recommended sampling |
  > |---|---|---|
  > | `qwen3.6:35b-a3b-q4_K_M` | 24 000, **empty** | 10 810 / 8 023, **valid** |
  > | `qwen3.6:35b-a3b-q8_0` | 16 677, valid | 4 109 / 12 604, **valid** |
  > | `gemma4:12b-nvfp4` | 24 000, **empty** | 3 278 / 2 525, **valid** |
  >
  > Note the q8_0 row: for **qwen3.6 only**, raising quantization also fixes it on its
  > own. That does not generalize — `gemma4:12b` still loops at F16 upstream — so fix
  > the sampling, which works on both.
  >
  > **Do not escalate `num_ctx` to chase this** — no context size fixes it, and above
  > ~90 K `num_predict` the 1800 s `HTTP_TIMEOUT` expires first, turning a cap into an
  > error with no data. Fix the sampling instead.
  >
  > Two further consequences for anyone reading think-on numbers: the failure is
  > **stochastic even at `temperature: 0`** (the same `gemma4:12b-nvfp4` finetext cell
  > converged at 2 761 tokens and capped at 28 672 on identical requests), so a single
  > capped observation proves nothing — run n ≥ 3. And precision is **not** implicated:
  > gemma4 31b converges at nvfp4/mxfp8/bf16 and nemotron3 33b at q4_K_M/q8/bf16, though
  > higher precision costs up to 2.3× more reasoning tokens, which tightens a fixed
  > `num_predict`. Full evidence:
  > [runaway-reasoning-under-think.md](../runaway-reasoning-under-think.md).
- Subtract each model's text baseline when reading `prompt_eval_count`, measured with
  **the same prompt as the probe** — a baseline taken with a different prompt puts the
  text-length difference into every row. Beware also that the prefix can tokenise
  differently once an image is attached: `nemotron3` reads 21 text-only but 20 inside an
  image request for the same prompt (`gemma4:31b`: 19 both ways), so prefer `measure.py`'s
  two-image calibration over any text-only number. Counts are grid-quantised — ignore ±2.
- Bbox scoring is dual-space: models emit their trained coordinate conventions
  regardless of prompt instructions — qwen3.6 answers in 0-1000 normalized (IoU ~0.95
  once decoded; near-perfect grounding), nemotron3 with reasoning answers in pixels
  (center-accurate, IoU ~0.3). The scorer tries both spaces, keeps the better, and
  reports `bbox_space` + `bbox_mean_iou` alongside center-hits. Accepted schema-key
  dialects: `bbox`, `bbox_2d` (qwen, and nemotron's self-chosen key), `box_2d`
  (gemma4/Gemini — note its [y1,x1,y2,x2] order, searched automatically), plus
  `name_bbox`/`name_bbox_2d` on the invoice. Measured dialect map (2026-08-02):
  qwen3.6 = bbox_2d, xyxy, norm-1000 (IoU ~0.95); gemma4 = box_2d, yxyx, norm-1000
  (IoU ~0.78); nemotron3 = bbox_2d, xyxy, norm-1000 of its input canvas — on the
  unpatched 512-letterbox payload the y-axis carries the padding offset, on dynres
  payloads the canvas is the image itself; under prompted reasoning it can emit
  coarse pixel-space boxes instead. Key choice alone did not change quality; the
  space/order decode did.

## Example: full grid against an isolated test server

```bash
python3 gen_scenes.py
RESTART_CMD="docker restart my-test-container" \
  MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m" \
  ./run_grid.sh http://127.0.0.1:11435 mytag
```

The isolated-container recipe (own port, model store mounted read-only, GPU
passthrough) is in [nemotron-test-image.md](../nemotron-test-image.md).

## Comparing against stock (use the control arm)

A bare stock-vs-fork comparison moves **two** variables at once:

1. **our vision token budget** — `visionServerArgs` adds `gemma4` and
   `nemotron_h_omni` branches that upstream does not have *at all* (checked
   2026-08-07: both `v0.32.5:llm/llama_server.go:994` and `v0.32.6:…:999` contain
   only `qwenVLServerArgs`, handling qwen arches); and
2. **the llama.cpp payload** — `LLAMA_CPP_VERSION` differs whenever the fork is not
   synced to the release the stock server runs. Measured 2026-08-07: fork `b10091`
   (v0.32.5) vs stock `b10242` (v0.32.6), 151 builds apart.

So "the fork detects the fine text that stock misses" and "the fork's bbox IoU is
worse than stock's" are individually uninterpretable — either could be ours or
upstream's drift.

`run_compare.sh` adds a third arm that pins the fork's budget to upstream's
effective defaults. A delta that **disappears** under the control was ours; a delta
that **survives** is the payload.

```bash
STOCK=http://127.0.0.1:11434 FORK=http://127.0.0.1:11435 \
  MODEL=gemma4:12b-it-q4_K_M CONTROL_MIN=40 CONTROL_MAX=280 \
  ./run_compare.sh mytag
```

Control values are **per-arch**, and wrong ones silently invalidate the arm:

| arch | control min | control max | why |
|---|---|---|---|
| `gemma4` | 40 | 280 | llama.cpp `set_limit_image_tokens(40, 280)` |
| `nemotron_h_omni` | 256 | 256 | unpatched payload is a structural flat 256; 002 makes it (256, 3328), so pinning both bounds reproduces stock |

The knobs are arch-gated, so on any other arch the control arm is a no-op that
duplicates the fork arm. That is a valid result — but do not read it as "no budget
effect" on an arch that was never wired into `visionServerArgs`.

**Backend caveat.** A CPU arm can differ from a Metal arm on identical inputs with
identical `prompt_eval_count` — greedy sampling diverges on backend floating point.
Always check `prompt_eval_count` before attributing such a delta to a patch.

## Runs archive and harness knobs (2026-08-02)

- `runs/` holds raw campaign logs plus `*.parsed.json` (one object per scored
  cell) for the 2026-08-02 campaign, max-context arm, true-stock baseline,
  and runaway bisect; as-run parsed files keep the scorer outputs of their
  time (Q4 is pre-correction in blocks scored before the dialect fix);
  `final-matrix-2026-08-02.json` is the merged, Q4-corrected dataset behind
  the published matrix.
- `ONLY_TESTS=scene_single[,document_single,...]` runs a subset of the suite —
  used by the bisect harness. `HTTP_TIMEOUT` (seconds, default 1800) bounds a
  single request — raise it for uncapped think-mode probes. `KV_CACHE_TYPE`
  passes a per-request `options.kv_cache_type` (fork feature, ADR 0005) —
  single type or K/V pair like `q8_0/f16`.
- Multi-image Q4 is scored dialect-aware like scene boxes (`q4_bbox_space`
  reports the matched space); models answer norm-1000 regardless of prompt.
- Caveat: with `OLLAMA_KV_CACHE_TYPE=q8_0`, qwen3.6 think-on inflates
  prompt-dependently: document unaffected, scene ~19K thinking tokens (vs
  3.3K at f16), multi no convergence within 131K (vs 9.0K at f16). Use f16
  KV for qwen reasoning runs; no practical num_predict rescues multi on
  q8_0. See vision-campaign-2026-08-02.md §6.

## Fine-text probe and coordinate-dialect guidance (2026-08-02)

- `finetext_probe.py` generates a 1568² dense-text page (20 reference codes at
  22/16/12/9/7 px, seeded) and scores exact-match recall per size tier — the
  test that separates real transcription from confabulation. `gen` regenerates
  the image; run form matches vision_suite env knobs.
- Prompt bounding boxes in **norm-1000**, not pixels: all three models answer
  norm-1000 natively; nemotron additionally OBEYS a pixel instruction when
  thinking and loses geometry doing so (IoU .39 pixel-prompt vs .81
  norm-1000-prompt, think on). The scorer's `bbox_space` field verifies what
  came back.

## Running an ARM (repeats, subsets, sampling overrides)

**Never hand-roll a loop over models.** `run_engine_compare.sh` is the only
runner, and it is the only thing that climbs the `num_ctx` ladder and derives
`num_predict` for think-on. Six bespoke loops were written in one week and none
of them did either, so every one silently measured think-on at the think-off cap
of 2200 and produced empty responses that looked like model failures.

```sh
# a repeated, subsetted arm with a sampling override
TEMPERATURE=0.01 REPEATS=3 TAG_PREFIX=lt \
  ONLY_TESTS=bbox_contract,bbox_contract_anchored \
  MODELS="gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_K_M" \
  RESTART_CMD='sh docs/maxusai/vision-suite/serve-apple-mlx.sh' THINK_MODES='false on' \
  ./run_engine_compare.sh http://127.0.0.1:11436
```

| knob | effect |
| --- | --- |
| `REPEATS` | n runs per cell, tagged `<prefix><rep>_<model>_think<mode>` |
| `TAG_PREFIX` | names the arm so its scores do not collide with the campaign's |
| `ONLY_TESTS` | subset passed to `vision_suite.py`; `finetext_probe.py` is skipped unless `finetext` is in the list |
| `TEMPERATURE`, `TOP_P`, … | inherited by `sampling.py`; the resulting `sampling_source` records the override |

Tags are unchanged when `REPEATS=1` and `TAG_PREFIX` is empty, so campaigns and
every summarizer keep working.

## Bounding-box contract probes (2026-08-16)

Eight probes measure whether a model's *declaration* of its coordinate convention
matches the numbers it emitted — a different axis from grounding, and one the
older scorers could not separate. Superseded guidance: the norm-1000 advice
above is right but was measured on three models; it is now seven. The normative
contract is
[SPEC: vision bounding-box response contract](../spec/vision-bbox-response-contract.md)
(C1–C11), decided in
[ADR 0027](../adr/0027-bbox-requests-pin-norm1000-and-carry-an-anchor.md).

| probe | condition | declaration | `contract_followed` |
| --- | --- | --- | --- |
| `bbox_contract` | single image | free choice, top-level | 5/7 (n=1) |
| `bbox_contract_multi` | + distractors, "ignore them" | free choice, top-level | **5/21** |
| `bbox_contract_reasoning` | + distractors, must USE them | free choice, top-level | 5/7 (n=1) |
| `bbox_contract_pinned` | + distractors, "ignore them" | pinned norm-1000, top-level | **21/21** |
| `bbox_contract_perobject` | + distractors, "ignore them" | pinned norm-1000, per object | **21/21** |
| `bbox_contract_anchored` | + distractors, "ignore them" | pinned, named keys, `__IMAGE__` anchor | **21/21** |
| `bbox_contract_adv_real` | as above, dimensions withheld | pinned **`real`** — resisted on purpose | **3/21** |
| `bbox_contract_adv_norm1` | as above | pinned **`norm1`** — resisted on purpose | **15/21** |

The two `adv_*` arms exist to make models mis-declare, so read them on
`hits_anchor` and `self_check`, **not** on `contract_followed` — a low
`hits_declared` there is the point. They establish that which convention you pin
is not free (norm-1000 21/21, norm-1 15/21, real 3/21), and that an
`__IMAGE__` anchor can *inherit* a false declaration rather than correct it.
`bbox_self_check` (range + aspect, response-only) separates usable from
unusable anchors 42/42 across those arms.

Reading the metrics: `hits_declared` scores grounding **only** in the dialect the
model named, `hits_bestfit` is the legacy search over type × order, and the gap
between them is the cost of a wrong declaration. `hits_bestfit` 6/6 with
`contract_followed` false means perfect vision and an unreliable
self-description — a different defect from poor grounding, and the one that
motivated these probes. `declaration_scope` records whether the declaration was
top-level, per-object, or absent; `anchor_implied_type` / `hits_anchor` record
what the `__IMAGE__` calibration entry says, independently of what the model
claimed.

Do **not** "fix" a failing cell by relaxing the scorer to best-fit. That
tolerance is what hid this class of error in the first place.
