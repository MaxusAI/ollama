# OCRBench in this fork: one test, one format, three hosts

Three sessions ran OCRBench against gemma4:31b on 2026-09-18/19 — mlx-metal, ROCm/GGUF
and CUDA (both engines) — and each wrote its own harness notes. This is the shared
entry point: what the test is, how a run is produced, how a result is reported, and where
every number lives. The per-host detail stays in its own document.

| host / engine | what it answers | detail |
|---|---|---|
| mlx-metal | does the mlx#3912 kernel fix change quality? (full 1000-item set) | [vision-campaign-2026-09-18-mlx8a7ba949-nvfp4.md](vision-campaign-2026-09-18-mlx8a7ba949-nvfp4.md), [ADR 0037](adr/0037-keep-the-mlx-3912-kernel-fix.md) |
| ROCm / GGUF | does q4→q8→bf16 cost quality, with bf16 as the unquantised control? | [ocrbench-gemma4-quant-ladder.md](ocrbench-gemma4-quant-ladder.md) |
| CUDA / mlx-cuda + GGUF | the same ladder on both engines, plus the vision-tower precision axis | [ocrbench-quantisation-ladder.md](ocrbench-quantisation-ladder.md) |

## The test

`vision-suite/extbench.py`, `echo840/OCRBench` test split, gold answer contained in the
prediction (lmms-eval's `ocrbench` semantics, including its whitespace-stripped
handwritten-maths case), think off, temperature 0 with `apply_sampling=False`,
`/api/generate`, one model loaded at a time.

```bash
LIMIT=200 OFFSET=0 THINK=false NUM_CTX=8192 NUM_PREDICT=512 SLEEP=1 TIMEOUT=1800 \
  python3 docs/maxusai/vision-suite/extbench.py http://127.0.0.1:<port> <tag> <model> ocrbench
```

The row slice is cached under `extimgs/ocrbench/rows_<offset>_<limit>.json` and the
images beside it, so a re-run answers the same questions and a transient DNS failure
cannot end a twenty-minute arm. `REFRESH_ROWS=1` re-fetches.

## The format

One renderer, `vision-suite/summarize_extbench.py`, pasted verbatim (SPEC
`vision-harness-reuse` H7). A bare argument is an arm; `label=tag,tag` is an arm with
repeats.

```bash
python3 docs/maxusai/vision-suite/summarize_extbench.py --paired --repeats --timing \
  --categories ocrbench "q4_K_M=ocr_q4,ocr_q4_r2" "bf16=ocr_bf16,ocr_bf16_r2"
```

| flag | table | why it is not optional in practice |
|---|---|---|
| (default) | scored / errors / empty / correct / accuracy, with an H13 provenance footer | a run whose host and build are unrecorded says so rather than inheriting a sibling's |
| `--paired` | both ✓, both ✗, A only, B only, exact McNemar p | a 200-item slice carries a ±0.024 standard error, so the accuracy column cannot separate arms three items apart |
| `--repeats` | accuracies per run, items that changed verdict | the control for the paired test: an arm re-run against itself must move less than the arms move against each other |
| `--timing` | ±1 s.e., mean and median s/item, mean prompt_eval | prompt_eval equal across arms is what proves they were shown the same image |
| `--categories` | correct/n per OCRBench question type | H19: rows 0–200 are four of ten categories, so the split is part of the result |

Two rules bind a result (SPEC `vision-harness-reuse`):

- **H17 — a checkpoint is its manifest digest**, not its tag. `gemma4:31b-nvfp4` was
  re-published with a bf16 vision tower under an unchanged tag and config blob, so two
  hosts holding that tag need not hold the same weights.
  `vision-suite/store_audit.py --digests` prints the digest to cite, and the same tool
  run on both hosts is what makes a cross-host comparison mean anything
  ([ADR 0038](adr/0038-a-model-is-identified-by-its-manifest-digest.md)).
- **H19 — a slice names its rows and its categories.** The set is ordered by task: rows
  0–200 are regular, irregular, artistic and handwriting recognition, 50 each, and
  contain no VQA, key-information extraction, digit strings or handwritten maths. A
  200-item score is not an OCRBench score and does not belong beside a published one.

## Where the numbers are

**Do not read down this table.** The 1000-item run covers all ten categories; the
200-item runs cover the four recognition ones, which the models find easier. The rows are
comparable within a slice, never across.

| slice | host / engine | model | correct / scored | accuracy |
|---|---|---|---|---|
| 1000, all categories | mlx-metal | `gemma4:31b-nvfp4`, 0.34.0 fixed kernel | 835 / 1000 | 0.835 |
| 1000, all categories | mlx-metal | `gemma4:31b-nvfp4`, 0.33.2 defective kernel | 833 / 1000 | 0.833 |
| 200, rows 0–200 | mlx-cuda | `gemma4:31b-nvfp4`, 4-bit tower | 172 / 200 | 0.860 |
| 200, rows 0–200 | mlx-cuda | `gemma4:31b-nvfp4`, bf16 tower (library) | 170 / 200 | 0.850 |
| 200, rows 0–200 | mlx-cuda | `gemma4:31b-mxfp8` | 169 / 200 | 0.845 |
| 200, rows 0–200 | CUDA GGUF | `gemma4:31b-it-q4_K_M` | 171 / 200 | 0.855 |
| 200, rows 0–200 | CUDA GGUF | `gemma4:31b-it-q8_0` | 170 / 200 | 0.850 |
| 200, rows 0–200 | CUDA GGUF | `gemma4:31b-it-bf16` | 171 / 200 | 0.855 |
| 200, rows 0–200 | ROCm GGUF | `gemma4:31b-it-q4_K_M` | 171 / 200 | 0.855 |
| 200, rows 0–200 | ROCm GGUF | `gemma4:31b-it-q8_0` | 169 / 200 | 0.845 |
| 200, rows 0–200 | ROCm GGUF | `gemma4:31b-it-bf16` | 169 / 200 | 0.845 |

## What all three runs agree on

**No quantisation difference resolves.** Every paired test run on every host — 4-bit
against 8-bit against bf16, on GGUF and on MLX, and the fixed kernel against the
defective one on 1000 items — returns p between 0.375 and 1.000. The arms differ by
single items in both directions. Two engines also reproduced their arms item for item
across repeats, so this is not noise swamping a signal; on this benchmark there is no
quality signal to find between these builds.

**What OCRBench cannot do is settle a kernel question.** The mlx-metal campaign says it
plainly: the full 1000 items cannot tell the fixed and defective `fp_qmm_t` apart
(p = 0.804), while the vision golden delta can, and does, exactly. A benchmark that
scores an answer string is a weak instrument for a numerical defect; keep the goldens for
that.

**Speed moves where quality does not.** On CUDA the vision tower's precision is worth 39 %
of the time per image — 1.4 s against 2.3 s for the same language model, identical images
and an identical 1115-token prefill — because MLX's quantised matmul on sm_120 runs at
about half the dense bf16 rate. That is the finding this ladder actually produced.
