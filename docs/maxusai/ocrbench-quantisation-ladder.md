# OCRBench across the quantisation ladder: gemma4:31b

**Status: open, numbers landing.** The `mlx-cuda` and `llama.cpp` rows are measured on the
CUDA host; the `mlx-metal` section is empty on purpose — it is for the Metal session to
fill from its own host. Nothing here mixes the two platforms into one table
([ADR 0012]; a missing same-platform baseline is stated, never substituted).

Glenn's ask, 2026-09-18: the Metal host scored an OCRBench slice on the 0.34.0 fixed
kernel — 175 correct of 200, 0.875 — so run the same slice on `mlx-cuda` and on the
llama.cpp GGUF path across the ladder (`nvfp4`, `mxfp8`, `bf16` on MLX; `q4_K_M`,
`q8_0`, `bf16` on GGUF) and compare.

---

## Read this first: a published tag is not a fixed artifact

`gemma4:31b-nvfp4` in this store and `gemma4:31b-nvfp4` in the library today are
**different models**. Same name, same config blob, 1248 layers of which 194 carry
different weights — all of them on the vision path, and each about 3.5× larger upstream,
which is exactly `nvfp4` (4 bits plus scales) widening to `bf16`. The library
re-published the tag with an **unquantised vision tower**; our copy still has the
quantised one.

`ollama show` reports neither the layer set nor a content digest, so nothing surfaces
the swap. The first symptom would have been a measurement that did not reproduce on a
host that pulled later.

Audit of this store against the registry, from `vision-suite/store_audit.py`:

| tag | local manifest | registry manifest | layers changed | where | size ratio | reading |
|---|---|---|---|---|---|---|
| `gemma4:12b-nvfp4` | `117d0d84cf2a` | `ded7a2735003` | 6 / 737 | vision 2, other 2, audio 1, blob 1 | 2.4× | — |
| `gemma4:26b-nvfp4` | `c8656f50f0a6` | `f0fc7e0ae494` | 190 / 1072 | vision 189, blob 1 | 3.55× | nvfp4 → bf16 |
| `gemma4:31b-nvfp4` | `637cc0ff1570` | `a22a363052da` | 194 / 1248 | vision 191, other 2, blob 1 | 3.52× | nvfp4 → bf16 |
| `qwen3.5:0.8b-mlx` | `6a48dd9c06e3` | `12f40a7f8473` | 473 / 481 | language 284, vision 171, audio 18 | — | tensors renamed |

39 of the store's 54 tags are byte-identical to the registry, 4 changed, 10 are local
builds the registry never served. The three gemma4 `nvfp4` tags all moved the same way:
the vision path is no longer quantised. On `qwen3.5:0.8b-mlx` the tensor *names* changed
too, so almost every layer differs and no size ratio is meaningful.

**What this costs us.** Every gemma4 vision measurement in this repo that names
`gemma4:*-nvfp4` — the #312 encoder work, the vision goldens in
`x/mlxrunner/testdata`, the fine-text tiers, this ladder — was taken against a
**quantised tower**. A host that pulls that tag today gets a bf16 tower and will not
reproduce them. Production is unaffected: `:11497` serves from the store as it stands,
and nothing re-pulls on its own.

**What to do about it.**

1. Cite the manifest digest, not the tag, in anything that records a number.
   `python3 vision-suite/store_audit.py --digests gemma4:31b` prints them.
2. Run the audit before trusting a cross-host comparison — the other host may hold a
   different artifact under the same name. That is the first thing the Metal session
   should run below.
3. Do not `ollama pull` a tag this repo measures without checking the audit first. A
   pull is silent and there is no local history to roll back to.

There is a second reading, and it is the interesting one: upstream moved gemma4's
vision path off `nvfp4` at some point after we pulled. That is the same direction as
[#312](https://github.com/MaxusAI/ollama/issues/312), where the Metal `fp_qmm_t` bug
(`K % 32 == 16`, which the tower's `mlp.down_proj` at K = 4304 hits) corrupted exactly
those weights. Whether upstream did it for that reason or for quality, a bf16 tower
sidesteps the whole class of problem.

---

## What is measured

The repo's own [`vision-suite/extbench.py`](vision-suite/extbench.py), the same tool and
the same scoring the Metal number came from: OCRBench's `echo840/OCRBench` test split,
rows 0–200, gold answer contained in the prediction (lmms-eval's `ocrbench` semantics,
including its whitespace-stripped handwritten-maths case).

```bash
LIMIT=200 OFFSET=0 THINK=false NUM_CTX=8192 NUM_PREDICT=512 SLEEP=1 TIMEOUT=1800 \
  python3 docs/maxusai/vision-suite/extbench.py http://127.0.0.1:11521 \
    <tag> gemma4:31b-nvfp4 ocrbench
```

| setting | value | why |
|---|---|---|
| items | 200, offset 0 | the slice the Metal run used |
| think | off | OCRBench answers are a word or a phrase |
| sampling | temperature 0, `apply_sampling=False` | the arms must differ by weights, not by sampling |
| `num_ctx` | 8192, every arm | one image and a short question; identical across arms so no arm truncates |
| endpoint | `/api/generate` | what the published slices used |
| loaded models | one at a time, unloaded between arms | GPU0 is shared; a resident model skews the next arm |
| engine | the deployed build `0.34.1-dynres-16-g16649e8` | GGUF arms therefore decode the image in one batch (ADR 0036) |

Each arm runs in a probe container on port 11521 against GPU0 with a 16 GiB overhead
reserve, never against `:11497`.

**The ladder varies one thing at a time — but only within an engine.** The library's
three MLX tags (`31b-nvfp4`, `31b-mxfp8`, `31b-mlx-bf16`) carry an **identical bf16
vision tower** and differ only in the language model. The three GGUF tags
(`31b-it-q4_K_M`, `31b-it-q8_0`, `31b-it-bf16`) carry an **F16 tower** in all three and
differ only in the language model — verified from the GGUF tensor types: 191 `v.*`
tensors at F16 and 165 at F32 in both `q4_K_M` and `q8_0`. Our local
`gemma4:31b-nvfp4`, the one the Metal comparison is against, is the exception: its tower
is nvfp4 too. It is reported as its own row, because it differs from the library's
`nvfp4` in the tower and not in the language model.

---

## Results — `mlx-cuda` (RTX PRO 6000 Blackwell, sm_120)

<!-- GENERATED by vision-suite/ocrbench_table.py; re-render, do not hand-edit. -->

| arm | engine | model | correct / scored | accuracy | ±1 s.e. | mean s/item | median | prompt_eval |
|---|---|---|---|---|---|---|---|---|
| nvfp4 (quantised tower) | mlx-cuda | `gemma4:31b-nvfp4` | 172 / 200 | **0.860** | 0.025 | 2.3 | 2.0 | 1115 |
| nvfp4 (bf16 tower) | mlx-cuda | `gemma4:31b-nvfp4` (library) | _pending_ | | | | | |
| mxfp8 | mlx-cuda | `gemma4:31b-mxfp8` | _pending_ | | | | | |
| bf16 | mlx-cuda | `gemma4:31b-mlx-bf16` | _pending_ | | | | | |

## Results — llama.cpp GGUF (same host, same GPU)

| arm | engine | model | correct / scored | accuracy | ±1 s.e. | mean s/item | median | prompt_eval |
|---|---|---|---|---|---|---|---|---|
| q4_K_M | llama.cpp | `gemma4:31b-it-q4_K_M` | 171 / 200 | **0.855** | 0.025 | 5.0 | 4.9 | 1115 |
| q8_0 | llama.cpp | `gemma4:31b-it-q8_0` | _pending_ | | | | | |
| bf16 | llama.cpp | `gemma4:31b-it-bf16` | _pending_ | | | | | |

**Paired, the two arms measured so far** (same 200 items):

| A | B | A | B | b (A only) | c (B only) | p | resolved |
|---|---|---|---|---|---|---|---|
| nvfp4 (mlx-cuda) | q4_K_M (llama.cpp) | 0.860 | 0.855 | 3 | 2 | 1.000 | no |

Five items out of 200 separate the two engines, three one way and two the other. On the
95 % of items where both are right or both are wrong they agree exactly, so the engines
are not reading these images differently; they differ on a handful of hard ones, in both
directions. MLX answers in 2.3 s against llama.cpp's 5.0 s, at an identical mean
prompt_eval of 1115 tokens — the same image budget, so that gap is engine speed and not
a different amount of image.

## Harness note (2026-09-19)

Two arms died mid-run when a DNS blip made the row fetch fail, so `extbench.py` now
caches the row slice under `extimgs/<bench>/rows_<offset>_<limit>.json` and retries HTTP
with backoff. A slice is a fixed set of items: fetching it once per host, rather than
once per arm, removes the dependency and makes a re-run answer the same questions.
`REFRESH_ROWS=1` re-fetches. A partial fetch is never cached.

## Results — `mlx-metal` (for the Metal session)

The Metal host's own numbers go here. Preliminary, from Glenn, 2026-09-18: **175 / 200,
0.875**, described as "0.34.0 fixed kernel" — the artifact it was measured against is
not yet identified, which the audit above makes load-bearing.

| arm | engine | model | manifest digest | correct / scored | accuracy | mean s/item |
|---|---|---|---|---|---|---|
| nvfp4 | mlx-metal | `gemma4:31b-nvfp4` | | | | |
| mxfp8 | mlx-metal | `gemma4:31b-mxfp8` | | | | |
| bf16 | mlx-metal | `gemma4:31b-mlx-bf16` | | | | |

**To add yours:**

1. `python3 docs/maxusai/vision-suite/store_audit.py gemma4` on that host, and paste the
   result. It says whether your `gemma4:31b-nvfp4` is the quantised-tower artifact this
   ladder's first row uses or the library's current bf16-tower one. Without that line the
   number cannot be placed on the ladder.
2. Run the command in "What is measured" per arm, one model loaded at a time.
3. Render with
   `python3 docs/maxusai/vision-suite/ocrbench_table.py --engine mlx-metal "nvfp4=<tag>"`
   and paste the table into this section. Same generator, so the columns line up.
4. State the build the runner came from, as the CUDA rows do.

---

## How to read these numbers

- **A 200-item slice resolves about 5 points, not 1.** At p ≈ 0.87 the binomial standard
  error is 0.024, so 0.860 and 0.875 — 3 items — are the same number. Do not rank arms on
  the accuracy column.
- **The arms answer the same items, so compare them paired.** The generator prints the
  discordant counts and an exact McNemar p for every pair. That resolves differences the
  marginals cannot, and it is the only column that supports a claim like "q8 beats q4".
- **MLX is not bit-reproducible run to run** (despite [ADR 0012] §4), so every MLX arm is run at least twice and the repeat table
  reports the spread and how many items changed verdict. A gap smaller than an arm's own
  repeat spread is not a finding.
- **`mlx-cuda` and `mlx-metal` never share a table.** Different kernels, different
  silicon, and — per the audit above — possibly different weights.
- **Speed here is not a throughput claim.** `SLEEP=1` between requests and a shared GPU
  make the seconds column a sanity check, not a benchmark.

## Provenance

- Score files: `docs/maxusai/vision-suite/ext_ocr31b_*_ocrbench.json` (one record per
  item: prompt, prediction, gold, verdict, timings), gitignored, kept on the array.
- Runner logs: `preflight-runs/ocrladder*.log` and `ocrbench-runner.log`.
- Manifest digests of the CUDA arms:

  | model | manifest digest |
  |---|---|
  | `gemma4:31b-nvfp4` (this store, quantised tower) | `sha256:637cc0ff15709212de4aa694be67e3af5e80533ca538c86ad69c53c511e40840` |
  | `gemma4:31b-it-q4_K_M` | `sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` |
  | `gemma4:31b-it-q8_0` | `sha256:53dd8459790f8795177444daa9e33f417e03c0d1cdedb80b6c73898603d20aef` |

  The pulled library tags are kept in a separate store on the array
  (`claude-scratch/ocrbench-store`) so that pulling them cannot overwrite the artifacts
  the fork's existing measurements were taken against; their digests land here with their
  rows.

[ADR 0012]: adr/0012-benchmark-report-templates.md
