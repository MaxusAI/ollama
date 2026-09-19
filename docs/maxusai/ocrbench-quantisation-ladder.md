# OCRBench across the quantisation ladder: gemma4:31b

> Part of the fork's OCRBench set — shared method, format and cross-host results in
> [ocrbench.md](ocrbench.md).

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
   `python3 vision-suite/store_audit.py --digests gemma4:31b` prints them. This is now
   [ADR 0037](adr/0038-a-model-is-identified-by-its-manifest-digest.md) and SPEC
   `vision-harness-reuse` H15.
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

## Which checkpoint produced the 0.860, and when the Metal number is comparable

The `mlx-cuda` 0.860 came from **this store's `gemma4:31b-nvfp4`, manifest
`sha256:637cc0ff15709212de4aa694be67e3af5e80533ca538c86ad69c53c511e40840`, pulled
2026-08-17** (blob mtimes; the manifest was written 00:57 that morning). Its vision tower
is **4-bit**: `model.vision_tower.encoder.layers.*.mlp.down_proj.linear.weight` is
2.79 MB per layer, which is nvfp4 for a [1152 × 4304] tensor. bf16 would be 9.92 MB, and
that is what the library serves under the same tag today
(`sha256:a22a363052da…`, the separate row in the table below).

**So the Metal 0.875 is comparable to the 0.860 only if the Metal host also holds a
4-bit tower.** The check is one command on that host:

```bash
python3 docs/maxusai/vision-suite/store_audit.py gemma4
```

- `gemma4:31b-nvfp4` listed as **changed, vision ~191, 3.5× — nvfp4 → bf16**: the host
  holds the 4-bit-tower artifact, the same class as ours, and the numbers are comparable.
  `--digests` then gives the exact digest to record; if it is `637cc0ff1570…` it is
  byte-identical to ours.
- `gemma4:31b-nvfp4` **not listed** (identical to the registry): the host holds the
  bf16-tower artifact, and its number belongs on the `nvfp4 / bf16 tower` row instead —
  ours scored 0.850 there, not 0.860.

A host that pulled after the re-publish cannot get the 4-bit tower back: the registry
serves only the current content, and no other tag carries it. If the Metal host still has
it, that copy is an archive worth keeping — see [ADR 0037](adr/0038-a-model-is-identified-by-its-manifest-digest.md).

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
| items | 200, offset 0 | the slice the Metal run used — see the warning below |
| think | off | OCRBench answers are a word or a phrase |
| sampling | temperature 0, `apply_sampling=False` | the arms must differ by weights, not by sampling |
| `num_ctx` | 8192, every arm | one image and a short question; identical across arms so no arm truncates |
| endpoint | `/api/generate` | what the published slices used |
| loaded models | one at a time, unloaded between arms | GPU0 is shared; a resident model skews the next arm |
| engine | the deployed build `0.34.1-dynres-16-g16649e8` | GGUF arms **on this host** decode the image in one batch (ADR 0036). That is a per-host fact, not a property of the build: read the logged `num_batch`. On gfx1151 the same build is refused the floor at every quantisation — 1024 for q4 at `-np 1`, 512 for q8 and bf16 — because `availableMemoryForLoad` takes its integrated-GPU branch and sizes the batch against 31 GiB of system RAM while the scheduler logs `available="95.4 GiB"` of GPU ([ocrbench-gemma4-quant-ladder.md](ocrbench-gemma4-quant-ladder.md#why-the-batch-differs-and-why-adr-0036-never-fires-here)) |

Each arm runs in a probe container on port 11521 against GPU0 with a 16 GiB overhead
reserve, never against `:11497`.

**What rows 0–200 actually contain.** The dataset is ordered by task, so this slice is
**not a sample of OCRBench** — it is four of its ten categories, 50 items each:

| Regular Text Recognition | Irregular Text Recognition | Artistic Text Recognition | Handwriting Recognition |
|---|---|---|---|
| 50 | 50 | 50 | 50 |

Digit-string, non-semantic text, scene-text VQA, doc-oriented VQA, key-information
extraction and handwritten maths are **absent**. So every number here — ours and the
Metal 0.875 — is an accuracy on OCRBench's *text-recognition half*, and must not be
called an OCRBench score or compared with a published one (those are out of 1000 across
all ten categories). It is still the right slice for this question, because the arms are
being compared with each other on identical items. Widening to the full 1000, or to a
stratified 20-per-category slice, is the obvious follow-up and costs about 80 minutes per
arm at these rates.

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

<!-- GENERATED by vision-suite/summarize_extbench.py; re-render, do not hand-edit. -->

| arm | engine | model | correct / scored | accuracy | ±1 s.e. | mean s/item | median | prompt_eval |
|---|---|---|---|---|---|---|---|---|
| nvfp4 / nvfp4 tower (run 1) | mlx-cuda | `gemma4:31b-nvfp4` | 172 / 200 | **0.860** | 0.025 | 2.3 | 2.0 | 1115 |
| nvfp4 / nvfp4 tower (run 2) | mlx-cuda | `gemma4:31b-nvfp4` | 172 / 200 | **0.860** | 0.025 | 2.2 | 2.0 | 1115 |
| nvfp4 / bf16 tower (run 1) | mlx-cuda | `gemma4:31b-nvfp4` (library) | 170 / 200 | **0.850** | 0.025 | 1.4 | 1.4 | 1115 |
| nvfp4 / bf16 tower (run 2) | mlx-cuda | `gemma4:31b-nvfp4` (library) | 170 / 200 | **0.850** | 0.025 | 1.5 | 1.5 | 1115 |
| mxfp8 / bf16 tower (run 1) | mlx-cuda | `gemma4:31b-mxfp8` | 169 / 200 | **0.845** | 0.026 | 1.3 | 1.2 | 1115 |
| mxfp8 / bf16 tower (run 2) | mlx-cuda | `gemma4:31b-mxfp8` | 169 / 200 | **0.845** | 0.026 | 1.3 | 1.2 | 1115 |
| bf16 | mlx-cuda | `gemma4:31b-mlx-bf16` | **not run** — refused admission, see below | | | | | |

**Repeats**

| arm | runs | accuracies | items that flipped |
|---|---|---|---|
| nvfp4 / nvfp4 tower | 2 | 0.860, 0.860 | 0 |
| nvfp4 / bf16 tower | 2 | 0.850, 0.850 | 0 |
| mxfp8 / bf16 tower | 2 | 0.845, 0.845 | 0 |

Not one item changed verdict between runs on any MLX arm either. On this workload — one
image, a short answer, temperature 0, no drafting under a grammar — the engine's
run-to-run movement does not appear at all, which is a stronger control than
[ADR 0012] §4's caveat led us to expect.

**Paired**

| A | B | A | B | b (A only) | c (B only) | p | resolved |
|---|---|---|---|---|---|---|---|
| nvfp4 / nvfp4 tower | nvfp4 / bf16 tower | 0.860 | 0.850 | 3 | 1 | 0.625 | no |
| nvfp4 / nvfp4 tower | mxfp8 / bf16 tower | 0.860 | 0.845 | 4 | 1 | 0.375 | no |
| nvfp4 / bf16 tower | mxfp8 / bf16 tower | 0.850 | 0.845 | 1 | 0 | 1.000 | no |

**The MLX bf16 arm did not run: the scheduler refused it, and the reserve is why.**
Verbatim from the runner, 2026-09-19 07:28:

```
gpu memory id=0 library=CUDA available="62.7 GiB" free="79.2 GiB" minimum="457.0 MiB" overhead="16.0 GiB"
Load failed error="model requires 75.0 GiB (weights 59.1 GiB + KV cache 1.4 GiB at num_ctx 8192
+ 14.5 GiB headroom) but only 62.7 GiB are available (after 16.4 GiB overhead)"
```

Other tenants held 15.8 GiB of the card, and the 16 GiB reserve ([ADR 0034](adr/0034-mlx-admission-prices-the-context-rung.md)
prices MLX admission as weights + KV + a per-architecture headroom) takes the budget to
62.7 GiB against a 75.0 GiB ask — short by 12.3 GiB. Lowering `num_ctx` does not close it:
KV is 1.4 GiB of the 75. It was not forced, because dropping the reserve on a shared card
to fit one benchmark arm trades other people's work for a row in this table.

It is not impossible, only not now: with the card otherwise idle the same ask is 75.0 GiB
against about 78.6 GiB available **with the reserve intact**, so the arm runs whenever
GPU0 is quiet. The GGUF bf16 arm did run, at the same 58 GiB of weights, because
llama.cpp's admission prices weights + KV without MLX's headroom term — which is worth
noting on its own: two engines, the same model size, one admitted and one refused.

**Quantising the vision tower costs 39 % of the time per image and buys nothing here.**
The two `nvfp4` rows share a language model and differ only in the tower: ours is nvfp4,
the library's is bf16, and the bf16 one answers in 1.4 s against 2.3 s on identical
images at an identical 1115-token mean prefill. The bf16 tower is 3.5× the bytes, so this
is not weight bandwidth — the tower is about half a gigabyte either way. It is kernel
throughput: the image encode is compute-bound, and MLX's quantised matmul on sm_120 runs
at roughly half the rate of the dense bf16 path, which is what
[`sm120-mixed-input-gemm`](../../x/mlxrunner/bench/qqmm) measured directly. Accuracy does
not move with it: three items separate the two towers, p = 0.625.

Language-model precision does not move accuracy either — `nvfp4` and `mxfp8` over the
same bf16 tower differ on a single item out of 200, at 1.4 s against 1.3 s.

**By question type** (first run of each arm)

| question type | n | nvfp4 / nvfp4 tower | nvfp4 / bf16 tower | mxfp8 / bf16 tower |
|---|---|---|---|---|
| Artistic Text Recognition | 50 | 49/50 | 49/50 | 48/50 |
| Handwriting Recognition | 50 | 34/50 | 33/50 | 33/50 |
| Irregular Text Recognition | 50 | 40/50 | 39/50 | 39/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 | 49/50 |

## Results — llama.cpp GGUF (same host, same GPU)

| arm | engine | model | correct / scored | accuracy | ±1 s.e. | mean s/item | median | prompt_eval |
|---|---|---|---|---|---|---|---|---|
| q4_K_M (run 1) | llama.cpp | `gemma4:31b-it-q4_K_M` | 171 / 200 | **0.855** | 0.025 | 5.0 | 4.9 | 1115 |
| q4_K_M (run 2) | llama.cpp | `gemma4:31b-it-q4_K_M` | 171 / 200 | **0.855** | 0.025 | 4.9 | 4.8 | 1115 |
| q8_0 (run 1) | llama.cpp | `gemma4:31b-it-q8_0` | 170 / 200 | **0.850** | 0.025 | 5.1 | 5.0 | 1115 |
| q8_0 (run 2) | llama.cpp | `gemma4:31b-it-q8_0` | 170 / 200 | **0.850** | 0.025 | 5.2 | 4.8 | 1115 |
| bf16 (run 1) | llama.cpp | `gemma4:31b-it-bf16` | 171 / 200 | **0.855** | 0.025 | 5.0 | 4.9 | 1115 |
| bf16 (run 2) | llama.cpp | `gemma4:31b-it-bf16` | 171 / 200 | **0.855** | 0.025 | 4.8 | 4.8 | 1115 |

All 61 layers of the bf16 model sit on the GPU (58 GiB of weights, 1.8 GiB of KV at
`num_ctx` 8192 — gemma4's sliding-window layers keep the cache small), leaving 19 GB free
on a card shared with other tenants, so no arm here ran partly on CPU.

**Repeats — the GGUF path is item-level deterministic**

| arm | runs | accuracies | items that flipped |
|---|---|---|---|
| q4_K_M | 2 | 0.855, 0.855 | 0 |
| q8_0 | 2 | 0.850, 0.850 | 0 |
| bf16 | 2 | 0.855, 0.855 | 0 |

Not one of 200 items changed verdict between runs on either arm. That is the positive
control the rest of the ladder needs: on this engine a difference between arms is the
weights, not run noise.

**Paired, the arms measured so far** (same items)

| A | B | A | B | b (A only) | c (B only) | p | resolved |
|---|---|---|---|---|---|---|---|
| nvfp4 (mlx-cuda) | q4_K_M (llama.cpp) | 0.860 | 0.855 | 3 | 2 | 1.000 | no |
| q4_K_M | q8_0 | 0.855 | 0.850 | 1 | 0 | 1.000 | no |
| q4_K_M | bf16 | 0.855 | 0.855 | 2 | 2 | 1.000 | no |
| q8_0 | bf16 | 0.850 | 0.855 | 1 | 2 | 1.000 | no |

**The GGUF ladder is flat from 4 bits to bf16.** `q4_K_M` and `bf16` score the same 171
of 200 and differ on four items, two each way; `q8_0` sits one item below both. Nothing
resolves, and the seconds per item are the same across all three — this workload is
prefill-bound with an eight-token answer, so tripling the weights costs no clock. Whatever
4-bit quantisation does to this model, it is not visible in text recognition, which is the
same conclusion the Qwen2.5-VL campaign reached at 32B. Against MLX, five items separate the engines, three one way and two the
other; where both are right or both are wrong they agree exactly, so the engines are not
reading these images differently, they differ on a handful of hard ones in both
directions. MLX answers in 2.3 s against llama.cpp's 5.0 s at an identical mean
prompt_eval of 1115 tokens — the same image budget, so that gap is engine speed, not a
different amount of image.

**By question type** (first run of each arm), from `summarize_extbench.py --categories`:

| question type | n | mlx-cuda nvfp4 | q4_K_M | q8_0 | bf16 |
|---|---|---|---|---|---|
| Artistic Text Recognition | 50 | 49/50 | 49/50 | 48/50 | 48/50 |
| Handwriting Recognition | 50 | 34/50 | 33/50 | 33/50 | 34/50 |
| Irregular Text Recognition | 50 | 40/50 | 40/50 | 40/50 | 40/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 | 49/50 | 49/50 |

Regular and artistic text are at ceiling for every arm. All the headroom is in
handwriting (16–17 misses) and irregular text (10 misses, the same count on all three),
and no arm is more than one item better than another in any category. If a quantisation
cost exists at 31b, this slice does not show it in text recognition.

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
   `python3 docs/maxusai/vision-suite/summarize_extbench.py --paired --repeats --timing --categories ocrbench "nvfp4=<tag>,<repeat>"`
   and paste the table into this section. Same generator, so the columns line up.
4. State the build the runner came from, as the CUDA rows do.
5. Label the number for what it is: rows 0–200 are OCRBench's four text-recognition
   categories, not the benchmark. `--categories` prints the per-type split.

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
  the fork's existing measurements were taken against:

  | model (library, pulled 2026-09-19) | manifest digest |
  |---|---|
  | `gemma4:31b-nvfp4` (bf16 tower) | `sha256:a22a363052da770302019d5afab29bd967f9318e4377bddd297421189ea09d7d` |
  | `gemma4:31b-mxfp8` | `sha256:9740f018f0d6bb64b439b6d701017bac85675511d519eae18992b7ed09d0dc6a` |
  | `gemma4:31b-it-bf16` | `sha256:236d76ae08745dbc143c31b9271b0f25750885199aa6039d0fc0113171606e6d` |

[ADR 0012]: adr/0012-benchmark-report-templates.md
