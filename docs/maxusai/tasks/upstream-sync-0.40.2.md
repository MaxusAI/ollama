# TASK: fold upstream v0.40.2 into main

Upstream [v0.40.2](https://github.com/ollama/ollama/releases/tag/v0.40.2) (tag `b061384d9`, 2026-10-08) folds on top
of `main` at `5273e495f` (#459), which carries the v0.35.0 fold.
- **Size:** v0.35.0..v0.40.2 is 38 commits plus 11 merges, 239 files, +31,564/−2,277. That is 106 files added and
  133 modified.
- **Where:** branch `task/upstream-sync-0.40.2`, worktree `.claude/worktrees/upstream-sync-0402` on the Metal host
  (`macbook-pro-m5-max-128GB/mlx-metal`).

**Scope, the maintainer's call (2026-10-10):**
- One fold, straight to v0.40.2, taking v0.35.1, v0.40.0 and v0.40.1 on the way.
- v0.35.1 is not folded on its own. Its pins (llama.cpp b11232, MLX `64ea011c`) are already superseded, so a separate
  0.35.1 fold would measure a payload nobody serves.

## Status (2026-10-11, 09:45 AEST)

| gate | state |
|---|---|
| 1, the merge | **done**: `b6f672df2`, 10 conflicted files ("The conflicts and their resolution" below). `go build` and `go vet` are clean on all 85 non-app packages, and `go test` passes the 52 that need no GPU (38 with tests). The fork's delta carries exactly, apart from the resolved files and one gofmt realignment |
| 2, docs and paths | **clean**: `check_source_paths.py --changed-since origin/main` resolves every reference. The full-tree run's 7 unresolved references are already on `main` |
| 3, the patch series | **done**: `948ef3aab`. At b11351 the series applied through 802; 805 and 903 failed and are re-cut ("Gate 3" below). All ten patches apply in the applier's order to a clean b11351, and in the real build |
| no-GPU harness gates | **green**: `test_verdicts.py` 238 OK, `test_summarizers.py`, `test_rescore.py` (1 skipped, as in CI), `test_mlx_test_gate.py` |
| 4, image | **built on Metal**: `0.35.0-dynres-27-g948ef3a` (an interim stamp: no `v0.40.2-dynres` tag yet), with llama-server at `631109b34` (b11351). `CLEAN_DEPS=1`, 3 min 40 s. A full native build: this fold cannot be a Go-only swap |
| 5, preflight | **Metal: PASS=23 SKIP=16** on the interim build, against the new `mlx-metal-0-40-2` ("Gate 5 on Apple Silicon" below). CUDA and ROCm: not run. Every profile needs a new measurement, because both payloads move (ADR 0032) |
| 6, campaigns | **Metal: done** ("Gate 6 on Apple Silicon" below). OCRBench rows 0–199 are 200/200 byte-identical, and five of the seven think-off cells are 28/28 byte-identical. Both qwen MLX cells moved, attributed to upstream's MLX tokenizer fix. CUDA and ROCm: each host's call |
| tag and deploy | **tagged** `v0.40.2-dynres` on `5623192cc` (the merge of #460, 2026-10-11). **Apple Silicon deployed it** at 09:29 AEST ("The deploy on Apple Silicon" below). CUDA and gfx1151: not deployed |

## Native inputs

Both payloads move, so ADR 0032 calls for new profiles, not widened ones.

| input | `main` | v0.35.1 | v0.40.2 |
|---|---|---|---|
| `LLAMA_CPP_VERSION` | b11081 | b11232 | **b11351** |
| `MLX_VERSION` | `59d600b5` | `64ea011c` | **`a59cc231`** |
| `MLX_C_VERSION` | `ebc88f10` | `ebc88f10` | `ebc88f10` |

Upstream's MLX carry patches are now applied through one `cmake/local.cmake` command for MLX and MLX-C.
- `cf6c9de62` carried a Metal residency-refresh patch.
- `cf2a313a2` dropped it again, because `a59cc231` contains it.

## What v0.40.x changes for the fork, as read so far

Each item here is a question for the fold, not yet an answer.

- **Runner-specific manifests in one tag, with MLX preferred on Apple Silicon** (`75b952780`).
  - A tag can now hold one child manifest per runner (`mlx`, `llamacpp`, `ggml`).
  - `runnerPreferences()` (`manifest/manifest.go:844`) orders them `mlx, llamacpp, ggml` on darwin/arm64, and
    `llamacpp, ggml, mlx` everywhere else.
  - So **one tag can resolve to different weights on the Metal host than on the CUDA and ROCm hosts**. Comparing hosts
    by tag, the "q4_K_M/q8 tags are GGUF" convention, and checkpoint identity by manifest digest all need a rule for
    which child was resolved.
- **Background conversion of legacy GGUFs** (`75b952780`, `compatmigrate/`).
  - On first load, a legacy (`ggml`) GGUF is converted into a `llamacpp` child, and the original is kept as a backup.
  - The converter covers every GGUF family the Metal host serves: gemma4, qwen35/qwen35moe, nemotron_h_omni,
    qwen3vl (chandra) and qwen2vl (olmocr2).
  - On that host, the production store holds 14 GGUF model files, 270 GiB in all, with 696 GiB free. A deploy would
    convert each model on its first load, add up to that much on disk, and change the file llama-server loads.
  - GGUF baselines taken on a converted model are new measurements.
- **The MLX tokenizer now matches the original tokenizers' semantics** (`195f4cdca`, +2,180/−495). This covers
  pretokenizer order, split behaviour, Unicode boundaries, added-token normalization and ranked BPE merges.
  **Measured: it fixes qwen on MLX.** Production's tokenizer matches 33 of upstream's 47 qwen reference cases, the
  fold's matches 47 of 47, and the qwen think-off answers move accordingly ("What moved the qwen cells" below). The
  preflight ladders do not move.
- **Slow first request after GPU idle** (`cf6c9de62`, upstream #18744). The MLX runner now sets
  `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` unless the environment sets it. Fork campaigns wait for 15 idle
  minutes before measuring, so whether their first cells paid this cost is worth checking. Not yet measured.
- **New MLX surface:**
  - System One and decision models on MLX (`a00d083bb`, `27a36513d`): Laya, Clef/Clef Flash, Tev1, Nimble and
    Strands.
  - Multimodal embeddings with `embeddinggemma-2` (`efe43c556`), which reuses gemma4's vision and audio towers.
- **System One now requires a decision model** (v0.35.1: #18708, #18737). `/v1/systemone` accepts only a model
  that declares the `decision` capability (GGUF `decision.type`, or an explicit capability at create). Production's
  0.35.0 answers System One on qwen3.6:35b-a3b-q4_K_M, the one model on the qwen3.5 renderer. On this fold it
  answers 400, "does not support decision", as upstream intends. No fork tool calls System One.
- **Outside the runners:** OpenAI tool-result handling (four commits), cloud usage/balance proxying, CLI onboarding,
  and `ollama launch claude` context settings.

## The conflicts and their resolution

The merge of v0.40.2 into `5273e495f` conflicted in 10 files, nine of them in `mlxrunner/`. v0.35.1 alone would
have merged with none.

- **`mlxrunner/runner.go`: both sides.** The fork's "report first, stop second" exit on a `fatalRunnerError` stays
  in the completion case, and upstream's new embedding case follows it. `runEmbed` never returns a
  `fatalRunnerError`, so the embedding case only logs, as upstream wrote it.
- **`mlxrunner/client.go`: both sides.** The fork's `Client` fields (the atomic `softContextLength`, `numCtxAuto`,
  `kvEstimate`) plus upstream's `embeddingDimensions`.
- **`server/sched_test.go`: the fork's form.** The fork runs this test under `synctest`, with plain receives and no
  context timeout. Upstream's change moves `b.ctxDone()` after the VRAM report; the resolution takes that move into
  the fork's form.
- **`mlxrunner/model/gemma4/`, seven files: the fork's pipeline, as in the v0.34.4 fold.** That fold kept upstream's
  per-image resolution out (ADR 0003, 0008, 0021), and upstream's `audio.go`, `process_audio.go`,
  `process_image.go` and their tests stayed deleted. They stay deleted here, along with two new upstream tests of
  that API (`multimodal_export_test.go`, `process_image_test.go`). `gemma4.go` auto-merged and takes upstream's
  `rope_type` check.

### embeddinggemma-2 is not in this fold

Upstream's multimodal embedding model (#18820, `efe43c556`) is built on gemma4's upstream towers. It calls
`gemma4.LoadVisionTower`, `ProcessImage`, `ImageGeometry`, `LoadAudioTower` and `ProcessAudio`. None of those
exists in the fork, whose gemma4 vision code is its own (`VisionTower.Forward` over patches, the budget-fill
pipeline). Taking it would mean one of two things:
- reversing the v0.34.4 decision on gemma4's image pipeline, which needs an ADR with a measurement;
- carrying upstream's towers a second time, beside the fork's, for one model.

So `mlxrunner/model/gemma4embedding/` is removed and unregistered. The generic `/api/embed` media path,
`mlxrunner/embed.go` and `create/gemma4embedding.go` stay. On this fold, an `embeddinggemma-2` model fails to load
with an unknown architecture. No host serves it. **Deferred**, with the gemma4 pipeline question it depends on.

### The carry

The fork's delta against v0.35.0 before the fold, compared line for line with its delta against v0.40.2 after it,
differs only in the resolved files above and in one gofmt realignment of a map literal in
`server/routes_generate_test.go`.

## Gate 3: the patch series at b11351

Applied in the applier's order (`cmake/apply-git-patches.cmake`, sorted by name) to a clean b11351 checkout:

| patch | at b11351 |
|---|---|
| 001 hooks, 002 nemotron dynres, 004, 005, 801, 802, 908 | apply as they are |
| 002-clef (upstream's, new) | applies, before the fork's 002. The two touch different files |
| 805 fattn force-tile | **re-cut, context only**. Upstream templated `flash_attn_mask_to_sparse_indices` and moved the kernel selector down 12 lines. The added and removed lines are identical to the b11081 cut |
| 903 MMQ padding | **re-cut, one semantic change**, below |

**903 is still needed.** b11351 still sizes both src1 allocations with `ggml_cuda_mmq_get_J_max()`.

One hunk failed on context, and b11351 changes one thing 903 depends on: `ggml_cuda_mmq_get_config()` gains a
`prec_src1` argument. On Blackwell, NVFP4 and MXFP4 now use the ampere table unless src1 is Q4. The re-cut
computes `J_pad` after `prec_src1` and passes it, so the padding comes from the table the launch uses, not the
default-precision one. Otherwise its added and removed lines match the b11081 cut.

The compile-time guard's table functions keep their `(type, J, fallback)` signature at b11351. **Not compiled
here**: the Metal host has no nvcc or hipcc, so the CUDA and ROCm hosts' builds are where the guard runs.

## On Apple Silicon

Every GPU step ran on the fold's own stage: `~/.ollama/binaries/stage-fold-0402`, holding its binary and its
`lib/ollama`. It ran under production's gate, which means production idle for 15 minutes and powermode 2, with a
step cancelled and re-run if production woke.

**The store is an APFS clone, `~/.ollama/models-mlx-fold0402` (`cp -cR`), not production's.** v0.40.2 converts a
legacy GGUF on its first load and writes the result into the store. Against production's own store, validation
would have rewritten production's manifests while 0.35.0 served them. The clone cost no disk until the conversion
wrote.

### The native gate

`go test ./mlx/... ./mlxrunner/... -p 1` with `OLLAMA_VISION_E2E=1`, on the fold's payload: **1,208 passed,
0 failed, 9 skipped**, 25 packages. `TestVisionGoldenParity` and `TestVisionEndToEnd` pass on real weights. The
0.34.4 and 0.35.0 runs passed 806, but they ran `./mlxrunner/...` alone, so the counts are not comparable. This run
adds `./mlx/...`, plus upstream's new System One, Laya, Clef and Strands tests.

### GGUF conversion, and the rollback it has to survive

One 1-token request per model on the fold's stage, then on production's deployed 0.35.0 build
(`0.35.0-dynres-0-g043f441`) against the same converted clone:

| model | converted in | the fold then loads | 0.35.0 afterwards |
|---|---|---|---|
| gemma4:31b-it-q4_K_M | 24 s | `sha256-3bf4a777…`, 17.4 GiB, new | answers, from the original `sha256-280af683…` |
| qwen3.6:35b-a3b-q4_K_M | 19 s | `sha256-3f87bb0b…`, 21.5 GiB, new | answers, from the original `sha256-f5ee307a…` |

- **Disk.** Each converted model is a second full copy: 17.4 + 21.5 GiB of model blobs, plus 1.9 GiB of other new
  blobs. The 14 GGUF files in production's store hold 270 GiB, and a deploy would convert each one on its first
  load. The original stays as the downgrade anchor.
- **The anchor is how rollback works.** The legacy manifest stays where it was and gains three
  `manifest.v2+json` layers. 0.35.0 reads it and loads the original blob.
- **One rollback wart, cosmetic.** 0.35.0 lists each converted child as its own model,
  `llamacpp:<digest>`: 29 models on the converted clone against production's 27. v0.40.2 hides them (#18874).

### Gate 5 on Apple Silicon

**`mlx-metal-0-40-2` is a new profile, measured on this payload.**
- The four ladders were measured with `measure_ladder.py`, one fresh stage per arch, and are byte-identical to 0-34-4's.
- Every engine init reported MLX `0.32.3-53-ga59cc23`, and llama-server reports `631109b34`.
- Its pattern admits only the `v0.40.2-dynres` tag's lineage.

**The interim build was run against a campaign copy of the file.** The interim build stamps `0.35.0-dynres-27-g948ef3a`,
which falls in 0-34-4's lineage. The campaign copy's only change pinned 0-40-2's pattern to that one stamp. Parsed and
compared field by field, once, before the run, the copy and the branch file agree everywhere else.

Preflight `--quality` on the stage: **PASS=23 SKIP=16**, the same shape as the v0.35.0 release run. The 16 skips
are the native path's usual ones: no container, no load_hparams line, and no quality, pinned-budget or poison
expectation recorded for these arches.

### The smoke

| request | result |
|---|---|
| think+format, two-pass, on MLX (`gemma4:26b-nvfp4`) | pass one drafted, 110 of 138 accepted; content parses as JSON |
| think+format, two-pass, on the converted GGUF (`gemma4:31b-it-q4_K_M`) | content parses as JSON; llama-server loads the converted blob |
| an image on the converted GGUF | `prompt_eval_count` 573 and a correct description, so the projector survived the conversion |
| System One on `qwen3.6:35b-a3b-q4_K_M` | 400, "does not support decision", as upstream intends since v0.35.1 |

### Gate 6 on Apple Silicon

This is what the v0.35.0 leg ran against the build production serves (`r0350fold_1`, `ocrk0350fold_c0`): the same
suite tree, driver, order and settings, with every request gated. No request overlapped production. Both payloads
moved, so byte equality was a finding here, not the bar.

**OCRBench v1, rows 0–199, gemma4:31b-nvfp4:**
- 174/200 correct on both builds;
- **200/200 predictions byte-identical**, with identical prompt and answer token counts.

**The seven think-off cells** (`drv/cmp_thinkoff_0402.py`; A is production's 0.35.0, B the fold):

| cell | answers identical | A: contract, hits, mean IoU | B: contract, hits, mean IoU |
|---|---|---|---|
| gemma4:12b-nvfp4 | 28/28 | 16/20, 96/120, 0.767 | the same |
| gemma4:26b-nvfp4 | 28/28 | 12/20, 72/120, 0.583 | the same |
| gemma4:31b-nvfp4 | 28/28 | 19/20, 114/120, 0.886 | the same |
| qwen3.6:35b-a3b-nvfp4 | **2/28** | 16/20, 99/120, 0.794 | **14/20, 92/120, 0.703** |
| qwen3.8:27b-nvfp4 | **8/28** | 19/20, 114/120, 0.936 | 19/20, 114/120, 0.937 |
| gemma4:31b-it-q4_K_M (converted GGUF) | 28/28 | 16/20, 97/120, 0.738 | the same |
| qwen3.6:35b-a3b-q4_K_M (converted GGUF) | 28/28 | 14/20, 89/120, 0.691 | the same |

Valid JSON is 189/189 on both builds.
- **The GGUF path did not move.** That is across llama.cpp b11081 → b11351 and the conversion to a llamacpp child.
- **Neither did gemma4 on MLX**, across MLX 59d600b5 → a59cc231.

**qwen3.6's quality change is two cases.** `bboxm_free_anc_named` and `bboxm_free_anc_pos` let the model choose its
box dialect. The fold's answers carry near-identical coordinates but declare `"bbox_type": "real"` where 0.35.0
declared `"norm1000"`. Read as pixels, the boxes score 1 of 6 hits at IoU 0.044, against 6 of 6 at 0.967. Two other
cases gained hits: `bbox_contract` 0 → 1 and `bbox_contract_real_1img` 0 → 2.

### What moved the qwen cells: the MLX tokenizer fix, not MLX

**The attribution run** used a hybrid stage: the fold's binary on production's MLX payload (engine init
`0.32.2-65-g59d600b`), running the same qwen3.6 cell (`attr-0402.sh`, tagged `r0402hyb_1`). Its answers equal the
fold's, **28/28**, and production's in 2/28. So the MLX move does not change these answers. The Go side does.

**The tokenizer is the Go-side change, and the fold's is the correct one.** Upstream rewrote the MLX tokenizer to
match the publishers' tokenizers (#18779, `195f4cdca`). Its `TestTokenizerReference` holds 47 qwen cases with the IDs
Hugging Face's tokenizers produce. Run with the qwen3.6 tokenizer from this store:
- production's v0.35.0 tokenizer matches **33/47**;
- the fold's matches **47/47**.

The 14 misses include runs of spaces before digits and punctuation, indented JSON (`{\n  "a": 12}`), NFC
normalization and combining marks. That is the shape of these prompts: seven qwen3.6 prompts change length by one or
two tokens. On v0.35.0, qwen on MLX has been reading prompts tokenized differently from how it was trained. The fold
fixes that, and the two dialect cases are near-ties that now land the other way.

gemma4's tokenizer does not take this path, and its cells did not move. qwen's GGUF cells tokenize in llama.cpp,
and they did not move either.

### The deploy on Apple Silicon

**The release build.** `0.40.2-dynres-0-g5623192` was built at the tag in a detached worktree, both halves, with
`CLEAN_DEPS=1`. All ten patches applied, and llama-server reports `631109b34`. The code equals the validated fold
build's, since `948ef3aab..5623192` touches only docs. The native libraries differ byte-wise from that build's, so
the release was checked by behaviour, on its own stage, against the APFS clone:

| check | result |
|---|---|
| preflight against the committed profile | PASS=23 SKIP=16, resolved by `mlx-metal-0-40-2` |
| the two-pass smoke | pass |
| the qwen3.6:35b-a3b-nvfp4 think-off cell, against the fold build's | 28/28 byte-identical (`r0402rel2_1`) |
| every GGUF model in the store | all 15 answer text, and every vision model describes the test image |
| rollback: 0.35.0 on the converted clone | every model answers, from its original |

**One check had to be re-run, and why.** The first run of the release's qwen cell (`r0402rel_1`) matched the fold's
in 16 of 28 answers. The 16 that matched are exactly the cases before a power-mode hold at 08:33. When the hold ends,
the driver restarts the server cold, so the 12 cases after it ran without the warm prefix state the fold's run had
at that point. Nothing drafted (no speculative-decode lines), and every prompt token count matched. Run again without
a hold, the cell matched 28/28. So a hold mid-cell changes greedy answers on this model, and the two runs are not
comparable.

**GGUF on v0.40.2, all 15 models in the store.**
- **Converted:** gemma4:26b-a4b-it-q4_K_M, gemma4:31b-it-q4_K_M, nemotron3:33b (bf16, q8, q4_K_M),
  qwen3.6:35b-a3b (q4_K_M, q8_0). Each took 19–65 s.
- **No conversion needed:** no migrator applies, so these run as they are. chandra (both), chandratables,
  gemma4:12b-it-q4_K_M, olmocr2 (q4_K_M, q8_0, and `richardyoung/olmocr2:7b-q8`), qwen3.8:27b-q4_K_M.
- **Conversion is silent when it does not apply.** `compatmigrate` returns without a log line, so a check must
  look for a "starting local compat GGUF migration" line rather than wait for a downgrade anchor.

**The swap**, production idle since 2026-10-01, single-purpose commands:
1. `launchctl bootout`;
2. `cp -p` the binary;
3. `rsync -a --delete` the payload into the checkout's `build/lib/ollama`;
4. `launchctl bootstrap`.

The plist is unchanged; its backup is `com.maxusai.ollama-fork.plist.0.35.0-dynres-0-g043f441`.

**After the swap:**
- `/api/version` reports `0.40.2-dynres-0-g5623192`.
- The process environment and the startup config carry `OLLAMA_FORMAT_TWO_PASS=1`, `OLLAMA_KV_CACHE_TYPE=f16`,
  `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` and the production store.
- Preflight on production, reading the log from the restart's `server config` line, gives PASS=23 SKIP=16.

Production's own GGUF models convert on their first load from now on.

**Rollback:**
1. `launchctl bootout`;
2. `cp -p ~/.ollama/binaries/ollama-0.35.0-dynres-0-g043f441` to the checkout's `ollama`;
3. `rsync -a --delete ~/.ollama/binaries/payload-0.34.4-dynres-0-gb43ee8e/` to `build/lib/ollama/`;
4. `launchctl bootstrap`.

Converted models stay converted, and 0.35.0 loads their originals through the anchor.

## Who does what

- **Metal host** (`macbook-pro-m5-max-128GB/mlx-metal`): gates 1–3, then gates 4–6 on Apple Silicon. GPU steps wait
  for production to be idle and for High Power mode, as usual.
- **CUDA and ROCm hosts:** gates 4–6 on their own hardware, when this branch is ready for them. No ask is sent until
  then.
