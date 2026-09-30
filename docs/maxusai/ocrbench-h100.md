# OCRBench on an H100 (sm_90): gemma4:31b on both engines, all 1000 items

On 2026-09-30, `0.34.4-dynres-0-gb43ee8e` ran the full `echo840/OCRBench` test split (1000
items, all ten categories) against `gemma4:31b` on one H100 SXM5 80GB: once through
llama.cpp with the GGUF `q4_K_M`, once through mlx-cuda with `nvfp4`. It is the first
1000-item OCRBench run on CUDA and the first on sm_90. The entry point for every OCRBench
number is [ocrbench.md](ocrbench.md).

## Result

`summarize_extbench.py --paired --timing --categories`, verbatim:

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 1000 | 0 | 0 | 825 | **0.825** | false | generate |
| `gemma4:31b-nvfp4` | 1000 | 0 | 0 | 832 | **0.832** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..1000.

host: http://127.0.0.1:11434 · build: 0.34.4-dynres-0-gb43ee8e

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| h100-gemma4-31b-it-q4_K_M vs h100-gemma4-31b-nvfp4 | 817 | 160 | 8 | 15 | 0.210 |

| arm | accuracy | ±1 s.e. | mean s/item | median | mean prompt_eval |
|---|---|---|---|---|---|
| h100-gemma4-31b-it-q4_K_M | 0.825 | 0.012 | 2.3 | 2.3 | 1122 |
| h100-gemma4-31b-nvfp4 | 0.832 | 0.012 | 1.2 | 1.1 | 1122 |

| question type | n | h100-gemma4-31b-it-q4_K_M | h100-gemma4-31b-nvfp4 |
|---|---|---|---|
| Artistic Text Recognition | 50 | 48/50 | 49/50 |
| Digit String Recognition | 50 | 29/50 | 31/50 |
| Doc-oriented VQA | 200 | 180/200 | 180/200 |
| Handwriting Recognition | 50 | 33/50 | 33/50 |
| Handwritten Mathematical Expression Recognition | 100 | 34/100 | 38/100 |
| Irregular Text Recognition | 50 | 40/50 | 39/50 |
| Key Information Extraction | 200 | 182/200 | 182/200 |
| Non-Semantic Text Recognition | 50 | 45/50 | 47/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 |
| Scene Text-centric VQA | 200 | 185/200 | 184/200 |

## What it says

**The engines do not separate.** 817 items right on both, 160 wrong on both, and 23 that
split 8 to 15, which the exact McNemar test puts at p = 0.210. No category moves by more than
four items. Each arm ran once, so there is no repeats control here; the two-run arms on
sm_120 flipped no items, which is a reason to expect little run-to-run movement, not a
measurement of it.

**Rows 0–200 tie.** Scored on the four recognition categories the 200-item rows use, both
arms get 170 / 200, disagreeing on one item each way. That is level with the sm_120
mlx-cuda arm on the same artifact (`nvfp4`, bf16 tower, 170 / 200) and one item behind
sm_120's GGUF `q4_K_M` (171 / 200), which is a different artifact (below).

**mlx-cuda answers in half the time.** 1.1 s median per item against 2.3 s for GGUF, over
identical prefills (mean prompt_eval 1122 in both arms), so the time is the engine, not the
input.

**The full-set mlx-metal arms are not a like-for-like comparison.** They scored 835 and 833
on 0.34.0 and 0.33.2; this document does not establish that they ran the same
`gemma4:31b-nvfp4` artifact (H17), so the gap of three items is not attributed.

## Setup

| setting | value |
|---|---|
| machine | GCP `a3-highgpu-1g`: one H100 SXM5 80GB (sm_90), Sapphire Rapids, 26 vCPUs, 230 GiB |
| PCIe | Gen4 x16 on a Gen5-capable GPU: the port above it tops out at Gen4 |
| build | `0.34.4-dynres-0-gb43ee8e`, a native build of `v0.34.4-dynres` with the CUDA 13 llama.cpp and MLX (`59d600b`) payloads |
| driver | 615.71.09, CUDA 13.4 |
| items | 1000, offset 0, all ten categories |
| think | off |
| sampling | temperature 0, `apply_sampling=False` |
| `num_ctx` | 8192, both arms |
| endpoint | `/api/generate` |
| loaded models | one at a time, unloaded between arms |
| runs per arm | 1 |
| GGUF image batch | `n_batch` = `n_ubatch` = 2048: ADR 0036's floor was granted, one batch |

The machine's full profile, in the `host-profile/1` format, is
`host-profile_h100-sm90-2026-09-30.json` beside the arms.

## Model identity (ADR 0038)

| tag | local manifest digest |
|---|---|
| `gemma4:31b-it-q4_K_M` | `sha256:17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97` |
| `gemma4:31b-nvfp4` | `sha256:a22a363052da770302019d5afab29bd967f9318e4377bddd297421189ea09d7d` |

The `nvfp4` digest is the registry manifest that
[ocrbench-quantisation-ladder.md](ocrbench-quantisation-ladder.md) identifies as the
re-publication with a bf16 vision tower ("library"). The `q4_K_M` digest is not the one the
ROCm ladder measured (`sha256:6316f062…`, [ocrbench-gemma4-quant-ladder.md](ocrbench-gemma4-quant-ladder.md)):
the tag moved between the two pulls, so the GGUF rows compare artifacts as well as hosts.

## Files and reproducing

The two arms are committed with the other published ones, as
`vision-suite/bench-runs/ocrbench/ext_h100-gemma4-31b-it-q4_K_M_ocrbench.json` and
`ext_h100-gemma4-31b-nvfp4_ocrbench.json`, with the host profile beside them. The tables
above render from them; the category split also needs extbench's cached row slice:

```bash
cd docs/maxusai/vision-suite
python3 summarize_extbench.py --dir bench-runs/ocrbench --paired --timing \
  --categories ocrbench h100-gemma4-31b-it-q4_K_M h100-gemma4-31b-nvfp4
```

Each arm is the command in [ocrbench.md](ocrbench.md#the-test) with `LIMIT=1000`. The slice's
images were all fetched before the first arm: the rows' image links expire an hour after the
rows are fetched, and an earlier attempt at this run lost both arms at item 682 to a 403.
