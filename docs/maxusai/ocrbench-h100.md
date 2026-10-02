# OCRBench on an H100 (sm_90): eight vision tags and a GGUF quantisation ladder, all 1000 items

On 2026-09-30, `0.34.4-dynres-0-gb43ee8e` ran the full `echo840/OCRBench` test split (1000
items, all ten categories) against `gemma4:31b` on one H100 SXM5 80GB: once through
llama.cpp with the GGUF `q4_K_M`, once through mlx-cuda with `nvfp4`. It is the first
1000-item OCRBench run on CUDA and the first on sm_90. On 2026-10-01 the same build on the
same machine ran the same test against the other six tags of the
[2026-09-30 vision campaign](vision-campaign-2026-09-30-h100-sm90.md), and against
`gemma4:31b-it-q8_0` and `gemma4:31b-it-bf16` to complete the GGUF ladder. The entry point for
every OCRBench number is [ocrbench.md](ocrbench.md).

## 2026-09-30: gemma4:31b on both engines

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

### What it says

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

## 2026-10-01: the campaign's eight tags

`summarize_extbench.py --paired --timing --categories`, verbatim. The first two arms are the
2026-09-30 ones.

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 1000 | 0 | 0 | 825 | **0.825** | false | generate |
| `gemma4:31b-nvfp4` | 1000 | 0 | 0 | 832 | **0.832** | false | generate |
| `gemma4:26b-nvfp4` | 1000 | 0 | 0 | 818 | **0.818** | false | generate |
| `gemma4:12b-nvfp4` | 1000 | 0 | 0 | 706 | **0.706** | false | generate |
| `qwen3.8:27b-nvfp4` | 1000 | 0 | 0 | 867 | **0.867** | false | generate |
| `qwen3.6:35b-a3b-nvfp4` | 1000 | 0 | 0 | 878 | **0.878** | false | generate |
| `qwen3.8:27b-q4_K_M` | 1000 | 0 | 0 | 857 | **0.857** | false | generate |
| `nemotron3:33b-q8` | 1000 | 0 | 0 | 879 | **0.879** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..1000.

host: http://127.0.0.1:11434 · build: 0.34.4-dynres-0-gb43ee8e

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| h100-gemma4-31b-it-q4_K_M vs h100-gemma4-31b-nvfp4 | 817 | 160 | 8 | 15 | 0.210 |
| h100-gemma4-31b-it-q4_K_M vs h100-gemma4-26b-nvfp4 | 788 | 145 | 37 | 30 | 0.464 |
| h100-gemma4-31b-it-q4_K_M vs h100-gemma4-12b-nvfp4 | 683 | 152 | 142 | 23 | 0.000 |
| h100-gemma4-31b-it-q4_K_M vs h100-qwen3.8-27b-nvfp4 | 751 | 59 | 74 | 116 | 0.003 |
| h100-gemma4-31b-it-q4_K_M vs h100-qwen3.6-35b-a3b-nvfp4 | 756 | 53 | 69 | 122 | 0.000 |
| h100-gemma4-31b-it-q4_K_M vs h100-qwen3.8-27b-q4_K_M | 762 | 80 | 63 | 95 | 0.013 |
| h100-gemma4-31b-it-q4_K_M vs h100-nemotron3-33b-q8 | 773 | 69 | 52 | 106 | 0.000 |
| h100-gemma4-31b-nvfp4 vs h100-gemma4-26b-nvfp4 | 789 | 139 | 43 | 29 | 0.125 |
| h100-gemma4-31b-nvfp4 vs h100-gemma4-12b-nvfp4 | 682 | 144 | 150 | 24 | 0.000 |
| h100-gemma4-31b-nvfp4 vs h100-qwen3.8-27b-nvfp4 | 760 | 61 | 72 | 107 | 0.011 |
| h100-gemma4-31b-nvfp4 vs h100-qwen3.6-35b-a3b-nvfp4 | 763 | 53 | 69 | 115 | 0.001 |
| h100-gemma4-31b-nvfp4 vs h100-qwen3.8-27b-q4_K_M | 767 | 78 | 65 | 90 | 0.054 |
| h100-gemma4-31b-nvfp4 vs h100-nemotron3-33b-q8 | 778 | 67 | 54 | 101 | 0.000 |
| h100-gemma4-26b-nvfp4 vs h100-gemma4-12b-nvfp4 | 684 | 160 | 134 | 22 | 0.000 |
| h100-gemma4-26b-nvfp4 vs h100-qwen3.8-27b-nvfp4 | 755 | 70 | 63 | 112 | 0.000 |
| h100-gemma4-26b-nvfp4 vs h100-qwen3.6-35b-a3b-nvfp4 | 755 | 59 | 63 | 123 | 0.000 |
| h100-gemma4-26b-nvfp4 vs h100-qwen3.8-27b-q4_K_M | 761 | 86 | 57 | 96 | 0.002 |
| h100-gemma4-26b-nvfp4 vs h100-nemotron3-33b-q8 | 774 | 77 | 44 | 105 | 0.000 |
| h100-gemma4-12b-nvfp4 vs h100-qwen3.8-27b-nvfp4 | 665 | 92 | 41 | 202 | 0.000 |
| h100-gemma4-12b-nvfp4 vs h100-qwen3.6-35b-a3b-nvfp4 | 670 | 86 | 36 | 208 | 0.000 |
| h100-gemma4-12b-nvfp4 vs h100-qwen3.8-27b-q4_K_M | 676 | 113 | 30 | 181 | 0.000 |
| h100-gemma4-12b-nvfp4 vs h100-nemotron3-33b-q8 | 671 | 86 | 35 | 208 | 0.000 |
| h100-qwen3.8-27b-nvfp4 vs h100-qwen3.6-35b-a3b-nvfp4 | 817 | 72 | 50 | 61 | 0.343 |
| h100-qwen3.8-27b-nvfp4 vs h100-qwen3.8-27b-q4_K_M | 806 | 82 | 61 | 51 | 0.395 |
| h100-qwen3.8-27b-nvfp4 vs h100-nemotron3-33b-q8 | 804 | 58 | 63 | 75 | 0.349 |
| h100-qwen3.6-35b-a3b-nvfp4 vs h100-qwen3.8-27b-q4_K_M | 798 | 63 | 80 | 59 | 0.089 |
| h100-qwen3.6-35b-a3b-nvfp4 vs h100-nemotron3-33b-q8 | 825 | 68 | 53 | 54 | 1.000 |
| h100-qwen3.8-27b-q4_K_M vs h100-nemotron3-33b-q8 | 800 | 64 | 57 | 79 | 0.071 |

| arm | accuracy | ±1 s.e. | mean s/item | median | mean prompt_eval |
|---|---|---|---|---|---|
| h100-gemma4-31b-it-q4_K_M | 0.825 | 0.012 | 2.3 | 2.3 | 1122 |
| h100-gemma4-31b-nvfp4 | 0.832 | 0.012 | 1.2 | 1.1 | 1122 |
| h100-gemma4-26b-nvfp4 | 0.818 | 0.012 | 1.4 | 1.3 | 1122 |
| h100-gemma4-12b-nvfp4 | 0.706 | 0.014 | 0.6 | 0.5 | 1122 |
| h100-qwen3.8-27b-nvfp4 | 0.867 | 0.011 | 1.0 | 0.5 | 863 |
| h100-qwen3.6-35b-a3b-nvfp4 | 0.878 | 0.010 | 0.9 | 0.3 | 863 |
| h100-qwen3.8-27b-q4_K_M | 0.857 | 0.011 | 1.2 | 0.9 | 1345 |
| h100-nemotron3-33b-q8 | 0.879 | 0.010 | 0.3 | 0.2 | 725 |

| question type | n | h100-gemma4-31b-it-q4_K_M | h100-gemma4-31b-nvfp4 | h100-gemma4-26b-nvfp4 | h100-gemma4-12b-nvfp4 | h100-qwen3.8-27b-nvfp4 | h100-qwen3.6-35b-a3b-nvfp4 | h100-qwen3.8-27b-q4_K_M | h100-nemotron3-33b-q8 |
|---|---|---|---|---|---|---|---|---|---|
| Artistic Text Recognition | 50 | 48/50 | 49/50 | 46/50 | 33/50 | 47/50 | 46/50 | 44/50 | 47/50 |
| Digit String Recognition | 50 | 29/50 | 31/50 | 34/50 | 13/50 | 39/50 | 44/50 | 37/50 | 40/50 |
| Doc-oriented VQA | 200 | 180/200 | 180/200 | 178/200 | 166/200 | 176/200 | 174/200 | 172/200 | 171/200 |
| Handwriting Recognition | 50 | 33/50 | 33/50 | 32/50 | 24/50 | 39/50 | 37/50 | 28/50 | 33/50 |
| Handwritten Mathematical Expression Recognition | 100 | 34/100 | 38/100 | 31/100 | 20/100 | 77/100 | 88/100 | 69/100 | 81/100 |
| Irregular Text Recognition | 50 | 40/50 | 39/50 | 40/50 | 28/50 | 47/50 | 47/50 | 43/50 | 47/50 |
| Key Information Extraction | 200 | 182/200 | 182/200 | 181/200 | 179/200 | 181/200 | 179/200 | 188/200 | 176/200 |
| Non-Semantic Text Recognition | 50 | 45/50 | 47/50 | 42/50 | 25/50 | 44/50 | 44/50 | 43/50 | 47/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 | 48/50 | 40/50 | 49/50 | 48/50 | 47/50 | 49/50 |
| Scene Text-centric VQA | 200 | 185/200 | 184/200 | 186/200 | 178/200 | 168/200 | 171/200 | 186/200 | 188/200 |

### What it says

**Two groups, and one category between them.**
- **`nemotron3:33b-q8`, `qwen3.6:35b-a3b-nvfp4` and `qwen3.8:27b` on both engines** score 857
  to 879 and do not separate: their six pairs give p between 0.071 and 1.000.
- **`gemma4:31b` on both engines and `gemma4:26b-nvfp4`** score 818 to 832 and do not separate
  either: p between 0.125 and 0.464.
- **Across the two groups, 11 of the 12 pairs resolve** at p < 0.05. The exception is
  `gemma4:31b-nvfp4` against `qwen3.8:27b-q4_K_M`, at 0.054.
- **`gemma4:12b-nvfp4` is last,** at 706, below every other arm at p < 0.001.

**The category is handwritten maths.** On its 100 items the gemma4 tags score 20 to 38 and
the other four 69 to 88. On the other 900 items, the seven larger tags land within 11 items
of one another, 787 to 798.

**That category scores notation as well as reading.** Its gold answers are LaTeX in the
dataset's own spaced notation, matched as a substring once whitespace is removed.
`qwen3.6:35b-a3b-nvfp4` writes that notation back (`x _ { 1 }`); gemma4 writes conventional
LaTeX, and `x_1` is not a substring of `x_{1}`. On 51 of the category's items qwen3.6 is
right and `gemma4:31b-nvfp4` wrong:
- **Some are misreadings.** On row 900 gemma4 reads `y = 1` for `y _ { 2 } = - 1`.
- **Some are notation alone.** On row 904 it writes `\frac{6\text{V}}{10\Omega} = 0.6\text{A}`,
  the gold's values, and is scored wrong.

This document does not count which is which, so the gap does not measure how well either
family reads maths.

**The two `qwen3.8:27b` arms saw different inputs.** mlx-cuda prefills 863 tokens per item and
llama.cpp 1345: the engines turn each image into different numbers of tokens, where the two
`gemma4:31b` arms both prefill 1122. Over all 1000 items they do not separate (p = 0.395), but
the categories trade. The MLX arm scores 182 against 162 on rows 0–200, the four recognition
categories, and 168 against 186 on scene-text VQA.

**`nemotron3:33b-q8` is the fastest arm and ties for the most accurate:** 0.3 s per item, on
the smallest prefill (725 tokens).

## 2026-10-01: gemma4:31b's GGUF ladder

`summarize_extbench.py --paired --timing --categories`, verbatim. `q4_K_M` is the 2026-09-30
arm.

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 1000 | 0 | 0 | 825 | **0.825** | false | generate |
| `gemma4:31b-it-q8_0` | 1000 | 0 | 0 | 829 | **0.829** | false | generate |
| `gemma4:31b-it-bf16` | 1000 | 0 | 0 | 833 | **0.833** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..1000.

host: http://127.0.0.1:11434 · build: 0.34.4-dynres-0-gb43ee8e

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| h100-gemma4-31b-it-q4_K_M vs h100-gemma4-31b-it-q8_0 | 820 | 166 | 5 | 9 | 0.424 |
| h100-gemma4-31b-it-q4_K_M vs h100-gemma4-31b-it-bf16 | 820 | 162 | 5 | 13 | 0.096 |
| h100-gemma4-31b-it-q8_0 vs h100-gemma4-31b-it-bf16 | 824 | 162 | 5 | 9 | 0.424 |

| arm | accuracy | ±1 s.e. | mean s/item | median | mean prompt_eval |
|---|---|---|---|---|---|
| h100-gemma4-31b-it-q4_K_M | 0.825 | 0.012 | 2.3 | 2.3 | 1122 |
| h100-gemma4-31b-it-q8_0 | 0.829 | 0.012 | 2.4 | 2.4 | 1122 |
| h100-gemma4-31b-it-bf16 | 0.833 | 0.012 | 2.0 | 2.0 | 1122 |

| question type | n | h100-gemma4-31b-it-q4_K_M | h100-gemma4-31b-it-q8_0 | h100-gemma4-31b-it-bf16 |
|---|---|---|---|---|
| Artistic Text Recognition | 50 | 48/50 | 48/50 | 48/50 |
| Digit String Recognition | 50 | 29/50 | 31/50 | 33/50 |
| Doc-oriented VQA | 200 | 180/200 | 182/200 | 180/200 |
| Handwriting Recognition | 50 | 33/50 | 33/50 | 33/50 |
| Handwritten Mathematical Expression Recognition | 100 | 34/100 | 36/100 | 38/100 |
| Irregular Text Recognition | 50 | 40/50 | 40/50 | 40/50 |
| Key Information Extraction | 200 | 182/200 | 182/200 | 182/200 |
| Non-Semantic Text Recognition | 50 | 45/50 | 44/50 | 47/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 | 49/50 |
| Scene Text-centric VQA | 200 | 185/200 | 184/200 | 183/200 |

### What it says

**No rung separates.** `q4_K_M` 825, `q8_0` 829, `bf16` 833, with exact McNemar p of 0.424,
0.096 and 0.424. The closest is `bf16` against `q4_K_M`, 13 items to 5. The 200-item ladders
on sm_120 and ROCm found no step either.

**Rows 0–200 are 170 / 200 on every rung,** as they are for `nvfp4`. `q8_0` and `bf16` are the
manifests the sm_120 and ROCm ladders measured, so those rows are one set of weights on three
hosts: sm_120 scored them 170 and 171, ROCm 169 and 169.

**The `bf16` rung decoded every image in pieces.**
- **It ran with a generation batch of 512,** so each image, about 1,080 tokens, decoded in
  three pieces. `q4_K_M` and `q8_0` ran with ADR 0036's floor of 2048 and decoded each image
  whole.
- **The scheduler grants that floor only with headroom:** a 2048 batch while the model's
  predicted memory is at most 60 % of free memory, 1024 at 75 %. At bf16 the 31B model needs
  about three quarters of this card.
- **The split cost nothing visible.** `bf16` scores the most of the three and is the fastest,
  2.0 s per item against 2.3 and 2.4, over the same 1122-token prefill.

## Setup

| setting | value |
|---|---|
| machine | GCP `a3-highgpu-1g`: one H100 SXM5 80GB (sm_90), Sapphire Rapids, 26 vCPUs, 230 GiB |
| PCIe | Gen4 x16 on a Gen5-capable GPU: the port above it tops out at Gen4 |
| build | `0.34.4-dynres-0-gb43ee8e`, a native build of `v0.34.4-dynres` with the CUDA 13 llama.cpp and MLX (`59d600b`) payloads |
| driver | 615.71.09, CUDA 13.4 |
| dates | 2026-09-30: `gemma4:31b-it-q4_K_M` and `gemma4:31b-nvfp4`. 2026-10-01: the other eight arms, in one queue. All ten on the same boot |
| harness | `extbench.py` from the `v0.34.4-dynres` checkout, every arm. Its scorer, prompt and request match main's; #429 changed only how rows and images are fetched |
| items | 1000, offset 0, all ten categories |
| think | off |
| sampling | temperature 0, `apply_sampling=False` |
| `num_ctx` | 8192, every arm |
| endpoint | `/api/generate` |
| loaded models | one at a time, unloaded between arms |
| runs per arm | 1 |
| GGUF image batch | `n_batch` = `n_ubatch` = 2048 for `gemma4:31b` `q4_K_M` and `q8_0`: ADR 0036's floor was granted, one batch per image. 512 for `bf16`, three pieces per image (above). 1024 for `qwen3.8:27b-q4_K_M` and `nemotron3:33b-q8`, with no split logged |

The machine's full profile, in the `host-profile/1` format, is
`host-profile_h100-sm90-2026-09-30.json` beside the arms, and
`host-profile_h100-sm90-2026-10-01.json` for the 2026-10-01 queue. The two differ only in when
they were collected and in Docker's patch version.

## Model identity (ADR 0038)

| tag | local manifest digest |
|---|---|
| `gemma4:31b-it-q4_K_M` | `sha256:17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97` |
| `gemma4:31b-nvfp4` | `sha256:a22a363052da770302019d5afab29bd967f9318e4377bddd297421189ea09d7d` |
| `gemma4:26b-nvfp4` | `sha256:f60799545325362bdaa15cbf38694f997514d3d46425c900b4eb06c1d6422d18` |
| `gemma4:12b-nvfp4` | `sha256:ded7a27350032202d9e9b2a6071e8aa89959ab156771b5228f30863c741c4970` |
| `qwen3.8:27b-nvfp4` | `sha256:5642e97495e1a088883805981563dcdc4a040c2f53388b7a41d1f24d3622cf7e` |
| `qwen3.6:35b-a3b-nvfp4` | `sha256:e92a3e94bbca90a85491dc34e9257bfee2318cedaa16360828c4d8edf14295b9` |
| `qwen3.8:27b-q4_K_M` | `sha256:25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e` |
| `nemotron3:33b-q8` | `sha256:74d89c84a4432530cbc5cac09b68d012bd01559196823db3a7a4e39da036cdf6` |
| `gemma4:31b-it-q8_0` | `sha256:53dd8459790f8795177444daa9e33f417e03c0d1cdedb80b6c73898603d20aef` |
| `gemma4:31b-it-bf16` | `sha256:236d76ae08745dbc143c31b9271b0f25750885199aa6039d0fc0113171606e6d` |

The `nvfp4` digest is the registry manifest that
[ocrbench-quantisation-ladder.md](ocrbench-quantisation-ladder.md) identifies as the
re-publication with a bf16 vision tower ("library"). The `q4_K_M` digest is not the one the
ROCm ladder measured (`sha256:6316f062…`, [ocrbench-gemma4-quant-ladder.md](ocrbench-gemma4-quant-ladder.md)):
the tag moved between the two pulls, so the GGUF rows compare artifacts as well as hosts.
The other six campaign tags are the manifests [the campaign](vision-campaign-2026-09-30-h100-sm90.md)
measured. `q8_0` and `bf16` were pulled during the 2026-10-01 queue, before their arms, and are
the manifests both the sm_120 ladder and the ROCm ladder measured.

## Files and reproducing

The ten arms are committed with the other published ones, as
`vision-suite/bench-runs/ocrbench/ext_h100-<tag>_ocrbench.json`, with the two host profiles
beside them. The tables above render from them. The category split also needs extbench's
cached row slice under the `--dir`, as `extimgs/ocrbench/rows_0_1000.json`: a 1000-item arm
caches it under `vision-suite/extimgs/ocrbench/`, and `.gitignore` keeps it out of the tree.

```bash
cd docs/maxusai/vision-suite
python3 summarize_extbench.py --dir bench-runs/ocrbench --paired --timing \
  --categories ocrbench h100-gemma4-31b-it-q4_K_M h100-gemma4-31b-nvfp4
python3 summarize_extbench.py --dir bench-runs/ocrbench --paired --timing --categories ocrbench \
  h100-gemma4-31b-it-q4_K_M h100-gemma4-31b-nvfp4 h100-gemma4-26b-nvfp4 h100-gemma4-12b-nvfp4 \
  h100-qwen3.8-27b-nvfp4 h100-qwen3.6-35b-a3b-nvfp4 h100-qwen3.8-27b-q4_K_M h100-nemotron3-33b-q8
python3 summarize_extbench.py --dir bench-runs/ocrbench --paired --timing --categories ocrbench \
  h100-gemma4-31b-it-q4_K_M h100-gemma4-31b-it-q8_0 h100-gemma4-31b-it-bf16
```

Each arm is the command in [ocrbench.md](ocrbench.md#the-test) with `LIMIT=1000`. The slice's
images were all fetched before the first arm: the rows' image links expire an hour after the
rows are fetched, and an earlier attempt at this run lost both arms at item 682 to a 403.
