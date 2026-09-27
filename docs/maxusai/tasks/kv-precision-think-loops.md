# TASK: KV precision and the attention path against the think-on loops

**Question (the maintainer, 2026-09-26):** does the KV cache's precision decide the think-on loops, and does the
flash-attention path? Every host runs the same cold captures on the cases that loop on it, and adds its results
here or as a comment on the pull request that opened this file. Production's KV cache is f16 on every platform
([ADR 0043](../adr/0043-production-runs-an-f16-kv-cache-on-every-platform.md)). This task measures whether f16
is enough, and whether the attention path matters.

## What is compared

Each looping case is captured **cold**, with every model evicted first, by the suite's own request
([`thinkcap.py`](../vision-suite/thinkcap.py)). The capture runs on the host's fold image, in the single-pass
flow, and in production's environment with only two knobs changed:

| KV cache \ flash attention | on | off |
|---|---|---|
| `q8_0` | the control, where it was the environment | does not exist: a quantized V cache needs flash attention |
| f16 | production (ADR 0043) | the flash-attention kernels against `mul_mat`, at the same storage |
| f32 | **equals f16, byte for byte**, on CUDA and HIP; not run ([ADR 0044](../adr/0044-an-f32-kv-cache-equals-f16-under-flash-attention.md)) | the most precise attention the build has |

- **Why f32 needs flash attention off.** At b11081, CUDA's and HIP's flash attention convert an f32 K/V cache to
  f16 before their kernels run (`ggml/src/ggml-cuda/fattn.cu`: `need_f16_K = K->type == GGML_TYPE_F32 …`; the
  tile and MMA kernels always take f16).
  - **Measured on both hosts: with flash attention on, f32 reproduces f16 byte for byte.** On gfx1151 that is
    qwen3.6 `real_1img`, 143,475 characters of thinking. On CUDA it is gemma4:26b, all six pairs.
  - So it is not run ([ADR 0044](../adr/0044-an-f32-kv-cache-equals-f16-under-flash-attention.md), SPEC H25).
  - Only with flash attention off does the attention itself run in f32. Metal's llama.cpp backend is unmeasured.
- **Budget.** `num_ctx` is 65536, so the capture has 57344 tokens (`thinkcap.py` sets `num_predict = num_ctx -
  8192`). ADR 0005 found qwen3.6's trajectories token-identical across `num_ctx`, so budgets compare as token
  counts.
- **Readout** ([`kvloop_read.py`](../vision-suite/kvloop_read.py)):
  - how the capture ended (`done_reason` and tokens);
  - the thinking's repetition profile: distinct lines in the whole and in the second half, and the line
    repeated most;
  - the answer, scored by the suite's own scorer.

  A loop shows as a second half with few distinct lines. For the adversarial arms (`adv_*`), read `hits_anchor`
  and not `contract_followed`, as the suite notes.
- **Loop onset.** `kvloop_read.py` also reports where the loop starts: the first 100-line window in which under 5%
  of the lines are new to the thinking. This finds cycles of any period. It gives the line and an estimate of the
  token, scaled by the capture's characters per token.
- **Flags.** `OLLAMA_FLASH_ATTENTION=0` makes the fork pass `--flash-attn off`. If it is unset, the fork passes
  `auto`, which enables flash attention on these GPUs. Record the runner's `--cache-type-k`, `--cache-type-v` and
  `--flash-attn` flags with each capture; [`kvloop.sh`](../vision-suite/kvloop.sh) logs them.

## How to run it

On a docker host (ROCm or CUDA), `kvloop.sh` starts one container per arm and captures every case:

```
IMG=<the fold image> STORE=<model store> GPU_ARGS="--gpus all" \
  ARMS="f16:1 f16:0 f32:0" \
  CASES="gemma4:26b-a4b-it-q4_K_M|bbox_contract_real_1img" \
  OUT=kvloop-<host> ./kvloop.sh
python3 kvloop_read.py kvloop-<host>/*.json
```

On ROCm, `GPU_ARGS` is `--device /dev/kfd --device /dev/dri` plus the host's numeric `--group-add` IDs for
`video` and `render`. On a native host, start the server once for each arm with `OLLAMA_KV_CACHE_TYPE` and
`OLLAMA_FLASH_ATTENTION` set. Then run this for each case:

```
python3 thinkcap.py <host> <model> <test> 65536 <out.json>
```

## gfx1151 (ROCm, `amd-server`)

The runs use the fold image `0.34.3-dynres-5-g29ae523` (b11081), single pass, `OLLAMA_NUM_PARALLEL=2`. The first
column is the aligned protocol's (#375), which ran with production's `q8_0` at the time. The other columns are cold
captures.

| case | `q8_0`, FA on, in the protocol | `q8_0`, FA on, cold | f16, FA on | f16, FA off | f32, FA off | f32, FA on |
|---|---|---|---|---|---|---|
| qwen3.6 `bbox_contract_real_1img` | never finishes at 131072; second half 35/2282 lines distinct | loops: all 24576 tokens at 32768; second half 49/436 | **loops**: all 57344 tokens, no answer; second half 77/1647 | **loops**: all 57344 tokens, no answer; second half 12/906 | **loops**: all 57344 tokens, no answer; second half 76/1643 | byte-identical to f16, FA on: loops, 143,475 characters |
| qwen3.6 `bbox_contract_adv_real` | never finishes at 131072; second half 27/3165 | **finishes**: 17,626 tokens, valid JSON, 6/6 labels | **finishes**: 12,120 tokens, valid JSON, 6/6 labels | — | — | — |
| gemma4:26b `bbox_contract_real_1img` | never finishes at 131072, in both flows | loops: all 24576 tokens at 32768; second half 10/381 | **loops**: all 57344 tokens, no answer; second half 26/1224 | **finishes**: 5,800 tokens, valid JSON, 6/6 labels | **loops**: all 57344 tokens, no answer; second half 20/1322 | — |

**Captured cold, no precision turns a loop into a finish, and the loop's start moves in both directions.** f32 with
flash attention off, the most precise attention the build has, loops *earliest* on qwen3.6 `real_1img`. Its thinking
diverges from both flash-attention runs 30 characters in, settles into a 76-line cycle after its first quarter, and
repeats "Let's assume the image is 1600x900." 79 times. Estimated token at which the loop starts:

| case | `q8_0`, FA on | f16, FA on | f16, FA off | f32, FA off |
|---|---|---|---|---|
| qwen3.6 `bbox_contract_real_1img` | about 14,650 (cold) and 14,890 (in the protocol) | about 23,000 | about 7,800 | about 7,390 |
| qwen3.6 `bbox_contract_adv_real` | about 8,750 in the protocol; cold, no loop, finishes at 17,626 | no loop; finishes at 12,120 | — | — |
| gemma4:26b `bbox_contract_real_1img` | about 3,290 (cold) | about 4,670 | no loop; finishes at 5,800 | about 3,000 |

On qwen3.6 `real_1img`, both flash-attention-off runs loop earlier than both flash-attention-on runs. That is one case
with one run per cell, so it is not a trend.

**Cold captures isolate the KV type. The protocol's cells can also carry the run's history.**

- qwen3.6 `real_1img`'s cold `q8_0` capture at 32768 is byte-identical to the protocol's thinking at 131072 for its
  whole length, 58,820 characters. That trajectory depends on neither history nor `num_ctx`, which matches ADR 0005.
  Its cold f16 capture diverges from the cold `q8_0` one 246 characters in ("Single JSON object" against "A single
  JSON object").
- qwen3.6 `adv_real` is different. Its cold `q8_0` capture takes the same 5,930 prompt tokens as the protocol's
  cell, and finishes. The protocol's thinking diverges from it 369 characters in, and never finishes. So the
  protocol's loop on this case came from the run's state and not from the KV type. The likeliest causes, which
  are inferred and not tested, are the prompt cache reused from the previous cell (the same first image and
  prompt head) and the parallel slot the request landed on.

**One run of the eight on the two looping cases escaped, and not on the most precise path.** gemma4:26b with f16
and flash attention off finished in 5,800 tokens. All six of its boxes are right in the 0–1000 frame, although its
declaration says pixels (`hits_bestfit` 6, `hits_declared` 1). f32 with flash attention off, which is more precise,
loops on the same case. Neither precision nor the attention path decides the outcome in one direction. Whether a
run escapes looks like chance in the numerics, and the prompt sets the trap every time.

**So no KV type or attention path reliably turns a cold case from a loop into a finish.** qwen3.6 `adv_real` finishes under
both KV types, 5,500 tokens sooner with f16. qwen3.6 `real_1img` and gemma4:26b `real_1img` loop under all three paths measured so far.
On both, f32 with flash attention off loops earliest. The numerical path changes where the loop starts, but not in one
direction. What decides these two loops is the prompt, which withholds the image size, and not the precision. gemma4's thinking locks up after its first quarter (157, 20, 18
and 26 distinct lines out of 612 per quarter), and repeats "Wait, I'm still getting the same numbers. Let me
re-examine the image." 77 times.

**What the looping thinking goes over, in both qwen3.6 cases.** The prompts ask for pixel coordinates without
giving the image's size, and the model keeps re-deciding it: "I will assume W=1920, H=1080 … Maybe the image is
smaller … Let's assume W=1200, H=675". In `adv_real` it commits to one size and finishes, under both KV types. Both
answers declare `ref_size` [1000, 600] against the true 1920×1080. Their `hits_anchor` is 0, and 3 of 6 boxes hit
only in the best-fitting dialect, norm-1000.

## Prompt and sampling check (gfx1151, 2026-09-27)

The KV and attention arms could not stop these loops, so the next question was whether the prompt, or production's
sampling, can. The case is `bbox_contract_real_1img`, captured cold on the fold image with f16 and flash attention
on, at 32768 (24,576 tokens), by [`promptcap.py`](../vision-suite/promptcap.py):

- **size:** the prompt's one unanswerable instruction ("If you resized the image internally, give the size YOU used,
  not the size you were sent") is replaced by the image's real size;
- **commit:** the same sentence is replaced by "choose your best estimate once, give it as ref_size, and do not
  revisit it";
- **card:** the original prompt at the model card's sampling, which is what production sends. It is sampled, so
  there are three runs per model.

The greedy original loops on both models (the table above).

| arm | qwen3.6 | gemma4:26b |
|---|---|---|
| greedy, **size** | finishes, 20,299 tokens; boxes in 0–1000 (2/6 as pixels) | **finishes, 2,081 tokens; 6/6 pixel boxes, IoU 0.71, contract followed** |
| greedy, **commit** | **loops** from about token 7,800 ("Let's assume the image is 1920x1080." x27) | **finishes, 4,345 tokens; 6/6, IoU 0.73, contract followed** |
| **card**, 3 runs | 3/3 finish (4,924, 15,637 and 22,947 tokens); pixel boxes 1/6, 5/6 and 2/6, mostly in the 0–1000 frame | 3/3 finish (1,680, 6,837 and 7,138); 6/6 pixel boxes in two runs, and 0–1000 in one |

**The reading:**

- **The unanswerable sentence is the loop's trigger.** Take it out, and gemma4:26b answers in a few thousand tokens
  with correct pixel boxes, whether or not it is given the size.
- **qwen3.6 does not produce pixel coordinates.** Even given the size, it answers in its own 0–1000 frame under a
  pixel declaration. For qwen3.6 a caller converts from 0–1000, which the bbox contract already requires
  ([SPEC C1](../spec/vision-bbox-response-contract.md), ADR 0027: pin norm-1000).
- **At production's sampling, none of the six runs loops.** The suite's greedy think-on is the worst case, as
  `sampling.py` says. The loops are real, but production users are unlikely to meet them on this prompt.

## A second trap: `multi_3img_anchored` (gfx1151, 2026-09-27)

`multi_3img_anchored` is `multi_3img`'s prompt plus one calibration paragraph. That paragraph ends with the same kind
of sentence: "If you resized image 1 internally, use the size YOU used." `promptcap.py` replaces it too. On the CUDA
host's fixed-history run, MLX's gemma4:26b leaves this case NOT CONVERGED in all 8 runs and converges on
`multi_3img` in all 8, some on a higher rung (#375). This host ran the GGUF leg of #375's open item 8: gemma4:26b-a4b q4_K_M on the fold image, captured cold and greedy, with
f16 and flash attention on, at 32768. The `orig` prompts' fingerprints equal the suite's `prompt_sha`.
`kvloop_read.py`, verbatim:

```
greedy-orig_gemma4_26b-a4b-it-q4_K_M_multi_3img_32768
  arm=greedy-orig  done=stop  tokens=3616  thinking=5618 chars  answer=3086 chars
  thinking: 90/102 lines distinct, second half 42/51, most repeated x3: '- **Key Objects:**'
  onset: no loop found
  score (multi_3img): json_valid=True q1_right=True q2_right=True q4_bbox_hit=True q4_bbox_space=norm1000/xyxy chart_values_found=5
greedy-orig_gemma4_26b-a4b-it-q4_K_M_multi_3img_anchored_32768
  arm=greedy-orig  done=length  tokens=24576  thinking=61234 chars  answer=0 chars
  thinking: 239/1717 lines distinct, second half 11/859, most repeated x288: 'The text "DYNAMO" is at y ~ 530.'
  onset: loop from line 280 of 1,717, about token 4,148
  score (multi_3img_anchored): json_valid=False q1_right=False q2_right=False q4_bbox_hit=False chart_values_found=0
greedy-size_gemma4_26b-a4b-it-q4_K_M_multi_3img_anchored_32768
  arm=greedy-size  done=stop  tokens=5875  thinking=9557 chars  answer=3510 chars
  thinking: 177/239 lines distinct, second half 93/120, most repeated x4: '"ANCHOR"'
  onset: no loop found
  score (multi_3img_anchored): json_valid=True q1_right=True q2_right=True q4_bbox_hit=True q4_bbox_space=norm1000/xyxy chart_values_found=5
greedy-commit_gemma4_26b-a4b-it-q4_K_M_multi_3img_anchored_32768
  arm=greedy-commit  done=stop  tokens=5633  thinking=10023 chars  answer=3431 chars
  thinking: 174/217 lines distinct, second half 107/109, most repeated x3: '- Key objects:'
  onset: no loop found
  score (multi_3img_anchored): json_valid=True q1_right=True q2_right=True q4_bbox_hit=True q4_bbox_space=norm1000/xyxy chart_values_found=5
```

- **The trap sentence alone makes the case loop, and replacing it ends the loop.** The control finishes in 3,616
  tokens. The anchored prompt loops from about token 4,148, re-deriving image 1's height. With only the sentence
  replaced, it finishes in 5,875 tokens (`size`) or 5,633 (`commit`), with every question right.
- **Under `q8_0` the same case finished**, in 8,331 tokens, both in the fold's protocol and cold. Here f16 is the side
  that loops. This is the pattern above again: the numerical path moves where a loop starts, in both directions, and
  the prompt sets the trap. It does not change ADR 0043. f16 is production's setting for parity and headroom, and the
  fix for the loop is the prompt.
- **The image that ships loops too, byte for byte.** On the 908 image (`0.34.3-dynres-22-g5584539`), both `orig`
  captures equal the fold image's in thinking, answer and token counts: the control finishes in 3,616 tokens, and
  the anchored prompt loops through the same 61,234 characters. 908 changes no gfx1151 kernel.

**The CUDA host's legs of item 8** (#375, fold record `5a6305139`), with the same tool and settings:
- **GGUF, CUDA's fold image:** the same answer as here. `orig` loops, from about token 2,129; `size`, `commit` and the
  control finish.
- **GGUF, the image CUDA ships (908):** all four finish. On CUDA the loop also needs `ce8caa6e6`'s flash-attention
  tiling, which 908 reverts. On gfx1151, whose RDNA tile table that commit does not touch, the sentence is enough.
  Again the numerical path decides whether the trap closes.
- **MLX, five cold draws per prompt:** `orig`, `size` and `commit` each finish 1 of 5, and the control 3 of 5. On MLX
  the sentence does not set the loop rate; stating the size changes only how the case loops. So the ladder's
  "converges on `multi_3img` in all 8" hides looping draws: each rung is one cold draw.

## The KV type against the fold's own change (gfx1151 protocol, 2026-09-27)

The fold's think-on protocol (#375) runs the whole suite on qwen3.6 in two arms on one fold image. `fold` is the
fold's single-pass structured output. `fold2p` is the same image with `OLLAMA_FORMAT_TWO_PASS=1`, ADR 0004's two-pass
flow. So `fold` against `fold2p` isolates the fold's one behaviour change. The cells are greedy
(`card:qwen3.6+temp0`) and climb the full ladder to 131072. The pair ran under `q8_0`, production's KV type at the
time, and runs again under f16. `fold2p` has now finished under both, so the KV type can be set beside the fold's
change.

[`cmp_scored.py`](../vision-suite/cmp_scored.py) prints, test by test, the quality fields that moved and the side
each move favours (`A+` or `B+`). It leaves out lengths and budgets, and compares only the tests that finished in
both runs.

The fold's change, both arms under `q8_0` (A = `fold`, B = `fold2p`):

```
bbox_contract_box2d_1img: iou_anchor 0.633->0.892 B+; iou_declared 0.633->0.892 B+; self_check False->True
bbox_contract_positional_1img: iou_anchor 0.635->0.616 A+; iou_declared 0.635->0.616 A+
bboxm_free_noanc_pos: iou_declared 0.836->0.957 B+
bboxm_pin_anc_pos: iou_anchor 0.57->0.968 B+; iou_declared 0.57->0.968 B+
multi_3img: q4_bbox_space 'norm1000/xyxy'->'pixel/xyxy'
quality moves: 5 favour B, 2 favour A, 2 label changes (A = scores_r0344p_fold_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json, B = scores_r0344p_fold2p_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json)
```

The KV type, both runs in the `fold2p` arm (A = `q8_0`, B = f16):

```
bbox_contract: iou_declared 0.89->0.956 B+
bbox_contract_anchored: iou_anchor 0.938->0.6 A+; iou_declared 0.938->0.6 A+; self_check True->False
bbox_contract_anchored_1img: iou_anchor 0.757->0.648 A+; iou_declared 0.757->0.648 A+; self_check False->True
bbox_contract_box2d_1img: iou_anchor 0.892->0.967 B+; iou_declared 0.892->0.967 B+
bbox_contract_multi: bestfit_dialect 'real/xyxy'->'norm1000/xyxy'; contract_followed False->True B+; declaration_matches_boxes False->True B+; declared_ref [1920, 1080]->[1000, 1000]; declared_type 'real'->'norm1000'; hits_bestfit 3->6 B+; hits_declared 3->6 B+; implied_scale 0.838->None; iou_at_implied_scale 0.489->None; iou_declared 0.202->0.931 B+
bbox_contract_perobject: iou_declared 0.954->0.964 B+
bbox_contract_pinned: iou_declared 0.959->0.966 B+
bbox_contract_positional_1img: iou_anchor 0.616->0.968 B+; iou_declared 0.616->0.968 B+
bbox_contract_reasoning: bestfit_dialect 'norm1/xyxy'->'norm1000/xyxy'; declared_ref [1000, 600]->[1000, 1000]; declared_type 'norm1'->'norm1000'; iou_declared 0.55->0.676 B+
bboxm_free_anc_named: iou_anchor 0.964->0.949 A+; iou_declared 0.964->0.949 A+
bboxm_free_anc_pos: iou_anchor 0.951->0.97 B+; iou_declared 0.951->0.97 B+
bboxm_free_noanc_named: contract_followed False->True B+; declaration_matches_boxes False->True B+; declared_ref [1920, 1080]->[1000, 1000]; declared_type 'real'->'norm1000'; hits_declared 1->6 B+; iou_declared 0.057->0.855 B+
bboxm_free_noanc_pos: contract_followed True->False A+; declaration_matches_boxes True->False A+; hits_bestfit 6->1 A+; hits_declared 6->1 A+; iou_declared 0.957->0.079 A+
bboxm_pin_anc_named: iou_anchor 0.966->0.692 A+; iou_declared 0.966->0.692 A+; self_check True->False
bboxm_pin_anc_pos: iou_anchor 0.968->0.919 A+; iou_declared 0.968->0.919 A+
bboxm_pin_noanc_named: iou_declared 0.719->0.959 B+
bboxm_pin_noanc_pos: contract_followed True->False A+; declaration_matches_boxes True->False A+; hits_bestfit 6->3 A+; hits_declared 6->3 A+; iou_declared 0.952->0.325 A+
document_single: name_bbox_mean_iou 0.577->0.733 B+
multi_3img: q4_bbox_space 'pixel/xyxy'->'norm1000/xyxy'
multi_3img_anchored: q4_bbox_hit False->True B+; q4_bbox_space None->'norm1000/xyxy'
scene_single_anchored: bbox_mean_iou 0.057->0.039 A+
bbox_contract_adv_real: finished only in B
quality moves: 22 favour B, 21 favour A, 15 label changes (A = scores_r0344p_fold2p_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json, B = scores_r0344pf16_fold2p_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json)
```

Derived from the output above, not generator output: 25 tests finish in both runs of each comparison.

- The fold's change moves the quality of 4 of them, 3 toward `fold2p`, and a label in 1 more.
- The KV type moves the quality of 20, 12 toward f16 and 8 toward `q8_0`, and a label in 1 more.
- `bbox_contract_adv_real` finishes only under f16. `bbox_contract_real_1img` finishes in none of the three runs.

On this host a greedy qwen3.6 run reproduces cell for cell: the v0.34.3 fold's control and candidate agreed in all
1009 cells ([upstream-sync-0.34.3.md](upstream-sync-0.34.3.md)). So both counts are effects, not noise.

**The reading:**

- **The KV type moves five times as many tests as the fold's change, and in no net direction.** The largest moves
  go both ways. With f16, `bboxm_free_noanc_named`'s IoU goes from 0.057 to 0.855, and `bbox_contract_multi`'s from
  0.202 to 0.931. With `q8_0`, `bboxm_free_noanc_pos` keeps 0.957 where f16 has 0.079, and `bboxm_pin_noanc_pos`
  keeps 0.952 where f16 has 0.325. A greedy trajectory that diverges early ends on a different answer, and the KV
  type is enough to make it diverge: the cold `real_1img` captures above part 246 characters in.
- **So f16 is not a quality setting.** It is production's setting for parity and headroom
  ([ADR 0043](../adr/0043-production-runs-an-f16-kv-cache-on-every-platform.md)), and this measures the parity half.
- **Compare arms only at one KV type, on one host and across hosts.** A KV mismatch between two arms, or between two
  hosts, moves more cells than the change under test. The `q8_0` pair stands as a pair. The f16 pair is compared
  within itself when its `fold` arm finishes.

## CUDA (`ai-server/mlx-cuda`)

**Run on the CUDA host on 2026-09-26** (#387): `kvloop.sh` at `0bbd5e67c`, unmodified. The captures are cold, at
`OLLAMA_NUM_PARALLEL=2`, single pass, on GPU0 (sm_120). Every one of the 24 captures has `--cache-type-k/v` and
`--flash-attn` flags that match its arm. The model is gemma4:26b-a4b-it-q4_K_M, on two builds: the fold as shipped
(b11081 with `ce8caa6e6`'s FA tiling) and the 908 image (the tiling reverted).

**Two byte-identities reduce the 24 captures to 12 distinct trajectories:**

- f32 with FA on reproduces f16 with FA on, byte for byte, on both builds and all three cases. That confirms that
  flash attention converts an f32 K/V cache to f16 first.
- With FA off, the two builds are byte-identical at both f16 and f32. That is a positive control for 908's scope,
  which is FA's MMA tiling only.

| case | fold, FA on (f16 = f32) | 908, FA on (f16 = f32) | FA off, f16 (both builds) | FA off, f32 (both builds) |
|---|---|---|---|---|
| `multi_3img_anchored` | **loops** from about token 2,105; second half 7/983 | finishes: 3,882, valid JSON | finishes: 8,547, valid JSON | **loops** from about 4,117; second half 5/1521 |
| `bbox_contract_real_1img` | **loops** from about 3,455; 12/923 | **loops** from about 1,543; 8/1452 | finishes: 3,340, 6/6 labels, `hits_bestfit` 6, `hits_declared` 1 | finishes: 5,363, the same answer |
| `bbox_contract_box2d_1img` | finishes: 2,377, IoU 0.973 | **loops** from about 4,247; 19/1367 | finishes: 4,332, IoU 0.962 | finishes: 2,378, IoU 0.962 |

- **Loop counts per path:** fold with FA on 2/3, 908 with FA on 2/3, f16 with FA off 0/3, f32 with FA off 1/3. Each
  path is one fixed trajectory, not a draw. The most precise path loops on `multi_3img_anchored`, where f16 with
  FA off finishes.
- **`real_1img` with f16 and FA off escapes the same way on both hosts.** All six boxes are right in the 0–1000 frame
  under a pixel declaration. It takes 3,340 tokens on CUDA and 5,800 on gfx1151.
- **The FA-on columns agree with #375's in-suite loop-rate run.**
- **Deploy source (ADR 0043, decision 1):** production does not set the variable and runs the f16 default (12 of 12
  KV allocations). The v0.34.4 deploy sets `OLLAMA_KV_CACHE_TYPE=f16` explicitly, and refuses if the live
  container carries a different value.

## Metal (`mlx-metal`)

**On MLX, neither knob exists** (the Metal host on #387, from the fold's code at `29ae52351`):

- `OLLAMA_KV_CACHE_TYPE` and `kv_cache_type` are resolved only for llama-server (`resolveKVCacheType`, called from
  `NewLlamaServer`). The MLX runner's `KVCache` allocates in the dtype of the model's own projections
  (`mlxrunner/cache/kvcache.go`).
- Every MLX text model's attention goes through `mlx.FastScaledDotProductAttention`, MLX's fused kernel
  (`mlxrunner/nn/sdpa.go`), and no variable switches it.
- On MLX, drafting and request history are what move the loops (#375). Varying the KV precision or the attention
  path there would take a code change.

**GGUF on llama.cpp's Metal backend does apply.** The arms can run there. The GPU time is the maintainer's call,
because the host's protocol campaign is using it.

**The protocol's GGUF runs on Metal (2026-09-27, #375).** They used the fold image, single pass, llama.cpp's Metal
backend at its defaults (f16 KV, flash attention auto), greedy, and the full ladder:
- gemma4:31b finishes 27 of 27 cases.
- qwen3.6 finishes 23 of 27. `scene_single`, `multi_3img`, `multi_3img_anchored` and `bbox_contract` never finish
  at 131072. On gfx1151's GGUF, at f16, qwen3.6 finishes all four of those.
- qwen3.6's `bbox_contract_real_1img` **finishes** on Metal, at 65536, in 34,337 tokens. It is the first path on
  which that case finished. It looped on every path gfx1151 tried: `q8_0` with flash attention on, f16 with it on and
  off, and f32 with it off.

Metal's llama.cpp and ROCm's are the same model, weights and prompts under another numerical path. They loop on
different cases, which is this task's finding again: the path moves where a loop starts, and sometimes whether it
starts at all.

**Deploy source (ADR 0043, decision 1):** production is a launchd agent. It sets `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`
and no `OLLAMA_KV_CACHE_TYPE`, and it could hold one. Setting f16 there is a production change for the maintainer,
and it affects only the GGUF models the server runs.
