# Vision model recommendations — settings per model, think mode, and use case

**Derived from measurements of 2026-08-20** — the `cudafull1` five-model CUDA
baseline (full CONTEXT ladder; [tasks/cuda-baseline-full-ladder-rerun.md](tasks/cuda-baseline-full-ladder-rerun.md)
holds the verbatim T1/T2 renders and the per-probe thinking-token map) and the
1,120-cell composable bbox factor matrix (`summarize_matrix.py mx` render,
same day). Host `http://10.8.0.6:11497`, build `0.32.14-rc0-dynres-0-ga5d6590`,
GGUF q4_K_M throughout. Every number below traces to those renders; nothing
here is a new measurement.

**Scope caveats.** One host, one build, one quantization, one repeat per cell
— and a think-on cell is one draw, not a constant (ADR 0012 conv. 4: measured
same-cell recall_9px 1/4 vs 2/4 across two draws). Apple/MLX and ROCm are out
of scope (deliberately skipped 2026-08-20). nemotron3 think-on token counts
are understated by an open server bug
([tasks/nemotron-thinkon-evalcount-undercount.md](tasks/nemotron-thinkon-evalcount-undercount.md));
its `s/req` is unaffected.

## Rules that apply to every model

1. **Sampling**: think-off runs greedy (temperature 0 — every baseline
   depends on it). Think-on uses the model card's sampling via `sampling.py`;
   greedy decoding under thinking is what produces runaway reasoning
   ([runaway-reasoning-under-think.md](runaway-reasoning-under-think.md)).
2. **Window**: think-off fits everywhere at `num_ctx` 16384 / `num_predict`
   2200. Think-on: never pin a fixed window — start at 16384 with
   `num_predict = num_ctx − 8192` and let the CONTEXT ladder escalate per
   cell (SPEC H4a/H4b). The rung a model needs is a result; the per-model
   sections below list the measured rungs.
3. **Prompt shape**: `pin + anchor + named coords` — state the actual image
   size, ask for an `__IMAGE__` calibration entry, use a named coordinate
   space — is the trustable arm for 7 of 9 measured model×mode groups
   (6.00/6, geometry spread 0.00 for most; factor matrix). The one measured
   exception: **nemotron3 think-on prefers positional over named coords by
   3.78/6**. The anchor is non-negotiable for multi-image bbox work: it flips
   qwen3.8 and nemotron3 ❌→✅ and is the difference between qwen3.6 think-on
   never terminating and finishing in 9,598 thinking tokens.
4. **Coordinate dialect is per model** and must be offered, never assumed:
   `box_2d` for gemma4, `bbox_2d` for qwen3.x and nemotron3
   ([vision-bbox-coordinate-conventions.md](vision-bbox-coordinate-conventions.md)).
   Never trust a model's *declared* coordinate space on adversarial input —
   nemotron3 declares real-pixel and emits norm-1000 (0/6 on the adversarial
   contract cell).
5. **Trust `done_reason`**: `stop` before using any cell as a result;
   `length` means the measurement is unfinished, not that the model failed
   (ADR 0012 conv. 9). For nemotron3 think-on it is the *only* reliable cap
   signal while the eval_count bug is open.
6. **Endpoint**: `/api/chat` with `format: "json"` (the calibrated campaign
   path; `/api/generate` drops reasoning text).

## Per-model settings

### gemma4:26b-a4b-it-q4_K_M (MoE)

| | think=false (recommended) | think=on |
|---|---|---|
| window | 16384 / 2200 | needs **65536** for scene (ladder-measured) |
| quality | scene IoU 0.973, doc bbox 0.756, fine text 4/4/4/3/3, multi ✅✅✅ | scene **collapses to 0.334, boxes 0/6**; fine text 4/4/4/4/3 |
| cost | 150 tok/s, 5.0 s/req, 725 req/h | 7,153 thinking tok on scene; 62 req/h (12× slower) |

**Use think-off.** Best quality-per-second in the fleet. If thinking is
required, use the full pin+anchor+named prompt shape (its matrix arm is
6.00/6 in both modes — the collapse is specific to free-form scene prompts)
and budget the 65536 window. Dialect `box_2d`.

### gemma4:31b-it-q4_K_M (dense)

| | think=false | think=on |
|---|---|---|
| window | 16384 / 2200 | **16384 suffices — never escalated** |
| quality | scene 0.962, fine text 4/4/4/4/3 (best small-tier OCR), multi ✅✅✅ | scene 0.962 (no think damage), fine text 4/4/4/4/3, multi ✅✅✅ |
| cost | 56 tok/s, 296 req/h | 833 thinking tok on scene; 102 req/h |

**The safe-either-mode model.** Only model with zero think-on degradation,
zero ladder escalation, and matrix 6.00/0.00 in both modes. Slowest decode —
choose it for correctness-critical work, small-glyph OCR (9px 4/4, 7px 3/4),
or wherever think-on is mandated. Dialect `box_2d`.

### qwen3.8:27b-q4_K_M (dense)

| | think=false | think=on |
|---|---|---|
| window | 16384 / 2200 | 16384 suffices |
| quality | scene **0.977 (best)**, doc bbox 0.550, multi q4 ❌ unanchored / ✅ anchored | scene 0.975, doc bbox **0.858 (best)**, multi q4 ❌ unanchored / ✅ anchored |
| cost | 65 tok/s, 366 req/h | frugal thinker: 431–1,421 tok per probe, 21,750 suite total; 188 req/h |

**Always anchor.** It grounds near-perfectly but in an internally rescaled
frame (measured ~1.22×, self-reported in its own thinking stream); without a
calibration entry any cross-frame consumer of its boxes scores it wrong.
**Avoid <9px text**: its small-tier misses are sub-glyph optical confusions
(M↔N, W↔K, digits 1↔3↔5↔9) in structurally correct codes — mode-independent,
so thinking cannot fix it. Best pick for document bbox work under thinking.
Dialect `bbox_2d`.

### nemotron3:33b-q4_K_M

| | think=false (recommended) | think=on |
|---|---|---|
| window | 16384 / 2200 | **32768 for multi; fine text needed the 65536 window** (terminated in 4,909 tok there after capping 24,576 at 32768 — window-dependent termination) |
| quality | scene 0.870, invoice fields 5/5 ✅✅, **doc bbox 0.044 — unusable** | scene 0.577, doc bbox 0.114, multi ✅✅❌ / anchored ✅✅✅ |
| cost | **209 tok/s gen, 4,797 prefill, 3.0 s/req, 1,196 req/h — fleet's fastest** | 148 req/h; token counts understated (open bug) |

**The throughput king for field extraction** — invoice items/totals/serials
all correct at 4× the req/h of anything else. **Never consume its document
bboxes** (0.044/0.114 both modes). Under thinking: use **positional coords**
(named costs it −3.78/6), anchor everything, expect the 32768 rung, and gate
on `done_reason`, not token counts. Dialect `bbox_2d`, but verify via anchor
— it misdeclares its space under adversarial prompts. 7px OCR: 0/4.

### qwen3.6:35b-a3b-q4_K_M (MoE)

| | think=false (recommended) | think=on |
|---|---|---|
| window | 16384 / 2200 | ≥32768; **three probes never terminate even at 131072/122,880** (`multi_3img` unanchored, two contract probes) |
| quality | scene 0.975, multi ✅✅✅, fine text 4/4/4/2/2 | scene 0.717 @32768; grounds 6.00/6 in *every* terminating matrix cell |
| cost | 95 tok/s, 545 req/h | **275,791 thinking tok** across the uncapped suite (12.7× qwen3.8); 15 req/h |

**Think-off by default.** Think-on is a termination lottery on free-form
prompts: when it stops, it is perfect; whether it stops depends on the prompt
shape. If think-on is required: anchored + pinned prompts only (anchored
multi finishes in 9,598 tok ✅✅✅; unanchored burns 122,880 and never stops),
window ≥32768, and treat any unanchored bbox request as a non-termination
risk. Dialect `bbox_2d`.

## By scenario

| scenario | pick | mode | key settings |
|---|---|---|---|
| Scene/object grounding, throughput matters | gemma4:26b-a4b | off | 16384/2200, `box_2d`, greedy |
| Scene grounding, max accuracy | qwen3.8 | off | 16384/2200, **anchored**, `bbox_2d` |
| Invoice/field extraction at scale (no boxes) | nemotron3 | off | 16384/2200; 1,196 req/h; ignore its bboxes |
| Document extraction **with** name bboxes | qwen3.8 | on | 16384, anchored, `bbox_2d` (0.858) |
| Small-print OCR (≤9px glyphs) | gemma4:31b | either | 16384/2200; the only model ≥3/4 at 7px |
| Multi-image cross-referencing | gemma4:31b or 26b-a4b | either | ✅✅✅ even unanchored; anchor anyway |
| Reasoning-mandated vision tasks | gemma4:31b | on | 16384 suffices, no think damage, cheap thinking (≤5,393 tok worst probe) |
| Reasoning + tight token budget | qwen3.8 | on | 16384, anchored; frugalest thinker |

## Overall

**gemma4:31b-it-q4_K_M is the overall recommendation** when one model must
cover everything: the only one with no think-mode degradation, no window
escalation, best small-glyph OCR, clean multi-image sweeps in both modes, and
matrix-perfect grounding under the pin+anchor+named shape — its price is
decode speed (56 tok/s). When throughput matters more than 7px text and
thinking is off, **gemma4:26b-a4b** delivers near-identical quality at 2.5×
the req/h. Split fleets pair **nemotron3 think-off** for high-volume field
extraction with **qwen3.8 anchored** for anything that consumes boxes.

And the single highest-leverage setting across every model and mode is not a
model choice at all: **ask for the `__IMAGE__` calibration anchor**. Measured
today, it turned two ❌ multi cells into ✅ sweeps, distinguished frame errors
from grounding failures, and converted qwen3.6's think-on non-termination
into a 9,598-token clean finish.
