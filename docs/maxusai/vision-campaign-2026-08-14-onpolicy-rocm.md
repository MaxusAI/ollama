# Vision campaign — on-policy think mode on gfx1151 (2026-08-14)

The ROCm reconfirmation [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md)
defers in its Alternatives section: *"that host is unreachable from here. The per-model
decisions above are scoped to the measured host and should be reconfirmed there."*

**Result: two of the three per-model verdicts transfer. The one ADR 0023 changed — gemma4 —
does not.** On this host gemma4's document IoU is negative in all three on-policy reps,
where Apple Silicon measured it non-negative in all three. Every other conclusion of ADR 0023
holds, including its central mechanism claim that nemotron3's regression is not a sampling
artefact.

## Provenance

Per [ADR 0012](adr/0012-benchmark-report-templates.md). Exploratory report (§5): the arms are
think-on/off × model × replicate rather than a T1 campaign matrix, so it is exempt from the
T1/T2 shapes but carries the provenance header and validity marks. Tables below are
generator-emitted from the committed scores files, not typed.

| | |
|---|---|
| Date | 2026-08-14, 00:35–03:27 (+10:00), 12/12 cells |
| Host | Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151** (RDNA 3.5), ROCm 7.2.1 |
| Server | `0.32.1-dynres-296eb020` (`v0.32.1-dynres.3`), image `maxusai-ollama:0.32.1-rocm-dynres-296eb020` |
| Payload | **b9888** + compat 001/002/004/005 |
| Suite | `47158ea0`, sampling via `vision-suite/sampling.py` (`843c5705`) |
| Config | `/api/generate`, `num_ctx 32768`, `num_predict 24000`, `HTTP_TIMEOUT 1800` |
| think-off | greedy (`temperature 0`), unchanged — 1 cell per model |
| think-on | card-sourced per model, **n = 3** per ADR 0023's admissibility rule |
| Isolation | cold container restart per **cell**, `OLLAMA_NUM_PARALLEL=2`, no other traffic |
| Raw data | [`vision-suite/runs/onpolicy-rocm-2026-08-14/`](vision-suite/runs/onpolicy-rocm-2026-08-14/) (12 scores files) |

Comparison target is ADR 0023's Apple Silicon run: server `0.32.5-maxusai-31a7f1ef`, payload
**b10353**. **Different host and different payload** — ADR 0023 forbids pooling the two, and
nothing here does. The comparison is of *deltas within each host*, never of absolute scores
across them.

## Admissibility

Every cell records `sampling_source`; none is `legacy-greedy`. Resolved values:

| model | `sampling_source` | think-on sampling |
|---|---|---|
| gemma4:31b-it-q4_K_M | `card:gemma4` | `temperature 1.0, top_p 0.95, top_k 64` |
| qwen3.6:35b-a3b-q4_K_M | `card:qwen3.6` | `temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5` |
| nemotron3:33b-q4_K_M | `packaged-defaults-no-card` | no overrides sent (model declares none) |

**Cap audit — applied before reading any score**, per ADR 0022's rule carried forward by 0023:

| cell | capped test | consequence |
|---|---|---|
| qwen3.6 rep 1 | `scene_single` @ 24 000 | **scene IoU 0.0 is inadmissible as a grounding score** |
| qwen3.6 rep 3 | `scene_single` @ 24 000 | **scene IoU 0.0 is inadmissible as a grounding score** |
| nemotron3 rep 1 | `finetext` @ 24 000 | fine-text cell void; scene/document unaffected |
| nemotron3 rep 3 | `finetext` @ 24 000 | fine-text cell void; scene/document unaffected |

No gemma4 cell capped. **No nemotron3 `scene_single` cell capped** — which matters, because
nemotron3's verdict rests entirely on that axis.

## Results

### gemma4:31b-it-q4_K_M — ADR 0023's permission does **not** transfer

| arm | scene IoU | document IoU | multi | finetext | scene tok | bbox space |
|---|---|---|---|---|---|---|
| off (greedy) | 0.961 | 0.728 | ✅ | ✅ | 530 | `norm1000/xyxy` |
| on rep 1 | 0.963 (+0.002) | **0.666 (−0.062)** | ✅ | ✅ | 7 573 | `norm1000/xyxy` |
| on rep 2 | 0.890 (**−0.071**) | 0.716 (−0.012) | ✅ | ✅ | 5 901 | `norm1000/xyxy` |
| on rep 3 | 0.960 (−0.001) | 0.718 (−0.010) | ✅ | ✅ | 8 713 | `norm1000/xyxy` |

12/12 cells valid, none capped, dialect stable throughout. This is the cleanest family in the
campaign and the result is unambiguous:

| | document-IoU deltas | mean |
|---|---|---|
| **gfx1151** (here) | −0.062, −0.012, −0.010 | **−0.028** — negative in every rep |
| Apple Silicon (ADR 0023) | +0.001, −0.001, +0.047 | +0.016 — negative in none |

Two of three deltas clear the ±0.01 noise floor, and the third sits on it; there is no rep in
which thinking did not cost document grounding here. Scene adds a −0.071 rep against two at
parity.

**Reasoning is also 3–4× more expensive here than there.** Scene: 5 901–8 713 tokens against
530 think-off = **11–16×**, where ADR 0023 measured ≈4× for this family. Same model, same card
values, same `num_predict`.

ADR 0023's finding that gemma4's greedy-measured cost was "an artefact of greedy decoding" is
correct **for its host**. On gfx1151 the cost survives on-policy sampling.

### qwen3.6:35b-a3b-q4_K_M — verdict confirmed, and worse than on Apple Silicon

| arm | scene IoU | document IoU | multi | scene tok | bbox space |
|---|---|---|---|---|---|
| off (greedy) | 0.953 | 0.320 | ✅ | 550 | `norm1000/xyxy` |
| on rep 1 | **capped** (inadmissible) | 0.290 (−0.030) | ✅ | **24 000** | none |
| on rep 2 | 0.537 (−0.416) | 0.277 (−0.043) | ✅ | 16 807 | **`pixel/xyxy`** |
| on rep 3 | **capped** (inadmissible) | 0.173 (−0.147) | ✅ | **24 000** | none |

**Two of three `scene_single` cells failed to terminate on-policy**, hitting `num_predict`
exactly with `json_valid: false`. ADR 0023 measured ~8% of cells non-terminating on Apple
Silicon (1 of 12); here it is 2 of 3 on this test alone. The single rep that did terminate
flipped the coordinate dialect from `norm1000` to `pixel` — the same instability ADR 0023
recorded in its rep 2, reproducing on different silicon.

Document IoU is negative in all three reps and, unlike scene, none of those cells capped, so
those deltas are admissible: −0.030, −0.043, −0.147.

`multi_3img` returned valid JSON in all three reps, confirming ADR 0023's correction that
multi-image non-termination — the mechanism ADR 0022 originally cited — is not the failure
mode. It is `scene_single`.

### nemotron3:33b-q4_K_M — verdict confirmed on the same terms ADR 0023 set

| arm | scene IoU | document IoU | multi | finetext | scene tok | bbox space |
|---|---|---|---|---|---|---|
| off (greedy) | 0.840 | 0.058 | ✅ | ✅ | 506 | `norm1000/xyxy` |
| on rep 1 | 0.736 (−0.104) | 0.000 | ✅ | **capped** | 10 827 | `norm1000/xyxy` |
| on rep 2 | **0.165 (−0.675)** | 0.198 | ✅ | ✅ | 5 332 | `norm1000/xyxy` |
| on rep 3 | 0.384 (−0.456) | 0.024 | ✅ | **capped** | 3 112 | `norm1000/xyxy` |

Scene grounding falls in every rep, **no scene cell capped**, and the coordinate dialect is
stable in all three. That is exactly ADR 0023's argument for this family: neither failure mode
that sampling explains is present, so the degradation is a real property. It reproduces on
different silicon, a different payload and a different sampling source.

Direction matches; spread is wider here (0.736 / 0.165 / 0.384 against Apple's 0.627 / 0.460 /
0.462).

Think-off document IoU is **0.058** — the floor, as ADR 0023 also found (0.045 there) — so that
axis carries no signal and the verdict rests on scene, as it does there.

**New here:** `finetext` hit the 24 000 cap in 2 of 3 reps. ADR 0023 reports 12/12 valid with no
caps for this family, so this is a gfx1151-specific observation, not a contradiction of it.

## What this changes

1. **`gemma4` — think stays off on this platform.** The permission ADR 0023 grants is
   host-scoped to its measurement and does not reproduce here. Enabling thinking for gemma4
   on gfx1151 costs document grounding in every measured rep and 11–16× the tokens.
2. **`qwen3.6` — unchanged, and reinforced.** Think off.
3. **`nemotron3` — unchanged, and reinforced.** Think off.

So on this lineage the operational rule is `think: false` for every family measured — the same
practical guidance [ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) gave, but now
reached per-model from on-policy measurements rather than as a blanket rule resting on a
greedy artefact. ADR 0022's *mechanism* was still wrong: its gemma4 figure and its "qwen3.6
multi-image does not terminate" claim are both superseded, and this campaign confirms the
replacements.

Recorded as [ADR 0025](adr/0025-think-stays-off-on-gfx1151.md).

## Limitations

- One host, one payload (b9888). The AMD upgrade gate pins this lineage, so a newer payload
  cannot be measured here without lifting it.
- gemma4 was measured at 31B only; ADR 0023's gemma4 evidence is also 31B, so the comparison
  is like-for-like, but neither covers 12B or 26B-a4b.
- n = 3 satisfies ADR 0023's rule and is enough to establish sign consistency; it is not
  enough to put a confidence interval on any individual delta.
- qwen3.6 scene grounding on-policy is **unmeasured** here rather than bad: two of three cells
  never produced a scorable answer. A larger `num_predict` was not attempted — ADR 0023's
  fourth trap notes that above ~90 K the HTTP timeout expires first, converting a cap into an
  error with no data.
