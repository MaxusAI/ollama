# Vision campaign 2026-09-19/20 — ROCm/gfx1151 on the promoted build

`0.34.1-dynres-16649e8c` (0.34.1 + compat 906 + ADR 0036), five GGUF vision models, both
think modes, `OLLAMA_NUM_PARALLEL=2`, one runner. Run to give the promoted production build a
complete set of numbers, and to re-measure [ADR 0025](adr/0025-think-stays-off-on-gfx1151.md)
against a payload it was not decided on: that ADR rests on b9888 measurements, and this host
now serves b10864.

## Think off — the on-policy arm

```
## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 0.972 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 1.000 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.976 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 548 | 59 | 572 | 13.8 | 260 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 12 | 266 | 54.2 | 66 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 10 | 1068 | 55.1 | 65 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 52 | 4020 | 10.8 | 334 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 64 | 5069 | 8.5 | 425 |

Provenance (from score files): host(s) http://127.0.0.1:11499 · build(s) 0.34.1-dynres-16649e8c · think=false
```

## Think on

```
## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384/32768/131072 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 3/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.996 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.963 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384/131072 ⚠ | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 16384/32768 ⚠ | 0.743 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 3/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384/32768/131072 ⚠ | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | capped | ≥122880 ⚠ | 33 | 501 | capped | capped |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 3 | 4 | 2 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1171 | 12 | 263 | 105.4 | 34 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2560 | 10 | 93 | 276.0 | 13 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384/131072 ⚠ | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | capped | — | 3645 | 48 | 354 | 80.3 | 45 |
| nemotron3:33b-q4_K_M | GGUF | 16384/32768 ⚠ | 3 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ❌ q4_bbox_hit | — | 3846 | 64 | 608 | 64.8 | 56 |

Provenance (from score files): host(s) http://127.0.0.1:11499 · build(s) 0.34.1-dynres-16649e8c · think=on
```

## What this does to ADR 0025

**The policy holds, and the evidence is now on the payload it governs.** Think-on buys nothing
and costs between 3x and 5x the latency:

| model | scene IoU off → on | s/req off → on | think-on verdict |
|---|---|---|---|
| `qwen3.6:35b-a3b` | 0.972 → **capped** | 13.8 → capped | **does not converge** |
| `qwen3.8:27b` | 1.000 → 0.996 | 54.2 → 105.4 | equal quality, 2x slower |
| `gemma4:31b` | 0.960 → 0.963 | 55.1 → **276.0** | equal quality, 5x slower |
| `gemma4:26b-a4b` | 0.976 → 0.973 | 10.8 → 80.3 | equal, and escalated to 131072 |
| `nemotron3:33b` | 0.862 → **0.743** | 8.5 → 64.8 | **worse**, and loses the serial |

Not one model is better with thinking on. Two are materially worse.

**`qwen3.6:35b-a3b` is the extreme case and deserves its own line.** It climbed the entire
context ladder — 16384, 32768, 65536, 131072 — over eight hours, emitted **334,892 characters
of thinking** and 122,880 generated tokens, and was still `capped` at the ceiling. There is no
converged think-on value for this model on this host; `capped` at `CTX_MAX` is not an unfinished
cell but a result, and the result is that the arm does not terminate. See
[runaway-reasoning-under-think.md](runaway-reasoning-under-think.md).

**But the two extremes sit under one policy, and that is worth noticing.** `qwen3.8` converges
in one rung at 0.996 against a think-off 1.000 — think-on there is merely pointless, not broken.
ADR 0025's blanket "think stays off for every measured family" is carrying both cases. It
remains the right *operational* rule — nothing here argues for turning thinking on — but
[ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md) says the decision is
per-model on on-policy measurement, and on that framework `qwen3.6` and `qwen3.8` are failing
for different reasons. Recorded rather than acted on.

## Thinking did engage everywhere — the `Think tok —` dash does not mean zero

Every model emitted thinking. The dash in the Think tok column is SPEC H14's
stamped-count-or-dash: the count comes only from `token_split.py`'s gate-proven split, which
was not run here, so the column is honestly blank rather than estimated. `thinking_chars` on
the scene arm shows what actually happened:

| model | thinking_chars | eval_count off → on |
|---|---|---|
| `qwen3.8:27b` | 1,878 | 544 → 1,171 |
| `gemma4:31b` | 3,873 | 538 → 2,560 |
| `gemma4:26b-a4b` | 7,067 | 536 → 3,645 |
| `nemotron3:33b` | 8,389 | 512 → 3,846 |
| `qwen3.6:35b-a3b` | **334,892** | 548 → **122,880** |

This is worth stating because a blank column reads as "no thinking" to anyone who does not know
H14, and the first draft of this campaign's status report made exactly that mistake about
`qwen3.8`.

## OCRBench — all five models, one campaign

Rows 0–200 of `echo840/OCRBench`, think off, `-np 1`, rendered by `summarize_extbench.py`:

```
| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `qwen3.6:35b-a3b-q4_k_m` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `qwen3.8:27b-q4_K_M` | 200 | 0 | 0 | 162 | **0.81** | false | generate |
| `gemma4:26b-a4b-it-q4_K_M` | 200 | 0 | 0 | 165 | **0.825** | false | generate |
| `nemotron3:33b-q4_K_M` | 200 | 0 | 0 | 175 | **0.875** | false | generate |

host: http://127.0.0.1:11499 · build: 0.34.1-dynres-16649e8c
```

The spread across five models (0.810–0.875) is barely wider than one model's own confidence
interval at n=200 (±0.025 at p≈0.85), so **this is not a ranking**. `nemotron3`'s 0.875 equals
mlx-metal's number on the same slice; nothing follows from that beyond both being inside the
same interval.

`gemma4:31b` was re-run to clear an H13 `MIXED` footer: its first file predated the `extbench.py`
provenance fix and carried no host or build, so a footer built from the other four would have
vouched for it. The re-run returned **171/200 again, to the item** — which also confirms the
slice is deterministic on this host under think off at temperature 0.
