# Vision campaign 2026-08-19 — qwen3.6 think-on does not terminate, and the anchor bounds it

Two hosts, two builds. Apple Silicon `0.32.14-maxusai-9594f81e` and CUDA
Blackwell `0.32.14-rc0-dynres-0-ga5d6590` (GGUF only — `nvfp4` is MLX and that
path is unstable on that build). Temperature 0, think-on, `multi_3img` and
`multi_3img_anchored`.

**The question.** The 18-model campaign recorded `qwen3.6:35b-a3b` think-on
failing `q4_bbox_hit`. That looked like the qwen3.8 frame artefact
([corrected earlier](vision-campaign-2026-08-17-qwen38-rocm.md)). It is not.

## 1. It is not a scoring artefact — it is non-termination

| arm | tokens generated | reasoning | answer | outcome |
|---|---|---|---|---|
| `multi_3img` | **122,880** (= `num_predict`) | 313,054 chars | **0 chars** | never finishes |
| `multi_3img_anchored` | **10,910** | — | complete | ✅ correct |

Measured on CUDA at `num_ctx` 131072. The ladder was raised to a 131072 rung
specifically for this: at the old 65536 ceiling the cell hit `eval_count 57344 ==
num_predict 57344`, which established only *"needs more than 57,344 tokens"* — a
floor, not a cost ([ADR 0012](adr/0012-benchmark-report-templates.md) rule 8).

Doubling the budget changed nothing. **Two ceilings a factor of two apart both
produced the identical outcome**, and reps 1 and 2 are byte-identical at
122,880 / 313,054 / 0. This is deterministic non-termination, not sampling.

**The difference is qualitative, not an efficiency gap.** Four lines of
calibration text — the `__IMAGE__` entry — turn an unbounded request into one
that completes in 10,910 tokens. For req/hour planning the unanchored
configuration has no rate: it never completes.

It reproduces on **both platforms and both builds**, so it is a property of the
model and prompt rather than of a backend.

## 2. Three quantisations, three different think-on behaviours

| tag | `multi_3img` | anchored |
|---|---|---|
| `q4_K_M` | ❌ non-termination | ✅✅✅ |
| `nvfp4` | ❌ non-termination | ✅✅❌ — `q4_bbox` still misses |
| `q8_0` | ✅✅✅ | ✅✅✅ *(salvaged)* |

Only `nvfp4` resembles the qwen3.8 frame artefact. `q8_0` never had a problem:
its ❌❌❌ was a **scoring** failure, recovered by `salvage.py` — the model had
serialised `answers` into a string inside an unrelated array, so the JSON parsed
and the key was simply absent. The recovered values match its own passing arm
exactly. See the C12 amendment in
[the contract SPEC](spec/vision-bbox-response-contract.md).

**Correction.** An earlier reading of this data said the anchor made `q8_0`
worse. That was the scorer, not the model. The anchor never hurt any cell.

## 3. Think-off is clean

All three tags, both arms, ✅✅✅ 5/5 at 16384 with no ladder escalation. Every
finding above is think-on only, which is consistent with
[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md).

## 4. Limits

`nvfp4` was measured on Apple only — it is an MLX tag and that path is unstable
on the CUDA build. The CUDA cells ran without `RESTART_CMD` (remote host), so
they are eviction-cold-started per [ADR 0031](adr/0031-model-residency-is-managed-client-side-on-remote-hosts.md)
and their `load_duration` is not comparable with `RESTART_CMD` runs. One rep was
lost to a container restart mid-generation and is recorded as an error, not a
❌ — the retry policy added in ADR 0031 decision 6 exists because of it.

This run does **not** reconcile with the 18-model campaign's `q8_0` row, which
recorded `q1`/`q2` failing where these cells pass. Build, test subset and power
provenance all differ, and until that is attributed the campaign row stands
uncorrected.
