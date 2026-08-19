# Vision campaign 2026-08-20 — the scene prompt, runaway reasoning, and a hypothesis that inverted

CUDA Blackwell `10.8.0.6:11497`, `0.32.14-rc0-dynres-0-ga5d6590`, GGUF Q4_K_M
only, eviction-cold per [ADR 0031](adr/0031-model-residency-is-managed-client-side-on-remote-hosts.md),
temperature 0, n=1. Five models × 2 think modes × 3 prompt arms.

**Read the process section too.** The headline claim from this work was published
as CONFIRMED, retracted as confounded, then refuted with the sign backwards. The
sequence is the most transferable part.

## 1. The arms

One variable each, verified programmatically before launch:

| arm | convention | image dims stated | `__IMAGE__` anchor |
|---|---|---|---|
| `scene_single` | ABSOLUTE PIXEL | ✅ | — |
| `scene_single_pinned` | norm-1000 | ✅ | — |
| `scene_single_anchored` | norm-1000 | ✅ | ✅ |

`scene_single` is **unchanged** — every published scene number was measured with
it, and rewriting it would silently invalidate the corpus.

## 2. Results

| model | arch | think | baseline (pixels) | pinned (norm-1000) | anchored (+`__IMAGE__`) | eval tok px / pin / anc |
|---|---|---|---|---|---|---|
| gemma4:26b-a4b | MoE | false | 0.973 | 0.973 | 0.973 | 536 / 536 / 611 |
| gemma4:26b-a4b | MoE | on | 0.334 | **0.000** | 0.711 | 7576 / 122880 / 4145 |
| qwen3.6:35b-a3b | MoE | false | 0.975 | 0.975 | 0.964 | 548 / 544 / 620 |
| qwen3.6:35b-a3b | MoE | on | 0.971 | **0.044** | 0.640 | 15518 / 4792 / 8423 |
| gemma4:31b | dense | false | 0.962 | 0.962 | 0.958 | 538 / 538 / 605 |
| gemma4:31b | dense | on | 0.962 | 0.964 | 0.749 | 1372 / 2377 / 3024 |
| nemotron3:33b | hybrid | false | 0.870 | 0.854 | 0.849 | 512 / 506 / 574 |
| nemotron3:33b | hybrid | on | 0.753 | **0.434** | 0.499 | 10163 / 6792 / 6636 |
| qwen3.8:27b | dense | false | 0.977 | 0.830 | 0.988 | 544 / 610 / 687 |
| qwen3.8:27b | dense | on | 0.973 | 0.981 | 0.962 | 1069 / 1914 / 1419 |

## 3. What this establishes

**The norm-1000 pin does not fix runaway reasoning on this task. It causes it.**
Worse than the pixel baseline in 3 of 5 models under thinking, catastrophically
in two: gemma4:26b-a4b scores **0.000** after burning the full **122,880**-token
budget, qwen3.6 scores **0.044** against a **0.971** baseline.

**Think-off is unaffected everywhere** (0.973 / 0.975 / 0.962 / 0.870 / 0.977),
so this is a reasoning-time effect, not a parsing or convention failure.

**gemma4:26b-a4b think-on on scene is genuinely poor** — 0.334 against its own
0.973 think-off — and **no prompt variant recovers it**. Anchored reaches only
0.711. That is a model property, and it is why its think-on throughput is 69
req/h against 618 think-off.

**qwen3.6 has no scene problem.** Baseline think-on is 0.971. Its runaway is
specific to `multi_3img`
([campaign](vision-campaign-2026-08-19-qwen36-anchored.md)), and the general
claim "qwen3.6 runs away under thinking" was over-broad.

**Observation, n=5, NOT a mechanism: the damage tracks architecture, not baseline
quality.** Both MoE models and the hybrid are harmed; both dense models are not.
Baseline quality does not predict it — qwen3.6 starts at 0.971 and collapses to
0.044, while gemma4:31b starts at 0.962 and is untouched. The gemma4 pair is the
cleanest evidence available: same family, quantisation, host and prompt, differing
only MoE vs dense — **0.334 vs 0.962** baseline, and the pin drives the MoE to
0.000 while the dense sibling holds at 0.964.

There is no account here of *why* sparse routing would interact with a normalized
coordinate space under thinking. Another architecture pair on a different family
would settle whether this is real or a two-family coincidence.

## 4. Process — how the claim inverted

Worth recording because the failure was methodological, not experimental.

1. **Published CONFIRMED from n=1.** gemma4:26b-a4b went 0.334 → 0.972 under the
   pin, with 4× less reasoning. Committed to SPEC §0.1 and the learnings log,
   generalised as *"suspect the convention before the model whenever think-on
   runs away."*
2. **Retracted the same day as confounded.** The `pinned` arm changed **two**
   variables: it swapped the convention *and* dropped "The image is exactly {w}
   pixels wide and {h} pixels tall". qwen3.8 then answered in its own
   **2500×1400** rescale frame — the ~1.30× frame
   [SPEC §4](spec/vision-bbox-response-contract.md) measured — scoring 0.088,
   which read as a model failure and was a prompt defect.
3. **Refuted by the corrected run.** With dimensions retained, the pin is worse in
   3 of 5 models. The original "fix" came from *removing the dimensions*, not from
   the pin.
4. **A fifth model was omitted.** `gemma4:31b` was left out of the first four and
   turned out to carry the most information, being the dense sibling of the worst
   affected model.

**The guard that works**: assert the single-variable property **with code before
launching**, not in prose afterwards. When finally applied it printed the
confound in four lines. Tracked in
[the learnings log](vision-learnings-log.md), where "generalised from one model"
now stands at four occurrences.

## 5. Limits

n=1 per cell. One task, one scorer, one fixture at 1920×1080 — `score_scene` has
no anchor-derived conversion, so a mis-declared space cannot be recovered the way
the bbox arms recover it, which is likely *why* the pin behaves so differently
here.

**None of this bears on the bbox contract**, where the norm-1000 pin stands at
**111 of 112** cells across four models and 14 geometries
([ADR 0030](adr/0030-bbox-conformance-is-scoped-to-image-geometry.md)).
`scene_single` is a different task with a different scorer.
