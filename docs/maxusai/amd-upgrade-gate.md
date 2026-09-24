# Upgrade gate for AMD / gfx1151: 0.32.5 is blocked

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-07-31 after
`0.32.5-gemma4budget-4259c191` produced degenerate output on the gfx1151 host and was rolled
back to `0.32.1-gemma4budget-85ebcb79`.

> **LIFTED 2026-09-19.** The AMD/gfx1151 deployment moved to `maxusai-ollama:0.34.1-rocm724-main-16649e8c`
> (llama.cpp b10864 + `llama/compat/906-revert-hip-integrated-flag.patch`) and has served `main`
> since; it runs `maxusai-ollama:0.34.2-rocm724-main-f67b1aef` from 2026-09-21. Clauses 3 and 4
> are satisfied on measurement; clauses 1 and 2 were **overridden on evidence** because they
> could never be satisfied as written — see [the 2026-09-19 decision](#decision-2026-09-19--the-gate-lifts-on-evidence).
> The history below is kept in full: it is why this fork does not trust a plumbing check.
>
> **What "not merely closed" was protecting against, and why it stopped applying.** Clauses 1
> and 2 required the two issues to be *fixed, with the fix in the target tag*. Both were
> closed **`not_planned`** — #17475 on 2026-08-12, #17459 on 2026-08-23 — so no fix exists to
> be in any tag, and the clauses became unsatisfiable rather than unsatisfied. Waiting on them
> was no longer waiting for anything. This gate is about the **upstream payload**, not about
> anything the fork changed — moving *within* a payload lineage, as the 2026-08-08 deploy did,
> is not an upgrade and is not gated.

## Status

| | |
|---|---|
| Deployed image | `maxusai-ollama:0.34.3-rocm724-main-650f8fda` (promoted 2026-09-25 07:32) |
| Deployed version | `0.34.3-dynres-0-g650f8fd`, a build of `main` at the `v0.34.3-dynres` tag (ADR 0032) |
| Build type | `Dockerfile.rocm` through `scripts/build_rocm.sh` (`ROCM_TOOLCHAIN=rocm7 AMDGPU_TARGETS=gfx1151`), on `rocm/dev-ubuntu-24.04:7.2.4-complete` ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md)). **This is the first Ubuntu-built production image on this host**, and the payload's `ROCM_IMAGE` stamp says so. It is gfx1151 only. Compat **001 + 002 + 004 + 005 + 801 + 802 + 903**: 802, the LM-graph node meter, is new here and inert unless `OLLAMA_LM_NODE_STATS` is set |
| Payload | **b10969** (`391fac164`), unchanged from 0.34.2; compat 906 **retired** — b10969 ships upstream's own HIP `prop.integrated` revert |
| Previous image | `maxusai-ollama:0.34.2-rocm724-main-f67b1aef` (`0.34.2-dynres-f67b1aef`, AlmaLinux-built, b10969) — **retained for rollback** as the stopped container `ollama-rocm-0.34.2-f67b1aef`; `0.34.1-rocm724-main-16649e8c` (b10864 + 906) and `0.32.1-rocm-dynres-5d5b7a72` (b9888) before it |
| Superseded pin | `0.32.1-dynres-296eb020` recorded here until 2026-09-19; the host was in fact running `5d5b7a72`, so this row had drifted from the host it describes |
| Blocked target | `0.32.5-gemma4budget-4259c191` (built, verified, **rolled back** 2026-07-31) — never unblocked; superseded, not cleared |
| Host | Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151**, ROCm, Linux |
| Gate status | **lifted 2026-09-19** on the decision below. Clauses 3 and 4 measured; 1 and 2 overridden as unsatisfiable |
| Issue status 2026-09-19 | #17459 **closed `not_planned`** 2026-08-23; #17475 **closed `not_planned`** 2026-08-12. Neither reproduces on this host |

The 2026-08-08 change swapped one b9888 image for another and **did not touch the gate**.
The overlay image it replaced could not carry `llama/compat/*.patch` at all, so shipping
002/004/005 required a full build — see [ADR 0006](adr/0006-release-lineage-is-never-merged-into-main.md)
and the deploy record in [nemotron-test-image.md](nemotron-test-image.md).

**Where these docs live.** `main` is the canonical home for fork documentation, including
this gate. `release/0.32.1-dynres` deliberately does **not** carry them — duplicating docs
across both lineages is the cost ADR 0006 exists to avoid, not something to chase. Read the
docs on `main`; build the artefact from the release branch.

## What was observed

Vision requests driven from `10.8.0.6` against `10.8.0.4`, `qwen3.6:35b-a3b-q4_k_m`
(`qwen35moe`), `/api/chat`:

| attempt | build | result |
|---|---|---|
| 1 | `0.32.5-…-4259c191` | 2 ok / 8 → then **6 degenerate** |
| 2 | `0.32.5-…-4259c191` | 0 ok → **degenerate inside row 1** |
| 3 | rollback landed mid-run | refused: `version_mismatch` |
| 4 | `0.32.1-…-85ebcb79` | **6 ok / 6, 0 degenerate** |

Every degeneration occurred on the new build; none on the old one. `prompt_eval` on the good
rows spanned 5,317–11,941, varying with image and tags, so vision was genuinely active
throughout rather than silently skipped.

**This is correlation, not a controlled experiment.** The rollback was not scheduled as part
of the test, and a container restart alone clearing the state cannot be excluded. It is
recorded as strong evidence, not proof.

## Ground-truth reproduction (2026-08-01)

The failure class was reproduced deterministically on this host with the ground-truth
vision suite in [nemotron-test-image.md](nemotron-test-image.md): on a b10091-payload
build, `gemma4:31b` — at budgets identical to the gated 0.32.1 build — dropped from
6/6 labels + a perfect invoice extraction + 5/5 chart values to 3/6 / 0/5 / 0/5, emitted
degenerate token salad on the multi-image test, and twice produced responses describing
a **previous request's** image (the #17475 shared-slot signature, also seen on
nemotron3). Temperature 0, same model blob, same prompts. The gate's "degenerate vision
output" observation is therefore not workload-specific and is now regression-testable.

## Corroborating upstream reports

Both open, both 0.32.5-specific, both matching this deployment:

**[#17459](https://github.com/ollama/ollama/issues/17459) — repeated-token degeneration.**
Reported on **Framework Desktop Max+ 395 / AMD Radeon 8060S** — the same SoC and the same
iGPU as this host — on ollama 0.32.5 via `/api/chat`: Gemma 4 emits repeated `<unused49>`
tokens when the request carries `think: false`. Same silicon, same version, same endpoint,
same symptom class.

**[#17475](https://github.com/ollama/ollama/issues/17475) — shared-slot cross-request
corruption.** Its runner config is nearly identical to ours — `-np 1`,
`--context-shift --keep 4`, `--direct-io`, vision + mmproj — and it names `qwen3.6:35b` among
the models involved. Reproduces 8/8 under concurrent submission + client aborts + model
swapping, and 0/6 when calls are serialized. Explains the
`erased invalidated context checkpoint` / `cached n_tokens = 0` churn seen in our logs.

## Why `--direct-io` is a no-go on AMD for now

[`72116baf`](https://github.com/ollama/ollama/pull/17286) (Daniel Hiltgen, upstream, not
MaxusAI) adds to `llm/llama_server.go`:

```go
if runtime.GOOS == "linux" && g.Integrated && (CUDA || ROCm) {
    params = append(params, "--direct-io")
}
```

**gfx1151 satisfies every clause** — Linux, integrated, ROCm — so the flag is applied
unconditionally here with no opt-out. Verified in the live runner cmdline: **present on
0.32.5, absent on 0.32.1.**

Its stated purpose is sound ("avoid double memory consumption by enabling direct IO for
iGPUs" — on shared-memory parts the page cache double-buffers weights). The objection is not
to the intent:

1. **It is unconditional and unexposed.** No env var or option disables it. On this host it
   arrives purely as a side effect of a version bump.
2. **It changes the weight-load path on the exact hardware class that regressed**, and
   O_DIRECT is alignment-sensitive — a silently short or misaligned read yields corrupted
   weights, whose signature is degenerate output rather than an error.
3. **It is present in #17475's runner config too**, on a DGX Spark GB10 — also a unified-memory
   part, also dio-enabled by the same commit. Two affected systems, both with dio on.
4. **It is unvalidated on ROCm iGPUs by us.** It was never measured on gfx1151 before it
   arrived; the 0.32.5 verification checked offload and throughput, not load-path integrity.

It is **not** proven to be the cause — #17475 attributes its corruption to slot sharing, and
degeneration could equally come from the llama.cpp payload bump. It is listed as a blocker
because it is an unvalidated, non-optional change to the load path on the affected hardware.

## What is *not* the cause

**The `qwen35`/`qwen35moe` 1024 image-token floor is exonerated.**
[`87cf1100`](https://github.com/MaxusAI/ollama/pull/10) was the *intended* change in the
0.32.5 cutover and drew the initial suspicion, wrongly. It alters prompt length
(313 → 1049 tokens for a 640×480 image), not output coherence, and appears in neither
upstream issue. The real risk was the **incidental payload** that rode along:

| layer | 0.32.1 → 0.32.5 | rebuilt by the overlay? |
|---|---|---|
| Fork runtime code | 1 commit (`87cf1100`) | yes (Go binary) |
| Upstream Go | 9 serving-path commits, incl. `72116baf` dio | yes (Go binary) |
| **llama.cpp payload** | **`b9888` → `b10091`** (~200 builds) | **no — comes from the base image** |

The overlay rebuilds only the Go binary, so the llama.cpp bump is never compiled or reviewed
here; it arrives wholesale inside `ollama/ollama:0.32.5-rocm`. That is the largest and least
inspected surface in any base bump, and it is where slot, checkpoint and prompt-cache logic
lives.

## The lift path has no owner — review this, do not wait on it

Checked 2026-08-16. Neither gate issue has a fix, a PR, or an assignee:

| | |
|---|---|
| #17459 | **open**. A contributor offered to take it; nothing landed. A maintainer has since questioned the reproduction itself ("the gemma output files show no sign of `<unused49>`. Was the model reloaded?"), so it is still in triage. |
| #17475 | **closed `not_planned`** after a *single* maintainer comment asking for server logs, image dimensions and the prompt. There is no sign the reporter replied. Closed for silence, not because it was judged harmless or repaired. |

So conditions 1 and 2 have no visible route to being satisfied. Treating this gate as
"wait until upstream fixes them" means waiting indefinitely.

That is not a reason to lift it — the defects are real and the 2026-07-31 rollback is the
evidence. It is a reason to give the gate an **owner-decided review date** rather than a
passive condition, and to decide deliberately between: keep gfx1151 on b9888 indefinitely;
validate a newer payload ourselves against conditions 3-5 and accept 1-2 as unmet with the
risk documented; or push the upstream issues forward.

On the last option, we are unusually well placed for #17475. Its reproduction needs
concurrent vision requests with client aborts across model swaps, on hardware where the
leak appears — and this fork already cold-restarts between every benchmark cell *because of
this bug* (`run_grid.sh` cites it by number). The maintainer asked for exactly what the
vision suite captures routinely. Reopening it with a controlled reproduction is a real
option; it is a third-party issue touching PII, so it is a decision, not a task.

## Spec — normative gate

Before moving the AMD/gfx1151 deployment past `0.32.1`, **all** must hold:

1. ~~[#17459](https://github.com/ollama/ollama/issues/17459) is **closed**, with the fix in the
   target tag.~~ **Superseded 2026-09-19** — closed `not_planned`, no fix exists. Replaced by:
   *#17459's symptom does not reproduce on this host on the target payload*, demonstrated across
   every payload tested (`rocm-gate-issues-result.md` § "#17459 — does not reproduce, on any payload").
2. ~~[#17475](https://github.com/ollama/ollama/issues/17475) is **closed**, with the fix in the
   target tag.~~ **Superseded 2026-09-19** — closed `not_planned`, no fix exists. Replaced by:
   *#17475's contamination does not reproduce under the reporter's protocols on this host*, 436
   victim extractions, three instrument corrections (same document, § "#17475 — no contamination").
   **This is the weaker of the two replacements** and it is stated as such: a clean run is not
   proof of absence for a race, and MaxusAI/ollama#313 still asks CUDA whether the HIP defect
   class reaches unified memory, which would make #17475 that bug in disguise.
3. `--direct-io` is either (a) absent for ROCm iGPUs in the target, (b) opt-out-able and
   disabled here, or (c) validated on gfx1151 — a load-path integrity check, not just
   throughput.
4. A vision A/B of **≥6 consecutive rows per build** on `qwen35moe` shows **0 degenerate** on
   the candidate, run with the rollback boundary controlled — the deficiency in the
   2026-07-31 evidence.
5. `make proof` passes against the new `BASETAG`, and `BASETAG`/`STOCK` moved together.

Gates 1–2 are upstream-dependent; 3–5 are ours. Record the outcome of each here when the gate
is next evaluated.

## Decision record

**Context.** `0.32.5-gemma4budget-4259c191` was built, verified (identity, 96 gfx1151
kernels, 100% GPU offload, budget behaviour) and deployed on 2026-07-31. It passed every
check we had. Those checks measured *plumbing* — flags, offload, token counts, throughput —
and none of them would detect degenerate output, which is what a downstream consumer hit
within the hour.

**Decisions.**

1. **Pin AMD/gfx1151 to `0.32.1-gemma4budget-85ebcb79`.** Recorded in
   `docker/ollama-rocm/.env` with the full image chain.
2. **Treat a base-image bump as a payload change, not a version bump.** The overlay makes
   base bumps look cheap — one build arg — while silently importing every llama.cpp change
   between tags. Sync deliberately, and read the upstream issue tracker for the target tag
   before deploying.
3. **Add output-quality checks to the deploy gate.** Plumbing checks passed while the build
   was producing garbage. Verification must include generating and inspecting real output,
   not just counting tokens and VRAM.
4. **Do not backport the 0.32.5 payload to get the qwen35 floor.** If the floor is wanted
   before the gate lifts, cherry-pick `87cf1100` onto `85ebcb79` and rebuild on the
   **0.32.1-rocm** base — one switch statement, none of the payload.

**Consequences.** The AMD host stays on `0.32.1`, so `qwen35`/`qwen35moe` keep the
llama.cpp default floor and small images stay cheap (313 rather than 1049 tokens at
640×480). Non-AMD deployments are **not** covered by this gate — `10.8.0.6` (Blackwell,
CUDA) is unaffected by clause 3 and may track a different version; note that #17475 was
reported on CUDA, so clauses 1–2 still apply there.

## Decision 2026-09-24 — ROCm images build on Ubuntu, not AlmaLinux

**Glenn, until further notice:** the fork's ROCm images build on AMD's Ubuntu 24.04 ROCm
images — `rocm/dev-ubuntu-24.04:7.2.4-complete` for `rocm7`, `rocm/dev-ubuntu-24.04:10.0.0-full`
for `rocm10` — and nothing the fork owns uses `rocm/dev-almalinux-8`. The recipe is
`Dockerfile.rocm` through `scripts/build_rocm.sh`; upstream's `Dockerfile` is no longer how this
host's images are built. Reasons and consequences:
[ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md).

**This is a toolchain change under an unchanged ROCm release, and it is gated like one.** The
same 7.2.4 runtime now comes from AMD's Ubuntu packages and links against glibc 2.39 and the
system GCC 13.3; preflight's `toolchain_build = "rocm-7.2.4"` passes either way. The first
Ubuntu-built image therefore passes clause 4 and the OCRBench slice **against the
AlmaLinux-built production image** before it is promoted — the v0.34.3 fold's gfx1151
regression run ([MaxusAI/ollama#372](https://github.com/MaxusAI/ollama/pull/372), recorded in its task doc).

## Decision 2026-09-25 — 0.34.3 promoted, the first Ubuntu-built production image

**Outcome: promoted.** `ollama-rocm` moved from `0.34.2-dynres-f67b1aef` to `0.34.3-dynres-0-g650f8fd` at 07:32
on 2026-09-25. It was down for about two seconds. The payload is unchanged: b10969 (`391fac164`). What moved is
the Go side (v0.34.3: thinking levels in the API) and the toolchain packaging (ADR 0042).

Rollback:

1. Stop the new container and rename it aside.
2. Rename the retained 0.34.2 container back to `ollama-rocm`.
3. Restore its restart policy and start it.

```
docker stop ollama-rocm && docker rename ollama-rocm ollama-rocm-0.34.3-rolledback &&
docker rename ollama-rocm-0.34.2-f67b1aef ollama-rocm &&
docker update --restart unless-stopped ollama-rocm && docker start ollama-rocm
```

The retained container has `--restart no`, so a reboot cannot start it against the new one on `:11434`.

**What the evidence covers, and why it covers the promoted image.** The v0.34.3 fold's gfx1151 regression run
([MaxusAI/ollama#372](https://github.com/MaxusAI/ollama/pull/372), recorded in
[upstream-sync-0.34.3.md](tasks/upstream-sync-0.34.3.md)) measured the candidate
`0.34.2-dynres-24-gef19770-rocm7-gfx1151` against this production image:

- Think off: every scored cell of five models is equal.
- OCRBench: every item is equal, 172 = 172.
- Think on (#377): the three greedy models are equal, and the two sampled models are flat as rates.

The promoted image is a separate build, at the tag. It carries that evidence because:

- **Native payload.** It is byte-identical to the candidate's. `llama-server`, `libllama-server-impl.so`,
  `libggml-hip.so`, `libggml-base.so` and `libmtmd.so` have the same sha256.
- **Payload structure.** `payload_diff.sh` finds 1863 = 1863 entries, no SONAME or symlink change, and 96/96
  gfx1151 rocBLAS kernels.
- **Go source.** It is identical too. The tag differs from the candidate's tree only in #374's build files
  (`Dockerfile.rocm`, `build_rocm.sh`, the ROCm presets, and a bundling regex that bundled nothing new here).

**Post-deploy check on the promoted container: 0 of 980 cells differ.** gemma4:26b-a4b, think off, was run through `run_engine_compare.sh` against `:11434` itself (server `0.34.3-dynres-0-g650f8fd`). Every scored cell equals the gate candidate's (`cmp_scores.py`). The check ran beside the v0.34.4 gate-6 campaign, on its own container; greedy scores on this host do not move with GPU contention.

### Clause outcomes

| clause | outcome |
|---|---|
| 1–2 | **As 2026-09-19.** Both overridden on evidence; nothing has changed upstream. |
| 3. `--direct-io` | **Satisfied as on 2026-09-21.** The payload did not move (b10969). |
| 4. Vision A/B, ≥6 consecutive rows, 0 degenerate | **PASSED.** All five models, think off, on the byte-identical candidate, equal to production in every scored cell. The promoted container itself was re-checked (above). |
| 5. `make proof` | **Waived — still does not exist.** |

**Not run: preflight (gate 5) for 0.34.3 on rocm7.** No rocm7 profile pins b10969 on a 0.34.3 stamp. The
`rocm7-0-34-4-dynres` profile added on the v0.34.4 fold branch (#378) matches the stamp but pins b11081, so it would
fail `payload_pin` here, by design. Every row it measures reproduced the b10864 and b10969 values unchanged.

**How the host was confirmed idle.** A client polls `GET /api/ps` about every 2 s from the docker bridge. The
promotion counted only working requests, and there were none in the two minutes before the swap. No model was
loaded.

## Decision 2026-09-21 — 0.34.2 promoted, ROCm 10.0.0 declined on measurement

**Outcome: promoted.** `ollama-rocm` moved from `0.34.1-dynres-16649e8c` to
`0.34.2-dynres-f67b1aef` (llama.cpp b10969, `391fac164`) at 09:48 on 2026-09-21. The previous
image is retained; rollback is one `docker run` with the same arguments and the old tag, and the
promotion script refuses to start unless **both** images are present, because discovering the
rollback is gone after `docker rm -f` is too late.

Compat **906 is not in this build and is not needed**: b10969 ships upstream's own revert of the
HIP `prop.integrated` change, so the carry-patch was retired (`3dade569`). Scene IoU is at or
above the 0.32.1 baseline on all three models the 906 defect destroyed.

### Clause outcomes

| clause | outcome |
|---|---|
| 1–2 | **As 2026-09-19.** Both overridden on evidence; nothing has changed upstream. |
| 3. `--direct-io` | **Satisfied under (c) on b10969.** Re-validated because the payload moved. 0/54 blocks differ between dio-on and dio-off, both arms verified to differ at the runner flag line. |
| 4. Vision A/B, ≥6 consecutive rows, 0 degenerate | **PASSED on the promoted image**, five models, think off. Zero degenerate rows. |
| 5. `make proof` | **Waived — still does not exist.** |

### What the promotion buys, and what it does not

`qwen3.6`'s 7px fine-text tier — one of the two losses the 2026-09-19 decision recorded as open —
**comes back** on this payload, N=5, zero variance. `nemotron3`'s 9px does not; it stays open.

Prefill on `gemma4:31b` also improves sharply in practice, 301.6s → 117.0s across the 27-block
suite, but **this is a KV-cache-hit change, not a faster encoder**: cold-prefill compute is
167 → 172 tok/s and what moved is 5 of 27 blocks hitting cache against 20 of 27. Recorded
because reading it as compute is exactly the error SPEC H22 was written to stop.

### Why not ROCm 10.0.0

ROCm 10.0.0 was built, run and measured on gfx1151 for the first time in this cycle, and it is
**sound**: all 13 preset architectures compile with device code verified in the binary, clause 3
passes 0/54, clause 4 passes with zero degenerate rows, preflight returns PASS 15/0/14 with all
three token ladders unmoved, and the two moved fine-text cells are identical to 7.2.4.

It was declined anyway, on the only question that matters for this deployment: **it is slower
where vision work is bound.** Against a ±0.4% noise floor measured from two direct-I/O pairs on
this host, prefill on blocks that actually encoded is down on **all five** models — −3.3% to
−10.4% — against a decode gain of +0.5% to +6.4%
([rocm-10-throughput-2026-09-20.md](rocm-10-throughput-2026-09-20.md)). It also exists only on an
unmerged branch, so promoting it would have made production non-reproducible from `main`.

Soundness is not superiority. The ROCm 10 evidence stands as a validated future option, and the
upgrade is declined on measurement rather than deferred on doubt. The surface is carried as
experimental, with a standing instance and deliberately no measured preflight profile:
[ADR 0040](adr/0040-rocm-10-is-experimental-until-it-is-faster.md).

## Decision 2026-09-19 — the gate lifts on evidence

**Outcome: promoted.** `ollama-rocm` moved from `0.32.1-dynres-5d5b7a72` (b9888) to
`0.34.1-dynres-16649e8c` (b10864 + compat 906) at 18:55 on 2026-09-19. The previous image is
retained; rollback is one `docker run` with the same arguments and the old tag.

### Clause outcomes, as the spec requires them to be recorded

| clause | outcome |
|---|---|
| 1. #17459 | **Overridden.** Closed `not_planned` 2026-08-23 — declined, not repaired, so no fix can be in any tag and the clause was unsatisfiable rather than unsatisfied. Replaced by: does not reproduce on this host on any payload tested, including the target. |
| 2. #17475 | **Overridden, and this is the weaker of the two.** Closed `not_planned` 2026-08-12. No contamination in 436 victim extractions under the reporter's protocols, after three instrument corrections. A clean run is not proof of absence for a race — see the open question below. |
| 3. `--direct-io` | **Satisfied under (c), validated.** A/B on the fixed build, both arms verified to differ at the runner flag line: dio-on `--load-mode dio`, dio-off no flag. `qwen3.6` 0.972 and `gemma4:31b` 0.960 in **both** arms, fine-text and invoice identical. Also satisfied under (b): `OLLAMA_IGPU_DIRECT_IO=0` exists (MaxusAI/ollama#318). The default is left **on**, because dio measurably helps load times here and costs nothing in quality. |
| 4. Vision A/B, ≥6 consecutive rows, 0 degenerate | **PASSED on the promoted image**, five models, think off, `num_ctx` 16384. Table below. |
| 5. `make proof` against the new `BASETAG` | **Waived — the check does not exist.** No `proof` target in either this repo or `amd-rocm-ollama`, and no `BASETAG`/`STOCK` variables anywhere. The `docker/ollama-rocm/.env` named in the 2026-07-31 decision record is not a real path. Clause 5 has been unimplementable since it was written; recorded as waived rather than passed. |

### Clause 4 evidence — the promoted image, not a proxy for it

Clause 4 first passed on the `hip906` image (`968762ca`). The artifact promoted is `16649e8c`,
which adds only an inert-unless-set env knob, an optimization measured inert on this host, and
docs. It was re-run anyway, because "near-certainly equivalent" is the reasoning that put a
plumbing-verified build into production on 2026-07-31.

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 0.972 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 1.000 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.976 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

Provenance (from score files): host(s) http://127.0.0.1:11499 · build(s) 0.34.1-dynres-16649e8c · think=false

Against the 2026-09-17 baseline: `qwen3.6` 0.953 → **0.972**, `qwen3.8` 0.991 → **1.000**,
`nemotron3` 0.857 → **0.862**, `gemma4:31b` 0.961 → **0.960** (one thousandth, noise). Zero
degenerate rows; boxes, labels and colours 6/6 on every model and the invoice 5/5 on every model.

A post-promotion smoke test against the live `:11434` returned a correct description of the
scene fixture naming all six shapes, `done_reason=stop`, `prompt_eval_count` 1122 — the 1120
budget, filled.

### The fine-text cost, measured rather than assumed

Scene IoU is at or above baseline on four of five models and one thousandth below on the
fifth. **Fine text is not uniformly at baseline**, and the first version of this record said
"no regression" on the strength of the IoU column alone.

Against the 0.32.1 baseline, think off, `num_ctx` 16384, **N=5 per arm per model**:

| model | tier | 0.32.1 baseline | promoted `16649e8c` | 0.34.2 `f67b1aef` | + ROCm 10.0.0 | |
|---|---|---|---|---|---|---|
| `nemotron3:33b` | 9px | 4/4/4/4/4 | 3/3/3/3/3 | 3/3/3/3/3 | 3/3/3/3/3 | **−1, still open** |
| `qwen3.6:35b-a3b` | 7px | 2/2/2/2/2 | 1/1/1/1/1 | **2/2/2/2/2** | **2/2/2/2/2** | **recovered** |
| `qwen3.8:27b` | 9px | 2/2/2/2/2 | 3/3/3/3/3 | 3 *(N=1)* | 3 *(N=1)* | **+1, gain held** |

The last two columns are N=5 per cell per arm, measured 2026-09-21
(`vision-suite/bench-runs/finetext-n5-two-moved-cells-2026-09-21.json`) on the two cells that
moved *down*, because a recovery claim was about to enter this record on one capture. `qwen3.8`'s
gain is carried at N=1 from the campaign's own suite block and is marked as such rather than
borrowed from its neighbours' N.

**One of the two losses is back.** `qwen3.6`'s 7px tier returns to its baseline value of 2 on the
b10969 payload, five times out of five, on both ROCm 7.2.4 and ROCm 10.0.0, with zero within-arm
variance — and the suite arm and the probe arm agree digit for digit on all four builds, so the
SPEC H16 disagreement is not in play. `nemotron3`'s 9px tier does **not** come back: it is 3 on
every rep of every arm. Neither cell is moved by ROCm 10.0.0 in either direction.

Every other tier on every model is identical across builds, and 22/16/12px is 4/4 everywhere.

**These are not noise, and the reason they were nearly dismissed as noise is worth recording.**
The initial reading was one capture per cell, and the fork's own spread finding —
`vision-learnings-log.md`, 2026-09-18 — says a single capture reports the minority mode about
1 in 8. That finding is about **MLX greedy decoding on Metal**. On ROCm with think off and
temperature 0 the fine-text probe is deterministic: five captures per cell returned the *same*
value five times out of five, in all six cells, on both builds. Applying another platform's
variance to this one would have retired a real regression as sampling noise.

**The trade, stated plainly.** The promotion costs one fine-text item at 9px on `nemotron3`
and one at 7px on `qwen3.6`, and gains one at 9px on `qwen3.8`, against a payload fix worth
0.065 → 1.000 scene IoU on `qwen3.8` and 0.161 → 0.862 on `nemotron3`. That is worth taking,
but it is a trade, not a free upgrade. **Half of that trade has since been refunded**: on the
b10969 payload `qwen3.6`'s 7px item returns, leaving `nemotron3`'s 9px as the only standing loss.
Neither movement is attributed — not to compat 906, not to b10864, not to ROCm — and the one that
recovered did so across the same payload bump that `llama.cpp` #23660's im2col demotion sits in
(`vision-learnings-log.md`, 2026-09-20). `nemotron3`'s 9px is recorded as open.

### What is still open, and what would reopen this

- **Whether the HIP defect class reaches CUDA unified memory** (MaxusAI/ollama#313). If it
  does, #17475 is that bug in disguise, compat 906 closes it, and clause 2 is satisfied on the
  merits instead of waived. This is the single measurement that would most strengthen the
  decision recorded here.
- **Clause 5 needs an implementation or a deletion.** A gate clause that cannot be run is not a
  safeguard; it is a line of text that makes the gate look stronger than it is.
- **Compat 906 is load-bearing.** b10864 misses upstream's revert by 78 minutes. Any payload bump
  must carry 906 or land after `d4389a4dd92`, or the vision regression returns silently — no
  crash, no warning, correct-looking token counts.

## The deployable line — superseded 2026-09-19

**AMD/gfx1151 is now served from `main`,** like every other platform: the promoted image is a
full `FLAVOR=rocm` build of `16649e8c`. The three platforms have converged — CUDA moved to
0.34.1 on 2026-09-18, mlx-metal in MaxusAI/ollama#324, ROCm here.

> **Until 2026-09-19 this section read:** `main` tracks llama.cpp **b10091** — the payload this
> gate blocks. The AMD/gfx1151 host is served from **`release/0.32.1-dynres`**, pinned to
> **b9888** and predating `--direct-io`. That lineage is permanent for as long as the gate
> holds, and it is **never merged into `main`**. Fixes reach AMD by cherry-pick, adapted to
> b9888; they will not arrive by merging.

`release/0.32.1-dynres` is therefore **retired and archived, not merged** — the lift condition
[ADR 0006](adr/0006-release-lineage-is-never-merged-into-main.md) always specified. It stays
buildable as the rollback target for as long as `maxusai-ollama:0.32.1-rocm-dynres-5d5b7a72`
is the image this decision names for rollback; archiving it is not the same as deleting it.

## See also

- [ADR 0006](adr/0006-release-lineage-is-never-merged-into-main.md) — why the release lineage
  is never merged into `main`, and how changes reach it
- [gemma4-budget-image.md](gemma4-budget-image.md) — build/verify/deploy runbook
- [vision-token-budget-measurements.md](vision-token-budget-measurements.md) — measured token
  cost, including the 313 → 1049 delta this gate defers
- [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) — why the budget is
  arch-gated
