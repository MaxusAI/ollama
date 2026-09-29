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
| Deployed image | `maxusai-ollama:0.34.4-rocm724-main-b43ee8e3` (promoted 2026-09-28 07:39) |
| Deployed version | `0.34.4-dynres-0-gb43ee8e`, a build of `main` at the `v0.34.4-dynres` tag (ADR 0032) |
| Build type | `Dockerfile.rocm` through `scripts/build_rocm.sh` (`ROCM_TOOLCHAIN=rocm7 AMDGPU_TARGETS=gfx1151`), on `rocm/dev-ubuntu-24.04:7.2.4-complete` ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md)), as every production image since 0.34.3. It is gfx1151 only. Compat **001 + 002 + 004 + 005 + 801 + 802 + 903 + 908**: 908 reverts the device half of `ce8caa6e6`, a CUDA flash-attention tiling, and changes no gfx1151 kernel. 802, the LM-graph node meter, is inert unless `OLLAMA_LM_NODE_STATS` is set |
| KV cache | **f16** since 2026-09-26 07:23 ([ADR 0005](adr/0005-per-model-kv-cache-type.md)). From the 2026-08-08 cutover until then it was `q8_0` by accident; see the [2026-09-26 decision](#decision-2026-09-26--productions-kv-cache-back-to-f16) |
| Think+format flow | **two-pass** (`OLLAMA_FORMAT_TWO_PASS=1`) since 2026-09-28, the maintainer's choice at the 0.34.4 deploy. v0.34.4's default is the single pass; see the [2026-09-28 decision](#decision-2026-09-28--0344-promoted-with-the-two-pass-flow) |
| Payload | **b11081** (`161755f29`) since 2026-09-28; b10969 (`391fac164`) for 0.34.2 and 0.34.3. Compat 906 stays **retired**: upstream ships its own HIP `prop.integrated` revert |
| Previous image | `maxusai-ollama:0.34.3-rocm724-main-650f8fda` (`0.34.3-dynres-0-g650f8fd`, b10969) — **retained for rollback** as the stopped container `ollama-rocm-0.34.3-650f8fda`, which runs f16. Before it, `0.34.2-rocm724-main-f67b1aef` (the stopped container `ollama-rocm-0.34.2-f67b1aef`, recreated with f16 on 2026-09-28), `0.34.1-rocm724-main-16649e8c` (b10864 + 906) and `0.32.1-rocm-dynres-5d5b7a72` (b9888) |
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

**The maintainer, until further notice:** the fork's ROCm images build on AMD's Ubuntu 24.04 ROCm
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

## Decision 2026-09-26 — production's KV cache back to f16

**Outcome: fixed.** `ollama-rocm` was recreated at 07:23:52 on 2026-09-26 with `OLLAMA_KV_CACHE_TYPE=f16`. The
image (`0.34.3-rocm724-main-650f8fda`) and every other argument are unchanged. It was down for about one second. A
model load now logs `--cache-type-k f16 --cache-type-v f16 --flash-attn on`.

**Why.** [ADR 0005](adr/0005-per-model-kv-cache-type.md) traced qwen3.6's think-mode runaway on grounding prompts
to a `q8_0` KV cache. On 2026-08-03 the operator recreated production with f16 by hand. That did not hold:

- The deployment's compose file, `docker/ollama-rocm/docker-compose.yml` in `MaxusAI/ollama-deployments`, had said
  `q8_0` since its first commit, and nobody changed it.
- The 2026-08-08 cutover went back through compose, so production returned to `q8_0`. The 2026-08-13 and
  2026-08-17 deploys used compose too.
- The 0.34.2 promotion (2026-09-21) and the 0.34.3 promotion (2026-09-25) copied the running container's
  arguments, and `q8_0` with them.

So production served qwen3.6 with a `q8_0` KV cache for seven weeks. The ROCm gate's `prod` profile
(`tasks/rocm-gate/gatelib.py`) and the 2026-09-17 ROCm baseline recorded `q8_0` as production's environment, which
was true when they ran.

**How it was found.** The v0.34.4 fold's think-on protocol on this host (#375) runs in production's environment.
In it, qwen3.6 `bbox_contract_real_1img` did not finish at the ladder's top, 131072: it stopped on the 122 880-token
budget, and the second half of its thinking was 35 distinct lines out of 2,282. That is ADR 0005's signature. The
llama-server command line had `--cache-type-k q8_0 --cache-type-v q8_0`, and no model sets `kv_cache_type`.

**What changed with it.**

- The compose file says f16 now (`MaxusAI/ollama-deployments` `31923a9`), with the reason in a comment.
- `gatelib.py`'s `PROD_ENV` says f16, so the next gate run reproduces production again.
- An f16 KV cache takes twice the memory of `q8_0`: about 3 GB → 6 GB per model at 32K context (ADR 0005). The host
  has 96 GiB of VRAM.

**No `q8_0` fallback remains (2026-09-28).** This decision first kept the `q8_0` container, as
`ollama-rocm-0.34.3-q8kv`, so the KV change could be rolled back. The maintainer removed it on 2026-09-28: production
runs f16 from here on, with no `q8_0` fallback. A model that wants a quantized cache sets it per model or per request
(ADR 0043). Every rollback container on this host is f16.

**The 0.34.2 rollback container said `q8_0` until 2026-09-28.** `ollama-rocm-0.34.2-f67b1aef` was created with
production's arguments on 2026-09-21, so starting it, as the 2026-09-25 rollback does, would have brought `q8_0` back.
**It was recreated with f16 on 2026-09-28**, at the maintainer's word.
- The only change is `OLLAMA_KV_CACHE_TYPE=f16`. The image, ports, devices, groups, security options, IPC, shm,
  mount and the rest of the environment are unchanged, compared field by field before the old container was removed.
- It is created and never started, with restart policy `no`.
- The 2026-09-25 rollback commands work as written again.

## Decision 2026-09-28 — 0.34.4 promoted, with the two-pass flow

**Outcome: promoted.** `ollama-rocm` moved from `0.34.3-dynres-0-g650f8fd` to `0.34.4-dynres-0-gb43ee8e` at 07:39 on
2026-09-28, on the maintainer's word ("deploy the gfx1151 release image with two-pass"). It was down for about one
second. The payload moved from b10969 to **b11081** (`161755f29`), with compat 908 added.

**The flow is two-pass.** The container sets `OLLAMA_FORMAT_TWO_PASS=1`, so think+format requests keep ADR 0004's
flow, not v0.34.4's single-pass default. The CUDA deploy made the same choice (MaxusAI/ollama#375, item 7). On
gfx1151's GGUF the fold's think-on protocol found no loop difference between the flows on any of five models, so
parity with CUDA decided it.
- The compose file carries the setting (MaxusAI/ollama-deployments `3354a42`), so a compose redeploy cannot drop it.
- **The compose `.env` names this image** since MaxusAI/ollama-deployments `55e2fa2` (2026-09-28). Until then it
  named `0.32.1-rocm-dynres-5d5b7a72`. `docker compose config` now renders production's image and settings. It
  differs from the running container only in `label=disable` (an SELinux option, with no effect on this host), an added
  healthcheck, and a compose network in place of the default bridge. Check the network before a compose redeploy.

Rollback:

1. Stop the new container and rename it aside.
2. Rename the retained 0.34.3 container back to `ollama-rocm`.
3. Restore its restart policy and start it.

```
docker stop ollama-rocm && docker rename ollama-rocm ollama-rocm-0.34.4-rolledback &&
docker rename ollama-rocm-0.34.3-650f8fda ollama-rocm &&
docker update --restart unless-stopped ollama-rocm && docker start ollama-rocm
```

The retained container runs f16 and has `--restart no`.

**What the evidence covers, and why it covers the promoted image.** The v0.34.4 fold's gfx1151 gates
([upstream-sync-0.34.4.md](tasks/upstream-sync-0.34.4.md), "Gates 4–6 on gfx1151") measured the fold image and the
908 image, `0.34.3-dynres-22-g5584539`:

- Think off: every scored cell of five models equals the 0.34.3 candidate's and production 0.34.2's.
- OCRBench: every item is equal, 172 = 172.
- Think on: on all five models the single pass leaves the same cases unfinished as two-pass.

The promoted image is a build at the `v0.34.4-dynres` tag, `maxusai-ollama:0.34.4-dynres-0-gb43ee8e-rocm7-gfx1151`,
tagged in production's naming as `0.34.4-rocm724-main-b43ee8e3`. It carries that evidence because:

- **Native payload.** It is byte-identical to the 908 image's. `llama-server`, `libllama-server-impl.so`,
  `libggml-hip.so`, `libggml-base.so` and `libmtmd.so` have the same sha256.
- **Payload structure.** `payload_diff.sh` finds 1863 = 1863 entries, no SONAME or symlink change, and 96/96
  gfx1151 rocBLAS kernels.
- **Go source.** It equals the 908 build's tree, apart from `mlxrunner/client_test.go`, which is test-only.

Those gate campaigns ran with production's KV type at the time, `q8_0`, and the promoted container runs f16. The KV
type alone moves greedy think-on cells in both directions (MaxusAI/ollama#387), so the think-on cells are not
expected to reproduce exactly under f16.
[2026-09-28's baseline](#2026-09-28s-baseline-productions-configuration-measured) measures f16 directly.

**Preflight (gate 5), on the release image and on the promoted container.** Profile `rocm7-0-34-4-dynres`, with
`--quality`: **PASS=20 SKIP=12** both times, the same 32 checks as the 908 image's run. On the promoted container,
all seven model loads show `--cache-type-k f16 --cache-type-v f16 --flash-attn on` (ADR 0043, decision 4). The run
records are `vision-suite/preflight/runs/preflight-rocm7-0344-release-gb43ee8e.json` and
`preflight-rocm7-0344-prod-gb43ee8e.json`. The baseline below adds rocm7's quality floors and fp16 canary, and the
same image then reads PASS=24 SKIP=8.

**One known loop, which production already had.** Under f16 and greedy think-on, gemma4:26b loops on
`multi_3img_anchored`, a prompt that asks for "the size YOU used" (MaxusAI/ollama#387). 0.34.3's image has the same
loop, byte for byte, so this promotion neither adds nor removes it. At the card's sampling it is a rate, and stating
the image's size ends it.

### Clause outcomes

| clause | outcome |
|---|---|
| 1–2 | **As 2026-09-19.** Both overridden on evidence; nothing has changed upstream. |
| 3. `--direct-io` | **Satisfied under (c) on b11081, validated.** Re-validated because the payload moved, on the promoted image in production's environment (2026-09-28): **0/54 blocks differ** between dio-on and dio-off (qwen3.6 and gemma4:31b, think off; every scored cell equal, 0 of 978 and 0 of 986). Both arms were verified at the runner flag line: dio-on passed `--load-mode dio` on both runner starts, dio-off (`OLLAMA_IGPU_DIRECT_IO=0`) on neither. Also satisfied under (b): the opt-out exists. |
| 4. Vision A/B, ≥6 consecutive rows, 0 degenerate | **PASSED on the byte-identical candidate.** Five models, qwen3.6 (`qwen35moe`) among them, think off, equal to the 0.34.3 candidate and production 0.34.2 in every scored cell. Zero degenerate rows. It ran under `q8_0`. |
| 5. `make proof` | **Waived — still does not exist.** |

**Clause 3 on b11081, 2026-09-28.** The A/B (`dio-ab.sh`, in the run directory) ran the vision suite with think off in
two arms of the promoted image, each model on a cold server, in production's environment: f16, flash attention on, two
slots, two-pass. Its output, verbatim:

```
##### runner flag lines, per arm 2026-09-28T09:02:07+10:00
  dio-on:       2 --load-mode dio
  dio-on: runner starts: 2
  dio-off: runner lines with --load-mode: 0
  dio-off: runner starts: 2
##### cmp_scores.py, dio-on against dio-off 2026-09-28T09:02:07+10:00
0 of 978 cells differ (scores_r0344dioon_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json vs scores_r0344diooff_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json)
0 of 986 cells differ (scores_r0344dioon_1_gemma4_31b-it-q4_K_M_thinkfalse.json vs scores_r0344diooff_1_gemma4_31b-it-q4_K_M_thinkfalse.json)
```

All 108 blocks, 27 for each model in each arm, finished with valid JSON on `0.34.4-dynres-0-gb43ee8e`.

**How the host was confirmed idle.** As for 0.34.3: no model was loaded, and there was no working request in the two
minutes before the swap. `deploy-0344.sh` refuses to swap otherwise.

### 2026-09-28's baseline: production's configuration, measured

The gate campaigns above ran under `q8_0`, and production runs f16. So, at the maintainer's word, this measures what
production runs. The image is the promoted one, with f16, flash attention on, two-pass and two slots, as in production.
It ran in a bench container on `:11497` (`base-canary.sh`, in the run directory), and production was not touched.

**The settings, read from the server logs.** All five model loads show `K (f16), V (f16)`, flash attention enabled
and `OLLAMA_FORMAT_TWO_PASS:true`. Each model ran on a cold server at its automatic batch: 2048 for both gemma4 models
(their image-chunk floor), and 1024 for the other three. None logged "images decode in pieces".

**Think off, five models** (`r0344base_1_<model>_thinkfalse`). All 135 blocks finished with valid JSON. None was
capped, and none errored.
- Against the gate's `q8_0` cells, the KV type moves 37 to 72 cells per model.
- More of the scored moves favour f16 than `q8_0` on four models. On qwen3.8 they are close to even: 8 favour f16
  and 9 favour `q8_0`.
- These moves are the KV type's. The gate cells ran on the fold image, without 908. But 908 changed no think-off cell
  on this host (the gate), and the promoted payload is byte-identical to the 908 image's.

```
##### cmp_scores.py, q8_0 gate cells (r0344fold_, 0.34.3-dynres-5-g29ae523) against the f16 baseline (r0344base_, 0.34.4-dynres-0-gb43ee8e)
37 of 978 cells differ (scores_r0344fold_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json vs scores_r0344base_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json)
45 of 976 cells differ (scores_r0344fold_1_qwen3_8_27b-q4_K_M_thinkfalse.json vs scores_r0344base_1_qwen3_8_27b-q4_K_M_thinkfalse.json)
39 of 986 cells differ (scores_r0344fold_1_gemma4_31b-it-q4_K_M_thinkfalse.json vs scores_r0344base_1_gemma4_31b-it-q4_K_M_thinkfalse.json)
51 of 980 cells differ (scores_r0344fold_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json vs scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json)
72 of 983 cells differ (scores_r0344fold_1_nemotron3_33b-q4_K_M_thinkfalse.json vs scores_r0344base_1_nemotron3_33b-q4_K_M_thinkfalse.json)
##### cmp_scored.py, the same pairs (A = q8_0 gate, B = f16 baseline)
quality moves: 8 favour B, 2 favour A, 0 neutral/changed-label (scores_r0344fold_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json = A, scores_r0344base_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json = B)
quality moves: 8 favour B, 9 favour A, 8 neutral/changed-label (scores_r0344fold_1_qwen3_8_27b-q4_K_M_thinkfalse.json = A, scores_r0344base_1_qwen3_8_27b-q4_K_M_thinkfalse.json = B)
quality moves: 5 favour B, 2 favour A, 1 neutral/changed-label (scores_r0344fold_1_gemma4_31b-it-q4_K_M_thinkfalse.json = A, scores_r0344base_1_gemma4_31b-it-q4_K_M_thinkfalse.json = B)
quality moves: 9 favour B, 2 favour A, 3 neutral/changed-label (scores_r0344fold_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json = A, scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json = B)
quality moves: 26 favour B, 14 favour A, 14 neutral/changed-label (scores_r0344fold_1_nemotron3_33b-q4_K_M_thinkfalse.json = A, scores_r0344base_1_nemotron3_33b-q4_K_M_thinkfalse.json = B)
```

**OCRBench**, gemma4:31b-it-q4_K_M, rows 0–200, think off, one slot (`r0344base_ocr_q4`). **f16 scores 172/200, as
`q8_0` did, on the same items.** 199 of the 200 answers are the same text. The one that differs, item 79, is a
refusal, and it is wrong under both. The MIXED banner is by design: the two arms differ in build and KV type.

```
| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

⚠ **MIXED — rows are not one campaign** (hosts: ['http://127.0.0.1:11497']; builds: ['0.34.3-dynres-5-g29ae523', '0.34.4-dynres-0-gb43ee8e'])

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| r0344fold_ocr_q4 vs r0344base_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |
```

**The preflight's output-quality floors** (`[quality.rocm7-0-34-4-dynres.*]` in `expectations.toml`). In the
think-off run, the three arches' models scored 6/6 labels on `scene_single` and 5/5 items on `document_single`, with
valid JSON. The floors are cuda-dynres-005's: JSON 1.0, label recall 0.70 and qty/price 0.70. They allow one miss per
test and fail on the second.

**The fp16 overflow canary on HIP** (`[poison.rocm7-0-34-4-dynres]`), its first run there. `canary_probe.py` ran the
preflight's own `check_poison_probe` twice, with the node meter on (`OLLAMA_CLIP_NODE_STATS=ffn_down`). The first run
had production's environment, so the fork's f32 gate applied. The second set `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` on the
container, which the gate leaves alone (an operator's value always wins).

```
##### canary_probe.py canary-default.json (checks.check_poison_probe; container label: default)
 status: PASS
 summary: 1.06x-ceiling trigger decodes healthily (52 chars, done_reason='stop'); slot clean after; node meter clean over 33 node(s)
 actual: The image is a black and white checkerboard pattern.
##### server-canary-default.log
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
CLIP_NODE_STATS name=ffn_down-31 op=MUL_MAT type=f32 n=15728640 max_abs=75478.2 hr=1.1523 n_gt32k=550 n_gt49k=58 n_gt60k=16 n_inf=0 n_nan=0
CLIP_NODE_STATS name=ffn_down op=MUL_MAT type=f32 n=6291456 max_abs=50.6 hr=0.0008 n_gt32k=0 n_gt49k=0 n_gt60k=0 n_inf=0 n_nan=0
##### canary_probe.py canary-f16.json (checks.check_poison_probe; container label: f16-override)
 status: FAIL
 summary: degenerate decode on the trigger request
 actual: done_reason='length' head='????????????????????????????????????????'
##### server-canary-f16.log
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
CLIP_NODE_STATS name=ffn_down-31 op=MUL_MAT type=f32 n=15728640 max_abs=64384.0 hr=0.9829 n_gt32k=541 n_gt49k=48 n_gt60k=10 n_inf=6 n_nan=0
CLIP_NODE_STATS name=ffn_down op=MUL_MAT type=f32 n=6291456 max_abs=51.4 hr=0.0008 n_gt32k=0 n_gt49k=0 n_gt60k=0 n_inf=0 n_nan=12288
```

- **The check tells the gate apart on gfx1151, as on CUDA.** Under the gate, block 31's `ffn_down` peaks at 75,478,
  1.15× fp16's 65,504, and stays finite in f32. Under f16 it holds 6 infinities, and the next metered node holds
  12,288 NaNs. The decode is question marks until `num_predict`.
- The "1.06x-ceiling" in the summary is the check's fixed wording, from CUDA's measurement (#214).

**Preflight with both entries, on the same image in a bench container** (profile `rocm7-0-34-4-dynres`, `--quality`):
**VERDICT PASS, PASS=24 SKIP=8**, against 20 and 12 before. The four checks that skipped for want of an expectation now
pass: output quality on all three arches (1.00 on every measure) and the canary. The eight that still skip are the
four Metal and MLX checks, which do not apply here; the aspect ladder on three arches, which no profile records; and
qwen35's pinned budget, which its arch does not take. The run record is
`vision-suite/preflight/runs/preflight-rocm7-0344-base-quality-gb43ee8e.json`, and the README's matrix now takes
gfx1151's row from it.

**The aspect ladder, 2026-09-29.** Next, the three arches' aspect ladders were measured on the same image and
environment (4:3, 4:1 and 1:1, the axis the 16:9 ladder cannot move), by the probe calls `check_aspect_ladder`
makes. A second preflight with them passed: **VERDICT PASS, PASS=27 SKIP=5**, with the aspect ladder 3/3 on every
arch, and quality and the canary as before.
- The five that still skip are the four Metal and MLX checks and qwen35's pinned budget. None of them applies on
  gfx1151, so every check that can run here now does.
- gemma4's row equals the Metal profiles' (1066, 1058, 1091).
- nemotron3 keeps each image's native grid (770, 902, 578).
- qwen3.8 raises every ratio to its 1024-token minimum (1038, 1026, 1026).
- The run record is `preflight-rocm7-0344-aspect-gb43ee8e.json`, and the README's matrix, whose columns are
  unchanged by it, now takes gfx1151's row from it.

**Think on, two-pass, f16** (`baseline-c.sh` in the run directory, 2026-09-28 15:26 to 2026-09-29 03:22). Four models
ran, each on a cold server, over the full ladder from 16384 to 131072. That is how the gate's `fold2p` arm ran under
`q8_0`, which is the comparison here.
- gemma4:31b and gemma4:26b ran greedy, with the batch pinned at 2048, the value their automatic batch takes here.
- qwen3.8 and nemotron3 ran twice each at their packaged sampling, so their counts are draws.
- qwen3.6 already has this configuration: `fold2p` under f16 in
  [kv-precision-think-loops.md](tasks/kv-precision-think-loops.md). That ran on the fold image, whose payload differs
  from the promoted one only by 908, and 908 changes no gfx1151 kernel.

**gemma4:31b and qwen3.8 finish every case, as under `q8_0`,** all at 16384. gemma4:31b is greedy, and 64 of its 984
cells differ from `q8_0`'s. One of them is a scored move: `scene_single_anchored`'s IoU falls from 0.965 to 0.724.

**nemotron3 finishes every case but one draw.** In run 2, `finetext` loops to the 131072 cap. Run 1's `finetext`
finishes, and so do all 54 cells of the gate's two `q8_0` draws. One loop in 54 cells against none in 54 cannot tell
the KV types apart with two draws.

**gemma4:26b is where f16 costs.** It is greedy, so each cell is one fixed trajectory. Three cases never finish,
against one under `q8_0`:
- `bbox_contract_real_1img` loops under both, as the gate found.
- `multi_3img_anchored` loops under f16 only. It is item 8's prompt trap ("the size YOU used"), which MaxusAI/ollama#387
  found looping under f16.
- `bbox_contract_anchored_1img` loops under f16 only, and this is new. All three `q8_0` runs finished it at 16384,
  in about 2,120 tokens. The loop check (2026-09-29) found a greedy knife edge, not a prompt trap:
  - A cold capture repeats the suite's thinking byte for byte, so the loop is the case's own trajectory.
  - f16 and `q8_0` part at one rounding of the anchor's corner (y=148 against y=149), 662 characters in.
  - Three card-sampled f16 draws of three finish, with 6/6 boxes.

  See [kv-precision-think-loops.md](tasks/kv-precision-think-loops.md#a-loop-with-no-prompt-trap-gemma426b-bbox_contract_anchored_1img-gfx1151-2026-09-29).

The cases both finish are 24. Under f16, three of them score worse, one scores marginally better, and the rest score
the same:
- `bbox_contract_adv_real` changes its coordinate convention and hits 1 of 6 boxes, against 6.
- `scene_single_pinned` hits 1 of 6 boxes, against 6.
- `scene_single_anchored`'s IoU falls from 0.948 to 0.756.
- `bbox_contract_positional_1img`'s IoU rises from 0.970 to 0.975.

The KV type is the only difference that matters in this comparison. Both arms are two-pass. Both run gemma4 at 2048,
here pinned and in the gate automatic. And 908 changes no gfx1151 kernel. MaxusAI/ollama#387 found that the KV type
moves greedy think-on cells both ways. On gemma4:26b here it moves them mostly one way. Production keeps f16
(ADR 0043; no `q8_0` fallback since MaxusAI/ollama#398), so these cells are what production now does.

**The batches.** Every launch ran at the batch its context calls for. No launch logged "images decode in pieces", so
the batch played no part in these comparisons:
- gemma4 ran at 2048, pinned, with two slots.
- qwen3.8 and nemotron3 ran at 1024 up to 32768 and at 2048 above, with one slot each. The scheduler logs "model
  architecture does not currently support parallel requests" for `qwen35` and `nemotron_h_omni`, and upstream's
  `sched.go` then forces one slot.

`step2_read.py`, `cmp_scored.py` on gemma4:26b, and `batch_check.py` (all in the run directory), verbatim:

```
== gemma4:31b
  f16 baseline       r0344base_1_           27 blocks, finished by rung {16384: 27}, valid JSON 27; never converged none
  q8_0 gate, fold2p  r0344p_fold2p_1_       27 blocks, finished by rung {16384: 27}, valid JSON 27; never converged none
  cmp_scores.py: 64 of 984 cells differ (scores_r0344p_fold2p_1_gemma4_31b-it-q4_K_M_thinkon.json vs scores_r0344base_1_gemma4_31b-it-q4_K_M_thinkon.json)
  cmp_scored.py: quality moves: 0 favour B, 1 favour A, 0 neutral/changed-label (scores_r0344p_fold2p_1_gemma4_31b-it-q4_K_M_thinkon.json = A, scores_r0344base_1_gemma4_31b-it-q4_K_M_thinkon.json = B)
== gemma4:26b
  f16 baseline       r0344base_1_           27 blocks, finished by rung {16384: 23, 32768: 1}, valid JSON 24; never converged ['bbox_contract_anchored_1img', 'bbox_contract_real_1img', 'multi_3img_anchored']
  q8_0 gate, fold2p  r0344p_fold2p_1_       27 blocks, finished by rung {16384: 23, 32768: 3}, valid JSON 26; never converged ['bbox_contract_real_1img']
  cmp_scores.py: 149 of 986 cells differ (scores_r0344p_fold2p_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json vs scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json)
  cmp_scored.py: quality moves: 2 favour B, 9 favour A, 2 neutral/changed-label (scores_r0344p_fold2p_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json = A, scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json = B)
== qwen3.8
  f16 baseline       r0344base_r1_1_        27 blocks, finished by rung {16384: 27}, valid JSON 27; never converged none
  f16 baseline       r0344base_r2_1_        27 blocks, finished by rung {16384: 27}, valid JSON 27; never converged none
  q8_0 gate, fold2p  r0344p_fold2p_r1_1_    27 blocks, finished by rung {16384: 27}, valid JSON 27; never converged none
  q8_0 gate, fold2p  r0344p_fold2p_r2_1_    27 blocks, finished by rung {16384: 27}, valid JSON 27; never converged none
== nemotron3
  f16 baseline       r0344base_r1_1_        27 blocks, finished by rung {16384: 19, 32768: 8}, valid JSON 27; never converged none
  f16 baseline       r0344base_r2_1_        27 blocks, finished by rung {16384: 20, 32768: 5, 65536: 1}, valid JSON 26; never converged ['finetext']
  q8_0 gate, fold2p  r0344p_fold2p_r1_1_    27 blocks, finished by rung {16384: 20, 32768: 6, 65536: 1}, valid JSON 27; never converged none
  q8_0 gate, fold2p  r0344p_fold2p_r2_1_    27 blocks, finished by rung {16384: 20, 32768: 7}, valid JSON 27; never converged none
```

```
bbox_contract_adv_real: anchor_implied_type 'norm1000'->'real'; contract_followed True->False A+; declaration_matches_boxes True->False A+; hits_anchor 6->1 A+; hits_declared 6->1 A+; iou_anchor 0.973->0.05 A+; iou_declared 0.973->0.05 A+; self_check True->False
bbox_contract_positional_1img: iou_anchor 0.97->0.975 B+; iou_declared 0.97->0.975 B+
scene_single_anchored: bbox_mean_iou 0.948->0.756 A+
scene_single_pinned: bbox_hits 6->1 A+; bbox_mean_iou 0.971->0.043 A+
bbox_contract_anchored_1img: finished only in A
multi_3img_anchored: finished only in A
quality moves: 2 favour B, 9 favour A, 2 neutral/changed-label (scores_r0344p_fold2p_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json = A, scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json = B)
```

```
20260928T170259-base.log           n/a                          num_ctx/slot=16384 -c=32768 -np=2 -b=2048 -ub=2048
20260928T175719-base.log           Qwen3.8 27B 0814             num_ctx/slot=16384 -c=16384 -np=1 -b=1024 -ub=1024
20260928T185025-base.log           Qwen3.8 27B 0814             num_ctx/slot=16384 -c=16384 -np=1 -b=1024 -ub=1024
20260928T194100-base.log           n/a                          num_ctx/slot=16384 -c=16384 -np=1 -b=1024 -ub=1024
20260928T200851-base.log           n/a                          num_ctx/slot=32768 -c=32768 -np=1 -b=1024 -ub=1024
20260928T205256-base.log           n/a                          num_ctx/slot=16384 -c=16384 -np=1 -b=1024 -ub=1024
20260928T211944-base.log           n/a                          num_ctx/slot=32768 -c=32768 -np=1 -b=1024 -ub=1024
20260928T213940-base.log           n/a                          num_ctx/slot=65536 -c=65536 -np=1 -b=2048 -ub=2048
20260928T221516-base.log           n/a                          num_ctx/slot=131072 -c=131072 -np=1 -b=2048 -ub=2048
20260928T225725-base.log           n/a                          num_ctx/slot=16384 -c=32768 -np=2 -b=2048 -ub=2048
20260928T232659-base.log           n/a                          num_ctx/slot=32768 -c=65536 -np=2 -b=2048 -ub=2048
20260929T003557-base.log           n/a                          num_ctx/slot=65536 -c=131072 -np=2 -b=2048 -ub=2048
20260929T032139-base-c-final.log   n/a                          num_ctx/slot=131072 -c=262144 -np=2 -b=2048 -ub=2048
```

Each server log is saved when the next launch replaces its container, so each line is the run before its timestamp.
In order, they are gemma4:31b; qwen3.8's two runs; nemotron3's run 1 at 16384 and 32768; nemotron3's run 2 at each rung
up to 131072; and gemma4:26b at each rung. The "n/a" rows are the models whose GGUF records `general.name` as "n/a".

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
