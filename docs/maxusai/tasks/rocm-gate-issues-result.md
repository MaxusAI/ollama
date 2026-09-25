# ROCm/gfx1151: reproduction results for #17459 and #17475

Answers [`rocm-gate-issues-17459-17475.md`](rocm-gate-issues-17459-17475.md). Run on this host
overnight 2026-09-17/18. Production was never touched: every test ran in a scratch container
named `ollama-gate` on **:11435**, removed after each run, while `ollama-rocm` on `:11434`
stayed up and held no model throughout.

> **Neither issue reproduces on this hardware.**
>
> **#17459** — 0 `<unused49>` tokens in **96 requests** across eight payload/environment
> combinations (b9888; b10091 fork *and* stock; b10864 fork dio-on, fork dio-off, stock),
> cold and warm, `think` both ways. On this host `think=false` is the arm that *works* — it
> returns a complete paragraph and stops — while `think=true` spends the whole 128-token cap
> on reasoning and returns no answer. That is the opposite of the report, on the same gfx1151
> silicon the reporter used.
>
> **#17475** — 0 contaminated victims in **436 extractions** (484 including superseded runs)
> across six configurations and all three protocols, with aborts and multi-model swapping
> verified to have actually occurred rather than assumed.
>
> **The consequential finding is incidental to both.** Ollama **0.32.x cannot see this box's
> VRAM**: the scheduler reports 27 GiB where 0.34.x reports 95 GiB, with the ROCm runtime held
> constant. Production is serving `gemma4:31b` with only **25–26 of its 61 layers on the GPU**.
> That is a standing, measured cost of staying behind the gate, and nobody had priced it.
>
> A clean run is not proof of absence for a concurrency race. The limits are in
> [Caveats](#caveats--what-would-change-these-answers), not buried.

## Payloads

| label | image | version | llama.cpp | HIP runtime |
|---|---|---|---|---|
| **b9888** | `maxusai-ollama:0.32.1-rocm-dynres-5d5b7a72` | `0.32.1-dynres-5d5b7a72` | b9888 | 7.2.70201 |
| **b10091 fork** | `maxusai-ollama:0.32.5-rocm-gemma4budget-4259c191` | `0.32.5-gemma4budget-4259c191` | b10091 | 7.2.70201 |
| **b10091 stock** | `ollama/ollama:0.32.5-rocm` | `0.32.5` | b10091 | 7.2.70201 |
| **b10864 fork** | `maxusai-ollama:0.34.1-rocm724-dynres-649fea9e-gated` | `0.34.1-dynres-649fea9e` | b10864 | 7.2.**70204** |
| **b10864 stock** | `ollama/ollama:0.34.1-rocm` | `0.34.1` | b10864 | 7.2.70201 |

b9888 is what production serves today. The b10864 fork image is `task/upstream-sync-0.34.1`
(PR #302) at `b1db10ef`, built here with `--build-arg ROCMVERSION=7.2.4`, plus one local commit
adding an `OLLAMA_IGPU_DIRECT_IO` knob. Its compat series (001, 002, 004, 005, 801, 903) was
verified to apply to a clean b10864 checkout with plain `git apply` before building, per the
fold's own hard-won advice. **It is a gate-evaluation artifact and was never deployed** — the
direct-I/O gate refuses that tree.

Two environment profiles were used, and both appear in the table below: `prod` reproduces
production's settings (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`,
`OLLAMA_NUM_PARALLEL=2`), `default` is a bare server. Every container ran with
`OLLAMA_NOPRUNE=1`, because a server that cannot parse a newer manifest could otherwise delete
production's blobs.

## #17459 — does not reproduce, on any payload

The reporter's request verbatim: `/api/chat`, `gemma4:31b` (locally `gemma4:31b-it-q4_K_M`,
digest `6316f0629137` — the tag the brief names), `"Explain DHCP in one paragraph."`,
`stream: true`, `num_predict: 128`. Twelve requests per combination — three repetitions of
cold `think=false`, warm `think=true`, cold `think=true`, warm `think=false`.

**The maintainer asked whether the model was reloaded. It was, and it changes nothing.** Cold
arms report a real `load_duration` — 3.8 s to 78 s depending on payload — and warm arms
0.00–0.33 s. Both are clean everywhere.

Degeneration is counted over **content and thinking together**: with `think=true` the whole
128-token cap is spent inside the thinking block, so a content-only check would score a flood
there as clean. The detector also flags any run of 20+ identical tokens, not just `<unused>`.

| payload | run | runner | requests | `<unused>` tokens | cold load s | warm load s | layers on GPU | VRAM seen | result |
|---|---|---|---|---|---|---|---|---|---|
| b10091 | `b10091_fork_prod` | `-c 524288 -np 2` | 12 | **0** | 11.58–16.13 | 0.24–0.27 | 27/61 | 27.5–27.6 GiB | clean |
| b10091 | `b10091_stock_default` | `-c 262144 -np 1` | 12 | **0** | — | — | 7/61 | 23.3–27.7 GiB | **12/12 failed to load** |
| b10091 | `b10091_stock_prod` | `-c 524288 -np 2` | 12 | **0** | 11.35–11.78 | 0.24–0.25 | 27/61 | 27.3–27.5 GiB | clean |
| b10864 | `b10864_fork_dio_prod` | `-c 524288 -np 2`, dio | 12 | **0** | 5.75–6.02 | 0.00 | 61/61 | 95.4 GiB | clean |
| b10864 | `b10864_fork_nodio_prod` | `-c 524288 -np 2` | 12 | **0** | 3.75–7.26 | 0.00 | 61/61 | 95.4 GiB | clean |
| b10864 | `b10864_stock_default` | `-c 262144 -np 1`, dio | 12 | **0** | 5.75–5.95 | 0.00 | 61/61 | 95.4 GiB | clean |
| b9888 | `b9888_fork_default` | `-c 262144 -np 1` | 12 | **0** | 8.24–18.77 | 0.25–0.27 | 36/61 | 27.4–27.6 GiB | clean |
| b9888 | `b9888_fork_prod` | `-c 524288 -np 2` | 12 | **0** | 32.19–78.02 | 0.27–0.33 | 25–26/61 | 27.1–27.4 GiB | clean |

Total: 96 requests, 0 `<unused>` tokens.

The `runner` column is there because the layer counts are only comparable if the context is:
within each profile every payload got the same `-c`/`-np`, so b9888's 25–26/61 and b10864's
61/61 are the same workload placed differently.

`b10091_stock_default` is not a clean result and not a degenerate one — it is a failure to
run. All twelve requests died with `llama-server process has terminated: signal: killed`
after placing 7 of 61 layers; that is the VRAM finding below, not the reported symptom.

## #17475 — no contamination, after three instrument corrections

Three protocols on `qwen3-vl:30b-a3b-thinking` (the reporter's model, pulled for this, digest
`eda0be100877`), `think: true`, `temperature 0`, `num_ctx 8192`, `num_predict 6144`, `-np 1`.
Synthetic documents only, 1568×1568 PNG — which also answers the maintainer's question about
image type and dimensions. The donor carries a 17-character VIN-style marker; the victim form
is asserted at generation time to share no 6-character window of it.

**Item 4 of the brief, answered from production's own logs:** its vision runner is
`-c 16000 -np 1` and its text runner `-c 32000 -np 2`, both with `--context-shift --keep 4`.
With a single slot, *every* request reuses the same KV slot — which is precisely the reuse path
#17475 alleges. These results bear on production directly, not by analogy.

### What had to be fixed before the protocols measured anything

Each of these produced a confident, wrong answer first:

1. **The abort never fired.** The report aborts the donor at an 8 s client timeout, which on
   their hardware landed mid-generation of a 6144-token request. This host answers the *same
   request* in **3.9 s**, so the timeout never triggered and V2/V3 silently degraded into V1
   with concurrency. The window was shortened to **2.5 s** — mid-generation here — rather than
   lengthening the prompt, and the cancel is confirmed in the server log.
2. **The swapping never happened.** V3's stated ingredient is continuous multi-model swapping,
   and V3 is the only protocol that contaminated for the reporter. With their noise models
   (17–22 GB) a cycle costs 60–75 s here, so across six configurations the loop cycled **1–3
   times** and `qwen3.6` never completed once. Victims now continue until the loop has cycled,
   and a fast-noise variant substitutes small models to exercise swapping properly.
3. **The marker collided with the documents.** The first marker was `ZX9SYNTHETMARK448`, which
   embeds `SYNTHET` — and both forms carry the header "SYNTHETIC TEST DOCUMENT". Victims that
   merely quoted their own header while reasoning scored as contaminated: **7 of 8 in one
   run**, a false positive whose shape matched the reported bug almost exactly. The marker is
   now `ZX9KRDVMBTLH47P2W`, with no dictionary fragment, and the generator asserts against
   *everything rendered on both forms* rather than the field values alone. Runs made with the
   old marker are re-scored below against only its provably donor-only windows.

The substitution of small noise models (`gemma4:e2b-it-q4_K_M` and `gemma4:e4b`) is a
**declared deviation** from the reporter's `gemma4:26b` and `qwen3.6:35b`. It was necessary to
reach their swapping rate at all. `qwen2.5vl:3b` was tried first and hung — the fork already
documents that as a poison-image serving bug.

| payload | run | protocol | victims | aborts fired | noise cycles | contaminated | note |
|---|---|---|---|---|---|---|---|
| b10091 | `b10091_stock_reporter` | V1 | 6 | 0 | — | **0** | re-scored |
| b10091 | `b10091_stock_reporter` | V2 | 6 | 6 | — | **0** | re-scored |
| b10091 | `b10091_stock_reporter` | V3 | 40 | 40 | 2 | **0** | re-scored |
| b10864 | `b10864_fork_dio` | V1 | 6 | 0 | — | **0** | re-scored |
| b10864 | `b10864_fork_dio` | V2 | 6 | 6 | — | **0** | re-scored |
| b10864 | `b10864_fork_dio` | V3 | 40 | 40 | 2 | **0** | re-scored |
| b10864 | `b10864_fork_fastnoise` | V3 | 8 | 9 | 429 | **0** | re-scored |
| b10864 | `b10864_fork_fastnoise2` | V3 | 24 | 24 | 1203 | **0** | clean marker |
| b10864 | `b10864_fork_nodio` | V1 | 6 | 0 | — | **0** | re-scored |
| b10864 | `b10864_fork_nodio` | V2 | 6 | 6 | — | **0** | re-scored |
| b10864 | `b10864_fork_nodio` | V3 | 36 | 37 | 3 | **0** | re-scored |
| b10864 | `b10864_nodio_fastnoise2` | V3 | 40 | 41 | 4 | **0** | clean marker |
| b10864 | `b10864_stock_reporter` | V3 | 40 | 40 | 3 | **0** | re-scored |
| b9888 | `b9888_fork_fastnoise` | V3 | 40 | 41 | 2 | **0** | re-scored |
| b9888 | `b9888_fork_fastnoise2` | V3 | 40 | 40 | 2 | **0** | clean marker |
| b9888 | `b9888_fork_prod` | V3 | 40 | 41 | 2 | **0** | re-scored |
| b9888 | `b9888_fork_prod_noabort` | V3 | 8 | 6 | 2 | **0** | superseded: no aborts fired |
| b9888 | `b9888_fork_reporter` | V1 | 6 | 0 | — | **0** | re-scored |
| b9888 | `b9888_fork_reporter` | V2 | 6 | 6 | — | **0** | re-scored |
| b9888 | `b9888_fork_reporter` | V3 | 40 | 40 | 1 | **0** | re-scored |
| b9888 | `b9888_fork_reporter_noabort` | V1 | 6 | 0 | — | **0** | superseded: no aborts fired |
| b9888 | `b9888_fork_reporter_noabort` | V2 | 6 | 0 | — | **0** | superseded: no aborts fired |
| b9888 | `b9888_fork_reporter_noabort` | V3 | 8 | 0 | 1 | **0** | superseded: no aborts fired |
| b9888 | `b9888_fork_reporter_noswap` | V1 | 6 | 0 | — | **0** | superseded: no swapping |
| b9888 | `b9888_fork_reporter_noswap` | V2 | 6 | 6 | — | **0** | superseded: no swapping |
| b9888 | `b9888_fork_reporter_noswap` | V3 | 8 | 8 | 1 | **0** | superseded: no swapping |

All runs: 484 victim extractions, 0 contaminated.
Excluding superseded runs: 436 victim extractions, 0 contaminated.

"Aborts fired" counts donor requests actually cancelled server-side. "Noise cycles" counts
completed model loads in the V3 window; a request still in flight when the window closed is
excluded, which is why some rows show one fewer cycle than requests attempted.

## Two findings neither issue was about

### 0.32.x cannot see this machine's VRAM

From Ollama's own scheduler, same host, same HIP runtime, same request:

```
0.32.1:  system memory total="31.0 GiB"   gpu memory available="27.1 GiB"
0.34.1:  system memory total="31.0 GiB"   gpu memory available="95.4 GiB"
```

0.32.x sizes the iGPU by system RAM and never sees the 96 GiB firmware carve. The consequence
is in the #17459 table above, where the context was held constant across payloads:
`gemma4:31b` places **25–26 of 61 layers** on the GPU under 0.32.1 and **61 of 61** under
0.34.1. Roughly half the model is running on the CPU in production today.

**The ROCm runtime is not the cause.** Stock `ollama/ollama:0.34.1-rocm` ships HIP
`7.2.70201`, the same as stock `0.32.5-rocm`, and still sees 95.4 GiB — so this is the Ollama
version, not the 7.2.4 bump in our own build. The mechanism is *not located*: the
`gpu.Integrated && systemInfo.FreeMemory < gpu.FreeMemory` clamp in `server/sched.go` exists in
both trees, and the device is reported integrated in both — 0.34.1 passes `--load-mode dio`,
which only happens for integrated devices. Observed, not explained (ADR 0024).

### Direct I/O is a large *benefit* on this host

The gate treats `--direct-io` as an unvalidated risk. Measured, it is also what makes repeated
model loading viable here. Same image, same protocol, same 900 s window, same noise models,
only the knob differing:

| run | payload | dio | completed cycles | every noise request, s |
|---|---|---|---|---|
| `b10864_fork_fastnoise2` | b10864 | **on** | **1203** | 0.1–9.4 over 1203 requests |
| `b10864_nodio_fastnoise2` | b10864 | off | **3** | 3, 900.1, 5.1, 237 |
| `b9888_fork_fastnoise2` | b9888 | off | **2** | 8.2, 329.8 |

Without direct I/O the noise loop does not stall consistently — it loads a model quickly, then
a later load takes four minutes or never returns inside the window. With it on, the same loop
cycled 1203 times. No degeneration was seen with dio on, in either issue's tests.

## What this means for the gate

Against the five clauses in [`amd-upgrade-gate.md`](../amd-upgrade-gate.md):

| clause | status after this work |
|---|---|
| 1. #17459 fixed, fix in target | **Cannot be satisfied as written.** Closed `not_planned` 2026-08-23; there is no fix to be in any tag. It does not reproduce here on any payload, including the target. |
| 2. #17475 fixed, fix in target | **Cannot be satisfied as written.** Closed `not_planned` 2026-08-12. Does not reproduce here in 436 extractions. |
| 3. `--direct-io` absent, opt-out-able, or validated | (a) no — present for ROCm iGPUs in b10864, confirmed in the runner line. (b) **now available** — `OLLAMA_IGPU_DIRECT_IO=0` removes the flag, verified on hardware in both tables; upstream pins that dio wins over `use_mmap=false`, so without that knob there is no opt-out. (c) partially — the dio-on b10864 arms ran 24 chat requests, 124 victim extractions and 1637 noise model loads with no corruption, but that is a by-product of these tests, not a targeted load-path integrity check. |
| 4. Vision A/B, ≥6 consecutive `qwen35moe` rows, 0 degenerate on the candidate | **Run 2026-09-18. FAILED on the candidate, PASSES with compat 906.** Unpatched b10864 dropped `qwen3.6:35b-a3b` scene IoU 0.953 → 0.273 and lost the multi-image question. With `906-revert-hip-integrated-flag.patch` it scores **0.972**, above the baseline, 0 degenerate. See [the addendum](#addendum-2026-09-18--clause-4-run-and-the-cause-found). |
| 5. `make proof` against the new `BASETAG` | **Outstanding.** This build used the full path, not the overlay. |

Clauses 1 and 2 are unsatisfiable on their own terms, because "fixed" requires someone upstream
to fix something nobody is working on. What the evidence changes is the nature of the block:
**the two bugs that justify the pin cannot be reproduced on the hardware they were reported on,
and the pin now has a measured cost** — half of `gemma4:31b` on the CPU, and model loads that
stall for minutes without direct I/O.

That is a decision, not a conclusion, and it is the maintainer's to make. What this document supports is
making it on evidence.

**That decision has since changed twice.** Clause 4 was run the next day and the candidate failed
it badly — for a reason unrelated to either gate issue. The cause was then found and fixed. See
the addendum below; the paragraph above is left as written because it was the honest reading of
the evidence available on 2026-09-17.

## Caveats — what would change these answers

- **A clean run is not proof of absence for #17475.** The reporter needed three protocols to
  surface it and it is a race. Our closest run to theirs — heavy swapping, aborts firing,
  b10864 — is 0/24.
- **Our V3 differs from theirs in the noise models**, by necessity, and the b9888 and dio-off
  cells reached only 2–4 noise cycles because model loading is slow without direct I/O. Those
  rows test concurrency and aborts, not swapping. Only the dio-on b10864 cells test swapping at
  a realistic rate, and that asymmetry is itself caused by the VRAM finding.
- **The 0.32.x #17459 arms ran mostly on CPU** — 25–36 of 61 layers — for the same reason. If
  the degeneration were GPU-kernel-related, those arms could mask it. The b10864 arms had 61/61
  on GPU and were clean, which is both the stronger evidence and the gate's actual target.
- **`b10091_stock_default` failed 12/12 to load.** Reported as a failure to run, not as clean
  and not as degenerate.
- **Different hardware from #17475's reporter**: they are on aarch64 DGX Spark GB10 with the
  CUDA v13 runner. A slot-reuse race can be backend-specific, and a negative here does not
  clear their configuration.
- **The re-scored rows are honest but weaker than the clean-marker rows.** They were run with
  the colliding marker and re-scored afterwards against its donor-only windows; the
  clean-marker rows were run with a marker that cannot collide by construction. Both are 0.

## Reproducing this

Harness in [`rocm-gate/`](rocm-gate/): `gen_forms.py` (synthetic documents, with the collision
assertion), `gatelib.py` (scratch-container plumbing), `repro_17459.py`, `repro_17475.py`, and
the two table generators `t17459.py` and `t17475.py` that produced the tables above. Raw
results and full server logs for every run are on the host, outside the repo, under the session
scratchpad.

## Addendum 2026-09-18 — clause 4 run, and the cause found

Clause 4 was run the day after this document was written. It changes the conclusion above, so it
is recorded here rather than left for the reader to discover from a PR thread.

### The candidate failed clause 4, then passed it

Five-model campaign against the [2026-09-17 baseline](../vision-campaign-2026-09-17-rocm-baseline.md),
same runner, same digests, same environment, think off. Scene bbox IoU:

| model | 0.32.1 / b9888 | 0.34.1 / b10864 | 0.34.1 + **906** |
|---|---|---|---|
| `qwen3.8:27b-q4_K_M` | 0.991 | **0.065** | **1.000** |
| `gemma4:31b-it-q4_K_M` | 0.961 | 0.922 *(3 of 6 objects)* | **0.960** |
| `nemotron3:33b-q4_K_M` | 0.857 | **0.161** | **0.862** |
| `qwen3.6:35b-a3b-q4_k_m` *(the gate model)* | 0.953 | **0.273** | **0.972** |
| `gemma4:26b-a4b-it-q4_K_M` | 0.973 | 0.966 | **0.975** |

Unpatched, three models also dropped invoice extraction 5/5 → **0/5** and fine-text OCR to zero
(`gemma4:31b` found 4 of 20 codes, against 20 of 20 on the baseline). With 906 every model
returns to **6/6 · 6/6 · 6/6** and **5/5 · 5/5 · ✅**, and `gemma4:31b` fine-text is digit-identical
to the baseline at 4/4/4/4/3, 20/20.

### The cause

Upstream `c7d8722922a` ([llama.cpp#24233](https://github.com/ggml-org/llama.cpp/pull/24233)) set
`info.devices[id].integrated = prop.integrated` on HIP builds. On an integrated/UMA device that
changes which buffer types the backend claims, and
[llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211) root-causes the result as an
**MMQ tile-barrier race on gfx115x producing wrong output with no crash and no warning, once a
decode exceeds `n_ubatch`**.

Upstream reverted it in [#28604](https://github.com/ggml-org/llama.cpp/pull/28604), merged
**2026-09-08T14:19:53Z**. Our payload b10864 is `5d806aa25`, **2026-09-08T13:01:03Z** — we are on
the wrong side of it by 78 minutes. Carried as `llama/compat/906-revert-hip-integrated-flag.patch`.

Two upstream reports describe this symptom on this exact silicon, both predating our measurements:
[#27419](https://github.com/ggml-org/llama.cpp/issues/27419) ("upper image content is lost while
Vulkan/CPU are correct", Ryzen AI Max+ 395 / Radeon 8060S / gfx1151 / Qwen3.8-27B) and #28211.

### Why every instrument said the build was healthy

Across 135 scored cells on the unpatched candidate: **0 amdgpu page faults, 0 GPU resets, 0 ring
timeouts, 0 HIP errors, 0 OOM kills, and 135/135 completions reporting `done_reason=stop`.** The
only GPU faults recorded all day belong to a `GGML_CUDA_NO_PINNED=1` experiment that aborted the
runner. Upstream notes that `llama-bench` cannot see the defect either, because throughput is
identical on both builds.

A plumbing check would have passed this image. Only scored output quality caught it — and even
there, `gemma4:31b` scene IoU read 0.932 against a healthy 0.961 while `name_bbox_mean_iou` on the
same run was 0.000. **The gate's own lesson 3 ("add output-quality checks to the deploy gate") is
the only reason this was caught.**

### What this does to clauses 1–3

- **Clauses 1 and 2 are unchanged.** Neither issue reproduces; both remain closed `not_planned`.
- **Clause 3 reads differently now.** Direct I/O was the gate's central suspicion for a month and
  is exonerated: it does not cause the vision regression (dio-off reproduces it exactly), and it is
  load-bearing — `GGML_CUDA_NO_PINNED=1`, the nearest thing to disabling the host-buffer path,
  aborts the runner on load. The `OLLAMA_IGPU_DIRECT_IO` knob is a convenience, not a safety valve.
- **The VRAM finding stands but reads differently.** 0.32.x seeing 27 GiB instead of 95 GiB is still
  a real cost of the pin. It is no longer an argument for upgrading on its own.

### Corrections to the body of this document

- The 25–36/61 layer-placement figures above are from the gate matrix at `-c 524288 -np 2`. At the
  vision campaign's `num_ctx 16384` **both builds run fully GPU-resident** (`offloaded 66/66
  layers` on `qwen3.8`), so placement is not a variable in the vision comparison.
- Hypotheses raised and refuted by experiment while chasing this, recorded so they are not re-run:
  fp16 accumulation in the vision tower; multi-sub-batch image decode; the b10864
  `RESIZE_ALGO_BILINEAR → BICUBIC` flip ([#27594](https://github.com/ggml-org/llama.cpp/pull/27594));
  the gemma4 token-budget bump ([#28335](https://github.com/ggml-org/llama.cpp/pull/28335)); and
  `GGML_CUDA_NO_PINNED=1`.

### Open

Whether this defect class reaches CUDA unified-memory devices — which bears directly on #17475,
reported on a DGX Spark GB10. Requested in MaxusAI/ollama#313.
