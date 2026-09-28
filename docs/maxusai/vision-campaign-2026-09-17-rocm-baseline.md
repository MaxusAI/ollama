# Vision baseline 2026-09-17: ROCm/gfx1151, the production build, five models, think off

> Verdict: **the production build is intact, and this is the regression reference for the
> next ROCm image.** All five models score 6/6 boxes, labels and colours, find the serial,
> extract the invoice 5/5 · 5/5 · ✅, and answer every multi-image question including the
> anchored arm; every cell converged at the 16384 start rung — no escalation, no cap, no OOM,
> no error. `qwen3.8:27b-q4_K_M` reproduces the 2026-08-17 ROCm campaign almost exactly
> (scene IoU 0.991, fine text 4/4/4/2/1), so neither the host nor the build moved in a month.
> `bcanchored` — the ADR 0027 request shape — is ✅ on all five. **Quality numbers are a
> baseline for every row; throughput is a baseline for three rows only** — see
> [Power and load](#power-and-load-read-before-quoting-any-rate).

## Provenance

- **Date / host:** 2026-09-17 22:36:32–23:56:07 local (80 min), `amd-server`,
  Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151**, 96 GiB firmware VRAM carve,
  30 GiB left to the OS. First vision run on this host since 2026-08-18, and the first since
  the APU was repasted.
- **Build under test:** `maxusai-ollama:0.32.1-rocm-dynres-5d5b7a72` — the image production
  serves, `0.32.1-dynres-5d5b7a72`, payload **b9888** (`cb295bf59`), ROCm **7.2.1**, compat
  001+002+004+005, no direct I/O. Unchanged since the 2026-08-17 deploy; `release/0.32.1-dynres`
  has moved only in docs and preflight since.
- **Server:** a scratch container of that same image on `:11499`, not production. Production's
  environment verbatim (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`,
  `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_DEBUG=1`) plus `OLLAMA_MAX_LOADED_MODELS=1` (AGENTS.md) and
  `OLLAMA_NOPRUNE=1`, sharing production's store read-write. Production (`ollama-rocm`,
  `:11434`) held no model for the whole run.
- **Driver:** `run_engine_compare.sh`, `THINK_MODES=false`, `TAG_PREFIX=rocm0321_`, cold
  restart per model (`docker restart ollama-bench`), context ladder at defaults. Tags are
  `rocm0321_1_<model>_thinkfalse`, so the generators take `--prefix rocm0321_1_`.
- **Models and digests:** `qwen3.8:27b-q4_K_M` `25b843619e94` (qwen35) ·
  `gemma4:31b-it-q4_K_M` `6316f0629137` · `nemotron3:33b-q4_K_M` `baa676a14e13`
  (nemotron_h_omni) · `qwen3.6:35b-a3b-q4_k_m` `07d35212591f` (qwen35moe) ·
  `gemma4:26b-a4b-it-q4_K_M` `5571076f3d70`. The first three are the `rocm-0-32-1-dynres`
  preflight arches; `qwen3.6` is the model the AMD gate was tripped on
  ([amd-upgrade-gate.md](amd-upgrade-gate.md)); `gemma4:26b` completes the pair the gate's
  #17475 noise loop uses.
- **Think off only.** It is the served mode on gfx1151
  ([ADR 0025](adr/0025-think-stays-off-on-gfx1151.md)); think-on on this host is not measured
  here.

## Power and load: read before quoting any rate

The run did not hold one power envelope, and the runner cannot say so on Linux: its
`powermode` stamp reads macOS `pmset` and prints `n/a` here. The segmentation below is
reconstructed from a 5-second telemetry log (k10temp, amdgpu hwmon, `/proc/stat`) and from
`ryzenadj --info` every 30 seconds.

**The box was in the firmware's Balanced preset: 85 W sustained, 120 W burst.**
`ryzenadj` read STAPM 85 W / slow PPT 85 W / fast PPT 120 W, with the live STAPM and slow
values sitting at 84.95 W and 84.99 W under GPU load — power-limited, at Tctl 68–72 °C.
`/sys/firmware/acpi/platform_profile` does not exist on this machine, and
`power-profiles-daemon`'s platform driver is `placeholder`, so the OS profile cannot move the
firmware limits. At **23:04:42** the SMU limits were raised to **120 / 120 / 140 W** with
`ryzenadj` (volatile — lost on reboot) and the OS profile set to `performance` (EPP
`performance` on all 32 threads). No firmware revert was observed for the rest of the night.

| model | window | package limit | mean draw | CPU >50% | throughput is a 120 W baseline? |
|---|---|---|---|---|---|
| `qwen3.8:27b` | 22:36–23:05 | **85 W** (last 16 s at 120) | 85 W | 0% | **no — Balanced** |
| `gemma4:31b` | 23:05–23:34 | 120 W | 117 W | **28%** | **no — overlapped a full image compile** (23:12–23:25) |
| `nemotron3:33b` | 23:34–23:42 | 120 W | 108 W | 0% | yes |
| `qwen3.6:35b` | 23:42–23:50 | 120 W | 111 W | 0% | yes |
| `gemma4:26b` | 23:50–23:56 | 120 W | 102 W | 0% | yes |

Mean draw sits below the limit because the window includes the cold restart and the idle
gap between arms. The last three rows ran beside network and disk I/O (model and image pulls)
and nothing else. Quality is read as unaffected in every row, with one qualification:
think-off is greedy decoding, which the eighteen-model campaign found power-invariant on
Apple silicon, but on this host the 2026-08-17 campaign measured it *near*-deterministic, not
exact (scene IoU 0.991 → 0.988 between two runs). Differences of that size between these rows
and a future run are noise, not regression.

**Pending:** an idle 120 W re-run of `qwen3.8:27b` and `gemma4:31b`, so every row has a
comparable rate.

### Thermals after the repaste

| load | package | Tctl (k10temp) | GPU edge | SMU "core" |
|---|---|---|---|---|
| idle | 7 W | 31.8 °C | 31 °C | — |
| GPU only, Balanced | 85 W | 68–72 °C | 67–73 °C | ~68 °C |
| GPU only, 120 W | 115–120 W | **98 °C** | up to **91 °C** | 77–80 °C, stable |
| image compile (50 compilers) + GPU, 120 W | **120 W** pinned | 82–92 °C | 67–71 °C | 68–88 °C |

- **Not thermally limited at the chassis's rated envelope.** Slow PPT held at 115–120 W
  through six minutes of GPU-only load and at 120.0 W through the whole combined window, with
  STAPM converging to 119.3 W; the SMU never pulled power back. The binding limit is the 120 W
  power cap, not temperature. There is no pre-repaste measurement at 120 W, so this is a
  statement about headroom, not an improvement figure.
- **k10temp's Tctl is a fast hotspot, and it is real.** It rose 29 °C within 5 s of the
  +33 W step and fell 32 °C within 10 s of idle; `ryzenadj`'s core value is a slower,
  filtered reading of the same die. The hotspot sits at the 98 °C `tctl-temp` limit under
  GPU-concentrated load.
- **The die runs cooler under combined load than under GPU load alone.** The 120 W is a
  shared budget: with 16 CPU cores taking a share, less of it concentrates in the iGPU region
  where the hotspot is.
- Memory peaked at 14.7 GiB of 30 during the compile, with 50 compiler processes observed at
  ninja's default parallelism — nowhere near an OOM beside a resident 25 GiB model.

## Against 2026-08-17

`qwen3.8:27b-q4_K_M` on the same image, a month apart
([vision-campaign-2026-08-17-qwen38-rocm.md](vision-campaign-2026-08-17-qwen38-rocm.md)):

| | 2026-08-17 (first think-off run) | 2026-09-17 |
|---|---|---|
| scene bbox IoU | 0.991 | 0.991 |
| fine text 22/16/12/9/7 px | 4/4/4/2/1 | 4/4/4/2/1 |
| multi q4-bbox | ✅ | ✅ |
| gen tok/s | 12.2 | 12 |
| prefill tok/s | 255–276 | 244 |

This run was at 85 W, observed. August's power envelope was not recorded, so the matching
throughput is consistent with the same Balanced preset but does not establish it.
`name_bbox` now renders as `5/5` in-band because the column changed (ADR 0012 rule 11), not
because the model did.

## Not measured here

- Think-on, on any model.
- A 120 W idle rate for `qwen3.8:27b` and `gemma4:31b` (pending, above).
- `qwen2.5vl` (its ROCm path is untested; its fp16 mitigation is CUDA-specific) and the small
  gemma4 (`e2b`, `e4b`).
- Anything about the 0.34.1 / ROCm 7.2.4 candidate — that is a separate record.

## Tables — generator output, rendered 2026-09-18T00:00:26

```
python3 summarize_engine_compare.py --prefix rocm0321_1_ --think false \
  qwen3.8:27b-q4_K_M gemma4:31b-it-q4_K_M nemotron3:33b-q4_K_M qwen3.6:35b-a3b-q4_k_m gemma4:26b-a4b-it-q4_K_M
python3 summarize_contract_matrix.py --prefix rocm0321_1_ --think false <same models>
```

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.991 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.961 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.857 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 0.953 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 544 | 12 | 244 | 56.4 | 64 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 530 | 9 | 608 | 61.7 | 58 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 62 | 5669 | 8.7 | 414 |
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 550 | 59 | 4363 | 9.9 | 362 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 532 | 49 | 363 | 15.5 | 232 |

Provenance (from score files): host(s) http://127.0.0.1:11499 · build(s) 0.32.1-dynres-5d5b7a72 · think=false

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8:27b-q4_K_M | GGUF | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 16384 |
| gemma4:31b-it-q4_K_M | GGUF | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 16384 |
| nemotron3:33b-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 16384 |
| qwen3.6:35b-a3b-q4_k_m | GGUF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 16384 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | 16384 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.
