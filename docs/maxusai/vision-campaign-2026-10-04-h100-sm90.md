# Vision campaign 2026-10-04 — 0.35.0 on the H100 (sm_90), the 908 gate on Hopper, and think-on across the fleet

One host, one build. The machine is [2026-09-30's](vision-campaign-2026-09-30-h100-sm90.md): a GCP
`a3-highgpu-1g` VM with one H100 SXM5 80GB (sm_90), collab label `gcp-a3-highgpu-1g/cuda`.

- **The build is `0.35.0-dynres-15-ga657392`,** `main` at `a657392`. The H100's production service has run
  it since 2026-10-03. It is a Go-only build on production's payload, which has not changed since the 0.34.4
  deploy of 2026-09-30:
  - llama.cpp `161755f29` (b11081), with production's compat patches, 908 among them;
  - MLX `59d600b`.
- **The server:** every cell but one ran on a second server on `127.0.0.1:11535`, of this build except where
  §4 says otherwise. It ran as the login user and restarted cold before every cell through `RESTART_CMD`, so
  production kept serving. One cell ran on production's own service (§4).
- **The harness** is `main` at `f566becf4`, which sends `prompt_cache_ram: 0` (#442).
- **Wall clock:** 2026-10-04 05:45 → 19:35, plus one cell at 22:12. Every invocation finished with `rc=0`.

**What it is.** Five questions, one section each:

1. **Does 0.35.0 change anything on this host?**
   - §1 repeats 2026-09-30's think-off run: the same eight tags, the same knobs.
   - §2 repeats its two think-on cells.
2. **Does think-on finish across the fleet?** §2 runs think-on for the six tags 2026-09-30 left out. One of them
   is descoped by policy.
3. **Does Hopper need 908?** §3 runs [the retirement register's](retirement-register.md) test for compat
   patch 908 twice.
4. **Did the build, the prompt cache or the serving path move `gemma4:31b-it-q4_K_M`'s prefill?** 2026-09-30's
   document left its drift unexplained (its §4). §4 here rules all three out.
5. **Does MLX-CUDA honour gemma4's image budgets?** That was never observed before
   ([expectations.toml](vision-suite/preflight/expectations.toml), `[expect.mlx-cuda.gemma4]`). §5 measures it.

OCRBench on the same build is in [ocrbench-h100.md](ocrbench-h100.md#2026-10-04-0350-with-the-prompt-cache-off).

**The request is 2026-09-30's, byte for byte.** `prompt_sha`, `images_sha` and the server's `prompt_eval_count`
equal 2026-09-30's on all 224 think-off arms, and its repeat's (2026-10-01).

Committed data, assembled from the run captures by script (ADR 0012 rule 8):

- `vision-suite/bench-runs/vision-campaign-2026-10-04-h100-sm90.json` holds all 35 cells, both finetext arms
  each, assembled by `bundle_campaign.py` with one `--set` per runner invocation.
- `vision-suite/bench-runs/vision-campaign-2026-10-04-h100-sm90.host-profile.json` is the machine, in the
  `host-profile/1` format.
  - It equals 2026-09-30's profile in every machine field.
  - It was collected after the runs, on 2026-10-05.
  - It differs only in Docker's patch version, the Ollama version, and a kernel-module field the probe did not
    read this time.
- `vision-suite/bench-runs/vision-campaign-2026-10-04-h100-sm90.mlx-gemma4-budgets.json` holds §5's readings.

**Checkpoints measured**, identified by manifest digest (SPEC H17). The eight campaign tags are 2026-09-30's
artifacts, manifest for manifest. The two GGUF tags §3 adds were pulled on 2026-10-04.

| checkpoint | manifest sha256 | config | size |
|---|---|---|---|
| `gemma4:12b-nvfp4` | `ded7a27350032202` | `f6127828935e` | 7.2 GiB |
| `gemma4:26b-nvfp4` | `f60799545325362b` | `3a77d9b825b3` | 15.0 GiB |
| `gemma4:31b-nvfp4` | `a22a363052da7703` | `b72c5344f12d` | 18.1 GiB |
| `qwen3.8:27b-nvfp4` | `5642e97495e1a088` | `25a98d24af80` | 16.9 GiB |
| `qwen3.6:35b-a3b-nvfp4` | `e92a3e94bbca90a8` | `79866ce67dc0` | 22.0 GiB |
| `gemma4:31b-it-q4_K_M` | `17ba34c06c801ba2` | `094d707563aa` | 19.0 GiB |
| `qwen3.8:27b-q4_K_M` | `25b843619e944cd0` | `492b2922d38e` | 16.5 GiB |
| `nemotron3:33b-q8` | `74d89c84a4432530` | `655f279147f2` | 34.0 GiB |
| `gemma4:26b-a4b-it-q4_K_M` | `5571076f3d700504` | `90294508afbb` | 16.8 GiB |
| `gemma4:e4b-it-q4_K_M` | `dc35e8d9c6061baa` | `7ca1ae564b24` | 6.1 GiB |

`gemma4:26b-a4b-it-q4_K_M` is the artifact the v0.34.4 fold's CUDA gates measured (`5571076f3d70`,
[its model digests](tasks/upstream-sync-0.34.4.md#model-digests)), so §3's loop counts compare like with like.
`gemma4:e4b-it-q4_K_M` is not: that fold read `c6eb396dbd59`, and the tag has been re-published since.

## 1. Think-off, eight tags: 0.35.0 equals 0.34.4

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.727** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.972** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.962** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.966** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| nemotron3:33b-q8 | GGUF | 16384 | 0.878 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 4 | 3 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 541 | 86 | 2714 | 6.9 | 519 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 94 | 2243 | 6.4 | 560 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 40 | 1785 | 14.4 | 251 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 44 | 6997 | 12.8 | 282 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 533 | 70 | 6159 | 8.1 | 446 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 150 | 2155 | 4.4 | 825 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 78 | 1645 | 8.6 | 421 |
| nemotron3:33b-q8 | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 245 | 4818 | 2.6 | 1361 |

Provenance (from score files): host(s) http://127.0.0.1:11535 · build(s) 0.35.0-dynres-15-ga657392 · think=false

**Arm by arm, against 2026-09-30 and its repeat.** An arm differs when any recorded field differs, timings and
stamps aside. The third column is the noise floor: two runs of the same build, two days apart.

| model | 10-04 against 09-30 | 10-04 against 10-01 | 10-01 against 09-30 |
|---|---|---|---|
| gemma4:12b-nvfp4 | 18 / 28 | 18 / 28 | 17 / 28 |
| gemma4:26b-nvfp4 | 17 / 28 | 13 / 28 | 17 / 28 |
| gemma4:31b-nvfp4 | 12 / 28 | 18 / 28 | 13 / 28 |
| qwen3.8:27b-nvfp4 | 7 / 28 | 8 / 28 | 7 / 28 |
| qwen3.6:35b-a3b-nvfp4 | 17 / 28 | 19 / 28 | 14 / 28 |
| gemma4:31b-it-q4_K_M | **0** / 28 | **0** / 28 | **0** / 28 |
| qwen3.8:27b-q4_K_M | **0** / 28 | **0** / 28 | **0** / 28 |
| nemotron3:33b-q8 | **0** / 28 | **0** / 28 | **0** / 28 |

- **GGUF: every arm equals 0.34.4's, in both of 0.34.4's runs.** llama.cpp on this H100 repeats itself exactly
  across days, servers and builds, as it does on the sm_120 host. A GGUF difference here would be real.
- **MLX moves no more than it moves against itself:** 71 of 140 arms against 2026-09-30, where 2026-10-01
  moved 68. In the tables:
  - **`gemma4:31b-nvfp4` reads 3 of 4 at 9 px instead of 4.** Its scene IoU is 0.962, against 0.965 in both
    0.34.4 runs. `gemma4:26b-nvfp4` moved by the same amount between those two runs.
  - **`gemma4:26b-nvfp4` repeats its 2026-10-01 values:** 3 at 9 px, scene IoU 0.972.
  - **`gemma4:12b-nvfp4`'s scene IoU, 0.727, falls inside the 0.724–0.731** that 2026-09-30's cell, reruns and
    repeat spanned.
  - **`qwen3.6:35b-a3b-nvfp4` answers in 533 tokens,** against 534 and 537. It is again the only answer length
    to move.
- **Rates: decode holds to within 2 tok/s, and prefill to within 3 %,** except `gemma4:31b-it-q4_K_M`'s prefill
  (2155 tok/s, against 1421 and 1740). §4 is about that one.

## 2. Think-on, seven tags

`gemma4:12b-nvfp4` is not measured think-on: it is in `DESCOPED_CELLS`, and the bundler records the skip.

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.998** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| gemma4:26b-nvfp4 | **MLX** | 16384/32768/131072 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/32768/65536 ⚠ | **0.964** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.964 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 1.000 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| nemotron3:33b-q8 | GGUF | 16384/32768 ⚠ | 0.842 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 5/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 4076 | 40 | 1813 | 101.9 | 35 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 986 | 44 | 6995 | 22.6 | 159 |
| gemma4:26b-nvfp4 | **MLX** | 16384/32768/131072 ⚠ | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | capped | capped | ≥122880 ⚠ | 94 | 775 | capped | capped |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/32768/65536 ⚠ | 4 | 4 | 4 | 2 | 3 | ✅ q1 + q2 + q4-bbox | ❌ q4_bbox_hit | — | 2710 | 76 | 5875 | 36.1 | 100 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1378 | 154 | 2198 | 9.7 | 372 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 1 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1084 | 78 | 1669 | 15.5 | 232 |
| nemotron3:33b-q8 | GGUF | 16384/32768 ⚠ | 4 | 4 | 4 | 4 | 1 | ✅ q1 + q2 + q4-bbox | ❌ q4_bbox_hit | — | 5509 | 245 | 4878 | 23.1 | 156 |

Provenance (from score files): host(s) http://127.0.0.1:11535 · build(s) 0.35.0-dynres-15-ga657392 · think=on

**Three cases never finish, all on MLX. Every GGUF case finishes.**

| model | engine | not converged at 131072 | finished at 16384 / 32768 / 65536 / 131072 | wall clock |
|---|---|---|---|---|
| gemma4:26b-nvfp4 | MLX | **2**: `scene_single`, `multi_3img_anchored` | 18 / 6 / 1 / 0 | 1 h 46 |
| qwen3.6:35b-a3b-nvfp4 | MLX | **1**: `bbox_contract` | 6 / 14 / 3 / 3 | 3 h 34 |
| gemma4:31b-nvfp4 | MLX | 0 | 27 / 0 / 0 / 0 | 21 min |
| qwen3.8:27b-nvfp4 | MLX | 0 | 27 / 0 / 0 / 0 | 15 min |
| gemma4:31b-it-q4_K_M | GGUF | 0 | 27 / 0 / 0 / 0 | 6 min |
| qwen3.8:27b-q4_K_M | GGUF | 0 | 27 / 0 / 0 / 0 | 8 min |
| nemotron3:33b-q8 | GGUF | 0 | 23 / 4 / 0 / 0 | 14 min |

- **All three are loops, not long reasoning.** Each spends the 131072 rung's 122,880 tokens repeating itself:
  82–182 distinct lines of 5,071–8,128, the top line repeated 613–665 times, and 8–24 distinct lines in the second
  half. qwen3.6's longest finished case reads 550 distinct lines of 1,175.
- **`gemma4:26b-nvfp4` `multi_3img_anchored` is a known loop.** On the sm_120 host it never converged in any of 8
  MLX-CUDA runs ([the v0.34.4 fold record](tasks/upstream-sync-0.34.4.md)).
- **qwen3.6 thinks long rather than looping.** 21 of its 27 cases do not finish within the first rung's 8,192
  tokens, and three finish only at 131072.
- **The two tags 2026-09-30 ran think-on lose nothing.** Every case finishes at the first rung. On 2026-09-30,
  two `gemma4:31b-nvfp4` cases (`bboxm_pin_anc_pos`, `scene_single_pinned`) needed 32768.
  - `qwen3.8:27b-nvfp4` scores 0.998 scene IoU against 0.994, reads one more 9 px code, and answers
    `q4_bbox_hit`.
  - 51 of these 56 arms differ from 2026-09-30, and think-on has no repeats control, so none of this is a finding.

## 3. The 908 gate on Hopper: four loops without 908, one with it

Compat patch 908 reverts the device half of llama.cpp `ce8caa6e6`, the flash-attention MMA tiling for head
dimensions 256 and 512. It is carried because, on the sm_120 host, that tiling left gemma4:26b think-on cases in
loops that never end. [The register](retirement-register.md) retires 908 when a fold's CUDA GGUF loop-rate run
on llama.cpp's own tiling matches 908's. Until now that was measured only on sm_120.

**The arms differ in one library.**
- **Arm A** is production's payload, with 908.
- **Arm B** is a full copy of that payload whose `cuda_v13/libggml-cuda.so` was rebuilt without 908:
  - **The source** is production's prepared llama.cpp source (b11081 with its compat patches applied), with 908
    alone reversed by `git apply -R`. The patch then re-applies cleanly.
  - **The configuration** comes from production's own CMake cache: every user-set entry is equal. Only the
    `ggml-cuda` target was built.
  - **The result:** both libraries hold 143 sm_90 ELF images and no PTX. They are 46,413,400 and 46,224,984
    bytes, sha256 `84232944d375b50c…` with 908 and `9d2023c67f7125ec…` without.
  - **The runner's memory map confirmed each arm loaded its own library.**

Each arm ran the register's test:
- the full think-on suite on gemma4:26b-a4b-it-q4_K_M, single pass, f16 KV, flash attention on, batch 2048;
- the ladder 16384 → 131072, from a cold server;
- then the think-off cells of gemma4:31b, 26b-a4b and e4b.

Both think-on cells then ran again (`with908rep_r1_`, `no908rep_r1_`).

| gemma4:26b-a4b-it-q4_K_M, think on | NOT CONVERGED at 131072 | `stop` | `json_valid` | `contract_followed` |
|---|---|---|---|---|
| **H100 (sm_90), with 908** | **1** | 26 / 27 | 26 / 27 | 17 / 20 |
| **H100 (sm_90), without 908** | **4** | 23 / 27 | 23 / 27 | 17 / 20 |
| sm_120, the device half reverted (= 908), b11081 | 1 | 26 / 27 | 25 / 27 | 16 / 20 |
| sm_120, the fold as shipped (no 908), b11081 | 6 | 21 / 27 | 21 / 27 | 15 / 20 |

The sm_120 rows are [the v0.34.4 fold's](tasks/upstream-sync-0.34.4.md#think-on-a-cuda-only-loop-from-ce8caa6e6s-device-half),
on the same gemma4:26b-a4b artifact.

- **Without 908, four cases never finish:** `multi_3img_anchored`, `bbox_contract_reasoning`,
  `bbox_contract_real_1img` and `bbox_contract_adv_real`. With it, one: `bboxm_pin_noanc_named`.
  - **The case behind 908 behaves as it did on sm_120.** `multi_3img_anchored` loops without 908 on both GPUs.
    With 908 it finishes, here at 32768 in 6,720 tokens.
  - `bbox_contract_adv_real` loops without 908 on both GPUs too. The other cases differ: which case loops moves
    with the numerics, as the fold record found.
  - **All five are loops:** 91–182 distinct lines of 4,317–9,321, the top line repeated 399–715 times.
- **The repeat is byte-identical.** Both arms' second runs equal their first in every field of all 28 arms, and
  every thinking trace matches byte for byte, the 122,880-token loops included. CUDA GGUF is deterministic on
  Hopper too, so one run per arm is the measurement.
- **Think off, removing 908 moves numbers, and the outcomes it moves cancel out.**
  - Without it, 53 of 84 arms change: 14 on gemma4:31b, 16 on 26b-a4b, 23 on e4b.
  - With it, gemma4:31b equals §1's cell in all 28 arms.
  - `cmp_scored.py` (moves under 0.005 ignored) finds 11 quality moves favouring the arm without 908 and 10
    favouring the arm with it. e4b holds 19 of them: it loses `bbox_contract_multi` (6 boxes to 0) and gains
    `bbox_contract_perobject` (5 to 6).
  - The headline tables move only in scene IoU: 26b-a4b 0.978 → 0.976 and e4b 0.679 → 0.678.

**So the retirement condition is not met on Hopper either.** On llama.cpp's own tiling the loop rate is four
times 908's here, against six times on sm_120. 908 stays.

## 4. gemma4:31b-it-q4_K_M's prefill: neither the build, the prompt cache nor the service path

2026-09-30's document left one rate unexplained: that tag's prefill rose a fifth between its campaign and the
repeat, and §1 here is faster again. Each table rate is one request's, the scene cell's. So this section uses the
median over all 28 requests of a cell.

Nine cells ran in three rounds, rotating the order each round, on one server, each from a cold restart:
- **`off`:** 0.35.0, `prompt_cache_ram 0`;
- **`def`:** 0.35.0, no option sent, so llama.cpp's 8192 MiB cache is on, as on 2026-09-30;
- **`old`:** `0.34.4-dynres-0-gb43ee8e`, 2026-09-30's build, on the same payload with the cache on.

One more cell, `pfprod`, ran through production's own service with the cache on. Rendered from the two bundles
by the snippet under "Reproducing":

| cell | build | host | prompt_cache_ram | prefill, median of 28 (ms) | scene prefill (ms) | request time, 28 requests (s) |
|---|---|---|---|---|---|---|
| 2026-09-30 | 0.34.4-dynres-0-gb43ee8e | 11434 | none sent | 1058 | 1185 | 140.0 |
| 2026-10-01 | 0.34.4-dynres-0-gb43ee8e | 11434 | none sent | 961 | 968 | 138.2 |
| 2026-10-04 | 0.35.0-dynres-15-ga657392 | 11535 | 0 | 780 | 782 | 129.7 |
| pfoff1 | 0.35.0-dynres-15-ga657392 | 11535 | 0 | 800 | 779 | 130.5 |
| pfdef1 | 0.35.0-dynres-15-ga657392 | 11535 | none sent | 866 | 779 | 138.5 |
| pfold1 | 0.34.4-dynres-0-gb43ee8e | 11535 | none sent | 770 | 769 | 135.4 |
| pfoff2 | 0.35.0-dynres-15-ga657392 | 11535 | 0 | 779 | 779 | 130.2 |
| pfdef2 | 0.35.0-dynres-15-ga657392 | 11535 | none sent | 783 | 768 | 135.6 |
| pfold2 | 0.34.4-dynres-0-gb43ee8e | 11535 | none sent | 770 | 789 | 135.4 |
| pfoff3 | 0.35.0-dynres-15-ga657392 | 11535 | 0 | 770 | 768 | 129.9 |
| pfdef3 | 0.35.0-dynres-15-ga657392 | 11535 | none sent | 781 | 782 | 135.1 |
| pfold3 | 0.34.4-dynres-0-gb43ee8e | 11535 | none sent | 773 | 768 | 135.5 |
| pfprod | 0.35.0-dynres-15-ga657392 | 11434 | none sent | 785 | 774 | 133.6 |

- **Every configuration measures the same prefill today.** That includes 2026-09-30's own: its build, cache on,
  even production's service. The 11 cells of 2026-10-04 give medians of 770–866 ms and scene prefills of 768–789
  ms.
- **So none of the three explains the drop.** Not the build, not the prompt cache, not the serving path.
  - 2026-09-30's 1058 ms and 2026-10-01's 961 ms came from something else on those days.
  - The journal for those hours has rotated, so this document does not say what.
- **The cache costs time outside the prefill timer.** With it on, a cell takes 134–139 s; with it off, 130 s.
  - That is about 4 % more: the per-request state copy that
    [llama-server-prompt-cache.md](llama-server-prompt-cache.md) measured. llama-server makes it before the
    request starts, so `prompt_eval_duration` never sees it.
  - On OCRBench every item is a new image. There this tag took 21 % less time per item with the cache off than
    on 2026-10-01 with it on, as that document's A/B predicts
    ([ocrbench-h100.md](ocrbench-h100.md#2026-10-04-0350-with-the-prompt-cache-off)).
- **The answers do not move.** All 11 cells agree with both 0.34.4 runs in every arm.

## 5. MLX-CUDA honours gemma4's image budgets, and clamps them to 70–1120

The MLX runner logs no budget line, so `[expect.mlx-cuda.gemma4]` carries its budgets
(`budgets_observed = false`). Here they are read from the runner's behaviour instead. Every reading is
`gemma4:31b-nvfp4` on 0.35.0, with the MLX limit pinned at 48 GiB as in the 2026-08-17 measurements.

- **The ladder holds.** `measure_ladder.py` reads 1102 image tokens at all five of preflight's geometries, with
  prefix and text-only both 19. That is the block's ladder exactly.
- **A pinned budget is honoured, and the default is the top one.** `image_min_tokens = image_max_tokens = pin`,
  sent through preflight's own `probes.Ollama` client, gives these image tokens (prompt_eval_count less the text
  prompt's 19):

| pin | none | 35 | 69 | 70 | 140 | 280 | 560 | 1120 | 1121 | 2240 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024x576 | 1102 | 68 | 68 | 68 | 122 | 266 | 529 | 1102 | 1102 | 1102 |
| 2048x1152 | 1102 | 68 | 68 | 68 | 122 | 266 | 529 | 1102 | 1102 | 1102 |

- **Pins below 70 read as 70, and pins above 1120 read as 1120.** So the runner clamps to the block's
  `budget_min_tokens` and `budget_max_tokens`.
- **The committed probe agrees.** `probe_mlx_image_budget.py` sweeps `image_max_tokens` alone on 2048x1152 and
  reads `KNOB LIVE`: 1122 / 1122 / 549 / 286 / 142 / 88 at none / 1120 / 560 / 280 / 140 / 70. Its prompt is a
  different sentence and every count is 20 higher, so the image tokens are the same 1102 / 529 / 266 / 122 / 68.

`budgets_observed` stays `false`. It records that no load log shows these values, which is still true;
`test_verdicts.py` holds every MLX block to it. The block now records this reading beside it.

## 6. Limits

- **The 908 rows from two GPUs differ in more than the GPU.**
  - The sm_120 arms ran in canary containers with a 16 GiB `OLLAMA_GPU_OVERHEAD`. These ran on a native server,
    on the same artifact.
  - Each GPU's count comes from one deterministic run per arm. The repeat proves that run repeats; it does not
    sample other numerics.
- **The think-off 908 arms use a newer `gemma4:e4b-it-q4_K_M`** than the sm_120 host's.
- **Think-on has no repeats control on MLX,** where runs differ, so the MLX think-on counts are one draw each.
- **2026-09-30's prefill drift is located, not explained** (§4).
- **The host profile was collected after the runs,** with the same values as 2026-09-30's.
- **Cross-host comparison is out of scope,** as in 2026-09-30's document, beyond the 908 rows.

## Reproducing

The tables render from the bundle alone, with `summarize_engine_compare.py --bundle`:

```bash
cd docs/maxusai/vision-suite
B=bench-runs/vision-campaign-2026-10-04-h100-sm90.json
python3 summarize_engine_compare.py --bundle $B --think false --prefix c0350_r1_ gemma4:12b-nvfp4 \
  gemma4:26b-nvfp4 gemma4:31b-nvfp4 qwen3.8:27b-nvfp4 qwen3.6:35b-a3b-nvfp4 gemma4:31b-it-q4_K_M \
  qwen3.8:27b-q4_K_M nemotron3:33b-q8
python3 summarize_engine_compare.py --bundle $B --think on --prefix c0350_r1_ gemma4:31b-nvfp4 \
  qwen3.8:27b-nvfp4 gemma4:26b-nvfp4 qwen3.6:35b-a3b-nvfp4 gemma4:31b-it-q4_K_M qwen3.8:27b-q4_K_M nemotron3:33b-q8
for p in with908_r1_ no908_r1_ with908rep_r1_ no908rep_r1_; do
  python3 summarize_engine_compare.py --bundle $B --think on --prefix $p gemma4:26b-a4b-it-q4_K_M
done
for p in with908_r1_ no908_r1_; do
  python3 summarize_engine_compare.py --bundle $B --think false --prefix $p gemma4:31b-it-q4_K_M \
    gemma4:26b-a4b-it-q4_K_M gemma4:e4b-it-q4_K_M
done
```

`cmp_scored.py` reads score files. Write the two cells out of the bundle first:

```bash
for p in with908_r1_ no908_r1_; do for m in gemma4_31b-it-q4_K_M gemma4_26b-a4b-it-q4_K_M gemma4_e4b-it-q4_K_M; do
  jq ".cells[] | select(.tag == \"${p}${m}_thinkfalse\") | .scores" $B > /tmp/scores_${p}${m}.json
done; done
for m in gemma4_31b-it-q4_K_M gemma4_26b-a4b-it-q4_K_M gemma4_e4b-it-q4_K_M; do
  python3 cmp_scored.py /tmp/scores_with908_r1_$m.json /tmp/scores_no908_r1_$m.json
done
```

The arm counts in §1 and §3, the identities in §3 and §4, and §4's table come from this, run in the same
directory:

```python
import json, statistics as st
cells = {}
for f in ("bench-runs/vision-campaign-2026-09-30-h100-sm90.json", "bench-runs/vision-campaign-2026-10-04-h100-sm90.json"):
    for c in json.load(open(f))["cells"]:
        cells[c["tag"]] = dict(c["scores"], finetext_probe=c["finetext_probe"])
STAMPS = {"host", "server_version", "tag", "cold_start", "capture_schema", "req_prompt_cache_ram", "retries", "powermode",
          "total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "gen_tps", "prefill_tps"}
REQUEST = ("prompt_sha", "images_sha", "prompt_eval_count")
def strip(arm): return {k: v for k, v in arm.items() if k not in STAMPS}
def cmp(a, b):   # arms that differ, and arms whose request differs
    a, b = cells[a], cells[b]
    return (sum(strip(a[k]) != strip(b[k]) for k in a), sum(any(a[k].get(f) != b[k].get(f) for f in REQUEST) for k in a))
T = "gemma4_31b-it-q4_K_M_thinkfalse"
print(cmp("c0350_r1_" + T, T), cmp("no908rep_r1_gemma4_26b-a4b-it-q4_K_M_thinkon", "no908_r1_gemma4_26b-a4b-it-q4_K_M_thinkon"))
for label, p in [("2026-09-30", ""), ("2026-10-01", "nf1_"), ("2026-10-04", "c0350_r1_")] + \
        [(f"pf{a}{n}", f"pf{a}{n}_r1_") for n in (1, 2, 3) for a in ("off", "def", "old")] + [("pfprod", "pfprod_r1_")]:
    a = list(cells[p + T].values())
    pc = a[0].get("req_prompt_cache_ram")
    print(f"| {label} | {a[0]['server_version']} | {a[0]['host'].split(':')[-1]} | {'none sent' if pc is None else pc} "
          f"| {st.median(x['prompt_eval_duration'] for x in a) / 1e6:.0f} "
          f"| {cells[p + T]['scene_single']['prompt_eval_duration'] / 1e6:.0f} "
          f"| {sum(x['total_duration'] for x in a) / 1e9:.1f} |")
```

§5's readings are in `vision-campaign-2026-10-04-h100-sm90.mlx-gemma4-budgets.json`: `measure_ladder` is the
tool's `--out` file, `pinned_sweep` and `pinned_bounds` hold one row per (geometry, pin), and
`probe_mlx_image_budget` is that probe's output. To read them again, on an MLX-CUDA server with gemma4:31b-nvfp4:

```bash
python3 probe_mlx_image_budget.py http://127.0.0.1:11434 mlx-cuda gemma4:31b-nvfp4 preflight/ladderimgs/2048x1152.png
```

The repetition profiles in §2 and §3 count the lines of the `think_<tag>_<test>.txt` traces `vision_suite.py`
writes next to its score files, which are not committed.
