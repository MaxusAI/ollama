# Vision campaign 2026-08-19 — bbox contract across image geometry

Server `0.32.14-maxusai-9594f81e` (b10434), Apple Silicon, powermode 2,
temperature 0, cold restart per model, n=1 per cell. Arms
`bbox_contract_anchored_1img` (norm-1000 pin) and `bbox_contract_real_1img`
(`real` + `ref_size` pin), both single-image. 14 geometries per
[SPEC §4.1](spec/vision-bbox-response-contract.md); decision in
[ADR 0030](adr/0030-bbox-conformance-is-scoped-to-image-geometry.md).

**What this measures.** Every previously published contract rate came from one
fixture at 1920×1080. This is the first measurement of the contract off that
point — at 320×320, at 3840×2160, in portrait, and at six non-round sizes drawn
once from `random.seed(20260818)` to stand in for images pasted into a chat
window.

**Not comparable to earlier campaigns.** These arms send **one** image; the
21/21 and 42/42 rates in the SPEC are three-image distractor-condition cells.
Throughput fields are omitted deliberately: part of this sweep ran while another
process held the host at loadavg ~148, so `gen_tps`/`prefill_tps` are not
comparable to clean cells. The quality metrics below are load-invariant.

## 1. The result in one line

Pinning norm-1000 makes image geometry stop mattering. Pinning `real` pixels
makes it matter enormously, and differently per model.

| pin | cells converting 6/6 |
|---|---|
| norm-1000 | **55 of 56** (14 geometries × 2 models × 2 think modes) |
| `real` + `ref_size` | qwen3.8 **14/14**, qwen3.6 **1/14** |

Under the norm-1000 pin the model never names the frame it works in, so its
internal resize cannot reach the coordinates. Under a `real` pin it must name
one, and that is exactly where the resize leaks.

## 2. Real-pinned — the frame arm

Ratio is the reported frame over the image actually sent. `chk anc/bf` is
`self_check`, `hits_anchor`, `hits_bestfit`. **⚠ marks a silent failure**:
`self_check` passed while the anchor converts fewer than 6 boxes.

Columns: qwen3.6:35b-a3b-q4_K_M, then qwen3.8:27b-q4_K_M, both think-off.

**bbox_contract_real_1img** (`frm-*`)

| geometry | sent | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf |
|---|---|---|---|---|---|---|---|
| `hd` | 1920×1080 | 1920×1080 | 1.00× | ❌ 1/6 | 2560×1440 | 1.33× | ✅ 6/1 |
| `hd_al32` | 1920×1088 | 1792×1024 | 0.93× | ❌ 1/6 | 2324×1322 | 1.21× | ✅ 6/1 |
| `hd_al48` | 1920×1104 | 1600×900 | 0.83× | ❌ 1/6 | 2304×1296 | 1.20× | ✅ 6/2 |
| `sq320` | 320×320 | 800×800 | 2.50× | ❌ 1/6 | *norm1000* | — | ✅ 6/6 |
| `vga` | 800×600 | 1000×750 | 1.25× | ✅ 3/6 ⚠ | 1000×750 | 1.25× | ✅ 6/3 |
| `portrait` | 1080×1920 | 800×1400 | 0.74× | ❌ 1/6 | 1000×1750 | 0.93× | ✅ 6/6 |
| `uhd` | 3072×1728 | 1707×960 | 0.56× | ❌ 1/6 | 2560×1440 | 0.83× | ✅ 6/3 |
| `uhd4k` | 3840×2160 | 2560×1440 | 0.67× | ❌ 0/6 | 2560×1440 | 0.67× | ✅ 6/1 |
| `paste1` | 1668×733 | 1600×800 | 0.96× | ❌ 1/6 | 2400×1000 | 1.44× | ✅ 6/1 |
| `paste2` | 2812×2135 | 1600×1200 | 0.57× | ✅ 2/6 ⚠ | 2337×1754 | 0.83× | ✅ 6/3 |
| `paste3` | 1235×1181 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste4` | 2750×2379 | 1600×1200 | 0.58× | ✅ 2/6 ⚠ | 2337×2137 | 0.85× | ✅ 6/4 |
| `paste5` | 3030×1549 | 1707×960 | 0.56× | ❌ 1/6 | 2560×1440 | 0.84× | ✅ 6/3 |
| `paste6` | 3011×2317 | 1400×1050 | 0.46× | ✅ 2/6 ⚠ | 2337×1754 | 0.78× | ✅ 6/1 |

- **qwen3_6_35b-a3b-q4_K_M/false** — 14 geometries: 1 convert 6/6, 9 rejected by C7, **4 silent failures** (C7 passed, anchor does not convert)
- **qwen3_8_27b-q4_K_M/false** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)


## 3. norm-1000 pinned — the control

Columns: qwen3.6 think-off, qwen3.8 think-off, qwen3.6 think-on, qwen3.8 think-on.

**bbox_contract_anchored_1img** (`geo-*`)

| geometry | sent | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hd` | 1920×1080 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `hd_al32` | 1920×1088 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `hd_al48` | 1920×1104 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `sq320` | 320×320 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 4/4 ⚠ | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `vga` | 800×600 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `portrait` | 1080×1920 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `uhd` | 3072×1728 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `uhd4k` | 3840×2160 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste1` | 1668×733 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste2` | 2812×2135 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste3` | 1235×1181 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste4` | 2750×2379 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste5` | 3030×1549 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste6` | 3011×2317 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |

- **qwen3_6_35b-a3b-q4_K_M/false** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)
- **qwen3_6_35b-a3b-q4_K_M/on** — 14 geometries: 13 convert 6/6, 0 rejected by C7, **1 silent failures** (C7 passed, anchor does not convert)
- **qwen3_8_27b-q4_K_M/false** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)
- **qwen3_8_27b-q4_K_M/on** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)


## 4. Findings

**C17 is settled, and the SPEC's earlier wording was wrong.** Frames *smaller*
than the input are not absent, they are common: 8 of 13 qwen3.6 cells and 6 of 13
qwen3.8 cells report a frame below 1.0×. The full ranges span BOTH directions —
0.46×–2.50× (qwen3.6) and 0.67×–1.44× (qwen3.8); quoting only the sub-1.0 subset
as if it were the range, as an earlier revision of this doc did, understates the
spread and hides that the same model both upscales and downscales. The claim that every observed anchor reported a
frame larger than the input was an artefact of only ever sending 1920×1080.

**The reported frame is not a smooth function of input size.** qwen3.8 returns
`2560×1440` for four different inputs (`hd`, `uhd`, `uhd4k`, `paste5`) and
`2337×1754` for two more. It snaps to canonical sizes rather than scaling, which
no "resize by a ratio" model predicts — and it is why a caller cannot infer the
frame and must read it from the anchor.

**C7's silent-failure rate is geometry-dependent.** ADR 0027's amendment records
one silent failure in 107 anchored cells. Under the `real` pin, qwen3.6 gives
**four in fourteen** — `vga`, `paste2`, `paste4`, `paste6`, all passing
`self_check` while converting 2–3 of 6. The mechanism is the known one (a
fabricated frame whose aspect matches the objects' extent defeats both checks);
geometry changes how often it fires. The 1-in-107 figure describes 16:9 HD.

**qwen3.6 at `hd` reports exactly `[1920, 1080]`** — the true size, answered
from knowledge rather than measured — and converts 1/6. That is the textbook C7
semantic-answer failure the SPEC documents for gemma4 MLX, now observed on
qwen3.6, and C7 correctly rejected it.

**The anchor's value inverts between models.** qwen3.8: anchor 6/6 everywhere
while best-fit reaches 1/6 in five geometries — the anchor is essential.
qwen3.6: anchor 6/6 once, best-fit 6/6 at all fourteen — the anchor is the worse
path. "The anchor rescues the response" was a qwen3.8 finding generalised too
far. C9 still stands: a caller with no ground truth cannot tell which model it
holds, so best-fit remains a diagnostic.

**Both models decline the `real` pin at `paste3`** (1235×1181, near-square) and
answer norm-1000 instead — and both then convert 6/6. Unexplained, and the only
geometry where either model overrides an explicit pin.

**`sq320` splits the models.** qwen3.8 answers norm-1000 and converts 6/6;
qwen3.6 claims `real/[800, 800]`, a 2.5× upscale of a 320×320 input, and
converts 1/6. The 2.5× is consistent with the fixed `--image-min-tokens 1024`
floor upscaling a below-floor image.

## 4b. gemma4, both families — added 2026-08-19

`gemma4:31b-it-q4_K_M` and `gemma4:26b-a4b-it-q4_K_M` (GGUF Q4_K_M), CUDA
Blackwell, eviction-cold per [ADR 0031](adr/0031-model-residency-is-managed-client-side-on-remote-hosts.md),
norm-1000 pin with named coordinates. **56 of 56 cells convert 6/6**, both think
modes, zero C7 rejections, zero silent failures.

Columns: 26b-a4b think-off, 26b-a4b think-on, 31b think-off, 31b think-on.

**bbox_contract_anchored_1img** (`g4-*`)

| geometry | sent | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf | frame | ratio | chk anc/bf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hd` | 1920×1080 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `hd_al32` | 1920×1088 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `hd_al48` | 1920×1104 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `sq320` | 320×320 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `vga` | 800×600 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `portrait` | 1080×1920 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `uhd` | 3072×1728 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `uhd4k` | 3840×2160 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste1` | 1668×733 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste2` | 2812×2135 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste3` | 1235×1181 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste4` | 2750×2379 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste5` | 3030×1549 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |
| `paste6` | 3011×2317 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 | *norm1000* | — | ✅ 6/6 |

- **gemma4_26b-a4b-it-q4_K_M/false** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)
- **gemma4_26b-a4b-it-q4_K_M/on** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)
- **gemma4_31b-it-q4_K_M/false** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)
- **gemma4_31b-it-q4_K_M/on** — 14 geometries: 14 convert 6/6, 0 rejected by C7, **0 silent failures** (C7 passed, anchor does not convert)

**The 26b MoE row is the load-bearing one.** Every measured `yxyx`-while-declaring-
`xyxy` flip in this corpus came from a gemma4 26b variant. Asked for named
`x1`/`y1`/`x2`/`y2`, it declared `norm1000`/`xyxy` and honoured it in **14/14**
cells in **both** think modes. That is C2 doing its job, and it does not
generalise to `box_2d` — no arm here requested a positional array.

Combined with §2–3, the pin now stands at **111 of 112 cells across four models**.

## 5. Limits

n=1 per cell — enough for the norm-1000 result (55 of 56 identical) and for the
qwen3.6/qwen3.8 split (14 cells each, consistent in direction), not enough to
quote a silent-failure *rate* precisely. Two qwen models on one platform; no
gemma4, so P1's budget-fill explanation is untested. Think-on is measured for
the norm-1000 arm only; the frame arm is think-off. One cell (`hd` think-on) was lost to a `NameError` when the suite module was
edited while the sweep was running against it; it was re-run separately and its
result is included above — the sweep is complete at 56/56 cells.
