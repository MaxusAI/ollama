# OCRBench on ROCm/GGUF: gemma4:31b across q4_K_M, q8_0 and bf16

> Part of the fork's OCRBench set — shared method, format and cross-host results in
> [ocrbench.md](ocrbench.md).

**Date:** 2026-09-19 · **Host:** gfx1151 (Strix Halo, 96 GiB GPU carve-out, 31 GiB system RAM)
**Build:** `0.34.1-dynres-16649e8c` — 0.34.1 + `906-revert-hip-integrated-flag` + `OLLAMA_IGPU_DIRECT_IO` + ADR 0036

Run to answer two questions at once:

1. **Cross-backend.** MLX-Metal reported OCRBench `correct: 175, accuracy: 0.875` on
   `gemma4:31b-nvfp4` with the post-#3912 kernel (200 scored, 0 errors, 0 empty). Where does
   the ROCm GGUF path land on the same slice?
2. **The unquantized control.** [ADR 0024](adr/0024-locate-faults-before-fixing-them.md) and the
   2026-09-18 learnings entry ("a tier move is not evidence about a kernel until the unquantized
   arm is measured") say a quantized score alone cannot tell a model's answer from a
   quantization artefact. The GGUF path had never had that control measured. bf16 supplies it.

## Result

Tables emitted by `vision-suite/summarize_extbench.py` (SPEC `vision-harness-reuse.md` H7) —
pasted verbatim, not retyped:

```
| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `gemma4:31b-it-q8_0` | 200 | 0 | 0 | 169 | **0.845** | false | generate |
| `gemma4:31b-it-bf16` | 200 | 0 | 0 | 169 | **0.845** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

host: pre-H11 run (not recorded) · build: pre-H11 run (not recorded)

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| ocr1np_q4_k_m vs ocr1np_q8_0 | 168 | 28 | 3 | 1 | 0.625 |
| ocr1np_q4_k_m vs ocr1np_bf16 | 168 | 28 | 3 | 1 | 0.625 |
| ocr1np_q8_0 vs ocr1np_bf16 | 169 | 31 | 0 | 0 | 1.000 |
```

## What the paired columns say that the accuracies cannot

**`q8_0` and `bf16` are identical on all 200 items.** Not "close" — zero discordant pairs. Every
item either both got right or both got wrong. On this benchmark, 8-bit quantization of
`gemma4:31b` costs nothing at all, and `q8_0` is a usable stand-in for the unquantized arm.

**`q4_K_M` differs from both by 3 items won and 1 lost, exact two-sided p = 0.625.** That is a
coin-flip disagreement, not a quality gap. The 0.855 / 0.845 split in the accuracy column is two
items wide and would be read as a real ordering by anyone comparing the aggregates alone.

This is the measurement the accuracy column *cannot* make. At n = 200 the 95% interval on 0.855
is about ±0.049, so every arm here — and MLX-Metal's 0.875 — falls inside every other arm's
interval. Only the shared row set makes the comparison sharp, and only for arms we ran ourselves.

## Against MLX-Metal

| backend | build | quant | correct | accuracy |
|---|---|---|---|---|
| MLX-Metal | 0.34.0, post-#3912 kernel | nvfp4 | 175/200 | 0.875 |
| ROCm GGUF | `0.34.1-dynres-16649e8c` | q4_K_M | 171/200 | 0.855 |
| ROCm GGUF | `0.34.1-dynres-16649e8c` | q8_0 | 169/200 | 0.845 |
| ROCm GGUF | `0.34.1-dynres-16649e8c` | bf16 | 169/200 | 0.845 |

**This table is not a ranking and must not be published as one.** A 4–6 item difference at
n = 200 is inside noise, and we cannot do better without MLX-Metal's per-item `ext_*_ocrbench.json`:
McNemar needs the discordant items, and an aggregate cannot supply them. The honest reading is
that the two backends are indistinguishable on this slice at this sample size.

Worth noting for its own sake: the ROCm ladder is *flat* while the MLX ladder was not. On Metal,
`31b-nvfp4` and `31b-mlx-bf16` differed by a 9 px fine-text tier once the kernel was fixed. Here
`bf16` and `q8_0` agree item for item and `q4_K_M` is within noise of both. Different
quantization schemes, different vision towers, different kernels — but it means the GGUF path has
no quantization artefact of the kind that misled the Metal investigation.

## Placement, and one confound

Every arm: `-c 16384 -np 1`, `offloaded 61/61 layers`, think off, `/api/generate`, median
`prompt_eval_count` **1113** in all three — so the image pipeline delivered identical token counts
and the budget-fill path is not a variable here.

The generation batch was **not** identical, and this is a confound between `q4_K_M` and the other
two:

| arm | `-b` / `-ub` | median s/req | wall |
|---|---|---|---|
| q4_K_M | 1024 | 7.7 | 26 min |
| q8_0 | 512 | 8.1 | 27 min |
| bf16 | 512 | 8.1 | 28 min |

So `q4_K_M`'s 3-vs-1 advantage could be quantization **or** batch size; these data cannot separate
them. `q8_0` vs `bf16` is clean — same batch, same everything, zero discordant items.

## Why the batch differs, and why ADR 0036 never fires here

All three arms logged ADR 0036's own diagnostic on every load:

```
msg="generation batch below the image chunk, images decode in pieces" num_batch=1024 image_chunk_batch=2048
```

The 2048 floor is requested and denied at every quantization. The cause is
`availableMemoryForLoad` (`server/sched.go`), which for an integrated GPU deliberately prefers the
live system-memory figure over the device baseline:

```go
if systemInfo.FreeMemory > 0 && sharedGPUFree > 0 && systemInfo.FreeMemory < sharedGPUFree {
    return discreteGPUFree + systemInfo.FreeMemory, gpuFree, true
}
```

On this host the scheduler logs `gpu memory … available="95.4 GiB"` and `system memory total="31.0 GiB"`,
so `availableMemory` resolves to **31 GiB**, not 95.4. Every rung decision then follows from 31 GiB:

| arm | predicted | 2048 needs ≤ 60% (18.6 GiB) | 1024 needs ≤ 75% (23.3 GiB) | result |
|---|---|---|---|---|
| q4_K_M | ~19 GiB | no | yes | 1024 |
| q8_0 | ~33 GiB | no | no (over the 80% threshold) | 512 |
| bf16 | ~58 GiB | no | no | 512 |

ADR 0036 already anticipates the step-down ("on a card without that headroom the batch steps down
and the split returns"). What is new is that **gfx1151 is not such a card** — it has 95.4 GiB free
and is being sized against 31 GiB of system RAM. The heuristic's comment justifies itself on the
grounds that iGPU free memory "can be a static or slowly refreshed device baseline"; on Strix Halo
with a fixed carve-out that assumption does not hold, and the cost is that the fork's own flagship
vision optimization never engages on the fork's own production hardware.

Two consequences:

- **ADR 0036 is inert on gfx1151 at every quantization**, `-np 1` and `-np 2` alike (512 at
  `-np 2`, 1024 at `-np 1` for q4). Its measured 1.5–1.9× prefill gain is not available here.
- **Compat 906 stays load-bearing for gemma4.** The image chunk is 1113 tokens against `n_ubatch`
  of 1024 or 512, so the decode still exceeds `n_ubatch` — the trigger condition for the HIP MMQ
  race. ADR 0036 does not incidentally mask the defect, which was a live worry when #320 landed.

## What the misses look like

29 wrong at q4_K_M, all near-misses on scene text and handwriting — the shape of a model reading a
blurry crop, not of a corrupted decode:

```
THREADS  → THREAD        CORNERS → CORNER S      GOOD → GOOOD
Scottynn → Scottlynn     CORONAD → CORONA        both → booth
```

For contrast, the HIP defect produced whole-image failures and `0.000` scene IoU, not off-by-one
transcriptions. Nothing in these records resembles it.

## Provenance caveat

The three score files carry **no `host` / `server_version`**: `extbench.py` did not persist them
until the change that accompanies this document, so the footer renders
`pre-H11 run (not recorded)` and the generator refuses to let a recorded file vouch for an
unrecorded one. The build in the heading is attested from the runner logs captured in the run log,
which is **inferred provenance, not recorded provenance** ([ADR 0024](adr/0024-locate-faults-before-fixing-them.md)).
Re-running the three arms purely to stamp them would cost ~5 hours of GPU and ~90 GiB of pulls;
future extbench runs record it for free.

## Reproducing

```sh
cd docs/maxusai/vision-suite
LIMIT=200 OFFSET=0 THINK=false \
  python3 extbench.py http://127.0.0.1:11499 ocr1np_q8_0 gemma4:31b-it-q8_0 ocrbench
python3 summarize_extbench.py ocrbench ocr1np_q4_k_m ocr1np_q8_0 ocr1np_bf16 --paired
```

The container must run `OLLAMA_NUM_PARALLEL=1` for the placement above; at `-np 2` the effective
context doubles to 32768 and q4's batch drops from 1024 to 512.
