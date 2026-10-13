# TASK: why is a muse-glimmer suite cell 1.6–3× slower than gemma4:31b's? Explain each part, with sources

Opened by the Metal host (`macbook-pro-m5-max-128GB/mlx-metal`) on 2026-10-13, for another agent to pick up.
Status: **open, unclaimed.** Context: #463 (onboarding).

## What was seen

The vision suite, thinking off, 27 tests per cell, `num_ctx` 16384, greedy. Per request, averaged from the score
files' `prompt_eval_*` / `eval_*`:
- glimmer: #464's build `0.40.2-dynres-1-g4cfad4a`;
- gemma4: the 0.40.2 fold build.

| model | prompt tok | prefill min | prefill tok/s | gen tok | decode min | decode tok/s | answer chars | cell min |
|---|---|---|---|---|---|---|---|---|
| muse-glimmer_30b-mlx | 4691 | 2.6 | 800 | 815 | 18.9 | 19.4 | 1089 | 21.7 |
| muse-glimmer_30b-mxfp8-dflash | 4691 | 2.8 | 752 | 890 | 38.3 | 10.5 | 1120 | 41.3 |
| muse-glimmer_30b-mxfp8 | 4691 | 2.9 | 731 | 890 | 39.0 | 10.3 | 1120 | 42.1 |
| muse-glimmer_30b-nvfp4 | 4691 | 2.5 | 849 | 815 | 17.7 | 20.7 | 1089 | 20.4 |
| muse-glimmer_30b-q4_K_M-dflash | 4691 | 5.2 | 405 | 853 | 15.8 | 24.3 | 1096 | 21.1 |
| muse-glimmer_30b-q8_0-dflash | 4691 | 5.2 | 409 | 884 | 18.1 | 21.9 | 1133 | 23.5 |
| muse-glimmer_30b-q8_0 | 4691 | 5.6 | 375 | 884 | 39.5 | 10.1 | 1133 | 45.3 |
| muse-glimmer_30b | 4691 | 6.0 | 350 | 837 | 28.6 | 13.2 | 1087 | 34.7 |
| gemma4_31b-it-q4_K_M | 2214 | 2.0 | 490 | 455 | 13.6 | 15.1 | 953 | 15.7 |
| gemma4_31b-nvfp4 | 2214 | 1.7 | 578 | 463 | 10.9 | 19.1 | 959 | 12.7 |

What the numbers already say. Each point is a hypothesis to confirm and explain:
1. **4-bit decode speed is not the gap.** Glimmer's nvfp4 decodes at 20.7 tok/s, gemma4's at 19.1.
2. **Glimmer generates ~1.8× the tokens for ~1.15× the answer characters**: 815–890 against ~460 tokens. Find where
   the extra ~350 tokens per request go. Candidates:
   - hidden tokens with `think: false` (the onboarding measured ~120 on a short prompt);
   - the message-header protocol (`<|start|>assistant to=user<|message|>`);
   - whitespace padding under the JSON grammar (padded to the bound on MLX; see #470);
   - a tokenizer that spends more tokens per character.
3. **Glimmer's prompts are ~2.1× longer**: 4691 against 2214 tokens. Glimmer images go up to 4096 tokens at native
   resolution; gemma4 caps at 1120.
4. **The 8-bit tags decode at ~10 tok/s, half the 4-bit tags: memory bandwidth, nearly settled.** The weights are
   ~1.9× the size and decode is ~2.0× slower. Weights × tok/s is near constant on MLX:
   - `nvfp4`: 17.2 GB × 20.7 ≈ 356 GB/s;
   - `mxfp8`: 32.6 GB × 10.3 ≈ 336 GB/s;
   - GGUF `q8_0`: 31 GB × 10.1 ≈ 313 GB/s.

   These are upper bounds, since the sizes include the vision encoder, which decode does not read. A constant figure is
   the signature of a bandwidth-bound dense model. Left to confirm:
   - **Dense or MoE:** confirm glimmer is dense, as this implies.
   - **The outlier:** GGUF `Q4_K_M` (18.1 GB × 13.2 ≈ 239 GB/s) falls short of the bound. That is likely K-quant
     dequantization cost on Metal, and its drafter hides it (`-dflash`: 24.3 tok/s).
5. **GGUF prefill on glimmer is about half MLX's** (350–409 against 731–849 tok/s). gemma4 is 490 on GGUF against 578
   on MLX.

## What to do

1. **Search online** for each point: the model card (ollama.com/library/muse-glimmer and the original release), its
   architecture (dense or MoE, layers, sliding window, vision encoder size and sparse attention), and llama.cpp and
   MLX issues or PRs on muse-glimmer performance. Is any of this documented or known? Cite each source.
2. **Count the hidden tokens.** On one cell, compare `eval_count` against the tokenized answer. Use a raw,
   unparsed decode (`/api/generate` `raw`) to see what the extra tokens are.
3. **Measure decode against the bandwidth bound**: weights in bytes per token against the machine's memory
   bandwidth, for the q8_0, mxfp8 and nvfp4 tags.
4. If a fork-side cause turns up (for example padding the grammar allows), open a fix PR. If it is the model, say so
   in this record with sources.

Rules: GPU work under the host's gate (production idle 15 min, powermode 2), on :11437 against an APFS store clone.
Never touch :11435. Pin `NUM_BATCH` for GGUF A/Bs.

## Done when

Each of the five points is confirmed or refuted, with a measurement or a cited source, in this record.
