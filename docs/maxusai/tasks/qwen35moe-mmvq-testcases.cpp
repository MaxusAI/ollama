// test-backend-ops cases for the mmvq.cu channel_y out-of-bounds read.
//
// SUPERSEDES an earlier draft that used n = 2040 (the image's token count). That
// was wrong: the crashing calls do NOT carry the image ubatch. Instrumentation on
// a live qwen3.6:35b-a3b run shows ggml_cuda_mul_mat_vec_q is reached with small
// column counts (n = 2, 4, 7), because routing there requires ne2 <= MMVQ_MAX_BATCH_SIZE (8).
// The image matters only in that it puts the server into the state where these
// calls happen; the faulting shapes themselves are tiny, which is what makes a
// standalone repro cheap.
//
// ---------------------------------------------------------------------------
// LIVE SHAPES, captured from the instrumented build (patch 900), qwen3.6:35b-a3b-q4_K_M,
// num_ctx 40960, one 1920x1080 image. 480 MUL_MAT_ID calls, two distinct kinds:
//
//   VULNERABLE (ffn_up / ffn_gate, 2048 -> 512):
//     type_a=12 (Q4_K)  ne0=512  ne1=8  ne2={2,4,7}  ne10=2048  ne11=1  ne02=256
//                                                              ^^^^^^  y is BROADCAST
//   SAFE (ffn_down, 512 -> 2048):
//     type_a=14 (Q6_K)  ne0=2048 ne1=8  ne2={2,4,7}  ne10=512   ne11=8  ne02=256
//                                                              ^^^^^^  y has one channel per expert
//
// The bug is in mmvq.cu:517
//     channel_y = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
// where for MUL_MAT_ID:
//     ncols_dst     = ne2   (2, 4 or 7 here)
//     nchannels_dst = ne1   (8 -> grid blockIdx.y spans 0..7)
//     nchannels_y   = ne11  (1 on the vulnerable call)
// With ncols_dst != 1 the modulo is skipped, so channel_y = blockIdx.y = 0..7 indexes
// an 8-channel stride into a y tensor holding ONE channel: a read up to 7 strides past
// the end of src1_q8_1. It is an over-READ, so it only faults when it crosses into
// unmapped memory — which is why it presents as non-deterministic and why num_ctx /
// num_batch (both of which change pool block sizes) appeared to be the trigger.
//
// TRIGGER CONDITION:  ids != nullptr && ncols_dst > 1 && nchannels_y < nchannels_dst
//
// ---------------------------------------------------------------------------
// MAPPING to test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k):
//     as  = (k, m, n_mats)          -> ne00=k    ne01=m    ne02=n_mats
//     b   = (k, b?1:n_used, n)      -> ne10=k    ne11=1 when broadcast   ne12=n
//     ids = (n_used, n)
//     out = (m, n_used, n)          -> ne0=m     ne1=n_used   ne2=n
// so the live vulnerable call is exactly:
//     n_mats=256  n_used=8  b=true  m=512  n={2,4,7}  k=2048
//
// EXPECTATIONS with the current code:
//   n = 1        PASSES — ncols_dst == 1, so the guard applies and the modulo runs
//   n = 2,4,7    FAILS  — the live crash shapes
//   n = 8        FAILS  — still <= MMVQ_MAX_BATCH_SIZE
//   n = 9+       PASSES — ne2 > MMVQ_MAX_BATCH_SIZE routes to mul_mat_q instead
//   b = false    PASSES — ne11 == n_used == nchannels_dst, no broadcast, no over-read
// That n=1 / n=2 pair and the b=true / b=false pair are the whole bug in four lines.

// --- live vulnerable shapes (ffn_up / ffn_gate), q4_K_M as deployed ---
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 2, 2048)); // live, 160 calls observed
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 4, 2048)); // live,  80 calls observed
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 7, 2048)); // live,  80 calls observed

// --- the boundary that isolates the faulty guard ---
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 1, 2048)); // expect PASS (ncols_dst == 1)
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 8, 2048)); // expect FAIL (still <= MMVQ_MAX)
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 9, 2048)); // expect PASS (routes to mul_mat_q)

// --- broadcast control: same shape, y given one channel per expert ---
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, false, 512, 4, 2048)); // expect PASS (ne11 == 8)

// --- live SAFE counterpart (ffn_down) — should pass, and confirms the suite is not
//     simply failing everything MoE ---
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q6_K, GGML_TYPE_F32, 256, 8, false, 2048, 4, 512)); // live, 80 calls observed

// --- CORRECTED: F16 does NOT reach mmvq at all on NVIDIA. ggml_cuda_mul_mat_id
//     only calls mul_mat_vec_q when ggml_is_quantized(src0->type); the non-quantised
//     branch calls mul_mat_vec_f only on AMD, so on CUDA an F16 src0 falls through to
//     mul_mat_q / mul_mat_f. This case therefore exercises a DIFFERENT path and is
//     expected to PASS — it is a routing control, not a quantisation control.
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16,  GGML_TYPE_F32, 256, 8, true,  512, 4, 2048)); // expect PASS (not mmvq)

// --- quantisation spread INSIDE the vulnerable path (all quantised => all reach mmvq) ---
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 256, 8, true,  512, 4, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 256, 8, true,  512, 4, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q6_K, GGML_TYPE_F32, 256, 8, true,  512, 4, 2048));

// --- nchannels_dst sweep: the over-read distance is (nchannels_dst - 1) strides,
//     so n_used = 1 must be clean even with b=true ---
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 1, true,  512, 4, 2048)); // expect PASS (no over-read)
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 2, true,  512, 4, 2048)); // 1 stride over
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32,  32, 8, true,  512, 4, 2048)); // fewer experts, same over-read

// ---------------------------------------------------------------------------
// PASSING CONFIGURATIONS — and why "passing" is not the same as "correct"
// ---------------------------------------------------------------------------
//
// At the SERVER level these configurations do not crash (all verified on
// qwen3.6:35b-a3b-q4_K_M, RTX PRO 6000 Blackwell, payload f8def7fe1):
//
//   num_ctx = 32768, any num_batch            -> no crash   (32769 crashes)
//   num_ctx = 40960, num_batch = 256..2039    -> no crash   (2040 crashes)
//   num_ctx = 40960, text-only prompt 4950 tok-> no crash    (no image => no MoE
//                                                            vision decode path)
//   num_ctx = 40960, gemma4 (dense arch)      -> no crash    (no MUL_MAT_ID at all)
//
// NONE of these can be expressed as test-backend-ops cases: the harness tests a
// single op in isolation and has no num_ctx, no num_batch and no KV cache. Those
// knobs do not change the op's SHAPES — they change the CUDA pool's block sizes,
// and therefore only whether the out-of-bounds read crosses into unmapped memory.
//
// That is the uncomfortable part. The faulty index is computed the same way in
// every one of the "passing" configurations above:
//
//     channel_y = channel_dst          // 0..nchannels_dst-1 = 0..7
//     nchannels_y = 1                  // y has ONE channel
//
// so the kernel still reads up to 7 channel-strides past the end of src1_q8_1. It
// simply lands inside memory the pool already owns, faults nothing, and feeds
// whatever it found into the expert matmul. A "passing" run is therefore a run
// whose MoE activations may be silently wrong, not a run where the bug is absent.
//
// test-backend-ops is exactly the instrument for that question, because it does
// not merely check for crashes: it compares against a reference backend with
// nmse() and max_nmse_err() (tbo.cpp:271, :1145). So on the vulnerable shapes
// expect one of two outcomes, BOTH of which are the bug:
//
//     - an illegal memory access (the over-read faults), or
//     - a PASS/FAIL on accuracy with a non-trivial NMSE (the over-read is
//       tolerated by the allocator but the numbers are wrong)
//
// The accuracy signal is the more valuable of the two for an upstream report,
// since it demonstrates silent corruption rather than a mere robustness issue,
// and it is deterministic where the crash is not.
//
// Suggested run:
//     ./build/bin/test-backend-ops -o MUL_MAT_ID
// and compare the vulnerable rows (b=true, n>1) against their controls
// (b=true n=1, b=false, and the Q6_K ffn_down shape) in the same output.
