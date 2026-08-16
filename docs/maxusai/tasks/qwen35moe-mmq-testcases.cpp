// test-backend-ops cases at the shapes that trigger the MMQ ids-padding
// under-allocation (mmq.cu:205-206). Insert next to the Llama-4-Maverick
// MUL_MAT_ID entries in tests/test-backend-ops.cpp.
//
// Shapes captured live from qwen3.6:35b-a3b-q4_K_M (qwen35moe, 256 experts,
// 8 used) on an RTX PRO 6000 Blackwell, payload f8def7fe1, via a per-node
// synchronised fault probe.
//
//   test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k)
//     as  = (k, m, n_mats)
//     b   = (k, b ? 1 : n_used, n)      <-- b=true  <=>  ne11 == 1, the broadcast case
//     out = (m, n_used, n)
//
// The live fault:
//   op=MUL_MAT_ID name=ffn_moe_gate-3
//   dst [512, 8, 2040, 1]  src0 q4_K [2048, 512, 256, 1]
//   src1 f32 [2048, 1, 2040, 1]  src2 i32 [8, 2040, 1, 1]
//   branch=mmq ne0=512 ne1=8 ne2=2040 ne02=256 ne11=1 ne12=2040 type=q4_K
//
// ---------------------------------------------------------------------------
// IMPORTANT: THESE MAY ALL PASS EVEN WHEN THE BUG IS PRESENT.
//
// The under-allocation causes an out-of-bounds READ into padding rows whose
// results the kernel discards. Output is unaffected (verified byte-identical
// across the fix), so the NMSE comparison in test-backend-ops will not flag it.
// It faults only when the read crosses an unmapped page, which depends on the
// pool's allocation history.
//
// Run these under a memory checker, not as a pass/fail oracle:
//   compute-sanitizer --tool memcheck ./build/bin/test-backend-ops -o MUL_MAT_ID
//
// A clean memcheck report is meaningful. A green test run is not.
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
// WHERE YOU PUT THESE MATTERS — a green run can mean they never executed.
//
// Insert into make_test_cases_eval(). Inserting after the LAST test_mul_mat_id
// line in the file lands them in make_test_cases_perf() instead, which
// `-o MUL_MAT_ID` compiles but never evaluates. The total stays at exactly
// 790/790 and reads as a clean pass while none of these cases ran at all.
// (Hit on ROCm/gfx1151 — see rocm-mmq-ids-padding-result.md §4.)
//
// Check the case count moves before believing any verdict: the total must rise
// by the number of cases added here.
//
// Building llama.cpp standalone from ollama's vendored tree also needs
// -I<ollama>/llama/compat, plus llama-ollama-compat.cpp and
// llama-ollama-compat-util.cpp added to the `llama` target, or the link fails
// on llama_ollama_compat:: symbols.
// ---------------------------------------------------------------------------

// --- the live crashing shapes: MoE gate/up, broadcast activations (ne11 == 1)
// b=true is the whole point: it sets ne11 == 1, which makes get_J_max() return 0
// and the tail padding vanish entirely.
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 2040, 2048)); // LIVE FAULT
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 1032, 2048)); // live, never faulted
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 1947, 2048)); // live, never faulted
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512, 2048, 2048)); // observed OK warm

// --- the live safe counterpart: ffn_down, NOT broadcast (ne11 == n_used == 8)
// Under-allocated too, just less: padding sized for 8 rows instead of n*n_used.
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q6_K, GGML_TYPE_F32, 256, 8, false, 2048, 2040, 512));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q6_K, GGML_TYPE_F32, 256, 8, false, 2048, 1032, 512));

// --- broadcast on/off control at one size: isolates the ne11 == 1 path
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, false, 512, 2040, 2048));

// --- n_used sweep: the flattened row count is n * n_used, so the shortfall
//     between get_J_max(ne11) and get_J_max(n*n_used) grows with n_used.
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 1, true,  512, 2040, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 2, true,  512, 2040, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 4, true,  512, 2040, 2048));

// --- quantisation spread through the same MMQ path
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 256, 8, true,  512, 2040, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 256, 8, true,  512, 2040, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q6_K, GGML_TYPE_F32, 256, 8, true,  512, 2040, 2048));

// --- expert-count sweep: ne02 feeds expert_bounds, not the padding term
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32,  32, 8, true,  512, 2040, 2048));
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 128, 8, true,  512, 2040, 2048));

// --- routing controls: these should NOT reach ggml_cuda_mul_mat_q
// F16 is unquantised, so on NVIDIA it never enters the MMQ path at all.
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16,  GGML_TYPE_F32, 256, 8, true,  512, 2040, 2048));
// Small n stays in mul_mat_vec_q (entry is gated on ne2 <= MMVQ_MAX_BATCH_SIZE == 8).
test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 256, 8, true,  512,    4, 2048));
