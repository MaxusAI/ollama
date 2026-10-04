// GPU-free test: the MMQ MUL_MAT_ID src1 tail padding must cover the tile the kernel launches.
// Material for llama.cpp#27044 against #29941; see docs/maxusai/upstream-mmq-submission-material.md.
//
// In the ids path, the last expert's last tile loads J blocks of block_q8_1_mmq from its first row
// (mmq.cuh: the src1 tile loads, `for (l0 < J*MMQ_TILE_Y_K)`, have no bound). If that tile holds r rows, it reads
// J - r blocks past the data in the last k-plane, so the padding must be at least J - 1 blocks for every routing.
// J is the tile width mul_mat_q_switch_J launches. Its selection loop is replicated below from mmq.cuh, because
// it sits in a template that queries the device; everything else is llama.cpp's own code.
//
// Three padding rules, the argument passed to ggml_cuda_mmq_get_J_max() in ggml_cuda_mul_mat_q()'s ids branch:
//   master  : ne11               (1 for broadcast gate/up, n_expert_used otherwise)
//   #29941  : ne12               (tokens in the batch)
//   #27044  : ne12*n_expert_used (rows the buffer actually holds)
//
// Build from the root of a llama.cpp checkout, with the CUDA toolkit's nvcc. It never touches a GPU:
//   nvcc -std=c++17 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda mmq-ids-padding-test.cu -o t && ./t
// The exit code is non-zero only if #27044's rule leaves a shape short.
//
// On llama.cpp master 05043961 (2026-10-04), it checked 2,438,400 shapes in 88 s:
//   master (ne11)                  FAIL: padding < J-1 in 2437760 shapes; worst 127 blocks short
//   #29941 (ne12)                  FAIL: padding < J-1 in 267960 shapes; worst 63 blocks short
//                                  (sm_75, q4_0, fallback=1, ne12=65, n_expert_used=2: J=128, padding=64)
//   #27044 (ne12*n_expert_used)    pass
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"   // MMVQ_MAX_BATCH_SIZE: below it MUL_MAT_ID takes MMVQ, not MMQ
#include <climits>
#include <cstdio>
#include <cstdlib>

void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
int ggml_cuda_get_device() { return 0; }  // never reached: nothing below queries a device
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }

// mul_mat_q_switch_J's choice, for NVIDIA (ncols_opt = ne12). Returns the launched tile width, 0 if none.
static int launched_J(ggml_type t, bool fb, int cc, size_t smpbo, int64_t ncols_opt) {
    int nt_best = INT_MAX, J_launch = 0;
    for (int J = 8; J <= 128 && nt_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, cc, GGML_PREC_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int nt = (ncols_opt + c.J - 1) / c.J;
        if (nt < nt_best) { nt_best = nt; J_launch = c.J; }
    }
    return J_launch;
}

int main() {
    // smpbo = max shared memory per block (opt-in), per architecture. sm_120 was read from an RTX PRO 6000 Blackwell.
    struct arch { const char * name; int cc; size_t smpbo; };
    const arch archs[] = {
        {"sm_75",  GGML_CUDA_CC_TURING,        65536},
        {"sm_86",  860,                       101376},
        {"sm_89",  GGML_CUDA_CC_ADA_LOVELACE, 101376},
        {"sm_120", GGML_CUDA_CC_BLACKWELL,    101376},
    };
    const ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                               GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    const char * type_name[] = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"};
    const char * rule_name[3] = {"master (ne11)", "#29941 (ne12)", "#27044 (ne12*n_expert_used)"};
    long checked = 0, short_n[3] = {0, 0, 0}; int worst[3] = {0, 0, 0};
    const char * ex_arch[3] = {"", "", ""}; int ex_t[3] = {}; int ex_fb[3] = {}, ex_ne12[3] = {}, ex_u[3] = {}, ex_J[3] = {}, ex_pad[3] = {};

    for (const arch & a : archs) for (int ti = 0; ti < 10; ++ti) for (int fb = 0; fb < 2; ++fb)
    for (int64_t ne12 = MMVQ_MAX_BATCH_SIZE + 1; ne12 <= 1024; ++ne12) {   // ne12 <= 8 goes to MMVQ, not MMQ
        const ggml_type t = types[ti];
        const int J = launched_J(t, fb, a.cc, a.smpbo, ne12);
        if (J == 0) continue;
        for (int n_used = 2; n_used <= 16; ++n_used) for (int bcast = 0; bcast < 2; ++bcast) {
            const int64_t arg[3] = { bcast ? 1 : n_used, ne12, ne12*n_used };
            ++checked;
            for (int r = 0; r < 3; ++r) {
                const int pad = ggml_cuda_mmq_get_J_max(t, fb, a.cc, arg[r]);
                const int shortfall = (J - 1) - pad;   // blocks the last tile can read past the allocation
                if (shortfall > 0) {
                    ++short_n[r];
                    if (shortfall > worst[r]) { worst[r] = shortfall; ex_arch[r] = a.name; ex_t[r] = ti; ex_fb[r] = fb;
                                                ex_ne12[r] = ne12; ex_u[r] = n_used; ex_J[r] = J; ex_pad[r] = pad; }
                }
            }
        }
    }
    printf("checked %ld shapes: 4 archs x 10 types x fallback 0/1 x ne12 9..1024 x n_expert_used 2..16 x broadcast 0/1\n", checked);
    for (int r = 0; r < 3; ++r) {
        printf("%-30s %s", rule_name[r], short_n[r] ? "FAIL" : "pass");
        if (short_n[r]) printf(": padding < J-1 in %ld shapes; worst %d blocks short (%s, %s, fallback=%d, ne12=%d, n_expert_used=%d: J=%d, padding=%d)",
                               short_n[r], worst[r], ex_arch[r], type_name[ex_t[r]], ex_fb[r], ex_ne12[r], ex_u[r], ex_J[r], ex_pad[r]);
        printf("\n");
    }
    return short_n[2] ? 1 : 0;   // the rule under test is #27044's
}
