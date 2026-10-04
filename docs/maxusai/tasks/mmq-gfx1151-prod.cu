// The MUL_MAT_ID shapes the ROCm host's production MoE GGUFs produce on gfx1151, under three forms of compat 903
// (main's #27044 rule, the widest tile, the widest padded tile), with the dispatch cut from the checkout by
// mmq-extract-dispatch.sh into dispatch.inc. The (type, ne01, n_used, n_exp) tuples below were read from the GGUF
// headers of the models production serves on 2026-10-05. Build like mmq-gfx1151-reach.cu.
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }
static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() { return g_info; }
int ggml_cuda_get_device() { return 0; }
#include "dispatch.inc"
static const size_t B = sizeof(block_q8_1_mmq);
static const int CC = GGML_CUDA_CC_OFFSET_AMD + 0x1151;
static const size_t SMPBO = 65536;
struct P { const char * where; ggml_type t; int ne01; int n_used; int n_exp; };
int main() {
    g_info.device_count = 1; g_info.devices[0].cc = CC; g_info.devices[0].smpbo = SMPBO; g_info.devices[0].warp_size = 32;
    const P ps[] = {
        {"qwen3.6:35b-a3b-q4_k_m gate/up", GGML_TYPE_Q4_K,  512, 8, 256}, {"qwen3.6:35b-a3b-q4_k_m down", GGML_TYPE_Q6_K, 2048, 8, 256},
        {"qwen3.6:35b-a3b-q8_0 gate/up",   GGML_TYPE_Q8_0,  512, 8, 256}, {"qwen3.6:35b-a3b-q8_0 down",   GGML_TYPE_Q8_0, 2048, 8, 256},
        {"qwen3-vl:30b-a3b gate/up",       GGML_TYPE_Q4_K,  768, 8, 128}, {"qwen3-vl:30b-a3b down q4_K",  GGML_TYPE_Q4_K, 2048, 8, 128},
        {"qwen3-vl:30b-a3b down q6_K",     GGML_TYPE_Q6_K, 2048, 8, 128},
        {"gemma4:26b-a4b gate_up",         GGML_TYPE_Q4_K, 1408, 8, 128}, {"gemma4:26b-a4b down q5_0",    GGML_TYPE_Q5_0, 2816, 8, 128},
        {"gemma4:26b-a4b down q8_0",       GGML_TYPE_Q8_0, 2816, 8, 128},
        {"nemotron3:33b-q4_K_M up",        GGML_TYPE_Q5_0, 1856, 6, 128}, {"nemotron3:33b-q4_K_M down q5_0", GGML_TYPE_Q5_0, 2688, 6, 128},
        {"nemotron3:33b-q4_K_M down q8_0", GGML_TYPE_Q8_0, 2688, 6, 128},
        {"nemotron3:33b-q8 up",            GGML_TYPE_Q8_0, 1856, 6, 128}, {"nemotron3:33b-q8 down",       GGML_TYPE_Q8_0, 2688, 6, 128},
    };
    printf("%-32s %4s %3s | %-28s | %s\n", "tensor", "fb", "mmvq", "J used (ne12 1..8192)", "src1 bytes past the allocation, worst: main / widest / amended");
    for (const P & p : ps) {
        const bool fb = p.ne01 % 128 != 0;
        const int mmvq = get_mmvq_mmid_max_batch(p.t, CC);
        int J_wide = ggml_cuda_mmq_get_J_max(p.t, fb, CC, 512), pad_amended = 0;
        for (int J = 8; J <= 128; J += 8) {
            ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(p.t, J, fb, CC);
            if (c.type == GGML_TYPE_COUNT) continue;
            const int bl = (int) ((GGML_PAD(c.J*B, c.nthreads*sizeof(int)) + B - 1)/B); if (bl > pad_amended) pad_amended = bl;
        }
        long worst[3] = {LONG_MIN, LONG_MIN, LONG_MIN}; long wne12[3] = {0, 0, 0}; bool usedJ[17] = {};
        for (int64_t ne12 = mmvq + 1; ne12 <= 8192; ++ne12) {
            if (!ggml_cuda_should_use_mmq(p.t, CC, ne12, p.n_exp)) continue;
            const int64_t ncols_opt = (ne12*p.n_used + p.n_exp - 1) / p.n_exp;
            int J = 0, nth = 0, nt_best = INT_MAX;
            for (int Jc = 8; Jc <= 128 && nt_best > 1; Jc += 8) {
                ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(p.t, Jc, fb, CC);
                if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, CC) > SMPBO) continue;
                const int nt = (ncols_opt + c.J - 1) / c.J; if (nt < nt_best) { nt_best = nt; J = c.J; nth = c.nthreads; }
            }
            usedJ[J/8] = true;
            const long T = (long) GGML_PAD(J*B, nth*sizeof(int));
            const int pads[3] = {ggml_cuda_mmq_get_J_max(p.t, fb, CC, ne12*p.n_used), J_wide, pad_amended};
            for (int r = 0; r < 3; ++r) { const long over = T - (long) B - (long) pads[r]*B; if (over > worst[r]) { worst[r] = over; wne12[r] = ne12; } }
        }
        char js[64] = ""; for (int i = 1; i <= 16; ++i) if (usedJ[i]) snprintf(js + strlen(js), sizeof(js) - strlen(js), "%d ", i*8);
        printf("%-32s %4d %4d | %-28s |", p.where, (int) fb, mmvq, js);
        for (int r = 0; r < 3; ++r) printf(" %6ld%s", worst[r], worst[r] > 0 ? "!" : " ");
        printf("   (negative = covered, by that many bytes at the tightest ne12)\n");
    }
    return 0;
}
