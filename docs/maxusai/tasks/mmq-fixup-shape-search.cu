// Find a native-FP4 MUL_MAT_ID shape whose stream-k launch actually runs the fixup kernel -- the only reader of
// y_scale that is not bounded by j_max. launch_mul_mat_q uses ntiles_dst blocks when they fill >= 90% of the
// waves, and then fixup_needed is false; only below 90% does it fall back to nsm blocks and need the fixup.
#include "ggml/src/ggml-cuda/mmq.cuh"
#include <cstdio>
#include <cstdlib>
#include <climits>
void ggml_cuda_error(const char*,const char*,const char*,int,const char*){abort();}
int ggml_cuda_get_device(){return 0;}
extern "C" void ggml_abort(const char*,int,const char*,...){abort();}
int main(int argc, char ** argv) {
    const int cc = GGML_CUDA_CC_BLACKWELL, nsm = argc > 1 ? atoi(argv[1]) : 188;
    const size_t smpbo = 101376;
    printf("nsm=%d  looking for: fixup_needed, mean rows/expert <= 2, J >= 16\n", nsm);
    printf("%-6s %-3s %6s %7s %7s %5s %5s %6s %7s %5s %6s\n",
           "type", "fb", "ne02", "n_used", "ne12", "m", "J", "I", "ntiles", "eff%", "rows");
    int found = 0;
    for (int fb = 0; fb < 2; ++fb)
    for (int ne02 = 8; ne02 <= 256 && found < 25; ne02 *= 2)
    for (int n_used = 1; n_used <= 8; ++n_used)
    for (int ne12 = 9; ne12 <= 64; ++ne12)
    for (int m = 128; m <= 1024; m += 64) {
        if (n_used > ne02) continue;
        if ((m % 128 != 0) != (bool) fb) continue;      // fallback is ne01 % 128 != 0
        // the launched tile, as mul_mat_q_switch_J picks it
        int J = 0, nt_best = INT_MAX, bI = 0; bool bsk = false;
        for (int Jc = 8; Jc <= 128 && nt_best > 1; Jc += 8) {
            ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, Jc, fb, cc, GGML_PREC_Q4);
            if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
            const int nt = (ne12 + c.J - 1) / c.J;
            if (nt < nt_best) { nt_best = nt; J = c.J; bI = c.I; bsk = c.stream_k; }
        }
        if (J < 16 || !bsk) continue;
        const int ntx = (ne12 + J - 1) / J;
        const int nty = (m + bI - 1) / bI;
        const int ntiles = ntx * nty * ne02;
        const int waves = (ntiles + nsm - 1) / nsm;
        const int eff = 100 * ntiles / (nsm * waves);
        const int blocks = eff >= 90 ? ntiles : nsm;
        if (ntiles % blocks == 0) continue;             // fixup not needed
        const double rows = (double) ne12 * n_used / ne02;
        if (rows > 2.0) continue;                        // need ~1 row in the last non-empty expert
        printf("%-6s %-3d %6d %7d %7d %5d %5d %6d %7d %5d %6.2f\n",
               "nvfp4", fb, ne02, n_used, ne12, m, J, bI, ntiles, eff, rows);
        ++found;
    }
    if (!found) printf("  none\n");
    return 0;
}
