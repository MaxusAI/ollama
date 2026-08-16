#include "ggml/src/ggml-cuda/mmq.cuh"
#include <cstdio>
#include <cstdlib>
void ggml_cuda_error(const char*,const char*,const char*,int,const char*){abort();}
int ggml_cuda_get_device(){return 0;}
extern "C" void ggml_abort(const char*,int,const char*,...){abort();}
int main() {
    const ggml_type types[] = {GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,GGML_TYPE_Q2_K,GGML_TYPE_IQ4_NL};
    int nz = 0, tot = 0, nvidia_bad = 0, amd_bad = 0;
    // real NVIDIA device-reported ccs: sm_50..sm_121
    for (int cc = 500; cc <= 1300; cc += 10)
        for (ggml_type t : types) for (int fb = 0; fb < 2; ++fb) {
            ++tot; if (ggml_cuda_mmq_get_J_max(t, fb, cc, 1) == 0) { ++nz; ++nvidia_bad; }
        }
    // real AMD device-reported ccs: OFFSET_AMD + gfx number
    for (int gfx = 900; gfx <= 1300; gfx += 1)
        for (ggml_type t : types) for (int fb = 0; fb < 2; ++fb) {
            const int cc = GGML_CUDA_CC_OFFSET_AMD + gfx;
            ++tot; if (ggml_cuda_mmq_get_J_max(t, fb, cc, 1) == 0) { ++nz; ++amd_bad; }
        }
    printf("  swept %d (cc x type x fallback) at ne11 = 1\n", tot);
    printf("  returned ZERO padding: %d  (nvidia %d, amd %d)\n", nz, nvidia_bad, amd_bad);
    printf("  any cc that gives non-zero padding at ne11=1? %s\n", nz == tot ? "NO - none" : "YES");
    return 0;
}
