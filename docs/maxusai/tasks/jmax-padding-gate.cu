#include "ggml/src/ggml-cuda/mmq.cuh"
#include <cstdio>
#include <cstdlib>

// stubs: the header drags these in, the gate never calls them
void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
int ggml_cuda_get_device() { return 0; }  // never reached: get_J_max does not query a device
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }
int main() {
    const int ccs[] = {
        GGML_CUDA_CC_PASCAL, GGML_CUDA_CC_VOLTA, GGML_CUDA_CC_TURING, GGML_CUDA_CC_AMPERE,
        GGML_CUDA_CC_ADA_LOVELACE, GGML_CUDA_CC_HOPPER, GGML_CUDA_CC_BLACKWELL,
        GGML_CUDA_CC_RDNA2, GGML_CUDA_CC_RDNA3, GGML_CUDA_CC_RDNA3_5, GGML_CUDA_CC_RDNA4,
        GGML_CUDA_CC_CDNA2, GGML_CUDA_CC_CDNA3, GGML_CUDA_CC_VEGA,
    };
    const ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q2_K};
    int zero_cases = 0, total = 0;
    int64_t worst_n = 0;
    for (int cc : ccs) for (ggml_type t : types) for (int fb = 0; fb < 2; ++fb)
        for (int64_t n = 1; n <= 16; ++n) {
            const int j = ggml_cuda_mmq_get_J_max(t, fb, cc, n);
            ++total;
            if (j == 0) { ++zero_cases; if (n > worst_n) worst_n = n; }
        }
    printf("  checked %d (cc x type x fallback x ne11 in 1..16)\n", total);
    printf("  returned ZERO padding in %d cases; largest ne11 with zero = %lld\n", zero_cases, (long long)worst_n);
    printf("  VERDICT: %s\n", zero_cases ? "INVARIANT VIOLATED (padding must be > 0 for ne11 >= 1)" : "ok");
    return 0;
}
