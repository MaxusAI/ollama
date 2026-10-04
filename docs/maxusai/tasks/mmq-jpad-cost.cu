#include "ggml/src/ggml-cuda/mmq.cuh"
#include <cstdio>
#include <cstdlib>
void ggml_cuda_error(const char*,const char*,const char*,int,const char*){abort();}
int ggml_cuda_get_device(){return 0;}
extern "C" void ggml_abort(const char*,int,const char*,...){abort();}
int main(){
    struct A { const char * n; int cc; } as[] = {{"sm_75",GGML_CUDA_CC_TURING},{"sm_86",860},
        {"sm_89",GGML_CUDA_CC_ADA_LOVELACE},{"sm_120",GGML_CUDA_CC_BLACKWELL},{"gfx1151",GGML_CUDA_CC_RDNA3_5}};
    const ggml_type ts[] = {GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,
                            GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K};
    const size_t B = sizeof(block_q8_1_mmq);
    printf("J_pad in blocks of %zu bytes: old rule = get_J_max(512), new rule = widest padded tile\n", B);
    const char * tn[] = {"q4_0","q4_1","q5_0","q5_1","q8_0","q2_K","q3_K","q4_K","q5_K","q6_K"};
    printf("%-9s %-3s %-6s %6s %6s %8s\n", "arch", "fb", "type", "old", "new", "extra B");
    for (auto & a : as) for (int fb = 0; fb < 2; ++fb) for (int ti = 0; ti < 10; ++ti) {
        const ggml_type t = ts[ti];
        const int o = ggml_cuda_mmq_get_J_max(t, fb, a.cc, 512);
        int n = 0;
        for (int J = 8; J <= 128; J += 8) {
            ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, a.cc);
            if (c.type == GGML_TYPE_COUNT) continue;
            const int blocks = (int)((GGML_PAD(c.J*B, c.nthreads*sizeof(int)) + B - 1)/B);
            if (blocks > n) n = blocks;
        }
        if (n != o) printf("%-9s %-3d %-6s %6d %6d %8zu\n", a.n, fb, tn[ti], o, n, (size_t)(n - o)*B);
    }
    printf("(only rows where the two rules differ)\n");
    return 0;
}
