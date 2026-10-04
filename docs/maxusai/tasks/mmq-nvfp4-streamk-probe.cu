#include "ggml/src/ggml-cuda/mmq.cuh"
#include <cstdio>
#include <cstdlib>
void ggml_cuda_error(const char*,const char*,const char*,int,const char*){abort();}
int ggml_cuda_get_device(){return 0;}
extern "C" void ggml_abort(const char*,int,const char*,...){abort();}
int main(){
    // Does the NVFP4 native-FP4 path use stream-k? Only the stream-k fixup write_back passes j_max = J, which is
    // what makes it read y_scale for the whole tile.
    for (int cc : {1200, 1210}) for (int fb = 0; fb < 2; ++fb) {
        printf("cc=%d fb=%d NVFP4 prec=Q4:", cc, fb);
        bool any = false;
        for (int J = 8; J <= 128; J += 8) {
            ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, fb, cc, GGML_PREC_Q4);
            if (c.type == GGML_TYPE_COUNT) continue;
            any = true;
            printf(" J=%d(nth=%d,stream_k=%d)", c.J, c.nthreads, (int) c.stream_k);
        }
        if (!any) printf(" no configs");
        printf("\n");
    }
    return 0;
}
