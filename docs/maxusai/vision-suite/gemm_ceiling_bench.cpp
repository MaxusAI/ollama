// Is there headroom in rocBLAS on gfx1151, or is it already near the metal?
//
// Everything measured so far is end-to-end: it says which configuration is
// fastest, never whether the fastest one is any good. This calls rocBLAS
// directly with the EXACT parameters captured from a real image request
// (transA=T transB=N, f16 a/b/c/d, compute f16) and reports achieved TFLOP/s,
// so the answer can be compared against the hardware ceiling rather than
// against another configuration.
//
// Shapes are the top rocBLAS consumers from gemm_shapes_gfx1151.tsv.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define HC(x) do { auto e=(x); if(e!=hipSuccess){printf("HIP fail %d @%d\n",(int)e,__LINE__);return 1;} } while(0)
#define RC(x) do { auto e=(x); if(e!=rocblas_status_success){printf("rocBLAS fail %d @%d\n",(int)e,__LINE__);return 1;} } while(0)

struct Shape { int m, n, k; const char* label; };

int main() {
    rocblas_handle h; RC(rocblas_create_handle(&h));
    hipDeviceProp_t prop; HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  WGPs=%d CUs=%d  clock=%.2f GHz\n",
           prop.gcnArchName, prop.multiProcessorCount, cus, prop.clockRate/1e6);
    // RDNA3.5 WMMA: 512 FP16 FLOP per CU per clock (256 MAC x 2).
    //
    // multiProcessorCount is NOT the CU count on RDNA -- HIP reports WGPs
    // (workgroup processors), each holding 2 CUs. Using it directly halved the
    // ceiling and produced "165% of peak", which is how the bug surfaced.
    // rocminfo reports 40 CUs for gfx1151 against multiProcessorCount = 20.
    const int cus = prop.multiProcessorCount * 2;
    double peak = cus * 512.0 * (prop.clockRate*1e3) / 1e12;
    printf("fp16 WMMA peak (theoretical): %.1f TFLOP/s\n\n", peak);

    std::vector<Shape> shapes = {
        {1152, 9900, 1152, "gemma4 tower attn-proj  x108"},
        {4304, 9900, 1152, "gemma4 tower ffn-up      x54"},
        {1152, 9900, 4304, "gemma4 tower ffn-down    x27"},
        {5376, 1100, 21504,"gemma4 LM ffn-down       x30"},
        {5120, 1024, 17408,"qwen3.8 LM ffn-down      x32"},
        {1280, 8170, 1280, "nemotron3 tower         x128"},
    };

    printf("%-30s %8s %8s %8s   %9s  %6s\n",
           "shape", "m", "n", "k", "TFLOP/s", "%peak");
    for (auto& s : shapes) {
        size_t na=(size_t)s.k*s.m, nb=(size_t)s.k*s.n, nc=(size_t)s.m*s.n;
        void *A,*B,*C;
        HC(hipMalloc(&A, na*2)); HC(hipMalloc(&B, nb*2)); HC(hipMalloc(&C, nc*2));
        HC(hipMemset(A,0,na*2)); HC(hipMemset(B,0,nb*2)); HC(hipMemset(C,0,nc*2));
        const _Float16 alpha=1.0f, beta=0.0f;
        // transA=T transB=N, lda=k ldb=k ldc=m -- exactly as captured.
        auto call=[&](){ return rocblas_gemm_ex(h,
            rocblas_operation_transpose, rocblas_operation_none,
            s.m, s.n, s.k, &alpha,
            A, rocblas_datatype_f16_r, s.k,
            B, rocblas_datatype_f16_r, s.k, &beta,
            C, rocblas_datatype_f16_r, s.m,
            C, rocblas_datatype_f16_r, s.m,
            rocblas_datatype_f16_r, rocblas_gemm_algo_standard, 0, 0); };
        for (int i=0;i<5;i++) RC(call());          // warm up
        HC(hipDeviceSynchronize());
        const int iters=20;
        std::vector<double> ts;
        for (int i=0;i<iters;i++) {
            hipEvent_t a,b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
            HC(hipEventRecord(a)); RC(call()); HC(hipEventRecord(b));
            HC(hipEventSynchronize(b));
            float ms; HC(hipEventElapsedTime(&ms,a,b)); ts.push_back(ms);
            HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
        }
        std::sort(ts.begin(), ts.end());
        double ms = ts[ts.size()/2];               // median, not mean
        double tf = 2.0*s.m*s.n*s.k/(ms*1e-3)/1e12;
        printf("%-30s %8d %8d %8d   %9.2f  %5.1f%%\n",
               s.label, s.m, s.n, s.k, tf, 100.0*tf/peak);
        HC(hipFree(A)); HC(hipFree(B)); HC(hipFree(C));
    }
    rocblas_destroy_handle(h);
    return 0;
}
