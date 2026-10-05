// Is a HIP VMM range safe to reserve again after it was freed? On gfx1151 (ROCm 7.2.1) it is not, and that is why
// mmq-debug-alloc.cuh keeps its reservations on HIP.
//
// Each iteration reserves size + one granule, maps a fresh allocation, places a buffer at the end of the mapping as
// the guard allocator does, and moves a pattern through it both ways: a copy engine writes and a kernel checks,
// then a kernel writes and a copy engine checks. A hipMalloc'd buffer churns in between, as ggml's legacy pool does.
// Sizes vary per iteration, as per-expert MMQ calls do.
//
//   hipcc --offload-arch=gfx1151 -O1 mmq-hip-vmm-reuse.hip -o mmq-hip-vmm-reuse
//   ./mmq-hip-vmm-reuse 200 keep   # unmap and release, never free the reservation: expect 0 and 0
//   ./mmq-hip-vmm-reuse 200 free   # also hipMemAddressFree, so ranges are reused: mismatches, maybe a GPU abort
//
// `free` writes through stale translations, and the writes can land in other allocations: on gfx1151 a run ended in
// HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION in an unrelated kernel. Run it only with no other GPU process on the device.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
__global__ void check(const int * p, size_t n, int v, unsigned long long * bad) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) if (p[i] != v + (int) i) atomicAdd(bad, 1ull); }
__global__ void fill(int * p, size_t n, int v) { for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) p[i] = v - (int) i; }
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <iterations> <keep|free>\n", argv[0]); return 2; }
    const int iters = atoi(argv[1]); const bool free_va = !strcmp(argv[2], "free");
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipMemAllocationProp prop = {}; prop.type = hipMemAllocationTypePinned; prop.location.type = hipMemLocationTypeDevice; prop.location.id = 0;
    size_t g; CK(hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum));
    unsigned long long * bad; CK(hipMalloc(&bad, sizeof(*bad)));
    srand(1); int bad_k = 0, bad_c = 0;
    for (int it = 0; it < iters; ++it) {
        const size_t size = (size_t) (40000 + rand() % 400000) & ~(size_t) 3, msize = (size + g - 1)/g*g, rsize = msize + g;
        void * churn; CK(hipMalloc(&churn, 4096 + rand() % 1000000));
        hipDeviceptr_t base; CK(hipMemAddressReserve(&base, rsize, g, 0, 0));
        hipMemGenericAllocationHandle_t h; CK(hipMemCreate(&h, msize, &prop, 0));
        CK(hipMemMap(base, msize, 0, h, 0)); CK(hipMemRelease(h));
        hipMemAccessDesc a = {}; a.location = prop.location; a.flags = hipMemAccessFlagsProtReadWrite;
        CK(hipMemSetAccess(base, msize, &a, 1));
        int * p = (int *) ((char *) base + ((msize - size) & ~(size_t) 127));
        const size_t n = size/sizeof(int);
        std::vector<int> host(n); for (size_t i = 0; i < n; ++i) host[i] = it*7 + (int) i;
        CK(hipMemcpyAsync(p, host.data(), size, hipMemcpyHostToDevice, s));
        CK(hipMemsetAsync(bad, 0, sizeof(*bad), s));
        hipLaunchKernelGGL(check, dim3(64), dim3(256), 0, s, p, n, it*7, bad);
        unsigned long long hb = 0; CK(hipMemcpyAsync(&hb, bad, sizeof(hb), hipMemcpyDeviceToHost, s)); CK(hipStreamSynchronize(s));
        if (hb && ++bad_k <= 3) printf("iter %d (%zu bytes): a kernel read %llu wrong ints after a host-to-device copy\n", it, size, hb);
        hipLaunchKernelGGL(fill, dim3(64), dim3(256), 0, s, p, n, it*11);
        CK(hipMemcpyAsync(host.data(), p, size, hipMemcpyDeviceToHost, s)); CK(hipStreamSynchronize(s));
        size_t wrong = 0; for (size_t i = 0; i < n; ++i) wrong += host[i] != it*11 - (int) i;
        if (wrong && ++bad_c <= 3) printf("iter %d (%zu bytes): a device-to-host copy read %zu wrong ints after a kernel write\n", it, size, wrong);
        CK(hipMemUnmap(base, msize));
        if (free_va) CK(hipMemAddressFree(base, rsize));
        CK(hipFree(churn));
    }
    printf("%s: %d iterations, kernel-read mismatches in %d, copy-read mismatches in %d\n",
           free_va ? "free (ranges reused)" : "keep (no range reused)", iters, bad_k, bad_c);
    return bad_k || bad_c;
}
