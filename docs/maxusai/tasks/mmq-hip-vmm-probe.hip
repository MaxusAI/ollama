// Does the guard-page method work on this AMD GPU? Reports what the guard allocator (mmq-debug-alloc.cuh) relies
// on -- HIP virtual memory support and its granularity -- and the device limits the MMQ checks assume, then maps
// one granule at the start of a two-granule reservation and reads it from a kernel, `past` ints beyond its end.
//
//   hipcc --offload-arch=gfx1151 -O1 mmq-hip-vmm-probe.hip -o mmq-hip-vmm-probe
//   ./mmq-hip-vmm-probe 0    # in bounds: expect "no error"
//   ./mmq-hip-vmm-probe 1    # one int past the mapping: expect "an illegal memory access was encountered"
//
// The second run faults the GPU on purpose. On gfx1151 / ROCm 7.2.1 that kills only this process, logs one
// "[gfxhub] page fault" in the kernel log and does not reset the GPU; check that on another device before running
// it beside a production server.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)
__global__ void rd(const int * p, int n, int * out) { int s = 0; for (int i = 0; i < n; ++i) s += p[i]; *out = s; }
int main(int argc, char ** argv) {
    const int past = argc > 1 ? atoi(argv[1]) : 0;
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
    int vmm = -1; CK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
    printf("gcnArchName=%s warpSize=%d sharedMemPerBlock=%zu sharedMemPerBlockOptin=%zu multiProcessorCount=%d VMM=%d\n",
           pr.gcnArchName, pr.warpSize, pr.sharedMemPerBlock, pr.sharedMemPerBlockOptin, pr.multiProcessorCount, vmm);
    if (vmm != 1) return 1;
    hipMemAllocationProp prop = {}; prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice; prop.location.id = 0;
    size_t gmin = 0, grec = 0;
    CK(hipMemGetAllocationGranularity(&gmin, &prop, hipMemAllocationGranularityMinimum));
    CK(hipMemGetAllocationGranularity(&grec, &prop, hipMemAllocationGranularityRecommended));
    printf("granularity min=%zu recommended=%zu\n", gmin, grec);
    hipDeviceptr_t base; CK(hipMemAddressReserve(&base, 2*gmin, gmin, 0, 0));
    hipMemGenericAllocationHandle_t h; CK(hipMemCreate(&h, gmin, &prop, 0));
    CK(hipMemMap(base, gmin, 0, h, 0)); CK(hipMemRelease(h));
    hipMemAccessDesc a = {}; a.location = prop.location; a.flags = hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(base, gmin, &a, 1)); CK(hipMemset(base, 0, gmin));
    int * out; CK(hipMalloc(&out, sizeof(int)));
    const int n = (int) (gmin/sizeof(int));
    printf("reading %d ints of a %zu-byte mapping, the last %d past its end\n", n, gmin, past); fflush(stdout);
    hipLaunchKernelGGL(rd, dim3(1), dim3(1), 0, 0, (const int *) base + past, n, out);
    const hipError_t e = hipDeviceSynchronize();
    printf("hipDeviceSynchronize -> %s\n", hipGetErrorString(e));
    return e == hipSuccess ? 0 : 1;
}
