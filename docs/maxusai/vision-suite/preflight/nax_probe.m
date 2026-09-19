// nax_probe — does this host enable the Metal tensor API (the M5 Neural
// Accelerators) for GGUF inference? Answers in ~100 ms cold, ~2 ms warm,
// without loading a model, and exits 0 when the answer is yes.
//
// WHY THIS EXISTS. The tensor path is worth 2.14x prefill on an M5 Max
// (docs/maxusai/m5-neural-accelerators.md) and it fails SILENTLY: ggml
// disables it on a failed runtime compile, the decision is not observable in
// the server log on this host, and the only symptom is half the prefill. A
// toolchain or OS regression would cost that with nothing turning red.
//
// It replicates ggml-metal-device.m's has_tensor decision at llama.cpp
// b10864, in the same order and with the same precedence:
//   1. [device supportsFamily:MTLGPUFamilyMetal4]   (enum 5002, hardcoded
//      there because older SDKs lack the symbol)
//   2. GGML_METAL_TENSOR_DISABLE wins outright
//   3. device-name allowlist M5/M6/A19/A20, bypassable with
//      GGML_METAL_TENSOR_ENABLE. Upstream gates on the NAME because the path
//      was "~5% slower" on M2 Ultra and no different on M4.
//   4. compile the dummy mpp::tensor_ops::matmul2d kernel below and build a
//      pipeline from it
//
// THE KERNEL BELOW IS COPIED VERBATIM from that file's src_tensor_f16 and
// MUST STAY IN SYNC. It is not reconstructable from memory: a hand-written
// version of it compiled with two errors here (a 6-argument
// matmul2d_descriptor, a 2-argument get_destination_cooperative_tensor, and a
// missing tensor_inline tag) and would have reported a false alarm. Re-extract
// it rather than editing it:
//
//   curl -sL https://raw.githubusercontent.com/ggml-org/llama.cpp/$(cat LLAMA_CPP_VERSION)/ggml/src/ggml-metal/ggml-metal-device.m \
//     | sed -n '/src_tensor_f16 = /,/^ *"}";/p'
//
// Build and run:
//   xcrun clang -fobjc-arc -framework Foundation -framework Metal nax_probe.m -o nax_probe && ./nax_probe
//
// Prints one JSON object; exit 0 = the accelerators will be used, 1 = they
// will not, with "error" carrying the compiler diagnostic when that is why.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;
static const NSUInteger MTLLanguageVersion4_0_GGML = 4 << 16;

static NSString *kDummy = @"\n#include <metal_stdlib> \n#include <metal_tensor> \n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n \nusing namespace metal; \nusing namespace mpp::tensor_ops; \n \nkernel void dummy_kernel( \n    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n    device float * C [[buffer(2)]], \n    uint2 tgid [[threadgroup_position_in_grid]]) \n{ \n    auto tA = A.slice(0, (int)tgid.y); \n    auto tB = B.slice((int)tgid.x, 0); \n \n    matmul2d< \n        matmul2d_descriptor(16, 16, dynamic_extent), \n        execution_simdgroups<4>> mm; \n \n    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n \n    auto sA = tA.slice(0, 0); \n    auto sB = tB.slice(0, 0); \n    mm.run(sB, sA, cT); \n \n    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n \n    cT.store(tC); \n}";

int main(void) { @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSString *name = [dev name];
  BOOL fam  = [dev supportsFamily:(MTLGPUFamily)MTLGPUFamilyMetal4_GGML];
  // Same precedence as the gate: DISABLE wins outright; ENABLE only bypasses
  // the device-name allowlist.
  BOOL disabled = getenv("GGML_METAL_TENSOR_DISABLE") != NULL;
  BOOL forced   = getenv("GGML_METAL_TENSOR_ENABLE")  != NULL;
  BOOL allow = forced || [name containsString:@"M5"] || [name containsString:@"M6"] ||
               [name containsString:@"A19"] || [name containsString:@"A20"];
  NSDate *t0 = [NSDate date];
  MTLCompileOptions *o = [MTLCompileOptions new];
  o.languageVersion = (MTLLanguageVersion)MTLLanguageVersion4_0_GGML;
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:kDummy options:o error:&err];
  id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"dummy_kernel"] : nil;
  id<MTLComputePipelineState> ps = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
  double ms = -[t0 timeIntervalSinceNow] * 1000.0;
  BOOL has_tensor = fam && allow && !disabled && ps != nil;
  printf("{\"device\":\"%s\",\"supports_metal4_family\":%s,\"name_allowlisted\":%s,"
         "\"dummy_kernel_compiles\":%s,\"pipeline_ms\":%.1f,\"has_tensor\":%s",
         [name UTF8String], fam?"true":"false", allow?"true":"false",
         ps?"true":"false", ms, has_tensor?"true":"false");
  if (!ps && err) printf(",\"error\":\"%s\"",
      [[[err localizedDescription] stringByReplacingOccurrencesOfString:@"\n" withString:@" "] UTF8String]);
  printf("}\n");
  return has_tensor ? 0 : 1;
}}
