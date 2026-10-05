// The MMQ src1 tail-padding requirement, checked per config instead of per shape: every config in every
// architecture's table against each padding rule. GPU-free, about 10 ms.
//
// Why per config is enough. A tile's y load copies GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int)) bytes
// from its first column (the loop in mul_mat_q_process_tile runs to the next multiple of nthreads ints), so a last
// tile holding one column reads that many bytes less one block past the data. The read depends only on the config's
// J and nthreads, and any config a launch can pick can end on one column: for MUL_MAT_ID the last expert holds one
// row whenever n_used < n_experts. So "every config's read fits the padding" is the whole requirement, with no
// shape sweep -- mmq-rules-check.cu's 9.4 million shapes come out the same, table by table.
//
// It calls each table function (ggml_cuda_mmq_get_config_<table>) directly rather than the cc dispatch, so it does
// not depend on the architectures the compiler targets: nvcc and hipcc evaluate it alike, with no gencode list.
//
// Build from a llama.cpp checkout with either compiler:
//   hipcc -std=c++17 -DGGML_USE_HIP --offload-arch=gfx1151 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//         mmq-padding-invariant.cu -o mmq-padding-invariant
//   nvcc  -std=c++17 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda mmq-padding-invariant.cu -o mmq-padding-invariant
// Exit 1 if a padded-tile rule (#29953 as of 3070d927f, or compat 903 as amended) leaves any config short.
#include "ggml/src/ggml-cuda/mmq.cuh"
#include <cstdio>
#include <cstdlib>

void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
int ggml_cuda_get_device() { return 0; }
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }

static const long B = (long) sizeof(block_q8_1_mmq);

typedef ggml_cuda_mmq_config (*table_fn)(ggml_type, int, bool);

// Every table ggml_cuda_mmq_get_config() can dispatch to. cc is a representative device, used only to size shared
// memory; smpbo is the largest opt-in shared memory among the devices that use the table, so "launchable" errs on
// the side of counting a config.
struct table { const char * name; table_fn get; int cc; size_t smpbo; };
static const table TABLES[] = {
    {"pascal_older (< sm_61)",    ggml_cuda_mmq_get_config_pascal_older, GGML_CUDA_CC_PASCAL,     49152},
    {"pascal_dp4a (sm_61)",       ggml_cuda_mmq_get_config_pascal_dp4a,  GGML_CUDA_CC_DP4A,       49152},
    {"ampere (sm_70 .. sm_90)",   ggml_cuda_mmq_get_config_ampere,       GGML_CUDA_CC_AMPERE,    232448},
    {"blackwell (sm_100, 120)",   ggml_cuda_mmq_get_config_blackwell,    GGML_CUDA_CC_BLACKWELL, 232448},
    {"gcn (gfx906)",              ggml_cuda_mmq_get_config_gcn,          GGML_CUDA_CC_VEGA20,     65536},
    {"cdna (gfx908 .. gfx950)",   ggml_cuda_mmq_get_config_cdna,         GGML_CUDA_CC_CDNA3,      65536},
    {"rdna2 (gfx1030)",           ggml_cuda_mmq_get_config_rdna2,        GGML_CUDA_CC_RDNA2,      65536},
    {"rdna3 (gfx1100 .. 1102)",   ggml_cuda_mmq_get_config_rdna3,        GGML_CUDA_CC_RDNA3,      65536},
    {"rdna3_5 (gfx1150, 1151)",   ggml_cuda_mmq_get_config_rdna3_5,      GGML_CUDA_CC_RDNA3_5,    65536},
    {"rdna4 (gfx1200, 1201)",     ggml_cuda_mmq_get_config_rdna4,        GGML_CUDA_CC_RDNA4,      65536},
};

enum { R_JBLOCKS, R_PADDED, R_WIDEST, R_WIDEST_PADDED, NR };
static const char * const RN[NR] = {
    "J blocks (#29953 as first posted)",
    "padded tile of J (#29953, 3070d927f)",
    "widest tile (903 before amending)",
    "widest padded tile (903 amended)",
};

int main() {
    int total = 0, short_all[NR] = {};
    printf("%-26s %7s | %-38s %s\n", "table", "configs", "rule", "short configs, launchable, worst blocks past");
    for (const table & t : TABLES) {
        int configs = 0, sn[NR] = {}, sl[NR] = {}, sw[NR] = {};
        for (int ti = 0; ti < GGML_TYPE_COUNT; ++ti) for (int fb = 0; fb < 2; ++fb) {
            const ggml_type type = (ggml_type) ti;
            long wide = 0, wide_padded = 0;
            for (int J = 8; J <= 128; J += 8) {
                const ggml_cuda_mmq_config c = t.get(type, J, fb);
                if (c.type == GGML_TYPE_COUNT) continue;
                const long T = (long) GGML_PAD(c.J*B, c.nthreads*sizeof(int));
                wide        = c.J*B     > wide        ? c.J*B     : wide;
                wide_padded = T/B*B     > wide_padded ? T/B*B     : wide_padded;   // 903 floors to whole blocks
            }
            for (int J = 8; J <= 128; J += 8) {
                const ggml_cuda_mmq_config c = t.get(type, J, fb);
                if (c.type == GGML_TYPE_COUNT) continue;
                ++configs;
                const bool launchable = mmq_get_nbytes_shared(c, t.cc) <= t.smpbo;
                const long T    = (long) GGML_PAD(c.J*B, c.nthreads*sizeof(int));  // bytes the y load copies
                const long need = T - B;                                            // past the data, one column left
                const long pad[NR] = { c.J*B, T, wide, wide_padded };
                for (int r = 0; r < NR; ++r) {
                    if (pad[r] >= need) continue;
                    ++sn[r];
                    sl[r] += launchable;
                    const int blocks = (int) ((need - pad[r] + B - 1)/B);
                    sw[r] = blocks > sw[r] ? blocks : sw[r];
                }
            }
        }
        total += configs;
        for (int r = 0; r < NR; ++r) {
            char cs[16];
            snprintf(cs, sizeof(cs), "%d", configs);
            printf("%-26s %7s | %-38s %4d %4d %3d\n", r ? "" : t.name, r ? "" : cs, RN[r], sn[r], sl[r], sw[r]);
            short_all[r] += sn[r];
        }
    }
    printf("\n%d configs in %zu tables\n", total, sizeof(TABLES)/sizeof(TABLES[0]));
    for (int r = 0; r < NR; ++r) printf("  %-40s %s\n", RN[r], short_all[r] ? "SHORT" : "covered");
    return short_all[R_PADDED] || short_all[R_WIDEST_PADDED] ? 1 : 0;
}
