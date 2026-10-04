// GPU-free check of the MMQ tail padding in ggml_cuda_mul_mat_q(), for every padding rule proposed so far.
// Material for a successor to llama.cpp#27044 on master dd266785c (#29941 merged).
//
// What the kernel reads past the end of its buffers (mmq.cuh at dd266785c; #29953 changes none of this):
//
//   src1_q8_1  A tile's y load is
//                  for (l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += nthreads) tile_y[l0 + tid] = by0[l0 + tid];
//              with no bound on l, so it reads GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int)) bytes
//              from the tile's first column -- exactly the size mmq_get_nbytes_shared() gives the shared-memory
//              tile it copies into. The buffer is laid out [k-plane][column], so in the last k-plane a tile
//              holding one column has all but one block of that past the data:
//                  need = GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int))/sizeof(block_q8_1_mmq) - 1
//              which exceeds J - 1 whenever nthreads*sizeof(int) does not divide J*sizeof(block_q8_1_mmq).
//              For MUL_MAT_ID the last expert's last tile really can hold a single row, at any batch size.
//
//   ids_dst    each tile loads J entries, `ids_dst_shared[j] = ids_dst[col_low + jt*J + j]` for j < J, again with
//              no bound, so the last tile reads up to J - 1 entries past ids_dst's ne12*n_expert_used.
//
// J is the tile width mul_mat_q_switch_J launches: of the J in 8..128 (step 8) that have a config fitting shared
// memory, the first with the fewest tiles for ncols_opt. That loop is replicated below because it sits in a
// template that queries the device; everything else is llama.cpp's own code.
//
// The rules, as the argument of ggml_cuda_mmq_get_J_max() unless stated:
//   #24127    : ne11                   (1 for broadcast gate/up, n_expert_used otherwise), b9992 .. dd266785c^
//   #29941    : ne12                   master, merged as dd266785c
//   #29953    : the launched J itself  (padding moved next to the tile-size choice, so the two cannot disagree)
//   #27044    : ne12*n_expert_used     our compat 903 until now
//   #448      : the widest tile that has a config, get_J_max(type, fallback, cc, 512)
//   #448+pad  : the widest padded tile, max over J of GGML_PAD(J*block, nthreads_J*4)/block
//
// Build from the root of a llama.cpp checkout, with the CUDA toolkit's nvcc. It never touches a GPU:
//   nvcc -Wno-deprecated-gpu-targets -std=c++17 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//        mmq-rules-check.cu -o mmq-rules-check -lcublas -lcuda
// It calls ggml_cuda_mmq_get_config() in its four-argument form, which exists at the fork's pin b11081 and is
// master's signature with prec_src1 defaulted, so the same source builds against either.
//   ./mmq-rules-check            the whole sweep
//   ./mmq-rules-check --design   shapes for test_mul_mat_id where only ids_dst is short
// The exit code is non-zero if the last rule leaves a shape short.
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"   // MMVQ_MAX_BATCH_SIZE
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
int ggml_cuda_get_device() { return 0; }  // never reached: nothing below queries a device
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }

static const size_t BLOCK = sizeof(block_q8_1_mmq);

// The largest MUL_MAT_ID batch MMVQ takes (mmvq.cu, get_mmvq_mmid_max_batch*; defined outside its header, so
// replicated for the ten types checked). A larger batch goes to MMQ.
//
// The dispatcher's NVIDIA branch is not "Turing and newer": Volta and Ada Lovelace and newer always take MMVQ up
// to MMVQ_MAX_BATCH_SIZE for every type, and only Turing..Ampere use the per-type table. Reading it as
// Turing-and-newer counted q2_K at 8 tokens and q3_K at 6..8 as MMQ shapes on sm_89 and sm_120, where MMVQ takes
// them -- caught by the ROCm host on MaxusAI/ollama#449.
static int mmvq_mmid_max(ggml_type t, int cc) {
    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
        switch (t) {   // get_mmvq_mmid_max_batch_rdna3
            case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: return 4;
            default: return MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return MMVQ_MAX_BATCH_SIZE;
    }
    switch (t) {   // get_mmvq_mmid_max_batch_turing_plus, Turing and Ampere only
        case GGML_TYPE_Q2_K: return 7;
        case GGML_TYPE_Q3_K: return 5;
        default:             return MMVQ_MAX_BATCH_SIZE;
    }
}

// Blocks of src1 a tile of width J with nthreads threads reads from its first column, minus the one that is data.
static int src1_need(int J, int nthreads) {
    if (J == 0) return 0;
    return (int) ((GGML_PAD(J*BLOCK, nthreads*sizeof(int)) + BLOCK - 1) / BLOCK) - 1;
}

// mul_mat_q_switch_J's choice: the launched tile width and its thread count.
static void launched(ggml_type t, bool fb, int cc, size_t smpbo, int64_t ncols_opt, int * J_out, int * nth_out) {
    int nt_best = INT_MAX;
    *J_out = 0; *nth_out = 0;
    for (int J = 8; J <= 128 && nt_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, cc);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int nt = (ncols_opt + c.J - 1) / c.J;
        if (nt < nt_best) { nt_best = nt; *J_out = c.J; *nth_out = c.nthreads; }
    }
}

// #448 as written: the widest tile that has a config, in blocks. And amended: the widest padded tile.
static void widest(ggml_type t, bool fb, int cc, int * J_out, int * pad_out) {
    *J_out = ggml_cuda_mmq_get_J_max(t, fb, cc, 512);
    size_t widest_bytes = 0;
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, cc);
        if (c.type == GGML_TYPE_COUNT) continue;
        // mmq-fix-amended.patch computes J_pad = nbytes_pad_y / sizeof(block_q8_1_mmq), a floor. That is still
        // enough -- floor(T/B) >= ceil(T/B) - 1 = the requirement -- and it is what the shipped patch does, so
        // score that rather than a ceiling the code does not use.
        const size_t nbytes = GGML_PAD(c.J*BLOCK, c.nthreads*sizeof(int));
        if (nbytes > widest_bytes) widest_bytes = nbytes;
    }
    *pad_out = (int) (widest_bytes / BLOCK);
}

static const char * const ARCH_NAMES[5] = {"sm_75", "sm_86", "sm_89", "sm_120", "gfx1151"};
static int arch_index(const char * name) { for (int i = 0; i < 5; ++i) if (!strcmp(name, ARCH_NAMES[i])) return i; return 0; }

struct tally {
    long n = 0; long n_arch[5] = {}; int worst = 0; char ex[220] = "";
    void add(int shortfall, const char * arch, const char * tname, int fb, long a, long b, long c, int J, int nth, int pad) {
        if (shortfall <= 0) return;
        ++n; ++n_arch[arch_index(arch)];
        if (shortfall > worst) {
            worst = shortfall;
            snprintf(ex, sizeof(ex), "%s, %s, fallback=%d, %ld/%ld/%ld: J=%d, nthreads=%d, padding=%d",
                     arch, tname, fb, a, b, c, J, nth, pad);
        }
    }
    void print(const char * label, const char * unit) const {
        printf("  %-34s %s", label, n ? "SHORT" : "covered");
        if (n) {
            printf(" in %ld shapes, worst %d %s past the allocation\n", n, worst, unit);
            printf("  %-34s  worst case: %s\n", "", ex);
            printf("  %-34s  by arch:", "");
            for (int i = 0; i < 5; ++i) printf(" %s %ld%s", ARCH_NAMES[i], n_arch[i], i < 4 ? "," : "");
        }
        printf("\n");
    }
};

enum { R24127, R29941, R29953, R27044, R448, R448P, NR };
static const char * const RN[NR] = {
    "#24127 (ne11)", "#29941 (ne12), master", "#29953 (the launched tile)",
    "#27044 (ne12*n_expert_used)", "#448 (widest tile)", "#448 amended (widest padded tile)"};

struct arch { const char * name; int cc; size_t smpbo; };
static const arch ARCHS[] = {
    {"sm_75",   GGML_CUDA_CC_TURING,        65536},
    {"sm_86",   860,                       101376},
    {"sm_89",   GGML_CUDA_CC_ADA_LOVELACE, 101376},
    {"sm_120",  GGML_CUDA_CC_BLACKWELL,    101376},
    {"gfx1151", GGML_CUDA_CC_RDNA3_5,       65536},
};
static const ggml_type TYPES[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                                  GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
static const char * const TN[] = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"};
static const int NE02S[] = {8, 32, 128, 256};   // only RDNA3/4 optimise J against ne12*n_expert_used/ne02

// --design: shapes for test_mul_mat_id where src1 is covered by EVERY rule and only ids_dst is short, so a fault
// can only come from the ids_dst read. test_mul_mat_id(type, f32, n_mats, n_used, b=false, m, ne12, k) gives
// ne11 = n_used, ne12 tokens, n_expert_used = n_used, ne02 = n_mats, ne01 = m (m % 128 != 0 -> fallback).
static void design(int cc, size_t smpbo, const char * arch_name) {
    printf("%s: test_mul_mat_id shapes where every rule's src1 padding is sufficient and only ids_dst is short\n",
           arch_name);
    printf("  %-6s %-3s %6s %7s %5s %5s %6s | %7s %7s %7s %7s %7s\n", "type", "fb", "ne12", "n_used", "J",
           "nthr", "need", "ne11", "ne12", "29953", "27044", "widest");
    int found = 0;
    for (int ti = 0; ti < 10; ++ti) for (int fb = 0; fb < 2; ++fb) {
        const ggml_type t = TYPES[ti];
        int J_wide, pad_wide_amended;
        widest(t, fb, cc, &J_wide, &pad_wide_amended);
        for (int64_t ne12 = mmvq_mmid_max(t, cc) + 1; ne12 <= 512; ++ne12)
        for (int n_used = 2; n_used <= 256; ++n_used) {
            int J, nth;
            launched(t, fb, cc, smpbo, ne12, &J, &nth);
            if (J == 0) continue;
            const int need = src1_need(J, nth);
            const int p[5] = {
                ggml_cuda_mmq_get_J_max(t, fb, cc, n_used),      // b = false, so ne11 = n_used
                ggml_cuda_mmq_get_J_max(t, fb, cc, ne12),
                J,
                ggml_cuda_mmq_get_J_max(t, fb, cc, ne12*n_used),
                J_wide,
            };
            bool all_ok = true;
            for (int i = 0; i < 5; ++i) if (p[i] < need) all_ok = false;
            if (!all_ok) continue;
            // 256 experts is what the MoE models in question use; keep the mean rows per expert well under one
            // tile so the last non-empty expert holds a partial tile and its last tile reads past ids_dst.
            const int n_mats = 256;
            if (n_used > n_mats || ne12*n_used > (int64_t) n_mats * (J/4)) continue;
            printf("  %-6s %-3d %6ld %7d %5d %5d %6d | %7d %7d %7d %7d %7d   n_mats=%d rows=%ld\n",
                   TN[ti], fb, (long) ne12, n_used, J, nth, need, p[0], p[1], p[2], p[3], p[4],
                   n_mats, (long) (ne12*n_used));
            if (++found >= 60) return;
        }
    }
    if (!found) printf("  none\n");
}

int main(int argc, char ** argv) {
    if (argc > 1 && !strcmp(argv[1], "--design")) {
        design(GGML_CUDA_CC_BLACKWELL, 101376, "sm_120");
        return 0;
    }

    tally ids_src1[NR][2], ids_dst[NR][2], dense[NR];
    long ids_shapes = 0, dense_shapes = 0;

    for (const arch & a : ARCHS) for (int ti = 0; ti < 10; ++ti) for (int fb = 0; fb < 2; ++fb) {
        const ggml_type t = TYPES[ti];
        const bool rdna = GGML_CUDA_CC_IS_RDNA3(a.cc) || GGML_CUDA_CC_IS_RDNA4(a.cc);
        int J_wide, pad_wide_amended;
        widest(t, fb, a.cc, &J_wide, &pad_wide_amended);

        // ids branch
        for (int64_t ne12 = mmvq_mmid_max(t, a.cc) + 1; ne12 <= 1024; ++ne12)
        for (int n_used = 1; n_used <= 16; ++n_used)
        for (int ei = 0; ei < (rdna ? 4 : 1); ++ei) {
            const int64_t ne02 = rdna ? NE02S[ei] : 256;
            if (n_used > ne02) continue;
            const int64_t ncols_opt = rdna ? (ne12*n_used + ne02 - 1) / ne02 : ne12;
            int J, nth;
            launched(t, fb, a.cc, a.smpbo, ncols_opt, &J, &nth);
            if (J == 0) continue;
            const int need_src1 = src1_need(J, nth);
            const int need_ids  = J - 1;
            for (int bcast = 0; bcast < 2; ++bcast) {
                if (bcast && n_used == 1) continue;   // one expert per token has nothing to broadcast
                ++ids_shapes;
                const int pad[NR] = {
                    ggml_cuda_mmq_get_J_max(t, fb, a.cc, bcast ? 1 : n_used),
                    ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne12),
                    J,
                    ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne12*n_used),
                    J_wide,
                    pad_wide_amended,
                };
                const int u = n_used == 1 ? 0 : 1;
                for (int r = 0; r < NR; ++r) {
                    ids_src1[r][u].add(need_src1 - pad[r], a.name, TN[ti], fb, ne12, n_used, ne02, J, nth, pad[r]);
                    // only #448 pads ids_dst, by the same count as src1
                    const int dst_pad = (r == R448 || r == R448P) ? pad[r] : 0;
                    ids_dst[r][u].add(need_ids - dst_pad, a.name, TN[ti], fb, ne12, n_used, ne02, J, nth, dst_pad);
                }
            }
        }

        // dense branch: MMVQ takes ne11 <= MMVQ_MAX_BATCH_SIZE
        for (int64_t ne11 = MMVQ_MAX_BATCH_SIZE + 1; ne11 <= 1024; ++ne11) {
            int J, nth;
            launched(t, fb, a.cc, a.smpbo, ne11, &J, &nth);
            if (J == 0) continue;
            ++dense_shapes;
            // the last tile holds `last` columns, so the read goes src1_need(J) - (last - 1) blocks past the data
            const int64_t last = ne11 - ((ne11 + J - 1) / J - 1) * J;
            const int need = src1_need(J, nth) - (int) (last - 1);
            const int pad[NR] = {
                ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne11), ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne11), J,
                ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne11), J_wide, pad_wide_amended,
            };
            for (int r = 0; r < NR; ++r) dense[r].add(need - pad[r], a.name, TN[ti], fb, ne11, 0, 0, J, nth, pad[r]);
        }
    }

    printf("ids branch: %ld shapes (sm_75/86/89/120 and gfx1151 x 10 types x fallback 0/1 x every MMQ batch up to\n"
           "            1024 tokens x n_expert_used 1..16 x broadcast 0/1; on gfx1151 also x 8/32/128/256 experts)\n",
           ids_shapes);
    printf(" src1_q8_1, one expert per token (n_expert_used = 1):\n");
    for (int r = 0; r < NR; ++r) ids_src1[r][0].print(RN[r], "blocks");
    printf(" src1_q8_1, two or more experts per token:\n");
    for (int r = 0; r < NR; ++r) ids_src1[r][1].print(RN[r], "blocks");
    printf(" ids_dst, any n_expert_used:\n");
    for (int r = 0; r < NR; ++r) {
        tally both = ids_dst[r][0].n ? ids_dst[r][0] : ids_dst[r][1];
        both.n = ids_dst[r][0].n + ids_dst[r][1].n;
        for (int i = 0; i < 5; ++i) both.n_arch[i] = ids_dst[r][0].n_arch[i] + ids_dst[r][1].n_arch[i];
        if (ids_dst[r][1].worst > both.worst) { both.worst = ids_dst[r][1].worst; snprintf(both.ex, sizeof(both.ex), "%s", ids_dst[r][1].ex); }
        both.print(RN[r], "int32 entries");
    }
    printf("dense branch: %ld shapes (same archs, types and fallbacks x every MMQ batch up to 1024 columns)\n",
           dense_shapes);
    for (int r = 0; r < NR; ++r) {
        if (r == R24127 || r == R29953 || r == R448 || r == R448P) dense[r].print(RN[r], "blocks");
    }

    const bool ok = ids_src1[R448P][0].n == 0 && ids_src1[R448P][1].n == 0
                 && ids_dst[R448P][0].n == 0 && ids_dst[R448P][1].n == 0 && dense[R448P].n == 0;
    return ok ? 0 : 1;
}
