// Finds MUL_MAT_ID shapes for which #29941's padding (ne12) makes test-backend-ops abort on a stock build, with no
// debug switch and no sanitizer: the src1_q8_1 buffer is the last allocation in a fresh VMM pool and ends so close
// to the pool's mapped end that the last tile's over-read leaves mapped memory.
//
// The pool model (ggml-cuda.cu, ggml_cuda_pool_vmm): allocations are bumped from the pool base, each rounded up to
// 128 bytes; the mapped size grows in multiples of the allocation granularity G (2 MiB on the GPUs measured) to cover
// the bump pointer. In the ids branch of ggml_cuda_mul_mat_q the allocations are, in order: ids_src1 and ids_dst
// (ne12*n_used int32 each), expert_bounds (n_mats+1 int32), then src1_q8_1. So in a process whose pool is fresh, the
// mapped end is E rounded up to G, where E is the end of src1_q8_1, and the slack is that distance.
//
// The last tile reads J - c blocks past the data, c being the valid columns in the globally last tile (the last
// non-empty expert's column count mod J, or J). test_mul_mat_id routes each token to n_used distinct experts drawn
// uniformly, so with many more experts than rows per expert, c is small. The read leaves mapped memory when
// (J - c - pad) * 144 > slack.
//
// One more allocation can follow src1_q8_1: launch_mul_mat_q's stream-k fixup buffer (nsm*J*I floats, 12 MiB on a
// 188-SM GPU). It is allocated when the config uses stream-k and the tiles do not divide into the blocks launched,
// which on NVIDIA is ntiles when they fill >= 90 % of their waves and nsm otherwise. Then the over-read lands in
// the fixup buffer. So a shape counts only if no fixup buffer follows on any of the SM counts listed below.
//
// Build like mmq-padding-check.cu, from a llama.cpp checkout:
//   nvcc -std=c++17 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda mmq-crash-shape.cu -o mmq-crash-shape
#include "ggml/src/ggml-cuda/mmq.cuh"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
int ggml_cuda_get_device() { return 0; }
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }

static int launched_J(ggml_type t, bool fb, int cc, size_t smpbo, int64_t ncols_opt) {
    int nt_best = INT_MAX, J_launch = 0;
    for (int J = 8; J <= 128 && nt_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, cc, GGML_PREC_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int nt = (ncols_opt + c.J - 1) / c.J;
        if (nt < nt_best) { nt_best = nt; J_launch = c.J; }
    }
    return J_launch;
}
static int64_t a128(int64_t x) { return (x + 127) / 128 * 128; }

// SM counts: RTX 3090, A100, RTX 4090, H100 SXM, RTX 5090, RTX PRO 6000 Blackwell.
static const int NSMS[] = {82, 108, 128, 132, 170, 188};
static bool fixup_follows(const ggml_cuda_mmq_config & c, int64_t ncols_max, int64_t nrows_x, int64_t nchannels, int nsm) {
    if (!c.stream_k) return false;
    const int64_t ntiles = ((ncols_max + c.J - 1) / c.J) * ((nrows_x + c.I - 1) / c.I) * nchannels;
    const int64_t nwaves = (ntiles + nsm - 1) / nsm;
    const int64_t eff = 100 * ntiles / (nsm * nwaves);
    const int64_t nblocks = eff >= 90 ? ntiles : nsm;
    return ntiles % nblocks != 0;
}

// P(c <= cmax) for the globally last tile, by simulation of test_mul_mat_id's routing.
static double p_small_c(int ne12, int n_used, int n_mats, int J, int cmax, int trials = 4000) {
    std::vector<int> cnt(n_mats), perm(n_mats);
    unsigned s = 12345;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return (s >> 8); };
    int ok = 0;
    for (int tr = 0; tr < trials; ++tr) {
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int t = 0; t < ne12; ++t) {
            for (int i = 0; i < n_mats; ++i) perm[i] = i;
            for (int i = 0; i < n_used; ++i) { int j = i + rnd() % (n_mats - i); std::swap(perm[i], perm[j]); cnt[perm[i]]++; }
        }
        int e = n_mats - 1; while (e >= 0 && cnt[e] == 0) --e;
        const int c = cnt[e] % J == 0 ? J : cnt[e] % J;
        ok += c <= cmax;
    }
    return double(ok) / trials;
}

int main(int argc, char ** argv) {
    const int cc = GGML_CUDA_CC_BLACKWELL; const size_t smpbo = 101376;   // sm_120, RTX PRO 6000
    const int64_t G = 2 << 20;
    const int64_t max_weights = argc > 1 ? atoll(argv[1]) << 20 : 512ll << 20;   // MiB
    struct tt { ggml_type t; const char * n; double bpw; int qk; };
    const tt types[] = {{GGML_TYPE_Q4_0, "q4_0", 18.0/32, 32}, {GGML_TYPE_Q4_K, "q4_K", 144.0/256, 256}};
    const int ms[] = {576, 640, 1408, 1536, 2880, 4096};   // multiples of 128 are non-fallback
    const int n_mats_s[] = {64, 128, 256, 512, 1024};
    printf("type m fb n_mats n_used ne12 k J pad29941 E slack_bytes over_read_blocks(c=1) weights_MiB p(crash)\n");
    for (const tt & T : types) for (int m : ms) for (int n_mats : n_mats_s)
    for (int n_used = 1; n_used <= 10; ++n_used) for (int ne12 = 9; ne12 <= 127; ++ne12)
    for (int64_t k = 512; k <= 16384; k += 512) {
        const bool fb = m % 128 != 0;
        const int J = launched_J(T.t, fb, cc, smpbo, ne12);
        const int pad = ggml_cuda_mmq_get_J_max(T.t, fb, cc, ne12);
        if (J == 0 || J - 1 - pad <= 0) continue;                    // #29941 not short here
        const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(T.t, J, fb, cc, GGML_PREC_Q8);
        bool masked = false;
        for (int nsm : NSMS) masked = masked || fixup_follows(cfg, ne12, m, n_mats, nsm);
        if (masked) continue;
        const double weights = double(n_mats) * m * k * T.bpw;
        if (weights > max_weights) continue;
        const int64_t rows = int64_t(ne12) * n_used;
        const int64_t used_before = 2*a128(rows*4) + a128(int64_t(n_mats + 1)*4);
        const int64_t S = a128(rows * k * 36 / 32 + int64_t(pad) * 144);
        const int64_t E = used_before + S;
        const int64_t slack = (G - E % G) % G;
        // the read leaves the mapping when (J - c - pad)*144 > slack, i.e. for c < J - pad - slack/144
        const int cmax = J - pad - int(slack / 144) - 1;
        if (cmax < 3) continue;
        const double p = p_small_c(ne12, n_used, n_mats, J, cmax, 300);
        if (p < 0.99) continue;
        printf("%s %d %d %d %d %d %lld %d %d %lld %lld %d %.0f %.3f\n", T.n, m, fb, n_mats, n_used, ne12, (long long) k, J, pad,
               (long long) E, (long long) slack, J - 1 - pad, weights / (1 << 20), p);
    }
    return 0;
}
