// One MUL_MAT_ID with a routing chosen by hand, on a GPU backend, checked against the CPU backend.
//
// test_mul_mat_id draws each token's experts uniformly, so on RDNA3 -- which picks the tile width J from the mean
// rows per expert -- a shape that launches a wide tile also gives its last expert about that many rows, and the
// last tile never holds the single column that the padding check's worst case needs. This program builds that
// worst case directly: token 0 is the only row of the last expert, every other token avoids it.
//
//   mmq-route <type> <n_exp> <n_used> <n_tokens> <m> <k> <bcast 0|1> <route last1|uniform> [backend, default ROCm0]
//
// Exit 0 when the GPU result matches the CPU's (NMSE < 5e-4); a read that leaves mapped memory aborts the process.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static ggml_type type_from_name(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && !strcmp(n, s)) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s);
    exit(2);
}

static std::vector<float> run(ggml_backend_t be, ggml_type type, int64_t n_exp, int64_t n_used, int64_t n_tok,
                              int64_t m, int64_t k, bool bcast, const std::vector<uint8_t> & qa,
                              const std::vector<float> & b, const std::vector<int32_t> & ids) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta   = ggml_new_tensor_3d(ctx, type, k, m, n_exp);
    ggml_tensor * tb   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, bcast ? 1 : n_used, n_tok);
    ggml_tensor * tids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * out  = ggml_mul_mat_id(ctx, ta, tb, tids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(ta, qa.data(), 0, ggml_nbytes(ta));
    ggml_backend_tensor_set(tb, b.data(), 0, ggml_nbytes(tb));
    ggml_backend_tensor_set(tids, ids.data(), 0, ggml_nbytes(tids));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(3);
    }
    std::vector<float> r(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s <type> <n_exp> <n_used> <n_tokens> <m> <k> <bcast 0|1> <last1|uniform> [backend]\n", argv[0]);
        return 2;
    }
    const ggml_type type = type_from_name(argv[1]);
    const int64_t n_exp = atoll(argv[2]), n_used = atoll(argv[3]), n_tok = atoll(argv[4]);
    const int64_t m = atoll(argv[5]), k = atoll(argv[6]);
    const bool bcast = atoi(argv[7]) != 0;
    const std::string route = argv[8];
    const char * be_name = argc > 9 ? argv[9] : "ROCm0";
    if (n_used > n_exp || (route == "last1" && n_used >= n_exp)) {
        fprintf(stderr, "last1 needs n_used < n_exp\n");
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(be_name);
    if (!dev) { fprintf(stderr, "no backend device %s\n", be_name); return 2; }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);

    std::vector<float> a((size_t) (k*m*n_exp));
    for (float & x : a) x = U(rng);
    std::vector<uint8_t> qa((size_t) (ggml_row_size(type, k)*m*n_exp));
    ggml_quantize_chunk(type, a.data(), qa.data(), 0, m*n_exp, k, nullptr);

    std::vector<float> b((size_t) (k*(bcast ? 1 : n_used)*n_tok));
    for (float & x : b) x = U(rng);

    // ids[t][j]: token t's j-th expert, distinct within a token
    std::vector<int32_t> ids((size_t) (n_used*n_tok));
    std::vector<int32_t> pool(n_exp);
    std::iota(pool.begin(), pool.end(), 0);
    for (int64_t t = 0; t < n_tok; ++t) {
        if (route == "last1") {
            // token 0 takes the last expert; nobody else does
            std::vector<int32_t> rest(pool.begin(), pool.end() - 1);
            std::shuffle(rest.begin(), rest.end(), rng);
            for (int64_t j = 0; j < n_used; ++j) {
                ids[t*n_used + j] = (t == 0 && j == 0) ? (int32_t) (n_exp - 1) : rest[t == 0 ? j - 1 : j];
            }
        } else {
            std::vector<int32_t> p = pool;
            std::shuffle(p.begin(), p.end(), rng);
            for (int64_t j = 0; j < n_used; ++j) ids[t*n_used + j] = p[j];
        }
    }
    std::vector<int64_t> per_exp(n_exp, 0);
    for (int32_t e : ids) per_exp[e]++;
    int64_t last = n_exp - 1;
    while (last > 0 && per_exp[last] == 0) --last;
    printf("%s n_exp=%ld n_used=%ld tokens=%ld m=%ld k=%ld bcast=%d route=%s: rows=%ld, mean rows/expert=%.2f, "
           "last non-empty expert %ld holds %ld rows\n",
           ggml_type_name(type), (long) n_exp, (long) n_used, (long) n_tok, (long) m, (long) k, (int) bcast,
           route.c_str(), (long) (n_used*n_tok), (double) (n_used*n_tok)/n_exp, (long) last, (long) per_exp[last]);
    fflush(stdout);

    const std::vector<float> rg = run(gpu, type, n_exp, n_used, n_tok, m, k, bcast, qa, b, ids);
    const std::vector<float> rc = run(cpu, type, n_exp, n_used, n_tok, m, k, bcast, qa, b, ids);
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < rg.size(); ++i) { err += (rg[i] - rc[i])*(rg[i] - rc[i]); ref += rc[i]*rc[i]; }
    const double nmse = err / std::max(ref, 1e-30);
    printf("NMSE vs CPU = %.3e -> %s\n", nmse, nmse < 5e-4 ? "OK" : "FAIL");

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return nmse < 5e-4 ? 0 : 1;
}
