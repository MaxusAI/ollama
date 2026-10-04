// DEBUG ONLY, never for upstream or for a shipped build.
//
// MMQ's MUL_MAT_ID path takes its src1 and ids_dst buffers from ggml's CUDA memory pool. The pool hands out
// suballocations of a much larger mapping, so a read a little past the end of either buffer lands in memory the
// process has already mapped: it neither faults nor shows up under compute-sanitizer, which only knows the pool's
// own allocation. This header replaces those two allocations so that such a read becomes observable.
//
//   MMQ_DEBUG_ALLOC=exact   each buffer gets its own cudaMalloc of exactly the size MMQ asked for, so
//                           compute-sanitizer memcheck reports a read past it. Needs the sanitizer to see anything.
//   MMQ_DEBUG_ALLOC=guard   each buffer is placed at the top of its own virtual-memory mapping with the next
//                           granule reserved but never mapped, so a read past the end faults on the device.
//                           Deterministic, and needs no sanitizer: the process dies with
//                           "an illegal memory access was encountered".
//
// guard mode is the stricter of the two and is what makes the difference between padding rules visible as a plain
// crash. The buffer's last byte sits at the last byte of the mapping, so the read has nowhere to land.
//
// Either mode takes an optional suffix naming one buffer, so a fault can be attributed: `guard:ids_dst`,
// `guard:src1`, `guard:src1_scale`, `exact:ids_dst`, ... Without a suffix all three are instrumented.

#pragma once

#include "common.cuh"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <cstdio>

enum mmq_dbg_mode { MMQ_DBG_OFF = 0, MMQ_DBG_EXACT = 1, MMQ_DBG_GUARD = 2 };
enum mmq_dbg_buf_id { MMQ_DBG_IDS_DST = 1, MMQ_DBG_SRC1 = 2, MMQ_DBG_SRC1_SCALE = 4 };

struct mmq_dbg_cfg {
    mmq_dbg_mode mode = MMQ_DBG_OFF;
    int          bufs = MMQ_DBG_IDS_DST | MMQ_DBG_SRC1 | MMQ_DBG_SRC1_SCALE;
};

static mmq_dbg_cfg mmq_dbg_cfg_init() {
    mmq_dbg_cfg c;
    const char * s = getenv("MMQ_DEBUG_ALLOC");
    if (s == nullptr || s[0] == '\0' || strcmp(s, "off") == 0) {
        return c;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%s", s);
    char * colon = strchr(buf, ':');
    if (colon) {
        *colon = '\0';
        const char * which = colon + 1;
        if (strcmp(which, "ids_dst") == 0) {
            c.bufs = MMQ_DBG_IDS_DST;
        } else if (strcmp(which, "src1") == 0) {
            c.bufs = MMQ_DBG_SRC1;
        } else if (strcmp(which, "src1_scale") == 0) {
            c.bufs = MMQ_DBG_SRC1_SCALE;   // native-FP4 y scales, read per whole tile by the stream-k fixup
        } else {
            fprintf(stderr, "MMQ_DEBUG_ALLOC: expected ids_dst, src1 or src1_scale after ':', got '%s'\n", which);
            exit(1);
        }
    }
    if (strcmp(buf, "exact") == 0) {
        c.mode = MMQ_DBG_EXACT;
    } else if (strcmp(buf, "guard") == 0) {
        c.mode = MMQ_DBG_GUARD;
    } else {
        fprintf(stderr, "MMQ_DEBUG_ALLOC: expected off, exact or guard, got '%s'\n", buf);
        exit(1);
    }
    return c;
}

static const mmq_dbg_cfg & mmq_dbg_get_cfg() {
    static const mmq_dbg_cfg c = mmq_dbg_cfg_init();
    return c;
}

// Non-zero when this buffer should come from mmq_dbg_alloc instead of the pool.
static mmq_dbg_mode mmq_dbg_get_mode(const mmq_dbg_buf_id id) {
    const mmq_dbg_cfg & c = mmq_dbg_get_cfg();
    return (c.bufs & id) ? c.mode : MMQ_DBG_OFF;
}

struct mmq_dbg_buf {
    void *      ptr   = nullptr; // what the kernel is given; its last byte is the mapping's last byte in guard mode
    CUdeviceptr base  = 0;       // guard mode: the reservation, one granule larger than the mapping
    size_t      msize = 0;       // guard mode: the mapped part of the reservation
    size_t      rsize = 0;       // guard mode: the whole reservation
};

// align: the alignment ptr must still satisfy. The smaller it is the tighter the guard, because the slack between
// the end of the buffer and the end of the mapping is what an over-read can get away with: at most align-1 bytes.
static mmq_dbg_buf mmq_dbg_alloc(const mmq_dbg_mode mode, const size_t size, const size_t align) {
    mmq_dbg_buf b;
    GGML_ASSERT(size > 0);

    if (mode == MMQ_DBG_EXACT) {
        CUDA_CHECK(cudaMalloc(&b.ptr, size));
        return b;
    }

    CUdevice dev;
    CU_CHECK(cuCtxGetDevice(&dev));

    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = dev;

    size_t gran = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    b.msize = ((size + gran - 1) / gran) * gran;
    b.rsize = b.msize + gran; // the extra granule is reserved so nothing else can map there, and left unmapped

    CU_CHECK(cuMemAddressReserve(&b.base, b.rsize, gran, 0, 0));

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, b.msize, &prop, 0));
    CU_CHECK(cuMemMap(b.base, b.msize, 0, handle, 0));
    CU_CHECK(cuMemRelease(handle)); // the mapping keeps it alive

    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(b.base, b.msize, &access, 1));

    // Put the end of the buffer at the end of the mapping, keeping ptr aligned.
    const size_t off = (b.msize - size) & ~(align - 1);
    b.ptr = (void *) (b.base + off);
    return b;
}

static void mmq_dbg_free(mmq_dbg_buf & b) {
    if (b.ptr == nullptr) {
        return;
    }
    if (b.base == 0) {
        CUDA_CHECK(cudaFree(b.ptr));
    } else {
        CU_CHECK(cuMemUnmap(b.base, b.msize));
        CU_CHECK(cuMemAddressFree(b.base, b.rsize));
    }
    b = mmq_dbg_buf();
}

// Stands in for ggml_cuda_pool_alloc<T> at the call sites, so the surrounding code is untouched.
template <typename T> struct mmq_dbg_ptr {
    T * ptr = nullptr;              // the dense branch passes src1_q8_1 as .ptr
    T * get() const { return ptr; } // the ids branch as .get()
};

// MMQ_DEBUG_PRINT=1: one line per MUL_MAT_ID call with the shape, the tile width the launch is about to pick, and
// how much padding this build actually reserved -- so the verdict for a rule comes from the build on the device
// rather than from a model of it. J_launch repeats mul_mat_q_switch_J's choice; it lives in a template that
// queries the device, so it cannot be called from here.
static bool mmq_dbg_print() {
    static const bool on = getenv("MMQ_DEBUG_PRINT") != nullptr;
    return on;
}

// How many blocks of padding src1 needs past its data, worst case.
//
// A tile's y load is `for (l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += nthreads) tile_y[l0 + tid] = by0[l0 + tid];` with no
// bound on l, so it reads GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int)) bytes from the tile's first
// column -- the same expression mmq_get_nbytes_shared() uses to size the shared-memory tile it copies into. When
// the last tile holds a single column, all but one block of that is past the data. So the requirement is one
// block less than that padded tile, which is more than J blocks whenever nthreads*sizeof(int) does not divide
// J*sizeof(block_q8_1_mmq).
static int mmq_src1_need_blocks(const int J, const int nthreads) {
    if (J == 0) {
        return 0;
    }
    const size_t bytes = GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int));
    return (int) ((bytes + sizeof(block_q8_1_mmq) - 1) / sizeof(block_q8_1_mmq)) - 1;
}

// ids_dst_n < 0 means the dense branch, which has no ids_dst.
static void mmq_dbg_report(
        const char * branch,
        const ggml_type type, const bool fallback, const ggml_prec prec_src1, const int cc, const size_t smpbo,
        const int64_t ne11, const int64_t ne12, const int64_t n_expert_used, const int64_t ne02,
        const size_t pad_bytes, const int64_t ids_dst_n, const int64_t ids_dst_entries) {
    if (!mmq_dbg_print()) {
        return;
    }

    const bool    dense     = ids_dst_n < 0;
    const bool    rdna      = GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc);
    const int64_t ncols_opt = dense ? ne11 : (rdna ? (ne12*n_expert_used + ne02 - 1) / ne02 : ne12);

    int J_launch = 0, ntiles_best = INT_MAX, nthreads = 0;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) {
            continue;
        }
        const int ntiles = (ncols_opt + c.J - 1) / c.J;
        if (ntiles < ntiles_best) {
            ntiles_best = ntiles;
            J_launch    = c.J;
            nthreads    = c.nthreads;
        }
    }

    const int pad_blocks = (int) (pad_bytes / sizeof(block_q8_1_mmq));
    int need_src1 = mmq_src1_need_blocks(J_launch, nthreads);
    if (dense && J_launch > 0) {
        // the dense branch's last tile holds ne11 - (ntiles - 1)*J columns, all but one of which is data
        const int64_t ntiles = (ne11 + J_launch - 1) / J_launch;
        need_src1 -= (int) (ne11 - (ntiles - 1)*J_launch - 1);
    }

    fprintf(stderr, "MMQ_%-5s %-5s fb=%d prec=%s ne11=%5ld ne12=%5ld n_used=%3ld ne02=%4ld | J=%3d nthreads=%3d"
            " | src1 need=%3d pad=%3d %-7s",
            branch, ggml_type_name(type), (int) fallback,
            prec_src1 == GGML_PREC_Q4  ? "q4"  : prec_src1 == GGML_PREC_Q8        ? "q8" :
            prec_src1 == GGML_PREC_F32 ? "f32" : prec_src1 == GGML_PREC_UNDEFINED ? "und" : "???",
            (long) ne11, (long) ne12, (long) n_expert_used, (long) ne02, J_launch, nthreads,
            need_src1, pad_blocks, pad_blocks >= need_src1 ? "covered" : "SHORT");
    if (dense) {
        fprintf(stderr, " | no ids_dst\n");
    } else {
        const int need_ids = J_launch - 1; // the last expert's last tile can hold a single row
        fprintf(stderr, " | ids_dst n=%7ld need=%3d pad=%3ld %-7s\n",
                (long) ids_dst_entries, need_ids, (long) (ids_dst_n - ids_dst_entries),
                ids_dst_n - ids_dst_entries >= need_ids ? "covered" : "SHORT");
    }
}
