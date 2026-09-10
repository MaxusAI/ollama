#!/usr/bin/env python
"""Drive the sm120 kernels (built by build.sh into .so files) from torch via ctypes.

Two kernels:
  --kind bf16   libsm120_bf16_dense.so : hand-instantiated sm120 TMA mainloop, bf16 x bf16 (the ceiling)
  --kind nvfp4  libsm120_nvfp4_bf16.so : the mixed-input derivative, bf16 x NVFP4 (e2m1 + e4m3 per 16)

Every run validates against an fp32 reference computed from the exact bf16 inputs (for nvfp4 the
dequantised weight is exactly representable in bf16, so the reference is exact up to fp32 summation),
then times with triton.testing.do_bench (median). cuBLAS bf16 (torch.matmul on the dequantised weight,
i.e. "not quantising at all") is timed in the same process on the same tensors so both numbers share
whatever contention the GPU is under. The GPU is shared with production; timings are contended.

Interpreter: /opt/conda/envs/python3.10_env_gemma4_cu128/bin/python (torch 2.7.1+cu128, triton 3.3.1)
"""
import argparse, ctypes, os, subprocess, sys, time
import torch
import triton.testing as tt

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, K, N) real layer shapes; M in {2048, 4096}
SHAPES = [
    ("gemma4-31b gate", 5376, 21504),
    ("gemma4-31b down", 21504, 5376),
    ("qwen3.8 gate",    5120, 17408),
    ("qwen3.8 down",    17408, 5120),
]

E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def gpu_state():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader", "-i", "0"], text=True)
        return out.strip()
    except Exception as e:  # pragma: no cover
        return f"nvidia-smi unavailable: {e}"


# ---------------------------------------------------------------------------------------------
# NVFP4 layouts
# ---------------------------------------------------------------------------------------------

def mlx_native_pack(codes):
    """Logical e2m1 codes [N, K] (uint8, 0..15) -> MLX/ModelOpt native uint32 [N, K/8]: element i at bits [4i+3:4i]."""
    N, K = codes.shape
    c = codes.view(N, K // 8, 8).to(torch.int64)
    shifts = (4 * torch.arange(8, device=codes.device)).view(1, 1, 8)
    w = (c << shifts).sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)   # bit pattern as int32


def mlx_native_unpack(w_u32, K):
    """MLX native uint32 [N, K/8] -> logical codes [N, K] uint8."""
    w = w_u32.to(torch.int64) & 0xFFFFFFFF
    shifts = (4 * torch.arange(8, device=w.device)).view(1, 1, 8)
    return ((w.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8).reshape(w.shape[0], K)


def preshuffle_weights(codes):
    """Logical codes [N, K] -> fragment-native Bp [N/8, K] int32 (see sm120_mma_tma_mixed_input.hpp).

    word (nt, kb*32 + lane), lane = 4*i + c (i = n%8, c = quad lane), nibble j = code[8nt+i][32kb + kk(c, j)],
    kk(c, j) = 16*(j//4) + (2c + j%4 if j%4 < 2 else 2c + 8 + j%4 - 2)  = the mma.m16n8k16 B-fragment k slots.
    """
    N, K = codes.shape
    NT, KB = N // 8, K // 32
    dev = codes.device
    q = codes.view(NT, 8, KB, 32)                                   # [nt, i, kb, k%32]
    c = torch.arange(4, device=dev).view(4, 1)
    j = torch.arange(8, device=dev).view(1, 8)
    jj = j % 4
    kk = 16 * (j // 4) + torch.where(jj < 2, 2 * c + jj, 2 * c + 8 + (jj - 2))   # [4, 8]
    g = q[:, :, :, kk]                                              # [nt, i, kb, c, j]
    words = (g.to(torch.int64) << (4 * j).view(1, 1, 1, 1, 8)).sum(-1)   # [nt, i, kb, c]
    words = words.permute(0, 2, 1, 3).reshape(NT, KB * 32)          # [nt, kb, i, c] -> lane = 4i + c
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()


def preshuffle_scales(scale_bits):
    """e4m3 scale bits [N, K/16] uint8 -> Sp [N/8, K/2] uint8: byte (nt, kb*16 + (n%8)*2 + s)."""
    N, G = scale_bits.shape
    NT, KB = N // 8, G // 2
    s = scale_bits.view(NT, 8, KB, 2).permute(0, 2, 1, 3).reshape(NT, KB * 16)
    return s.contiguous()


def make_nvfp4(N, K, dev, seed=0):
    """Random NVFP4 weight: codes, e4m3 scale bits, exact bf16 dequantised weight, and both packed layouts."""
    g = torch.Generator(device=dev); g.manual_seed(seed)
    codes = torch.randint(0, 16, (N, K), device=dev, dtype=torch.uint8, generator=g)
    sc = torch.empty(N, K // 16, device=dev).uniform_(0.02, 1.0, generator=g).to(torch.float8_e4m3fn)
    sc_bits = sc.view(torch.uint8)
    table = torch.tensor(E2M1_TABLE, device=dev)
    W = table[codes.long()] * sc.float().repeat_interleave(16, dim=1)    # fp32, exact
    W_bf16 = W.to(torch.bfloat16)
    assert torch.equal(W_bf16.float(), W), "dequantised NVFP4 weight must be exactly representable in bf16"
    native = mlx_native_pack(codes)
    assert torch.equal(mlx_native_unpack(native, K), codes)
    Bp = preshuffle_weights(mlx_native_unpack(native, K))     # i.e. the repack path from MLX's layout
    Sp = preshuffle_scales(sc_bits)
    return codes, sc_bits, W_bf16, native, Bp, Sp


# ---------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------

class Lib:
    def __init__(self, name, kind):
        self.lib = ctypes.CDLL(os.path.join(HERE, name))
        self.kind = kind
        info = [ctypes.c_int() for _ in range(5)]
        getattr(self.lib, f"sm120_{kind}_gemm_info")(*[ctypes.byref(i) for i in info])
        self.stages, self.tm, self.tn, self.tk, self.smem = [i.value for i in info]

    def desc(self):
        return f"{self.kind}: stages={self.stages} tile={self.tm}x{self.tn}x{self.tk} smem={self.smem} B"

    def bf16(self, A, B, D):
        err = ctypes.create_string_buffer(256)
        rc = self.lib.sm120_bf16_gemm(ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(B.data_ptr()), ctypes.c_void_p(D.data_ptr()),
                                      A.shape[0], B.shape[0], A.shape[1],
                                      ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), err, 256)
        if rc != 0:
            raise RuntimeError(f"sm120_bf16_gemm rc={rc}: {err.value.decode()}")

    def nvfp4(self, A, Bp, Sp, D, N, alpha=1.0):
        err = ctypes.create_string_buffer(256)
        rc = self.lib.sm120_nvfp4_gemm(ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(Bp.data_ptr()), ctypes.c_void_p(Sp.data_ptr()),
                                       ctypes.c_void_p(D.data_ptr()), A.shape[0], N, A.shape[1], ctypes.c_float(alpha),
                                       ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), err, 256)
        if rc != 0:
            raise RuntimeError(f"sm120_nvfp4_gemm rc={rc}: {err.value.decode()}")


def check(D, ref, label):
    D32 = D.float()
    err = (D32 - ref).abs()
    scale = ref.abs().max().item()
    rel_max = err.max().item() / scale
    rel_mean = (err / (ref.abs() + 1e-3 * scale)).mean().item()
    nbad = int((err > 5e-2 * scale).sum().item())
    print(f"    {label:<24} max|err|/max|ref| = {rel_max:.3e}   mean rel = {rel_mean:.3e}   #>5% = {nbad}")
    return rel_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="bf16", choices=["bf16", "nvfp4"])
    ap.add_argument("--lib", default=None)
    ap.add_argument("--dense-lib", default="libsm120_bf16_dense.so", help="dense sm120 bf16 kernel to time alongside nvfp4")
    ap.add_argument("--ms", default="2048,4096")
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="only validate at small shapes")
    ap.add_argument("--alpha", type=float, default=1.0, help="global scale for nvfp4 (epilogue alpha)")
    args = ap.parse_args()
    lib_name = args.lib or {"bf16": "libsm120_bf16_dense.so", "nvfp4": "libsm120_nvfp4_bf16.so"}[args.kind]

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
    print(f"GPU before: {gpu_state()}")
    lib = Lib(lib_name, args.kind)
    print(f"kernel {lib_name}: {lib.desc()}")
    dense = None
    if args.kind == "nvfp4" and not args.quick and os.path.exists(os.path.join(HERE, args.dense_lib)):
        dense = Lib(args.dense_lib, "bf16")
        print(f"dense  {args.dense_lib}: {dense.desc()}")

    if args.quick:
        shapes = [("small", 512, 768), ("small2", 2048, 1024), ("odd-M", 640, 384)]
        Ms = {"small": [256], "small2": [300], "odd-M": [77]}
    else:
        shapes = SHAPES
        Ms = {name: [int(m) for m in args.ms.split(",")] for name, _, _ in shapes}

    rows = []
    for name, K, N in shapes:
        for M in Ms[name]:
            A = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            D = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            print(f"  {name}  M={M} K={K} N={N}")
            if args.kind == "bf16":
                B = (torch.randn(N, K, device=dev) * 0.5).to(torch.bfloat16)
                ref = A.float() @ B.float().T
                run_k = lambda: lib.bf16(A, B, D)
            else:
                codes, sc_bits, B, native, Bp, Sp = make_nvfp4(N, K, dev)
                ref = args.alpha * (A.float() @ B.float().T)
                run_k = lambda: lib.nvfp4(A, Bp, Sp, D, N, args.alpha)
            run_k(); torch.cuda.synchronize()
            e_k = check(D, ref, f"sm120 {args.kind} kernel")
            e_c = check(args.alpha * (A @ B.T), ref, "cuBLAS bf16 (dequant W)")
            if e_k > 2e-2:
                print("    FAILED numerics; not timing"); rows.append((name, M, K, N, None, None, None)); continue
            if args.quick:
                continue
            flops = 2.0 * M * N * K
            ms_k = tt.do_bench(run_k, warmup=args.warmup, rep=args.rep, return_mode="median")
            ms_c = tt.do_bench(lambda: torch.matmul(A, B.T, out=D), warmup=args.warmup, rep=args.rep, return_mode="median")
            tf_k, tf_c = flops / ms_k / 1e9, flops / ms_c / 1e9
            tf_d = None
            if dense is not None:
                ms_d = tt.do_bench(lambda: dense.bf16(A, B, D), warmup=args.warmup, rep=args.rep, return_mode="median")
                tf_d = flops / ms_d / 1e9
            line = f"    sm120 {args.kind:<5} {ms_k:8.3f} ms {tf_k:7.1f} TFLOP/s | cuBLAS bf16 {ms_c:8.3f} ms {tf_c:7.1f} TFLOP/s | ratio {tf_k/tf_c:.2f}"
            if tf_d is not None:
                line += f" | dense sm120 bf16 {tf_d:7.1f} TFLOP/s ratio {tf_k/tf_d:.2f}"
            print(line)
            rows.append((name, M, K, N, tf_k, tf_c, tf_d))
            del A, B, D, ref
    print(f"GPU after: {gpu_state()}")
    if not args.quick:
        hdr = f"| shape | M | K | N | sm120 {args.kind} TFLOP/s | cuBLAS bf16 TFLOP/s | ratio |" + (" dense sm120 bf16 TFLOP/s | ratio |" if dense else "")
        print("\n" + hdr)
        print("|---|---|---|---|---|---|---|" + ("---|---|" if dense else ""))
        for name, M, K, N, tk, tc, td in rows:
            if tk is None:
                print(f"| {name} | {M} | {K} | {N} | FAIL | | |")
            else:
                extra = f" {td:.1f} | {tk/td:.2f} |" if td is not None else ""
                print(f"| {name} | {M} | {K} | {N} | {tk:.1f} | {tc:.1f} | {tk/tc:.2f} |{extra}")


if __name__ == "__main__":
    main()
