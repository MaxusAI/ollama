"""Ceiling measurement for a *streaming* nvfp4 x bf16 GEMM on sm120.

MLX runs `SM80_16x8x16_F32BF16BF16F32_TN` with Ampere-era tiling and reaches
55-73 TFLOP/s at 4096 rows on the fork's layer shapes; plain bf16 cuBLAS on the
same shapes reaches 100-160. This asks what a *well-tiled* mixed-input kernel --
4-bit weights unpacked in registers, bf16 activations, bf16 tensor cores, no
activation quantisation and no full-size dequantised copy in memory -- can reach
on this chip. That number is the ceiling for the "make the streaming kernel
fast" option, because the 4-bit tensor cores need *both* operands in 4-bit.

Deviations from a production kernel, both noted in the report:
  * per-16 scales are held as bf16 rather than decoded from e4m3 in-kernel
    (one extra cheap op per 16 values);
  * weights are packed with the two nibbles K/2 apart, so both halves of the
    K loop load contiguously. A real kernel pre-shuffles similarly (Marlin does).
"""
import os, sys, torch, triton, triton.language as tl

# Pin the PTX ISA level (e.g. 87 for CUDA 12.8) so a newer ptxas can be A/B'd on identical PTX:
# Triton 3.3.1 cannot derive a PTX version from a CUDA 13 ptxas and raises instead.
PTX_VERSION = int(os.environ.get("TRITON_FORCE_PTX_VERSION", "0")) or None

E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])          # magnitudes by (e,m)
BOUNDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])      # round-to-nearest midpoints


def quantise_nvfp4(w: torch.Tensor, group: int = 16):
    """w [N, K] -> packed uint8 [N, K//2], scales bf16 [N, K//group]."""
    n, k = w.shape
    g = w.float().reshape(n, k // group, group)
    scale = g.abs().amax(-1).clamp(min=1e-8) / 6.0
    q = (g / scale[..., None]).clamp(-6, 6)
    mag = torch.bucketize(q.abs(), BOUNDS.to(q.device)).to(torch.uint8)   # 0..7, on device
    code = (mag | ((q < 0).to(torch.uint8) << 3)).reshape(n, k)
    lo, hi = code[:, : k // 2], code[:, k // 2 :]                          # K/2-apart packing
    packed = (lo | (hi << 4)).contiguous().cuda()
    return packed, scale.to(torch.bfloat16).contiguous(), _dequantise(packed, scale, k, group)


def _dequantise(packed, scale, k, group):
    """Exact reference: rebuild bf16 weights from the packed form."""
    n = packed.shape[0]
    lo = (packed & 0xF).int()
    hi = ((packed >> 4) & 0xF).int()
    code = torch.cat([lo, hi], dim=1)
    sign, e, m = (code >> 3) & 1, (code >> 1) & 3, code & 1
    mag = torch.where(e == 0, m.float() * 0.5, (1 + 0.5 * m.float()) * torch.exp2((e - 1).float()))
    val = torch.where(sign == 1, -mag, mag)
    s = scale.float().cuda().repeat_interleave(group, dim=1)
    return (val * s).to(torch.bfloat16)


def _configs():
    out = []
    for bm, bn, bk in [(128, 128, 64), (128, 256, 64), (256, 128, 64), (128, 128, 128), (64, 256, 64), (128, 64, 128)]:
        for s, w in [(3, 8), (4, 8), (4, 4), (5, 4)]:
            out.append(triton.Config({"BM": bm, "BN": bn, "BK2": bk}, num_stages=s, num_warps=w))
    return out


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def mixed_gemm(a_ptr, b_ptr, s_ptr, c_ptr, M, N, K,
               sa_m, sa_k, sb_n, sb_k, ss_n, ss_g, sc_m, sc_n,
               GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK2: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    half = K // 2
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, half, BK2):
        i = k0 + tl.arange(0, BK2)
        # activations: the two nibble halves are K/2 apart, so both loads are contiguous
        a_lo = tl.load(a_ptr + offs_m[:, None] * sa_m + i[None, :] * sa_k,
                       mask=(offs_m[:, None] < M) & (i[None, :] < half), other=0.0)
        a_hi = tl.load(a_ptr + offs_m[:, None] * sa_m + (half + i)[None, :] * sa_k,
                       mask=(offs_m[:, None] < M) & ((half + i)[None, :] < K), other=0.0)
        # one packed byte carries both
        bp = tl.load(b_ptr + offs_n[None, :] * sb_n + i[:, None] * sb_k,
                     mask=(offs_n[None, :] < N) & (i[:, None] < half), other=0)
        lo, hi = (bp & 0xF).to(tl.int32), ((bp >> 4) & 0xF).to(tl.int32)

        s_lo = tl.load(s_ptr + offs_n[None, :] * ss_n + (i[:, None] // GROUP) * ss_g,
                       mask=(offs_n[None, :] < N), other=0.0)
        s_hi = tl.load(s_ptr + offs_n[None, :] * ss_n + ((half + i)[:, None] // GROUP) * ss_g,
                       mask=(offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a_lo, (_decode(lo) * s_lo.to(tl.float32)).to(tl.bfloat16))
        acc += tl.dot(a_hi, (_decode(hi) * s_hi.to(tl.float32)).to(tl.bfloat16))

    tl.store(c_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _decode(code):
    """e2m1 nibble -> float32 magnitude with sign, in registers."""
    sign, e, m = (code >> 3) & 1, (code >> 1) & 3, (code & 1).to(tl.float32)
    base = tl.where(e == 0, 0.0, tl.exp2((e - 1).to(tl.float32)))
    mag = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * base)
    return tl.where(sign == 1, -mag, mag)


def run(a, packed, scale, k, group=16):
    m, n = a.shape[0], packed.shape[0]
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(m, META["BM"]), triton.cdiv(n, META["BN"]))
    extra = {"ptx_version": PTX_VERSION} if PTX_VERSION else {}
    mixed_gemm[grid](a, packed, scale, c, m, n, k,
                     a.stride(0), a.stride(1), packed.stride(0), packed.stride(1),
                     scale.stride(0), scale.stride(1), c.stride(0), c.stride(1), GROUP=group, **extra)
    return c


SHAPES = [("gemma4-31b gate", 5376, 21504), ("gemma4-31b down", 21504, 5376),
          ("qwen3.8 gate", 5120, 17408), ("qwen3.8 down", 17408, 5120)]

if __name__ == "__main__":
    torch.manual_seed(0)
    ms = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2048", "4096"])]
    print(f"{torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}  triton {triton.__version__}")
    print(f"{'shape':18} {'M':>5} {'triton mixed':>13} {'bf16 cuBLAS':>12} {'vs cuBLAS':>10} {'rel err':>9}")
    for name, k, n in SHAPES:
        print(f"  [{name}] quantising K={k} N={n} ...", flush=True)
        w = (torch.randn(n, k, device="cuda") * 0.02)
        packed, scale, wdq = quantise_nvfp4(w, 16)
        torch.cuda.synchronize()
        for m in ms:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            got = run(a, packed, scale, k)
            ref = a @ wdq.T                                   # same weights, cuBLAS bf16
            err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
            t_tri = triton.testing.do_bench(lambda: run(a, packed, scale, k), warmup=50, rep=200, return_mode="median")
            t_ref = triton.testing.do_bench(lambda: a @ wdq.T, warmup=50, rep=200, return_mode="median")
            f = lambda t: 2 * m * n * k / (t * 1e-3) / 1e12
            print(f"{name:18} {m:5} {f(t_tri):8.1f} TF/s {f(t_ref):8.1f} TF/s {f(t_tri)/f(t_ref):9.2f}x {err:9.2e}")
        del w, packed, scale, wdq; torch.cuda.empty_cache()
