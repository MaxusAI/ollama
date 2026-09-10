#!/usr/bin/env python
"""A/B several builds of the mixed-input kernel in ONE process: per shape, every version is validated, then
timed round-robin (kernel_1, kernel_2, ..., cuBLAS, dense) R times; the median of per-round medians is
reported. Interleaving is what makes versions comparable on a GPU whose contention drifts by 2x within minutes.

  python compare_versions.py --libs libsm120_nvfp4_bf16.so,libsm120_nvfp4_v3e.so,libsm120_nvfp4_v3.so,libsm120_nvfp4_v4.so
"""
import argparse, os, statistics, sys
import torch
import triton.testing as tt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_sm120 as B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", required=True)
    ap.add_argument("--dense-lib", default="libsm120_bf16_dense.so")
    ap.add_argument("--ms", default="2048,4096")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--swizzle", type=int, default=4)
    args = ap.parse_args()
    B.RASTER, B.SWIZZLE = 0, args.swizzle
    dev = torch.device("cuda:0")
    libs = [B.Lib(n, "nvfp4") for n in args.libs.split(",")]
    dense = B.Lib(args.dense_lib, "bf16")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  GPU before: {B.gpu_state()}")
    for l, n in zip(libs, args.libs.split(",")):
        print(f"  {n}: {l.desc()}")
    print(f"  dense: {dense.desc()}   rounds={args.rounds} swizzle={args.swizzle} rep={args.rep}ms")
    shapes = [t for t in B.SHAPES if not args.shapes or any(f in t[0] for f in args.shapes.split(","))]
    table = []
    for name, K, N in shapes:
        for M in [int(m) for m in args.ms.split(",")]:
            A = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            D = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            codes, sc_bits, W, native, Bp, Sp8 = B.make_nvfp4(N, K, dev)
            Sp16 = B.preshuffle_scales(sc_bits, as_bf16=True)
            ref = A.float() @ W.float().T
            print(f"  {name}  M={M} K={K} N={N}")
            fns = []
            ok = True
            for l, n in zip(libs, args.libs.split(",")):
                Sp = Sp16 if l.scale_bf16 else Sp8
                fn = (lambda l=l, Sp=Sp: l.nvfp4(A, Bp, Sp, D, N, 1.0))
                fn(); torch.cuda.synchronize()
                e = B.check(D, ref, n.replace("libsm120_nvfp4_", "").replace(".so", ""))
                ok &= e < 2e-2
                fns.append((n, fn))
            fns.append(("cuBLAS bf16", lambda: torch.matmul(A, W.T, out=D)))
            fns.append(("dense sm120 bf16", lambda: dense.bf16(A, W, D)))
            if not ok:
                print("    numerics FAILED for a version; skipping timing"); continue
            times = {n: [] for n, _ in fns}
            for _ in range(args.rounds):
                for n, fn in fns:
                    times[n].append(tt.do_bench(fn, warmup=25, rep=args.rep, return_mode="median"))
            flops = 2.0 * M * N * K
            row = {"shape": name, "M": M}
            for n, _ in fns:
                med = statistics.median(times[n]); best = min(times[n])
                row[n] = (flops / med / 1e9, flops / best / 1e9)
                print(f"    {n:<28} median {med:8.3f} ms  {flops/med/1e9:7.1f} TFLOP/s   best {flops/best/1e9:7.1f}   rounds {[round(x,3) for x in times[n]]}")
            table.append(row)
            del A, D, W, ref
    print(f"GPU after: {B.gpu_state()}")
    names = [n for n, _ in fns]
    print("\n| shape | M | " + " | ".join(n.replace("libsm120_nvfp4_", "").replace(".so", "") + " (median / best TFLOP/s)" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for r in table:
        print(f"| {r['shape']} | {r['M']} | " + " | ".join(f"{r[n][0]:.0f} / {r[n][1]:.0f}" for n in names) + " |")

if __name__ == "__main__":
    main()
