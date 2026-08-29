#!/usr/bin/env python3
"""Static fp16-safety audit for a vision tower's matmul weights.

For every F16 weight W used in a plain (non-PREC_F32) matmul, the CUDA path
forms products in fp16.  A single input channel c with
    |x_c| > 65504 / max_row |W[row, c]|
produces inf on one multiply -- before any accumulation -- which NaNs the whole
dot product.  This audit reports that threshold per tensor, so a model can be
checked WITHOUT a GPU, without inference, and without a trigger image.

usage: fp16_audit.py <model.gguf> [activation_ceiling]
"""
import struct, sys, numpy as np
FP16_MAX = 65504.0
CEIL = float(sys.argv[2]) if len(sys.argv) > 2 else 50688.0   # measured peak, #214

f = open(sys.argv[1], "rb"); assert f.read(4) == b"GGUF"
struct.unpack("<I", f.read(4)); nt, nkv = struct.unpack("<QQ", f.read(16))
def rstr():
    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
def rval(t):
    if t in (0,1): return struct.unpack("<b" if t==0 else "<B", f.read(1))[0]
    if t in (2,3): return struct.unpack("<h" if t==2 else "<H", f.read(2))[0]
    if t in (4,5): return struct.unpack("<i" if t==4 else "<I", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return struct.unpack("<?", f.read(1))[0]
    if t == 8: return rstr()
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12)); return [rval(et) for _ in range(n)]
    if t in (10,11): return struct.unpack("<q" if t==10 else "<Q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(t)
align = 32
for _ in range(nkv):
    k = rstr(); t, = struct.unpack("<I", f.read(4)); v = rval(t)
    if k == "general.alignment": align = v
infos = {}
for _ in range(nt):
    name = rstr(); nd, = struct.unpack("<I", f.read(4))
    dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
    tt, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8))
    infos[name] = (dims, tt, off)
base = (f.tell() + align - 1) // align * align

rows, unsafe = [], 0
for name, (dims, tt, off) in sorted(infos.items()):
    if tt != 1 or not name.startswith("v.") or name.endswith(".bias"): continue
    if len(dims) != 2: continue                      # skip conv kernels
    n = int(np.prod(dims))
    f.seek(base + off)
    W = np.frombuffer(f.read(n*2), dtype=np.float16).reshape(tuple(reversed(dims))).astype(np.float32)
    colmax = np.abs(W).max(axis=0)
    thr = FP16_MAX / np.maximum(colmax, 1e-9)
    bad = int((thr < CEIL).sum())
    unsafe += bad > 0
    rows.append((name, W.shape, float(np.abs(W).max()), float(thr.min()), bad, len(thr)))

print(f"fp16 matmul safety audit — activation ceiling {CEIL:,.0f} (fp16 max {FP16_MAX:,.0f})\n")
print(f"{'tensor':<34}{'shape':>14}{'|w|max':>8}{'min |x| to inf':>16}{'unsafe ch':>12}")
print("-" * 86)
for name, shape, wmax, tmin, bad, tot in rows:
    flag = "  <-- UNSAFE" if bad else ""
    print(f"{name:<34}{str(shape):>14}{wmax:>8.3f}{tmin:>16,.0f}{f'{bad}/{tot}':>12}{flag}")
print(f"\n{unsafe} of {len(rows)} F16 matmul weights can produce inf at |x| = {CEIL:,.0f}")
print("Any of these running WITHOUT GGML_PREC_F32 is a latent degenerate-decode source.")
