#!/usr/bin/env python3
"""List a served MLX model's quantized weights with their logical K, and flag every K that is not a multiple of 32.

Why this exists (2026-09-18, #312): ml-explore/mlx#3912 fixed a Metal kernel that silently corrupted every matrix-sized
nvfp4 matmul whose quantized dimension satisfied K % 32 == 16. The gemma4 vision tower's mlp.down_proj has K = 4304,
so 26b and 31b moved across the v0.34.1 fold on Metal while 12b, whose vision weights have K = 3840 and 6912, did not.
The pin move had been folded as an opaque range; the fix's selection condition was never checked against the shapes
the fork serves. This tool is the shape half of that check. The other half is reading the pin range's kernel commits
(gh api repos/ml-explore/mlx/compare/<old>...<new>) for their conditions.

Reads the store directly (safetensors headers only, no tensor data), so it runs on any host with the models dir:

    OLLAMA_MODELS=/path/to/models python3 quant_dims.py gemma4:31b-nvfp4 gemma4:26b-nvfp4 gemma4:12b-nvfp4
    python3 quant_dims.py --unaligned gemma4:31b-nvfp4      # only the rows with K % 32 != 0

K is derived from the .scale sibling (scale columns × group size, group size from the blob's __metadata__ or the
quant type's default) and cross-checked against the packed weight (columns × values per uint32). A mismatch is
reported as a row with `?` rather than guessed.
"""
import argparse
import json
import os
import struct
import sys

TENSOR_MEDIA_TYPE = "application/vnd.ollama.image.tensor"

# Values packed per uint32 and the default group size, by quant type (x/quant/quant.go's table).
QUANT_PARAMS = {
    "nvfp4": (8, 16),
    "mxfp4": (8, 32),
    "mxfp8": (4, 32),
    "int4": (8, 64),
    "int8": (4, 64),
    "affine4": (8, 64),
    "affine8": (4, 64),
}


def models_dir(explicit=None):
    return explicit or os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")


def manifest_path(root, model):
    """`name`, `name:tag`, `ns/name:tag` or `host/ns/name:tag` → the manifest file under the store."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library"] + parts
    elif len(parts) == 2:
        parts = ["registry.ollama.ai"] + parts
    return os.path.join(root, "manifests", *parts, tag)


def read_header(blob_path):
    with open(blob_path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return header, meta


def quantized_rows(root, model):
    """One row per quantized weight (a U32 tensor with a `.scale` sibling in the same blob)."""
    with open(manifest_path(root, model)) as f:
        manifest = json.load(f)
    rows = []
    for layer in manifest["layers"]:
        if layer.get("mediaType") != TENSOR_MEDIA_TYPE:
            continue
        blob = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
        header, meta = read_header(blob)
        for name, t in header.items():
            if t["dtype"] != "U32" or name + ".scale" not in header:
                continue
            rows.append(describe(name, t, header[name + ".scale"], name + ".global_scale" in header, meta))
    rows.sort(key=lambda r: r["tensor"])
    return rows


def describe(name, weight, scale, has_global, meta):
    quant = (meta.get("quant_type") or "?").lower()
    pack, default_group = QUANT_PARAMS.get(quant, (None, None))
    group = int(meta["group_size"]) if meta.get("group_size") else default_group
    shape = weight["shape"]
    experts = shape[0] if len(shape) == 3 else None
    n = shape[-2] if len(shape) >= 2 else None
    packed_cols, scale_cols = shape[-1], scale["shape"][-1]
    k = None
    if pack and group:
        k = packed_cols * pack
        if k != scale_cols * group:
            k = None  # the header contradicts the quant type's packing; do not guess
    row = {"tensor": name, "quant": quant, "experts": experts, "N": n, "K": k, "group": group,
           "global_scale": has_global, "k_mod_32": (k % 32) if k is not None else None}
    return row


def render(model, rows, unaligned_only=False):
    shown = [r for r in rows if unaligned_only is False or r["K"] is None or r["k_mod_32"] != 0]
    out = [f"### {model}: {len(rows)} quantized weights, "
           f"{sum(1 for r in rows if r['K'] is not None and r['k_mod_32'] != 0)} with K % 32 != 0, "
           f"{sum(1 for r in rows if r['K'] is None)} undetermined"]
    out.append("| tensor | quant | experts | N | K | group | global_scale | K % 32 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in shown:
        k = "?" if r["K"] is None else r["K"]
        km = "?" if r["k_mod_32"] is None else r["k_mod_32"]
        flag = " **←**" if r["K"] is not None and r["k_mod_32"] != 0 else ""
        out.append(f"| `{r['tensor']}` | {r['quant']} | {r['experts'] or ''} | {r['N']} | {k} | {r['group'] or '?'} "
                   f"| {'yes' if r['global_scale'] else ''} | {km}{flag} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", help="model names as `ollama list` shows them")
    ap.add_argument("--models-dir", default=None, help="store root (default: $OLLAMA_MODELS or ~/.ollama/models)")
    ap.add_argument("--unaligned", action="store_true", help="show only rows whose K is not a multiple of 32")
    args = ap.parse_args(argv)
    root = models_dir(args.models_dir)
    for i, model in enumerate(args.models):
        if i:
            print()
        print(render(model, quantized_rows(root, model), args.unaligned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
