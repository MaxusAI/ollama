#!/usr/bin/env python3
"""quant_dims.py reads shapes out of a synthetic store; the shapes here are fixtures, not claims about any model.

    python3 test_quant_dims.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_dims  # noqa: E402

DTYPE_BYTES = {"U32": 4, "U8": 1, "F32": 4, "BF16": 2}


def write_blob(root, name, tensors, meta):
    """A minimal safetensors file: 8-byte header length, JSON header, zero data. Returns the manifest layer."""
    header, offset = {}, 0
    for tname, (dtype, shape) in tensors.items():
        size = DTYPE_BYTES[dtype]
        for d in shape:
            size *= d
        header[tname] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    if meta:
        header["__metadata__"] = meta
    body = json.dumps(header).encode()
    digest = "sha256:" + ("%064x" % abs(hash(name)))
    os.makedirs(os.path.join(root, "blobs"), exist_ok=True)
    with open(os.path.join(root, "blobs", digest.replace(":", "-")), "wb") as f:
        f.write(struct.pack("<Q", len(body)) + body + b"\0" * offset)
    return {"mediaType": quant_dims.TENSOR_MEDIA_TYPE, "digest": digest, "size": 8 + len(body) + offset, "name": name}


def make_store(root):
    layers = [
        # the #3912 shape: nvfp4, K = 538 × 8 = 269 × 16 = 4304, K % 32 == 16, with a global scale
        write_blob(root, "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
                   {"model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight": ("U32", [4, 538]),
                    "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight.scale": ("U8", [4, 269]),
                    "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight.global_scale": ("F32", [])},
                   {"quant_type": "nvfp4"}),
        # aligned nvfp4 with an explicit group_size and no global scale: K = 144 × 8 = 72 × 16 = 1152
        write_blob(root, "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
                   {"model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight": ("U32", [4, 144]),
                    "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight.scale": ("U8", [4, 72])},
                   {"quant_type": "nvfp4", "group_size": "16"}),
        # mxfp8 expert bank: 4 values per uint32, group 32: K = 288 × 4 = 36 × 32 = 1152, 8 experts
        write_blob(root, "model.language_model.layers.0.experts.down_proj",
                   {"model.language_model.layers.0.experts.down_proj.weight": ("U32", [8, 4, 288]),
                    "model.language_model.layers.0.experts.down_proj.weight.scale": ("U8", [8, 4, 36])},
                   {"quant_type": "mxfp8", "group_size": "32"}),
        # a header that contradicts its quant type's packing: reported as undetermined, never guessed
        write_blob(root, "model.language_model.odd.weight",
                   {"model.language_model.odd.weight": ("U32", [4, 100]),
                    "model.language_model.odd.weight.scale": ("U8", [4, 7])},
                   {"quant_type": "nvfp4"}),
        # not quantized: no .scale sibling, must not appear
        write_blob(root, "model.language_model.norm.weight",
                   {"model.language_model.norm.weight": ("BF16", [4])}, {}),
        # a non-tensor layer, skipped by media type
        {"mediaType": "application/vnd.ollama.image.json", "digest": "sha256:" + "0" * 64, "size": 2},
    ]
    mdir = os.path.join(root, "manifests", "registry.ollama.ai", "library", "gemma4")
    os.makedirs(mdir)
    with open(os.path.join(mdir, "fixture-nvfp4"), "w") as f:
        json.dump({"schemaVersion": 2, "layers": layers}, f)


class QuantDimsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        make_store(self.root)
        self.rows = {r["tensor"]: r for r in quant_dims.quantized_rows(self.root, "gemma4:fixture-nvfp4")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_quantized_weights_are_listed(self):
        self.assertEqual(sorted(self.rows), [
            "model.language_model.layers.0.experts.down_proj.weight",
            "model.language_model.odd.weight",
            "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
            "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
        ])

    def test_the_3912_shape_is_flagged_with_its_global_scale(self):
        r = self.rows["model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"]
        self.assertEqual((r["quant"], r["N"], r["K"], r["group"], r["k_mod_32"], r["global_scale"], r["experts"]),
                         ("nvfp4", 4, 4304, 16, 16, True, None))

    def test_aligned_nvfp4_reads_the_explicit_group_size(self):
        r = self.rows["model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight"]
        self.assertEqual((r["K"], r["group"], r["k_mod_32"], r["global_scale"]), (1152, 16, 0, False))

    def test_mxfp8_expert_bank_packs_four_per_word(self):
        r = self.rows["model.language_model.layers.0.experts.down_proj.weight"]
        self.assertEqual((r["quant"], r["experts"], r["N"], r["K"], r["group"], r["k_mod_32"]),
                         ("mxfp8", 8, 4, 1152, 32, 0))

    def test_contradictory_header_is_undetermined_not_guessed(self):
        r = self.rows["model.language_model.odd.weight"]
        self.assertIsNone(r["K"])
        self.assertIsNone(r["k_mod_32"])

    def test_manifest_path_forms(self):
        p = quant_dims.manifest_path
        self.assertEqual(p("/s", "gemma4:31b-nvfp4"), "/s/manifests/registry.ollama.ai/library/gemma4/31b-nvfp4")
        self.assertEqual(p("/s", "gemma4"), "/s/manifests/registry.ollama.ai/library/gemma4/latest")
        self.assertEqual(p("/s", "maxusai/gemma4:x"), "/s/manifests/registry.ollama.ai/maxusai/gemma4/x")
        self.assertEqual(p("/s", "h.example/ns/m:t"), "/s/manifests/h.example/ns/m/t")

    def test_render_summary_counts_and_unaligned_filter(self):
        rows = quant_dims.quantized_rows(self.root, "gemma4:fixture-nvfp4")
        full = quant_dims.render("gemma4:fixture-nvfp4", rows)
        self.assertIn("4 quantized weights, 1 with K % 32 != 0, 1 undetermined", full)
        self.assertIn("| 4304 | 16 | yes | 16 **←** |", full)
        only = quant_dims.render("gemma4:fixture-nvfp4", rows, unaligned_only=True)
        self.assertIn("down_proj", only)
        self.assertIn("odd.weight", only)  # undetermined rows stay visible under the filter
        self.assertNotIn("up_proj", only)
        self.assertNotIn("experts.down_proj", only)

    def test_cli_reads_the_models_dir_flag_over_the_env(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = quant_dims.main(["--models-dir", self.root, "--unaligned", "gemma4:fixture-nvfp4"])
        self.assertEqual(rc, 0)
        self.assertIn("### gemma4:fixture-nvfp4:", buf.getvalue())
        self.assertIn("4304", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
