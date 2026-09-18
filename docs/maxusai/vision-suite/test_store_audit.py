#!/usr/bin/env python3
"""store_audit.py against synthetic manifests — no network, no store.

    python3 test_store_audit.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store_audit  # noqa: E402


def layer(name, size, digest):
    return {"mediaType": "application/vnd.ollama.image.tensor", "digest": f"sha256:{digest}",
            "size": size, "name": name}


def manifest(layers, config="sha256:cfg"):
    return {"schemaVersion": 2, "config": {"digest": config, "size": 417}, "layers": layers}


class TestClassify(unittest.TestCase):
    def test_vision_and_audio_paths_are_not_language(self):
        self.assertEqual(store_audit.classify("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"), "vision")
        self.assertEqual(store_audit.classify("model.embed_vision.embedding_projection.weight"), "vision")
        self.assertEqual(store_audit.classify("v.blk.3.attn_q.weight"), "vision")
        self.assertEqual(store_audit.classify("model.embed_audio.embedding_projection.weight"), "audio")
        self.assertEqual(store_audit.classify("model.language_model.layers.4.mlp.up_proj.weight"), "language")
        self.assertEqual(store_audit.classify("blk.4.ffn_up.weight"), "language")

    def test_unnamed_layer_is_a_blob(self):
        # GGUF models carry one unnamed `model` layer; it has no tensor structure to group.
        self.assertEqual(store_audit.classify(None), "blob")


class TestCompare(unittest.TestCase):
    def test_identical_manifests_report_nothing_moved(self):
        m = manifest([layer("a", 10, "aa"), layer("b", 20, "bb")])
        c = store_audit.compare(m, m)
        self.assertEqual(c["moved"], 0)
        self.assertTrue(c["config_same"])

    def test_vision_tower_requantised_to_bf16(self):
        """The gemma4 case: language layers untouched, every tower layer 3.55x bigger."""
        local = manifest([layer("model.language_model.layers.0.mlp.up_proj.weight", 1000, "l0"),
                          layer("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight", 2790000, "v0"),
                          layer("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", 750000, "v1")])
        remote = manifest([layer("model.language_model.layers.0.mlp.up_proj.weight", 1000, "l0"),
                           layer("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight", 9920000, "V0"),
                           layer("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", 2650000, "V1")])
        c = store_audit.compare(local, remote)
        self.assertEqual(c["moved"], 2)
        self.assertEqual(c["groups"], {"vision": 2})
        self.assertTrue(c["config_same"], "the config blob is unchanged — this is what hides the swap")
        self.assertAlmostEqual(c["ratio"], 3.54, places=2)
        self.assertEqual(store_audit.dtype_reading(c["ratio"]), "nvfp4 → bf16")

    def test_ratio_is_none_when_names_do_not_match(self):
        # A GGUF model is one unnamed layer: it can differ, but there is no per-tensor ratio.
        local = manifest([{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:g1", "size": 20e9}])
        remote = manifest([{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:g2", "size": 21e9}])
        c = store_audit.compare(local, remote)
        self.assertEqual(c["moved"], 1)
        self.assertIsNone(c["ratio"])
        self.assertEqual(store_audit.dtype_reading(None), "")

    def test_changed_config_is_reported(self):
        local = manifest([layer("a", 10, "aa")], config="sha256:one")
        remote = manifest([layer("a", 11, "bb")], config="sha256:two")
        self.assertFalse(store_audit.compare(local, remote)["config_same"])


class TestRenderAndStore(unittest.TestCase):
    def test_render_one_row(self):
        c = {"moved": 194, "total": 1248, "groups": {"vision": 191, "other": 3},
             "config_same": True, "ratio": 3.52, "names": []}
        out = store_audit.render([("gemma4:31b-nvfp4", "sha256:" + "a" * 64, "sha256:" + "b" * 64, c)])
        self.assertIn("`gemma4:31b-nvfp4`", out)
        self.assertIn("194 / 1248", out)
        self.assertIn("vision 191", out)
        self.assertIn("3.52×", out)
        self.assertIn("nvfp4 → bf16", out)

    def test_read_store_returns_manifests_and_digests(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifests", "registry.ollama.ai", "library", "gemma4")
            os.makedirs(p)
            body = manifest([layer("a", 10, "aa")])
            with open(os.path.join(p, "31b-nvfp4"), "w") as f:
                json.dump(body, f)
            got = store_audit.read_store(d)
            self.assertIn("gemma4:31b-nvfp4", got)
            m, dig = got["gemma4:31b-nvfp4"]
            self.assertEqual(m["layers"][0]["name"], "a")
            self.assertTrue(dig.startswith("sha256:"))
            self.assertEqual(len(dig), 71)

    def test_missing_store_is_empty_not_an_error(self):
        self.assertEqual(store_audit.read_store("/nonexistent/store"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
