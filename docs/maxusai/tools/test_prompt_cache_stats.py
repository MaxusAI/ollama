#!/usr/bin/env python3
"""Tests for prompt_cache_stats.py, on journal lines captured from an H100 with identifiers removed."""
import contextlib
import io
import json
import os
import tempfile
import unittest

import prompt_cache_stats as pcs

NEMOTRON = "c34ddc773b0216ae33354c284466b69e733cfabae6db940668ef1a84a48a74f6"
GEMMA = "cd20f2b3f2e30239023189c04e65754b0cc7c38959665ac727461ee470df16cf"
MMPROJ = "ab" * 32

# Captured 2026-10-01/02 from `journalctl -u ollama -o short-iso-precise` on a GCP a3-highgpu-1g VM
# (llama.cpp b11081, -np 1, --cache-ram unset), the host renamed and the launch commands trimmed.
# nemotron3:33b-q8 during an OCRBench arm: three cache updates, the third of which restored a
# prompt, and journald's notice that it dropped lines for the unit. Then gemma4:31b-it-q4_K_M during
# RefCOCO: a slot reused by prefix, and one update.
JOURNAL = f"""\
2026-10-01T05:19:31.196695+00:00 gpu-box-7 ollama[394624]: time=2026-10-01T05:19:31.196Z level=INFO source=llama_server.go:438 msg="starting llama-server" cmd="/usr/local/lib/ollama/llama-server --model /models/blobs/sha256-{NEMOTRON} --port 39429 --host 127.0.0.1 -c 8192 -np 1 --mmproj /models/blobs/sha256-{MMPROJ}"
2026-10-01T05:19:40.576253+00:00 gpu-box-7 ollama[394624]: srv    load_model: prompt cache is enabled, size limit: 8192 MiB
2026-10-01T05:26:28.893473+00:00 gpu-box-7 ollama[394624]: slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 67569297773
2026-10-01T05:26:28.893473+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: updating prompt cache
2026-10-01T05:26:28.893473+00:00 gpu-box-7 ollama[394624]: srv   prompt_save:  - saving prompt with length 301, total state size = 49.385 MiB (draft: 0.000 MiB)
2026-10-01T05:26:28.924557+00:00 gpu-box-7 ollama[394624]: srv        update:  - cache state: 84 prompts, 8187.665 MiB (limits: 8192.000 MiB, 8192 tokens, 32017 est)
2026-10-01T05:26:28.925925+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: prompt cache update took 29.91 ms
2026-10-01T05:26:29.095747+00:00 gpu-box-7 ollama[394624]: slot print_timing: id  0 | task 2501 | prompt eval time =     136.60 ms /   516 tokens (    0.26 ms per token,  3777.54 tokens per second)
2026-10-01T05:26:29.095747+00:00 gpu-box-7 ollama[394624]: slot print_timing: id  0 | task 2501 |       total time =     172.22 ms /   525 tokens
2026-10-01T05:26:30.120068+00:00 gpu-box-7 ollama[394624]: slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 67570527815
2026-10-01T05:26:30.120068+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: updating prompt cache
2026-10-01T05:26:30.120068+00:00 gpu-box-7 ollama[394624]: srv   prompt_save:  - saving prompt with length 524, total state size = 50.694 MiB (draft: 0.000 MiB)
2026-10-01T05:26:30.153198+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: prompt cache update took 30.30 ms
2026-10-01T05:26:30.276165+00:00 gpu-box-7 ollama[394624]: slot print_timing: id  0 | task 2513 |       total time =     125.63 ms /   322 tokens
2026-10-01T05:26:31.299231+00:00 gpu-box-7 ollama[394624]: slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 67571708224
2026-10-01T05:26:31.299231+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: updating prompt cache
2026-10-01T05:26:31.299231+00:00 gpu-box-7 ollama[394624]: srv   prompt_save:  - saving prompt with length 321, total state size = 49.502 MiB (draft: 0.000 MiB)
2026-10-01T05:26:31.329759+00:00 gpu-box-7 ollama[394624]: srv          load:  - found better prompt with f_keep = 0.896, f_sim = 0.911
2026-10-01T05:26:31.337541+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: prompt cache update took 37.37 ms
2026-10-01T05:26:31.468845+00:00 gpu-box-7 ollama[394624]: slot print_timing: id  0 | task 2520 |       total time =     132.10 ms /   298 tokens
2026-10-01T05:26:44.000000+00:00 gpu-box-7 systemd-journald[543]: Suppressed 14515 messages from ollama.service
2026-10-02T00:33:28.762663+00:00 gpu-box-7 ollama[394624]: time=2026-10-02T00:33:28.762Z level=INFO source=llama_server.go:438 msg="starting llama-server" cmd="/usr/local/lib/ollama/llama-server --model /models/blobs/sha256-{GEMMA} --port 41215 -c 8192 -np 1 --mmproj /models/blobs/sha256-{MMPROJ}"
2026-10-02T01:00:00.143238+00:00 gpu-box-7 ollama[394624]: slot print_timing: id  0 | task 11636 |       total time =    1902.80 ms /  1163 tokens
2026-10-02T01:00:00.201552+00:00 gpu-box-7 ollama[394624]: slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.948 (> 0.100 thold), f_keep = 0.924
2026-10-02T03:05:00.953122+00:00 gpu-box-7 ollama[394624]: slot print_timing: id  0 | task 56687 |       total time =    1842.36 ms /  1179 tokens
2026-10-02T03:05:01.011682+00:00 gpu-box-7 ollama[394624]: slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 145482385206
2026-10-02T03:05:01.011682+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: updating prompt cache
2026-10-02T03:05:01.011766+00:00 gpu-box-7 ollama[394624]: srv   prompt_save:  - saving prompt with length 1179, total state size = 892.136 MiB (draft: 0.000 MiB)
2026-10-02T03:05:01.574027+00:00 gpu-box-7 ollama[394624]: srv  get_availabl: prompt cache update took 562.05 ms
"""

WINDOW = "ocrbench rows 0..1000=2026-10-01T05:19:00/2026-10-01T05:42:00"


def manifests(root, entries):
    """An ollama store's manifests tree: {"name:tag": model blob hex}."""
    for name, blob in entries.items():
        model, tag = name.split(":")
        d = os.path.join(root, "registry.ollama.ai", "library", model)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, tag), "w") as f:
            json.dump({"layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + blob},
                {"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:" + MMPROJ}]}, f)
    return root


class Parse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = manifests(self.tmp.name, {"nemotron3:33b-q8": NEMOTRON, "gemma4:31b-it-q4_K_M": GEMMA})
        self.d = pcs.parse(io.StringIO(JOURNAL), pcs.manifest_models(root), [pcs.parse_window(WINDOW)])
        self.rows = {(r["model"], r["window"]): r for r in self.d["rows"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_window_counts_what_llama_cpp_logged(self):
        r = self.rows[("nemotron3:33b-q8", "ocrbench rows 0..1000")]
        self.assertEqual((r["requests"], r["cache_updates"], r["restores"], r["slot_reuse"]), (3, 3, 1, 0))
        self.assertEqual((r["update_ms_median"], r["update_ms_p90"]), (30, 37))
        self.assertEqual((r["state_mib_median"], r["prompt_len_median"], r["mib_per_token"]), (49.5, 321, 0.154))
        self.assertEqual(r["compute_ms_median"], 132)

    def test_a_request_is_the_model_of_its_launch_not_its_projector(self):
        # The launch line names the projector blob after the model blob; the --model one counts.
        self.assertEqual({m for m, _ in self.rows}, {"nemotron3:33b-q8", "gemma4:31b-it-q4_K_M"})

    def test_requests_outside_every_window_are_other(self):
        r = self.rows[("gemma4:31b-it-q4_K_M", "other")]
        self.assertEqual((r["requests"], r["cache_updates"], r["slot_reuse"]), (2, 1, 1))
        self.assertEqual((r["update_ms_median"], r["state_mib_median"], r["mib_per_token"]), (562, 892.1, 0.757))

    def test_totals_cover_the_whole_log(self):
        t = self.d["totals"]
        self.assertEqual((t["cache_updates"], t["restores"]), (4, 1))
        self.assertEqual((t["update_ms_mean"], t["update_ms_median"], t["update_ms_max"]), (165, 34, 562))

    def test_journald_drops_mark_the_row_they_fall_in(self):
        r = self.rows[("nemotron3:33b-q8", "ocrbench rows 0..1000")]
        self.assertEqual(r["journal_lines_dropped"], 14515)
        self.assertNotIn("journal_lines_dropped", self.rows[("gemma4:31b-it-q4_K_M", "other")])
        self.assertEqual(self.d["totals"]["journal_lines_dropped"], 14515)

    def test_an_unknown_blob_is_named_by_its_prefix(self):
        d = pcs.parse(io.StringIO(JOURNAL), {}, [])
        self.assertIn("blob " + NEMOTRON[:12], {r["model"] for r in d["rows"]})

    def test_the_host_is_never_written(self):
        self.assertNotIn("gpu-box-7", json.dumps(self.d))

    def test_render_leaves_other_out_of_the_window_table(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pcs.render(self.d)
        text = out.getvalue()
        self.assertIn("| `nemotron3:33b-q8` | ocrbench rows 0..1000 | 3 | 3 | 1 | 0 | 49.5 MiB | 0.154 | 30 ms | 37 ms "
                      "| 132 ms |", text)
        self.assertNotIn("| other |", text)
        self.assertIn("| all | 4 | 1 | 165 ms |", text)
        self.assertIn("journald dropped lines in `nemotron3:33b-q8` ocrbench rows 0..1000 (14515 lines): "
                      "those rows' counts are lower bounds.", text)


class AB(unittest.TestCase):
    def arm(self, tmp, name, secs, preds, model="m:1", offset=4000):
        recs = [{"i": offset + k, "secs": x, "pred": y, "ok": True} for k, (x, y) in enumerate(zip(secs, preds))]
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            json.dump({"summary": {"model": model, "benchmark": "refcoco", "dataset": "d", "split": "val",
                                   "offset": offset, "requested": len(recs)}, "records": recs}, f)
        return path

    def test_a_pair_reports_means_without_the_load_and_differing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            on = self.arm(tmp, "on.json", [30.0, 2.0, 3.0], ["a", "b", "c"])
            off = self.arm(tmp, "off.json", [29.0, 1.5, 2.5], ["a", "b", "x"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                pcs.ab([on, off])
        self.assertIn("| `m:1` | refcoco 4000..4003 | 2 | 2.500 | 2.000 | -20.0 % | 1 |", out.getvalue())

    def test_a_pair_must_be_the_same_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            on = self.arm(tmp, "on.json", [1.0, 1.0], ["a", "b"])
            off = self.arm(tmp, "off.json", [1.0, 1.0], ["a", "b"], offset=0)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                pcs.ab([on, off])
            with self.assertRaises(ValueError):
                pcs.ab([on])


if __name__ == "__main__":
    unittest.main()
