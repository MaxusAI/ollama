#!/usr/bin/env python3
"""ocrbench_table.py against synthetic score files.

    python3 test_ocrbench_table.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocrbench_table  # noqa: E402


def write(directory, tag, verdicts, model="gemma4:31b-nvfp4"):
    """A score file whose i-th record is correct iff verdicts[i]."""
    recs = [{"i": i, "pred": "x", "secs": 2.0, "prompt_eval_count": 1100,
             "eval_count": 3, "gold": ["x"], "ok": bool(v)} for i, v in enumerate(verdicts)]
    doc = {"summary": {"tag": tag, "model": model, "benchmark": "ocrbench",
                       "dataset": "echo840/OCRBench", "split": "test", "offset": 0,
                       "requested": len(verdicts), "scored": len(verdicts), "errors": 0,
                       "empty_responses": 0, "think_env": "false", "think_on": False,
                       "endpoint": "generate", "correct": sum(bool(v) for v in verdicts),
                       "accuracy": round(sum(bool(v) for v in verdicts) / len(verdicts), 4)},
           "records": recs}
    with open(os.path.join(directory, f"ext_{tag}_ocrbench.json"), "w") as f:
        json.dump(doc, f)


class TestStats(unittest.TestCase):
    def test_stderr_matches_the_binomial_form(self):
        self.assertAlmostEqual(ocrbench_table.stderr(174, 200), 0.0238, places=3)
        self.assertEqual(ocrbench_table.stderr(0, 0), 0.0)

    def test_mcnemar_is_symmetric_and_exact(self):
        self.assertEqual(ocrbench_table.mcnemar_p(0, 0), 1.0)
        self.assertEqual(ocrbench_table.mcnemar_p(3, 3), ocrbench_table.mcnemar_p(3, 3))
        # 10 discordant pairs all one way is the strongest 10-pair evidence: 2 * 0.5**10
        self.assertAlmostEqual(ocrbench_table.mcnemar_p(10, 0), 2 / 1024, places=6)
        # a 3-item gap on a 200-item slice, spread over many discordant pairs, resolves nothing
        self.assertGreater(ocrbench_table.mcnemar_p(14, 11), 0.05)


class TestTables(unittest.TestCase):
    def test_arm_table_reports_accuracy_and_engine(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "a", [1] * 172 + [0] * 28)
            arms = {"nvfp4": [("a", ocrbench_table.load(d, "a"))]}
            out = ocrbench_table.arm_table(arms, "mlx-cuda")
            self.assertIn("172 / 200", out)
            self.assertIn("**0.860**", out)
            self.assertIn("mlx-cuda", out)

    def test_repeats_count_flipped_items(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "r1", [1, 1, 0, 0])
            write(d, "r2", [1, 0, 1, 0])          # items 1 and 2 flipped
            arms = {"nvfp4": [("r1", ocrbench_table.load(d, "r1")),
                              ("r2", ocrbench_table.load(d, "r2"))]}
            out = ocrbench_table.repeat_table(arms)
            self.assertIn("| nvfp4 | 2 |", out)
            self.assertIn("| 2 |", out)

    def test_paired_table_counts_discordant_pairs_both_ways(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "a", [1, 1, 0, 0, 1])
            write(d, "b", [1, 0, 1, 0, 1])
            arms = {"A": [("a", ocrbench_table.load(d, "a"))],
                    "B": [("b", ocrbench_table.load(d, "b"))]}
            out = ocrbench_table.paired_table(arms)
            self.assertIn("| A | B |", out)
            self.assertIn("| 1 | 1 |", out)        # b = 1, c = 1
            self.assertIn("| no |", out)           # one pair each way resolves nothing

    def test_main_reports_missing_score_files_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "present", [1, 0])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ocrbench_table.main(["--dir", d, "here=present", "gone=absent"])
            self.assertIn("Missing score files", buf.getvalue())
            self.assertIn("ext_absent_ocrbench.json", buf.getvalue())

    def test_main_exits_when_nothing_is_readable(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                ocrbench_table.main(["--dir", d, "x=nothing"])


class TestCategories(unittest.TestCase):
    def test_question_types_come_from_the_cached_slice(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "extimgs", "ocrbench"))
            with open(os.path.join(d, "extimgs", "ocrbench", "rows_0_3.json"), "w") as f:
                json.dump([{"question_type": "Handwriting Recognition"},
                           {"question_type": "Regular Text Recognition"},
                           {}], f)
            types = ocrbench_table.question_types(d, 0, 3)
            self.assertEqual(types[0], "Handwriting Recognition")
            self.assertEqual(types[2], "unlabelled", "a row without a type is still an item")

    def test_missing_cache_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ocrbench_table.question_types(d, 0, 200), {})

    def test_category_table_counts_per_arm(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "a", [1, 0, 1, 1])
            write(d, "b", [1, 1, 1, 0])
            arms = {"A": [("a", ocrbench_table.load(d, "a"))],
                    "B": [("b", ocrbench_table.load(d, "b"))]}
            types = {0: "hand", 1: "hand", 2: "regular", 3: "regular"}
            out = ocrbench_table.category_table(arms, types)
            self.assertIn("| hand | 2 | 1/2 | 2/2 |", out)
            self.assertIn("| regular | 2 | 2/2 | 1/2 |", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
