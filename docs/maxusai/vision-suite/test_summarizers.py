#!/usr/bin/env python3
"""Offline tests for what the summarizers CLAIM their numbers are.

A summarizer is a labelling machine. Its arithmetic is mostly trivial and its
failure mode is not a wrong number but a right number under a wrong name — and
that failure survives review, because the table looks exactly as it should.
`summarize_engine_compare.py` rendered eval_count under "Answer tok" in both
think modes; a think-on gemma4:12b row read 5588 against an answer of a few
hundred tokens, and it was published that way
(../vision-campaign-2026-08-17-eighteen-model.md).

So these tests assert the pairing of label to field, which is the thing nothing
else checks. The values here are fixtures, not measurements — nothing in this
file is a claim about any model.

    python3 test_summarizers.py
"""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarize_engine_compare as sec  # noqa: E402

# One model, two think cells. Numbers chosen so every assertion below can name
# which field it came from: the think-on count is deliberately far larger than
# the think-off one, which is what reasoning tokens do to eval_count.
THINKOFF = {
    "scene_single": {"eval_count": 412, "prompt_eval_count": 2613, "gen_tps": 40.0,
                     "prefill_tps": 1000.0, "num_ctx": 16384, "num_predict": 2200,
                     "bbox_mean_iou": 0.9, "bbox_hits": 6, "object_count": 6,
                     "labels_found": 6, "labels_total": 6, "colors_right": 6,
                     "serial_found": True},
    "document_single": {"items_found": 5, "items_total": 5, "qty_price_right": 5,
                        "total_right": True, "name_bbox_hits": 4, "num_ctx": 16384},
    "multi_3img": {"q1_right": True, "q2_right": True, "q4_bbox_hit": True,
                   "num_ctx": 16384},
    "finetext": {"recall_22px": 4, "recall_16px": 4, "recall_12px": 4,
                 "recall_9px": 1, "recall_7px": 0, "num_ctx": 16384},
}
THINKON = json.loads(json.dumps(THINKOFF))
THINKON["scene_single"]["eval_count"] = 5588      # thinking + answer
# run_engine_compare.sh raises num_predict for think-on precisely because the
# think-off cap sits below a reasoning budget. Leaving it at 2200 here would
# make every think-on row read as capped, which is a different code path and
# would hide the label under a "≥" -- as this fixture did until the test said so.
THINKON["scene_single"]["num_predict"] = 8192


def write(rundir, model, think, scores):
    tag = sec.tag_for(model, think)
    with open(os.path.join(rundir, f"scores_{tag}.json"), "w") as fh:
        json.dump(scores, fh)


def render(rundir, model, think):
    argv = ["summarize_engine_compare.py", "--dir", rundir, "--think", think, model]
    out = io.StringIO()
    with mock_argv(argv), contextlib.redirect_stdout(out):
        sec.main()
    return out.getvalue()


@contextlib.contextmanager
def mock_argv(argv):
    old, sys.argv = sys.argv, argv
    try:
        yield
    finally:
        sys.argv = old


def t2_header(rendered):
    """The throughput table's header row: the second table's first | line."""
    tables = [l for l in rendered.split("\n") if l.startswith("| Model |")]
    return tables[1]


class TestTokenColumn(unittest.TestCase):
    """eval_count is every generated token; only think-off makes it the answer."""

    def test_think_off_says_answer(self):
        self.assertEqual(sec.token_column("false"), "Answer tok")

    def test_think_on_does_not_say_answer(self):
        self.assertEqual(sec.token_column("on"), "Gen tok")

    def test_an_unrecognised_mode_fails_towards_the_wider_label(self):
        """"Gen tok" is never wrong; "Answer tok" is wrong whenever reasoning
        ran. An unknown mode must land on the one that cannot mislead."""
        for mode in ("true", "on ", "1", "", None):
            self.assertEqual(sec.token_column(mode), "Gen tok", f"mode={mode!r}")


class TestRenderedTables(unittest.TestCase):
    """End to end, because the header is built once and the bug was in that
    construction, not in token_column (which did not exist)."""

    MODEL = "gemma4:12b-it-q4_K_M"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write(self.dir, self.MODEL, "false", THINKOFF)
        write(self.dir, self.MODEL, "on", THINKON)

    def test_think_off_table_reports_the_answer_length(self):
        head = t2_header(render(self.dir, self.MODEL, "false"))
        self.assertIn("Answer tok", head)

    def test_think_on_table_never_says_answer_tok(self):
        rendered = render(self.dir, self.MODEL, "on")
        self.assertIn("Gen tok |", t2_header(rendered))
        self.assertNotIn("Answer tok", rendered)

    def test_the_number_is_eval_count_in_both_modes(self):
        """The label changes; the field must not. A rename that also changed
        which field is rendered would break every published comparison."""
        for think, scores in (("false", THINKOFF), ("on", THINKON)):
            rendered = render(self.dir, self.MODEL, think)
            row = [l for l in rendered.split("\n")
                   if l.startswith(f"| {self.MODEL} |")][1]
            self.assertIn(f"| {scores['scene_single']['eval_count']} |", row,
                          f"think={think}")

    def test_gen_tok_per_s_column_is_untouched(self):
        """The adjacent "Gen tok/s" column must not be confused with the new
        "Gen tok" header -- they are a rate and a count."""
        head = t2_header(render(self.dir, self.MODEL, "on"))
        self.assertIn("| Gen tok | Gen tok/s | Prefill tok/s |", head)

    def test_a_capped_cell_still_refuses_to_quote_throughput(self):
        """Capping is orthogonal to the label and must survive the change."""
        capped = json.loads(json.dumps(THINKON))
        capped["scene_single"]["eval_count"] = capped["scene_single"]["num_predict"]
        write(self.dir, self.MODEL, "on", capped)
        row = [l for l in render(self.dir, self.MODEL, "on").split("\n")
               if l.startswith(f"| {self.MODEL} |")][1]
        self.assertIn(f"≥{capped['scene_single']['num_predict']} ⚠", row)
        self.assertEqual(row.count("capped"), 2)   # s/req and req/h


class TestSiblingSummarizersDoNotClaimAnswerTokens(unittest.TestCase):
    """The same defect class, across every script that renders the same field.

    Scoped to markdown CELLS, not to the words: a docstring may discuss "Answer
    tok" freely -- this file does, and so does token_column -- but a string
    carrying both the phrase and a "|" is a table header, and a table header
    that names the answer must be reached only through the think-mode gate.
    """

    DIR = os.path.dirname(os.path.abspath(__file__))

    def unconditional_headers(self, path):
        tree = ast.parse(open(path).read())
        gated = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "token_column"]
        bad = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if "Answer tok" not in node.value or "|" not in node.value:
                continue
            if any(g.lineno <= node.lineno <= g.end_lineno for g in gated):
                continue
            bad.append((node.lineno, node.value.strip()))
        return bad

    def test_no_summarizer_hardcodes_an_answer_token_header(self):
        checked = 0
        for name in sorted(os.listdir(self.DIR)):
            if not (name.startswith("summarize_") and name.endswith(".py")):
                continue
            checked += 1
            bad = self.unconditional_headers(os.path.join(self.DIR, name))
            self.assertEqual(bad, [], f"{name}: header claims answer tokens "
                                      f"without consulting the think mode")
        self.assertGreater(checked, 1, "the sweep found nothing to check")


if __name__ == "__main__":
    unittest.main(verbosity=2)
