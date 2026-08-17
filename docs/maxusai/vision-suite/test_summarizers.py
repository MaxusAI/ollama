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
import summarize_reps as reps  # noqa: E402

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


class TestRepsInterval(unittest.TestCase):
    """A rendered interval must contain the data it summarises.

    `mean ±(max-min)/2` does not. It is symmetric about the mean, so it equals
    [min, max] only when the mean is the midrange -- true for every n=2 arm,
    which is why this survived a campaign before it bit. At n=5 the qwen3.8
    name_bbox arm rendered "0.742 ±0.103", excluding the observed 0.631 that the
    campaign's own argument quotes.
    """

    def runs(self, values, section="s", key="k"):
        return [{section: {key: v}} for v in values]

    def test_the_interval_contains_every_observation(self):
        for values in ([0.900, 0.900, 0.990],
                       [0.981, 0.992, 0.979, 0.983, 1.000],
                       [0.790, 0.631, 0.838, 0.742, 0.710],
                       [0.5, 0.5, 0.5, 0.9]):
            rendered, _ = reps.cell(self.runs(values), "s", "k", "float")
            lo, hi = (float(x) for x in
                      rendered.split("[")[1].rstrip("]").split("–"))
            self.assertLessEqual(lo, min(values), rendered)
            self.assertGreaterEqual(hi, max(values), rendered)

    def test_the_published_cell_now_shows_the_value_the_argument_cites(self):
        rendered, spread = reps.cell(
            self.runs([0.790, 0.631, 0.838, 0.742, 0.710]), "s", "k", "float")
        self.assertEqual(rendered, "0.742 [0.631–0.838]")
        self.assertAlmostEqual(spread, 0.207, places=3)

    def test_no_plus_minus_anywhere(self):
        """± asserts symmetry this data does not have; the range asserts nothing."""
        rendered, _ = reps.cell(self.runs([0.1, 0.2, 0.9]), "s", "k", "float")
        self.assertNotIn("±", rendered)

    def test_an_identical_arm_renders_a_bare_mean(self):
        rendered, spread = reps.cell(self.runs([0.75, 0.75]), "s", "k", "float")
        self.assertEqual(rendered, "0.750")
        self.assertEqual(spread, 0)

    def test_spread_is_still_the_full_range_not_half(self):
        """The second return value feeds the "bar any claim must clear" list;
        halving it there understated the bar in the campaign record's prose."""
        _, spread = reps.cell(self.runs([0.631, 0.838]), "s", "k", "float")
        self.assertAlmostEqual(spread, 0.207, places=3)

    def test_ratios_are_not_ranked_against_counts(self):
        """Four count metrics moving by 2 must not crowd out an IoU moving by
        0.2 -- they are different units, and the top-4 list is per unit."""
        floats = [m for m in reps.METRICS if m[3] == "float"]
        ints = [m for m in reps.METRICS if m[3] == "int"]
        self.assertTrue(floats and len(ints) >= 4, "fixture assumes both kinds")
        runs = []
        for i in range(2):
            r = {}
            for section, key, _, kind in floats:
                r.setdefault(section, {})[key] = 0.5 + 0.2 * i
            for section, key, _, kind in ints:
                r.setdefault(section, {})[key] = 10 * i
            runs.append(r)
        out = io.StringIO()
        with mock_argv(["summarize_reps.py"]), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            reps_main_with(runs)
        printed = out.getvalue()
        self.assertIn("ratios —", printed)
        self.assertIn("counts —", printed)
        for section, key, label, kind in floats:
            self.assertIn(label, printed, f"{label} crowded out by count metrics")


def reps_main_with(runs):
    """Run summarize_reps' spread report over in-memory runs.

    main() reads the filesystem, so the arm is injected at load_arm rather than
    written to disk -- the report itself is the thing under test.
    """
    real = reps.load_arm
    reps.load_arm = lambda tag: (runs, [f"scores_{tag}.json"])
    try:
        sys.argv = ["summarize_reps.py", "arm"]
        reps.main()
    finally:
        reps.load_arm = real


class TestSiblingSummarizersDoNotClaimAnswerTokens(unittest.TestCase):
    """The same defect class, across every script that renders the same field.

    Scoped to markdown CELLS, not to the words: a docstring may discuss "Answer
    tok" freely -- this file does, and so does token_column -- but a string
    carrying both the phrase and a "|" is a table header, and a table header
    that names the answer must be reached only through the think-mode gate.
    """

    DIR = os.path.dirname(os.path.abspath(__file__))

    def unconditional_headers(self, path):
        with open(path) as fh:
            tree = ast.parse(fh.read())
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
