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
import subprocess
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarize_engine_compare as sec  # noqa: E402
import summarize_head_to_head as shh  # noqa: E402
import summarize_reps as reps  # noqa: E402
import vision_suite as vs  # noqa: E402

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
        self.assertEqual(row.count("capped"), 3)   # Think tok, s/req, req/h


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


class TestRungRow(unittest.TestCase):
    """The ⚠ must mean "this arm escalated", not "finetext exists".

    finetext takes its own window (vision_suite.py:1136 defaults it to 32768
    where every other section defaults to 16384), so any arm measured without
    NUM_CTX exported differs section-to-section by design. Folding that into the
    warning made it fire on every arm ever run.
    """

    def arm(self, *runs):
        return list(runs)

    def run_(self, suite, finetext=None, key="num_ctx"):
        r = {"scene_single": {f"req_{key}": suite},
             "document_single": {f"req_{key}": suite}}
        if finetext is not None:
            r["finetext"] = {f"req_{key}": finetext}
        return r

    def test_finetext_alone_is_annotated_not_flagged(self):
        out = reps.rung(self.arm(self.run_(16384, 32768), self.run_(16384, 32768)))
        self.assertEqual(out, "16384 (finetext 32768)")
        self.assertNotIn("⚠", out)

    def test_a_suite_section_moving_between_repeats_is_flagged(self):
        out = reps.rung(self.arm(self.run_(16384, 32768), self.run_(32768, 32768)))
        self.assertIn("⚠", out)
        self.assertIn("16384/32768", out)

    def test_an_arm_that_held_still_is_a_bare_number(self):
        out = reps.rung(self.arm(self.run_(16384), self.run_(16384)))
        self.assertEqual(out, "16384")

    def test_finetext_moving_alone_still_does_not_raise_the_alarm(self):
        """A finetext window that differs BETWEEN repeats is worth showing, but
        it is not the escalation the ⚠ is reserved for."""
        out = reps.rung(self.arm(self.run_(16384, 32768), self.run_(16384, 16384)))
        self.assertNotIn("⚠", out)
        self.assertIn("finetext", out)

    def test_requested_wins_over_served(self):
        """Pre-#153 files recorded the suite default in num_ctx for finetext
        while req_num_ctx kept what was asked; served-first would report the
        block at a window it did not run at."""
        runs = [{"scene_single": {"req_num_ctx": 16384, "num_ctx": 16384},
                 "finetext": {"req_num_ctx": 32768, "num_ctx": 16384}}]
        self.assertEqual(reps.rung(runs), "16384 (finetext 32768)")

    def test_num_predict_uses_the_same_rule(self):
        runs = [self.run_(2200, 4000, key="num_predict"),
                self.run_(4400, 4400, key="num_predict")]
        out = reps.npred(runs)
        self.assertIn("2200/4400 ⚠", out)
        self.assertIn("(finetext 4000/4400)", out)


class TestArmSpec(unittest.TestCase):
    """One arm may pool several tag families under a name it chooses.

    Needed because the repeat runner and this file disagree on where the rep
    number goes -- run_engine_compare.sh writes it as a PREFIX, load_arm expects
    a `-rep<N>` SUFFIX -- so a repeated arm pools only via a glob, and a glob
    then becomes the column header of a published table. The qwen3.8 ROCm n=5
    arm pools two families and cannot be expressed as one glob at all; it was
    rendered by copying files to matching names by hand.
    """

    def test_a_bare_tag_is_unchanged(self):
        self.assertEqual(reps.parse_arm("rocm-n3-thinkoff"),
                         ("rocm-n3-thinkoff", ["rocm-n3-thinkoff"]))

    def test_a_named_arm_pools_several_patterns(self):
        label, pats = reps.parse_arm("think-on=a,b-np4400,c-rep*")
        self.assertEqual(label, "think-on")
        self.assertEqual(pats, ["a", "b-np4400", "c-rep*"])

    def test_an_equals_in_the_label_is_fatal_not_a_short_arm(self):
        """It yields a plausible pattern matching nothing, so the arm would be
        dropped with a warning and the table would render one column short."""
        with self.assertRaises(SystemExit) as cm:
            reps.parse_arm("think-on (n=5)=c-rep*")
        self.assertIn("may not contain", str(cm.exception))

    def test_a_blank_pattern_is_fatal(self):
        with self.assertRaises(SystemExit):
            reps.parse_arm("think-on=a,,b")

    def test_the_column_header_is_the_label_not_the_glob(self):
        """The reason the syntax exists: a published table must not carry a
        shell pattern as a column header. parse_arm alone does not pin this --
        rendering the spec instead of the label passes every test above."""
        tmp = tempfile.mkdtemp()
        scores = {"scene_single": {"bbox_mean_iou": 0.9, "req_num_ctx": 16384}}
        for name in ("a-rep1", "a-rep2"):
            with open(os.path.join(tmp, f"scores_{name}.json"), "w") as fh:
                json.dump(scores, fh)
        real_dir, reps.DIR = reps.DIR, tmp
        out = io.StringIO()
        try:
            with mock_argv(["summarize_reps.py", "think-on=a-rep*"]), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                reps.main()
        finally:
            reps.DIR = real_dir
        header = [l for l in out.getvalue().split("\n") if l.startswith("| metric")][0]
        self.assertIn("think-on (n=2)", header)
        self.assertNotIn("*", header)


class TestFinetextWindowIsSettableAlone(unittest.TestCase):
    """finetext's window must be pinnable independently of the suite's.

    It reads NUM_CTX when nothing more specific is set, so a runner that exports
    NUM_CTX drags finetext with it -- which is how the ROCm qwen3.8 arms (finetext
    at 32768, NUM_CTX unset) and any run_engine_compare.sh arm (finetext pulled to
    16384) ended up measuring small-text recall through different windows.
    """

    DIR = os.path.dirname(os.path.abspath(__file__))

    WINDOW_VARS = ("NUM_CTX", "NUM_PREDICT",
                   "FINETEXT_NUM_CTX", "FINETEXT_NUM_PREDICT")

    def finetext_opts(self, **env):
        """Resolve the finetext options in a subprocess with a clean window env.

        Every window variable is REMOVED first rather than blanked: the suite
        does int(os.environ.get(...)) and an empty string is present-but-unparsable,
        so NUM_CTX="" raises ValueError at import. Unset and empty are different
        states here, and this needs the former.
        """
        src = ("import sys; sys.path.insert(0, %r)\n"
               "import vision_suite as vs\n"
               "print([t[4] for t in vs.tests if t[0] == 'finetext'][0])" % self.DIR)
        clean = {k: v for k, v in os.environ.items() if k not in self.WINDOW_VARS}
        out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                             text=True, cwd=self.DIR,
                             env={**clean, **env, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(out.returncode, 0, out.stderr)
        return ast.literal_eval(out.stdout.strip())

    def test_unset_takes_the_finetext_default(self):
        """The ROCm qwen3.8 arms ran exactly this way."""
        self.assertEqual(self.finetext_opts()["num_ctx"], 32768)

    def test_num_ctx_alone_still_drags_it(self):
        """Unchanged behaviour: the coupling is the default, not a bug to break."""
        opts = self.finetext_opts(NUM_CTX="16384", NUM_PREDICT="2200")
        self.assertEqual(opts, {"num_ctx": 16384, "num_predict": 2200})

    def test_the_specific_override_wins(self):
        opts = self.finetext_opts(NUM_CTX="16384", NUM_PREDICT="2200",
                                  FINETEXT_NUM_CTX="32768",
                                  FINETEXT_NUM_PREDICT="4000")
        self.assertEqual(opts, {"num_ctx": 32768, "num_predict": 4000})


class TestLabelsMatchWhatIsGated(unittest.TestCase):
    """Labels corrected from the 2026-08-18 audit, each pinned to its field."""

    MODEL = "gemma4:31b-it-q4_K_M"

    def render_with(self, mu):
        d = tempfile.mkdtemp()
        scores = json.loads(json.dumps(THINKOFF))
        scores["multi_3img"] = dict(scores["multi_3img"], **mu)
        write(d, self.MODEL, "false", scores)
        return render(d, self.MODEL, "false")

    def test_the_multi_cell_names_the_three_questions_it_gates(self):
        """The prompt asks FOUR questions; q3 is never scored, so "all Qs" was a
        claim the cell cannot make."""
        out = self.render_with({"q1_right": True, "q2_right": True,
                                "q4_bbox_hit": True})
        self.assertIn("✅ q1 + q2 + q4-bbox", out)
        self.assertNotIn("all Qs", out)

    def test_name_bbox_header_does_not_promise_iou(self):
        """vision_suite scores it as a coarse pixel-band match, not an IoU."""
        out = self.render_with({})
        self.assertIn("name_bbox in-band", out)
        self.assertNotIn("name_bbox hits", out)

    def test_ctx_prefers_the_requested_window(self):
        """Pre-#153 files recorded the SUITE default in num_ctx for finetext
        while req_num_ctx kept what was asked, so served-first reports a block
        at a window it never ran at -- and hides the mixed-window warning."""
        self.assertEqual(sec.ctx_for({"req_num_ctx": 32768, "num_ctx": 16384}),
                         "32768")
        self.assertEqual(sec.ctx_for({"num_ctx": 16384}), "16384")
        self.assertIn("⚠", sec.ctx_for({"req_num_ctx": 16384},
                                       {"req_num_ctx": 32768}))

    def test_reps_calls_unmatched_entries_what_they_are(self):
        """total_found - sum(recall) includes malformed and duplicate entries,
        not only invented ones."""
        labels = [m[2] for m in reps.METRICS]
        self.assertIn("finetext unmatched", labels)
        self.assertNotIn("finetext fabricated", labels)


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


class TestProvenanceFooter(unittest.TestCase):
    """The footer must never vouch for rows it has no provenance for.

    A score file that predates H11 carries no host/server_version. The footer
    aggregates over sets, so such a file used to contribute *nothing* — and a
    table mixing an H11-era row with a pre-H11 row rendered one clean host, as
    if both rows were that campaign. Measured 2026-08-20: the five-model CUDA
    T2 quoted g4full1 (pre-H11 gemma) columns under cudafull1's host line.
    A loaded pre-H11 file must surface as its own footer entry and trip MIXED.
    """

    HOST, VER = "http://10.0.0.1:11497", "0.32.14-test"

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def with_h11(self, host=None, ver=None):
        s = json.loads(json.dumps(THINKOFF))
        s["scene_single"]["host"] = host or self.HOST
        s["scene_single"]["server_version"] = ver or self.VER
        return s

    def render_many(self, *models):
        argv = ["summarize_engine_compare.py", "--dir", self.dir,
                "--think", "false", *models]
        out = io.StringIO()
        with mock_argv(argv), contextlib.redirect_stdout(out):
            sec.main()
        return out.getvalue().splitlines()[-1]

    def test_one_campaign_renders_a_clean_footer(self):
        write(self.dir, "a:1b", "false", self.with_h11())
        write(self.dir, "b:1b", "false", self.with_h11())
        footer = self.render_many("a:1b", "b:1b")
        self.assertIn(self.HOST, footer)
        self.assertNotIn("MIXED", footer)
        self.assertNotIn("pre-H11", footer)

    def test_a_pre_h11_row_next_to_an_h11_row_is_mixed(self):
        write(self.dir, "a:1b", "false", self.with_h11())
        write(self.dir, "b:1b", "false", THINKOFF)
        footer = self.render_many("a:1b", "b:1b")
        self.assertIn(self.HOST, footer)
        self.assertIn("pre-H11", footer)
        self.assertIn("MIXED", footer)

    def test_all_pre_h11_is_unvouched_but_not_mixed(self):
        write(self.dir, "a:1b", "false", THINKOFF)
        write(self.dir, "b:1b", "false", THINKOFF)
        footer = self.render_many("a:1b", "b:1b")
        self.assertIn("pre-H11", footer)
        self.assertNotIn("MIXED", footer)

    def test_two_hosts_are_mixed(self):
        write(self.dir, "a:1b", "false", self.with_h11())
        write(self.dir, "b:1b", "false", self.with_h11(host="http://10.0.0.2:11497"))
        footer = self.render_many("a:1b", "b:1b")
        self.assertIn("MIXED", footer)


class TestT2CappedCells(unittest.TestCase):
    """ADR 0012 convention 9 in T2: a capped block renders "capped", not a score.

    T1 has done this since the ladder landed; T2 didn't, and published qwen3.6
    think-on multi as ❌ ❌ ❌ 0/5 (16384) — a grounding failure — when
    eval_count had hit req_num_predict, i.e. the cell never terminated at that
    rung (2026-08-20, cudafull1). "Cannot ground" and "cannot stop" must never
    share one rendering.
    """

    def render_t2(self, rundir, tag):
        argv = ["summarize_head_to_head.py", "--dir", rundir, "--tags", tag]
        out = io.StringIO()
        with mock_argv(argv), contextlib.redirect_stdout(out):
            shh.main()
        return out.getvalue()

    def rows(self, scores):
        rundir = tempfile.mkdtemp()
        with open(os.path.join(rundir, "scores_x.json"), "w") as fh:
            json.dump(scores, fh)
        rendered = self.render_t2(rundir, "x")
        return {l.split("|")[2].strip(): l.split("|")[3].strip()
                for l in rendered.splitlines() if l.startswith("|")}

    def test_a_capped_scene_hides_score_and_latency_but_not_tps(self):
        s = json.loads(json.dumps(THINKOFF))
        s["scene_single"]["eval_count"] = s["scene_single"]["num_predict"]
        r = self.rows(s)
        self.assertEqual(r["bbox IoU"], "capped (16384)")
        self.assertEqual(r["labels / serial"], "capped (16384)")
        self.assertEqual(r["s/req (unique image)"], "capped")
        self.assertEqual(r["req/h (serial)"], "capped")
        self.assertEqual(r["gen tok/s"], "40")   # a rate survives truncation
        self.assertIn("5/5", r["items / qty+price / total / invoice"])  # other blocks untouched

    def test_an_uncapped_scene_still_renders_its_score(self):
        r = self.rows(THINKOFF)
        self.assertEqual(r["bbox IoU"], "0.900 (16384)")
        self.assertEqual(r["req/h (serial)"], "279")  # 3600 / (412/40 + 2613/1000)


class TestResumeNeverSkipsCapped(unittest.TestCase):
    """A capped block is an UNFINISHED measurement (ADR 0012 conv 9), so the
    suite's resume must re-run it. Treating capped as done is the defect that
    silenced the context ladder on 2026-08-20: run_engine_compare.sh escalated
    the rung, vision_suite skipped every arm as "already scored", and
    cudafull1's think-on cells froze at (16384, 8192) — while g4full1, run
    hours earlier without the resume logic, climbed to 131072. Per-cell
    escalation only works if "done" means "finished", not "present"."""

    def test_finished_block_is_done(self):
        self.assertTrue(vs.arm_done({"eval_count": 412, "num_predict": 8192}))

    def test_capped_block_is_not_done(self):
        self.assertFalse(vs.arm_done({"eval_count": 8192, "num_predict": 8192}))

    def test_error_block_is_not_done(self):
        self.assertFalse(vs.arm_done({"error": "connection reset"}))

    def test_missing_block_is_not_done(self):
        self.assertFalse(vs.arm_done(None))
        self.assertFalse(vs.arm_done({}))


class TestWasCappedPrefersDoneReason(unittest.TestCase):
    """The server's stop verdict outranks the token arithmetic (SPEC H4b).

    done_reason "length" is capped even when eval_count sits under the
    recorded cap (thinking continuations can overshoot it), and "stop" is
    finished even at the exact cap boundary. Blocks without done_reason —
    every cell recorded before 2026-08-20, plus connection-closed finals,
    which serialize with the field omitted — keep the
    eval_count >= num_predict fallback, so no historical file changes verdict."""

    def test_length_is_capped_whatever_the_arithmetic_says(self):
        self.assertTrue(sec.was_capped(
            {"done_reason": "length", "eval_count": 100, "num_predict": 8192}))

    def test_stop_is_finished_even_at_the_exact_cap_boundary(self):
        self.assertFalse(sec.was_capped(
            {"done_reason": "stop", "eval_count": 8192, "num_predict": 8192}))

    def test_absent_done_reason_keeps_the_arithmetic_fallback(self):
        self.assertTrue(sec.was_capped({"eval_count": 8192, "num_predict": 8192}))
        self.assertFalse(sec.was_capped({"eval_count": 412, "num_predict": 8192}))

    def test_resume_trusts_the_verdict_through_arm_done(self):
        self.assertTrue(vs.arm_done(
            {"done_reason": "stop", "eval_count": 8192, "num_predict": 8192}))
        self.assertFalse(vs.arm_done(
            {"done_reason": "length", "eval_count": 100, "num_predict": 8192}))


class TestT1CampaignPrefix(unittest.TestCase):
    """--prefix renders a TAG_PREFIX campaign; unset, tags derive as before (H4).

    Without it the 2026-08-20 cudafull1 five-model baseline had no T1 render:
    resolve_tag knew only bare model-derived tags, so the canonical campaign
    matrix could not be produced for any prefixed campaign."""

    MODEL = "gemma4:12b-it-q4_K_M"

    def write_prefixed(self):
        d = tempfile.mkdtemp()
        tag = "cudafull1_" + sec.tag_for(self.MODEL, "false")
        with open(os.path.join(d, f"scores_{tag}.json"), "w") as fh:
            json.dump(THINKOFF, fh)
        return d

    def row(self, rendered):
        return [l for l in rendered.splitlines()
                if l.startswith(f"| {self.MODEL} |")][0]

    def test_prefix_finds_the_campaign_file(self):
        d = self.write_prefixed()
        argv = ["summarize_engine_compare.py", "--dir", d, "--think", "false",
                "--prefix", "cudafull1_", self.MODEL]
        out = io.StringIO()
        with mock_argv(argv), contextlib.redirect_stdout(out):
            sec.main()
        self.assertIn("0.900", self.row(out.getvalue()))  # scene IoU, not a dash row

    def test_without_prefix_the_prefixed_file_stays_invisible(self):
        d = self.write_prefixed()
        self.assertNotIn("0.900", self.row(render(d, self.MODEL, "false")))


class TestT1CappedQualityCells(unittest.TestCase):
    """T1 quality cells render "capped", never a score (ADR 0012 conv 9).

    The scene-derived throughput cells were guarded; the quality cells were
    not — measured 2026-08-20, the capped qwen3.6 think-on multi ceiling cell
    rendered "❌ q1_right, q2_right, q4_bbox_hit", a grounding failure, for a
    cell that never terminated. Same defect class fixed in T2 that morning."""

    MODEL = "gemma4:12b-it-q4_K_M"

    def rows(self, scores):
        d = tempfile.mkdtemp()
        write(d, self.MODEL, "false", scores)
        return [l for l in render(d, self.MODEL, "false").splitlines()
                if l.startswith(f"| {self.MODEL} |")]

    def test_a_capped_multi_renders_capped_not_grounding_failure(self):
        s = json.loads(json.dumps(THINKOFF))
        s["multi_3img"].update(eval_count=8192, num_predict=8192,
                               q1_right=False, q2_right=False, q4_bbox_hit=False)
        throughput_row = self.rows(s)[1]
        self.assertIn("| capped |", throughput_row)
        self.assertNotIn("❌", throughput_row)

    def test_a_capped_scene_hides_every_grounding_cell(self):
        s = json.loads(json.dumps(THINKOFF))
        s["scene_single"]["eval_count"] = s["scene_single"]["num_predict"]
        grounding_row = self.rows(s)[0]
        self.assertNotIn("0.900", grounding_row)
        self.assertNotIn("6/6", grounding_row)
        self.assertIn("capped", grounding_row)


class TestT1ThinkTokColumn(unittest.TestCase):
    """"Think tok" is token_split.py's exact reasoning-token count, or "—".

    Only the gate-proven split may fill it; absence (split not run, or the
    gate refused every tokenizer offered, as for nemotron3 on 2026-08-20)
    renders "—", never a char-proportional estimate — token_split measured
    21% control tokens on one response, so proportions lie."""

    MODEL = "gemma4:12b-it-q4_K_M"

    def render_scene(self, extra):
        d = tempfile.mkdtemp()
        s = json.loads(json.dumps(THINKON))
        s["scene_single"].update(extra)
        write(d, self.MODEL, "on", s)
        rendered = render(d, self.MODEL, "on")
        return (t2_header(rendered),
                [l for l in rendered.splitlines()
                 if l.startswith(f"| {self.MODEL} |")][1])

    def test_split_counts_render_between_multi_and_total(self):
        head, row = self.render_scene({"thinking_tokens": 4321})
        self.assertIn("| Think tok | Gen tok |", head)
        self.assertIn("| 4321 | 5588 |", row)

    def test_an_unsplit_block_renders_a_dash_not_an_estimate(self):
        _, row = self.render_scene({})
        self.assertIn("| — | 5588 |", row)


class TestT1AnchoredMultiColumn(unittest.TestCase):
    """The anchored multi variant is its own T1 column, never folded into the
    unanchored one — the pair is the evidence that a q4-bbox miss was a frame
    error rather than a grounding failure. Measured 2026-08-20, qwen3.8 AND
    nemotron3 think-on both flip ❌→✅ under the calibration entry."""

    MODEL = "gemma4:12b-it-q4_K_M"

    def test_the_pair_renders_side_by_side_and_independent(self):
        d = tempfile.mkdtemp()
        s = json.loads(json.dumps(THINKOFF))
        s["multi_3img"]["q4_bbox_hit"] = False
        s["multi_3img_anchored"] = {"q1_right": True, "q2_right": True,
                                    "q4_bbox_hit": True, "num_ctx": 16384}
        write(d, self.MODEL, "false", s)
        rendered = render(d, self.MODEL, "false")
        self.assertIn("| Multi-image (3 imgs) | Multi anchored |", t2_header(rendered))
        row = [l for l in rendered.splitlines()
               if l.startswith(f"| {self.MODEL} |")][1]
        self.assertIn("❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox |", row)

    def test_a_missing_anchored_block_is_a_dash(self):
        d = tempfile.mkdtemp()
        write(d, self.MODEL, "false", THINKOFF)
        row = [l for l in render(d, self.MODEL, "false").splitlines()
               if l.startswith(f"| {self.MODEL} |")][1]
        self.assertIn("✅ q1 + q2 + q4-bbox | — |", row)


class TestCappedDiscipline(unittest.TestCase):
    """ADR 0012 conv 9: a capped cell is an unfinished measurement, and
    sec.was_capped must be the only reachable path to pooling (SPEC H5)."""

    CLEAN = {"bbox_mean_iou": 0.5, "eval_count": 400, "num_predict": 2200,
             "done_reason": "stop"}
    CAPPED = {"bbox_mean_iou": 0.9, "eval_count": 2200, "num_predict": 2200,
              "done_reason": "length"}

    def test_reps_cell_excludes_capped_from_pooled_mean(self):
        runs = [{"scene_single": dict(self.CLEAN)},
                {"scene_single": dict(self.CAPPED)}]
        s, _ = reps.cell(runs, "scene_single", "bbox_mean_iou", "float")
        self.assertTrue(s.startswith("0.500"), s)

    def test_reps_cell_all_capped_renders_capped(self):
        runs = [{"scene_single": dict(self.CAPPED)}]
        s, _ = reps.cell(runs, "scene_single", "bbox_mean_iou", "float")
        self.assertEqual(s, "capped")

    def test_matrix_capped_defers_to_done_reason(self):
        import summarize_matrix as smx
        self.assertTrue(smx.capped({"req_num_predict": 8192, "eval_count": 500,
                                    "done_reason": "length"}))
        self.assertFalse(smx.capped({"req_num_predict": 8192, "eval_count": 8290,
                                     "done_reason": "stop"}))

    def test_geometry_pooled_excludes_capped(self):
        import summarize_geometry as sg
        got = [dict(self.CLEAN, gen_tps=40.0), dict(self.CAPPED, gen_tps=90.0)]
        kept = sg.pooled(got)
        self.assertEqual([s["gen_tps"] for s in kept], [40.0])


class TestCappedArms(unittest.TestCase):
    """The driver's escalation decision as an importable, tested function —
    the shell heredoc it replaces ignored done_reason (blueprint P0-1)."""

    def _scores(self, blocks):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "scores_x.json")
        json.dump(blocks, open(p, "w"))
        return p

    def test_stop_overshoot_does_not_escalate(self):
        p = self._scores({"a": {"eval_count": 8290, "num_predict": 8192,
                                "req_num_predict": 8192, "done_reason": "stop"}})
        self.assertEqual(sec.capped_arms(p), [])

    def test_synthetic_length_below_cap_escalates(self):
        p = self._scores({"a": {"eval_count": 500, "num_predict": 8192,
                                "req_num_predict": 8192, "done_reason": "length"}})
        self.assertEqual(sec.capped_arms(p), ["a"])

    def test_context_overflow_error_escalates(self):
        p = self._scores({"a": {"error": "num_ctx too small: prompt 9000"}})
        self.assertEqual(sec.capped_arms(p), ["a"])

    def test_blown_budget_error_escalates(self):
        # A bigger rung raises num_predict AND the derived timeout, so a
        # blown budget is cured by escalation — ending the ladder silently
        # left the cell re-timing-out at the same rung on every resume.
        p = self._scores({"a": {"error": "blown generation budget: timeout "
                                         "after 6444s (num_predict=122880)"}})
        self.assertEqual(sec.capped_arms(p), ["a"])

    def test_other_errors_do_not_escalate(self):
        p = self._scores({"a": {"error": "HTTP 500: boom"}})
        self.assertEqual(sec.capped_arms(p), [])

    def test_legacy_block_without_any_cap_abstains(self):
        # No num_predict, no req_num_predict, no done_reason: was_capped
        # abstains. The first version injected the driver's CURRENT $np here,
        # judging old-campaign blocks against a rung they never ran at.
        p = self._scores({"a": {"eval_count": 7370}})
        self.assertEqual(sec.capped_arms(p), [])

    def test_ceiling_marked_arm_does_not_drive_escalation(self):
        p = self._scores({"a": {"eval_count": 8192, "num_predict": 8192,
                                "done_reason": "length",
                                "ladder_not_converged_at": 131072}})
        self.assertEqual(sec.capped_arms(p, 131072), [])
        # Raising CTX_MAX above the recorded ceiling reopens the arm.
        self.assertEqual(sec.capped_arms(p, 262144), ["a"])

    def test_missing_file_is_no_arms(self):
        self.assertEqual(sec.capped_arms("/nonexistent/scores_x.json"), [])


class TestLadderCeilingMarker(unittest.TestCase):
    """NOT CONVERGED must be machine-readable so ceiling cells stop
    re-climbing the whole ladder on every resume (blueprint P0-2). The skip
    is the DRIVER's decision (ceiling_standing, which knows CTX_MAX) —
    arm_done stays SPEC H4b verbatim: a capped block ALWAYS re-runs, marker
    or not."""

    def _scores(self, blocks):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "scores_x.json")
        with open(p, "w") as fh:
            json.dump(blocks, fh)
        return p

    CAPPED = {"eval_count": 8192, "num_predict": 8192, "done_reason": "length"}

    def test_mark_not_converged_stamps_named_blocks(self):
        p = self._scores({"a": {"eval_count": 1}, "b": {"eval_count": 2}})
        sec.mark_not_converged(p, ["a"], 131072)
        with open(p) as fh:
            data = json.load(fh)
        self.assertEqual(data["a"]["ladder_not_converged_at"], 131072)
        self.assertNotIn("ladder_not_converged_at", data["b"])

    def test_mark_only_ever_raises_the_ceiling(self):
        p = self._scores({"a": dict(self.CAPPED,
                                    ladder_not_converged_at=131072)})
        sec.mark_not_converged(p, ["a"], 65536)
        with open(p) as fh:
            self.assertEqual(json.load(fh)["a"]["ladder_not_converged_at"],
                             131072)

    def test_mark_skips_error_blocks(self):
        p = self._scores({"a": {"error": "num_ctx too small"}})
        sec.mark_not_converged(p, ["a"], 131072)
        with open(p) as fh:
            self.assertNotIn("ladder_not_converged_at", json.load(fh)["a"])

    def test_mark_never_raises_on_missing_file(self):
        sec.mark_not_converged("/nonexistent/scores_x.json", ["a"], 131072)

    def test_arm_done_is_h4b_verbatim_marker_or_not(self):
        # SPEC H4b: a capped block always re-runs — the ceiling marker is
        # NOT a resume exception (one briefly existed and made a capped
        # block read as finished at any window <= the marker).
        self.assertFalse(vs.arm_done(dict(self.CAPPED,
                                          ladder_not_converged_at=131072)))
        self.assertFalse(vs.arm_done(dict(self.CAPPED)))

    def test_ceiling_standing_true_only_when_verdict_covers_ctx_max(self):
        p = self._scores({
            "a": dict(self.CAPPED, ladder_not_converged_at=131072),
            "b": {"eval_count": 400, "num_predict": 8192,
                  "done_reason": "stop"},           # finished arm
        })
        self.assertTrue(sec.ceiling_standing(p, 131072))
        # Raising CTX_MAX past the recorded ceiling reopens the cell.
        self.assertFalse(sec.ceiling_standing(p, 262144))

    def test_ceiling_standing_false_with_unfinished_unmarked_arm(self):
        p = self._scores({
            "a": dict(self.CAPPED, ladder_not_converged_at=131072),
            "b": dict(self.CAPPED),                  # capped, no verdict
        })
        self.assertFalse(sec.ceiling_standing(p, 131072))

    def test_ceiling_standing_false_on_error_blocks_and_missing_file(self):
        p = self._scores({"a": dict(self.CAPPED, ladder_not_converged_at=131072),
                          "b": {"error": "boom"}})
        self.assertFalse(sec.ceiling_standing(p, 131072))
        self.assertFalse(sec.ceiling_standing("/nonexistent/x.json", 131072))


class TestImpliedScaleDialectGate(unittest.TestCase):
    """implied_scale is a real-frame diagnostic; on a norm-dialect response it
    fabricates a frame error on a perfect cell (blueprint P0-4, observed in
    mlx0330nv: 0.721 / IoU 0.078 against iou_declared 0.956)."""

    FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "bbox_contract", "preexisting")

    def test_norm_dialect_response_carries_no_implied_scale(self):
        # The scorer seeds every field, so "not applicable" is the value None
        # (its existing idiom — cf. anchor_implied_type), never a number.
        with open(os.path.join(self.FIX, "norm1000_clean.txt")) as fh:
            s = vs.score_bbox_contract(fh.read())
        self.assertIsNone(s["implied_scale"])
        self.assertIsNone(s["iou_at_implied_scale"])

    def test_real_dialect_wrong_frame_still_measured(self):
        with open(os.path.join(self.FIX, "uniform_scale_130x.txt")) as fh:
            s = vs.score_bbox_contract(fh.read())
        self.assertIsNotNone(s["implied_scale"])

    def test_total_miss_keeps_the_diagnostic(self):
        # A frame error so large that NOTHING scores leaves bestfit_dialect
        # None — exactly the "right shape, wrong frame" case the diagnostic
        # exists to recover, and exactly what a real-only gate suppressed.
        with open(os.path.join(self.FIX, "norm1000_clean.txt")) as fh:
            resp = json.loads(fh.read())
        resp["bbox_type"] = "real"
        for o in resp["objects"]:
            o["box_2d"] = [v * 100 for v in o["box_2d"]]
        s = vs.score_bbox_contract(json.dumps(resp))
        self.assertIsNone(s["bestfit_dialect"])
        self.assertIsNotNone(s["implied_scale"])

    def test_anchor_tally_uses_declared_order_on_norm_responses(self):
        # Regression for the od leak: binding od inside the gate left the
        # best-fit loop's trailing "yxyx" driving the anchor tally on every
        # norm-dialect response, transposing hits_anchor/iou_anchor.
        with open(os.path.join(self.FIX, "norm1000_clean.txt")) as fh:
            resp = json.loads(fh.read())
        resp["objects"].insert(0, {"label": "__IMAGE__",
                                   "box_2d": [0, 0, 1000, 1000]})
        s = vs.score_bbox_contract(json.dumps(resp))
        self.assertTrue(s["anchor_present"])
        self.assertEqual(s["anchor_implied_type"], "norm1000")
        self.assertEqual(s["hits_anchor"], s["hits_declared"])
        self.assertEqual(s["hits_anchor"], 6)


class TestFinetextParity(unittest.TestCase):
    """ft_ blocks must carry the same capture schema as suite blocks
    (blueprint P0-5): fingerprints, durations, throughput, req_* window."""

    def test_ft_block_carries_fingerprints_and_throughput(self):
        import finetext_probe as ftp
        r = {"_prompt_sha": "aa", "_images_sha": "bb", "_host": "h",
             "_server_version": "v", "_num_predict": 4000, "_num_ctx": 32768,
             "prompt_eval_count": 100, "eval_count": 50, "done_reason": "stop",
             "total_duration": 2_000_000_000, "load_duration": 1,
             "prompt_eval_duration": 1_000_000_000,
             "eval_duration": 1_000_000_000,
             "response": "x", "thinking": ""}
        blk = ftp.ft_block(r, model="M", tag="T", think=False,
                           num_ctx=32768, num_predict=4000)
        self.assertEqual(blk["prompt_sha"], "aa")
        self.assertEqual(blk["images_sha"], "bb")
        self.assertEqual(blk["req_num_ctx"], 32768)
        self.assertEqual(blk["gen_tps"], 50.0)
        self.assertEqual(blk["prefill_tps"], 100.0)

    def test_ft_done_skips_finished_and_reruns_capped_or_error(self):
        import finetext_probe as ftp
        self.assertTrue(ftp.ft_done({"eval_count": 10, "num_predict": 4000,
                                     "done_reason": "stop"}))
        self.assertFalse(ftp.ft_done({"eval_count": 4000, "num_predict": 4000,
                                      "done_reason": "length"}))
        self.assertFalse(ftp.ft_done({"error": "boom"}))
        self.assertFalse(ftp.ft_done(None))


class TestIncompleteRenderSaysSo(unittest.TestCase):
    """A mid-run scores file must not render as a finished report.

    Before the suite persisted after every arm, a scores file only ever
    reached disk COMPLETE, so the renderer never had to ask. It does now: a
    file is routinely well-formed and mid-run for the whole duration of a
    rung, and an arm that has not run yet is otherwise indistinguishable from
    a measured em-dash. summarize_matrix.py has had this guard since ADR 0012
    rule 8; this is the same rule for the other summarizer.
    """

    MODEL = "gemma4:12b-it-q4_K_M"
    ALL = list(sec.RENDERED_ARMS)

    def test_a_cell_missing_only_multi_3img_anchored_is_not_flagged(self):
        """47 historical cells on the benchmark host are missing that arm and
        nothing else, several of them published verbatim. Flagging those reads
        as "Do not publish" over a finished campaign, and a guard that cries
        wolf on the archive gets trained away."""
        r = self._render_with(["scene_single", "document_single", "multi_3img",
                               "multi_3img_anchored", "finetext"])
        self.assertNotIn("INCOMPLETE", r)
        r2 = self._render_with(["scene_single", "document_single", "multi_3img",
                                "finetext"])
        self.assertNotIn("INCOMPLETE", r2)

    def test_the_sentinel_is_still_the_last_arm_the_suite_runs(self):
        """The narrowed expectation rests on finetext running LAST: any run cut
        short is missing it. If the suite is reordered so something follows it,
        that reasoning breaks and this tuple has to be reconsidered."""
        names = [t[0] for t in vs.tests]
        self.assertEqual(names[-1], "finetext",
                         "finetext is no longer the last arm; RENDERED_ARMS "
                         "relies on it as the truncation sentinel")
        for arm in sec.RENDERED_ARMS:
            self.assertIn(arm, names)

    def _render_with(self, arms, extra=()):
        d = tempfile.mkdtemp()
        scores = {a: json.loads(json.dumps(THINKOFF.get(a, {"num_ctx": 16384})))
                  for a in arms}
        write(d, self.MODEL, "false", scores)
        argv = ["summarize_engine_compare.py", "--dir", d, "--think", "false",
                *extra, self.MODEL]
        out = io.StringIO()
        with mock_argv(argv), contextlib.redirect_stdout(out):
            sec.main()
        return out.getvalue()

    def test_a_mid_run_file_is_marked_incomplete(self):
        r = self._render_with(self.ALL[:2])
        self.assertIn("⚠ **INCOMPLETE**", r)
        self.assertIn("Do not publish", r)
        for absent in self.ALL[2:]:
            self.assertIn(absent, r, f"{absent} missing but not named")

    def test_the_marker_precedes_the_tables(self):
        """"Do not publish" has to reach the reader before the thing not to
        publish; a warning under the tables is one a skimmer never sees."""
        r = self._render_with(self.ALL[:2])
        self.assertLess(r.index("INCOMPLETE"), r.index("## Scene grounding"),
                        "the marker renders after the tables")

    def test_a_complete_cell_is_not_marked(self):
        self.assertNotIn("INCOMPLETE", self._render_with(self.ALL))

    def test_a_scoped_run_can_declare_its_scope(self):
        """ONLY_TESTS campaigns are complete for what they ran. A guard that
        cried wolf on every scoped render would be trained away within a week.
        """
        arms = self.ALL[:2]
        self.assertNotIn("INCOMPLETE",
                         self._render_with(arms, ("--expect", ",".join(arms))))

    def test_an_error_block_counts_as_run(self):
        """An arm that ran and failed HAS a result: it re-runs on resume and
        the tables mark it. Only an arm with no block at all is unrendered
        work, and conflating the two would flag every campaign with a failure.
        """
        d = tempfile.mkdtemp()
        scores = {a: json.loads(json.dumps(THINKOFF.get(a, {"num_ctx": 16384})))
                  for a in self.ALL}
        scores["multi_3img"] = {"error": "boom"}
        write(d, self.MODEL, "false", scores)
        argv = ["summarize_engine_compare.py", "--dir", d, "--think", "false",
                self.MODEL]
        out = io.StringIO()
        with mock_argv(argv), contextlib.redirect_stdout(out):
            sec.main()
        self.assertNotIn("INCOMPLETE", out.getvalue())


class TestIncrementalPersist(unittest.TestCase):
    """A killed rung must keep the arms it already paid for.

    vision_suite wrote scores_<tag>.json once, after the whole arm loop. The
    driver invokes it per CONTEXT-ladder rung, so the blast radius is one rung
    — but at the 131072 ceiling a single qwen3.6:35b-a3b think-on arm runs ~30
    minutes and the rung re-runs 9 of them. A Ctrl-C or OOM at minute 260
    discarded every completed arm, because nothing had reached disk.
    finetext_probe.py has always persisted its ft_ block per run.
    """

    ARMS = ["scene_single", "document_single", "multi_3img"]

    def _arm_by_prompt(self):
        """Map each arm-under-test to its own prompt, from `vision_suite.tests`
        — the same table the loop iterates. Identity, not call order."""
        by = {}
        for entry in vs.tests:
            if entry[0] not in self.ARMS:
                continue
            prompt = entry[1]() if callable(entry[1]) else entry[1]
            self.assertNotIn(prompt, by, f"{by.get(prompt)} and {entry[0]} "
                                         "share a prompt; the stub cannot tell "
                                         "them apart")
            by[prompt] = entry[0]
        self.assertEqual(sorted(by.values()), sorted(self.ARMS),
                         "an arm under test is not in vision_suite.tests")
        return by

    def _run_suite(self, tmpdir, fail_on=None):
        """Drive the real arm loop with a stubbed generator.

        The stub names each arm from ITS OWN PROMPT, and RECORDS every call
        rather than refusing to run past an expected count. Both matter, and
        the previous positional version made this harness blind to the exact
        regression it exists to catch:

        with `arm_done` mutated to return False — the severest possible
        resume-skip regression — a resume re-runs all three arms. Naming them
        positionally against a one-element `order` mislabelled the first call
        and raised AssertionError on the second; the loop's own
        `except Exception` swallowed that and wrote it as an error block over
        a previously-good result. `calls` therefore still read
        ["multi_3img"], and the key-only completeness assertion still passed,
        because an error block has a key too. The test stayed green while the
        contract it names was destroyed, and the good arm was overwritten in
        the process.

        So: never raise from the stub (the loop eats it), and let an overrun
        show up as extra entries in the returned list where the caller's
        assertion can see it.

        KeyboardInterrupt (not Exception) is the fault injected: it is what a
        Ctrl-C or an OOM kill actually raises, and it deliberately bypasses the
        loop's `except Exception` guard, so it reaches disk-state the same way
        a real kill does. An Exception would be caught and turned into an error
        block, which tests a different path.
        """
        by_prompt = self._arm_by_prompt()
        calls = []

        def fake_gen(prompt, images, **kw):
            # Unknown prompt means the loop ran something outside ONLY_TESTS.
            # Record it; do not raise, or the guard hides it.
            name = by_prompt.get(prompt, f"<unknown arm {prompt[:40]!r}>")
            calls.append(name)
            if name == fail_on:
                raise KeyboardInterrupt("simulated kill mid-rung")
            return {"response": "{}", "_host": "http://h:1", "_server_version": "v",
                    "_prompt_sha": "abc", "_images_sha": "def",
                    "prompt_eval_count": 10, "eval_count": 20,
                    "done_reason": "stop", "_num_ctx": 16384, "_num_predict": 2200,
                    "total_duration": 1, "eval_duration": 1,
                    "prompt_eval_duration": 1, "load_duration": 1}

        env = {"ONLY_TESTS": ",".join(self.ARMS)}
        argv = ["vision_suite.py", "http://h:1", "persisttag", "m:tag"]
        with mock.patch.object(vs, "DIR", tmpdir), \
                mock.patch.object(vs, "gen", fake_gen), \
                mock.patch.object(vs.client, "persist", lambda *a, **k: {}), \
                mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                vs.main()
            except KeyboardInterrupt:
                pass
        return calls

    def test_completed_arms_survive_a_kill_mid_rung(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = self._run_suite(tmp, fail_on="multi_3img")
            self.assertEqual(calls, self.ARMS, "all three arms should be attempted")
            path = os.path.join(tmp, "scores_persisttag.json")
            self.assertTrue(os.path.exists(path),
                            "arms 1-2 completed; nothing was written to disk")
            data = sec.load(path)
            self.assertIn("scene_single", data)
            self.assertIn("document_single", data)
            self.assertNotIn("multi_3img", data,
                             "the killed arm must not appear as a result")

    def test_persisted_arms_are_valid_resume_input(self):
        """SPEC H4b: what survives must be recognised as FINISHED, so a resume
        skips it rather than paying for it twice."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run_suite(tmp, fail_on="multi_3img")
            data = sec.load(os.path.join(tmp, "scores_persisttag.json"))
            for arm in ("scene_single", "document_single"):
                self.assertTrue(vs.arm_done(data[arm]),
                                f"{arm} persisted but does not read as done")
                self.assertFalse(sec.was_capped(data[arm]))

    def test_file_is_never_truncated_by_a_kill(self):
        """Every write is atomic (tmp + os.replace), so no reader can observe a
        half-written scores file — the artefact a campaign cannot rebuild."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run_suite(tmp, fail_on="multi_3img")
            path = os.path.join(tmp, "scores_persisttag.json")
            with open(path) as fh:
                json.load(fh)  # raises if truncated
            self.assertEqual([n for n in os.listdir(tmp) if n.endswith(".tmp")], [],
                             "atomic write left a .tmp file behind")

    def test_full_run_still_writes_every_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_suite(tmp)
            data = sec.load(os.path.join(tmp, "scores_persisttag.json"))
            self.assertEqual(sorted(data), sorted(self.ARMS))

    def test_a_resume_skips_what_survived_and_reruns_only_the_rest(self):
        """The whole point, end to end: kill mid-rung, then resume. What
        survived must cost nothing the second time, and the arm that never
        finished must run again. This is the resume contract the incremental
        write has to leave intact, not just the arm_done() verdict on it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            first = self._run_suite(tmp, fail_on="multi_3img")
            self.assertEqual(first, self.ARMS)
            second = self._run_suite(tmp)
            self.assertEqual(second, ["multi_3img"],
                             "resume re-ran arms that were already finished")
            data = sec.load(os.path.join(tmp, "scores_persisttag.json"))
            self.assertEqual(sorted(data), sorted(self.ARMS),
                             "resume did not complete the cell")
            # CONTENT, not just keys. An arm re-run and then failed leaves an
            # {"error": ...} block behind — same key, destroyed result — which
            # is exactly what the key-only assertion above cannot see, and
            # exactly what a broken resume produces.
            for arm in self.ARMS:
                self.assertNotIn("error", data[arm],
                                 f"{arm} came back as an error block; a resume "
                                 "overwrote a finished arm")
                self.assertIn("req_num_ctx", data[arm],
                              f"{arm} carries no generation stamp; it is not a "
                              "real result")


class TestPersistRegressions(unittest.TestCase):
    """Two ways per-arm persistence lost data that the end-of-rung write did not.

    Self-contained driver rather than TestIncrementalPersist's, so this stays
    additive against the open summarizer PR that rewrites that helper.
    """

    def drive(self, tmpdir, arms, fail_on=None, env=None, seed=None):
        if seed is not None:
            sec.save(os.path.join(tmpdir, "scores_regr.json"), seed)
        calls = []

        def fake_gen(prompt, images, **kw):
            name = arms[len(calls)]
            calls.append(name)
            if name == fail_on:
                raise RuntimeError("connection reset by peer")
            return {"response": "{}", "_host": "h", "_server_version": "v",
                    "prompt_eval_count": 10, "eval_count": 20,
                    "done_reason": "stop", "_num_ctx": 16384,
                    "_num_predict": 2200, "total_duration": 1,
                    "eval_duration": 1, "prompt_eval_duration": 1,
                    "load_duration": 1}

        e = {"ONLY_TESTS": ",".join(arms)}
        e.update(env or {})
        with mock.patch.object(vs, "DIR", tmpdir), \
                mock.patch.object(vs, "gen", fake_gen), \
                mock.patch.object(vs.client, "persist", lambda *a, **k: {}), \
                mock.patch.dict(os.environ, e, clear=False), \
                mock.patch.object(sys, "argv",
                                  ["vs.py", "http://h:1", "regr", "m:tag"]), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                vs.main()
            except BaseException:
                pass
        return sec.load(os.path.join(tmpdir, "scores_regr.json"))

    FINISHED = {"eval_count": 20, "num_predict": 2200, "done_reason": "stop"}
    CAPPED = {"eval_count": 8192, "num_predict": 8192, "done_reason": "length",
              "num_ctx": 16384, "gen_tps": 31.4}

    def test_force_plus_a_kill_keeps_arms_outside_the_run(self):
        """FORCE=1 skips the resume read, so `results` started EMPTY and the
        first per-arm write replaced the whole file with one arm. Before
        per-arm persistence an aborted FORCE run was harmless — the abort was
        the escape hatch."""
        with tempfile.TemporaryDirectory() as tmp:
            seed = {"document_single": dict(self.FINISHED),
                    "scene_single": dict(self.FINISHED),
                    "multi_3img": dict(self.FINISHED)}
            data = self.drive(tmp, ["scene_single", "multi_3img"],
                              fail_on="multi_3img", env={"FORCE": "1"},
                              seed=seed)
            self.assertIn("document_single", data,
                          "an arm this run never touched was destroyed")

    def test_force_still_reruns_everything(self):
        """The other half of the FORCE fix: seeding `results` from disk must not
        turn FORCE into a resume. A finished arm must still re-run."""
        with tempfile.TemporaryDirectory() as tmp:
            ran = []
            sec.save(os.path.join(tmp, "scores_regr.json"),
                     {"scene_single": dict(self.FINISHED)})

            def fake_gen(prompt, images, **kw):
                ran.append(1)
                return {"response": "{}", "_host": "h", "_server_version": "v",
                        "prompt_eval_count": 10, "eval_count": 20,
                        "done_reason": "stop", "_num_ctx": 16384,
                        "_num_predict": 2200, "total_duration": 1,
                        "eval_duration": 1, "prompt_eval_duration": 1,
                        "load_duration": 1}

            with mock.patch.object(vs, "DIR", tmp), \
                    mock.patch.object(vs, "gen", fake_gen), \
                    mock.patch.object(vs.client, "persist", lambda *a, **k: {}), \
                    mock.patch.dict(os.environ, {"ONLY_TESTS": "scene_single",
                                                 "FORCE": "1"}, clear=False), \
                    mock.patch.object(sys, "argv",
                                      ["vs.py", "http://h:1", "regr", "m:tag"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                vs.main()
            self.assertEqual(len(ran), 1, "FORCE did not re-run a finished arm")

    def test_an_error_does_not_destroy_the_measurement_it_replaces(self):
        """finetext_probe.py has carried a `prior` guard for exactly this since
        a capped rung-1 measurement was lost to a rung-2 transport failure."""
        with tempfile.TemporaryDirectory() as tmp:
            data = self.drive(tmp, ["scene_single"], fail_on="scene_single",
                              seed={"scene_single": dict(self.CAPPED)})
            blk = data["scene_single"]
            self.assertIn("error", blk)
            self.assertEqual(blk.get("prior", {}).get("eval_count"), 8192,
                             "the capped measurement it replaced is gone")

    def test_an_error_over_a_capped_arm_still_escalates(self):
        """capped_arms drops non-escalatable errors, and the driver breaks its
        rung loop when nothing is capped — so an arm that was capped and then
        hit a transport error left the ladder silently, and the cell read as
        converged with no NOT-CONVERGED marker."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scores_x.json")
            sec.save(path, {"scene_single": {"error": "connection reset",
                                             "prior": dict(self.CAPPED)}})
            self.assertEqual(sec.capped_arms(path), ["scene_single"])

    def test_an_error_with_no_capped_history_still_does_not_escalate(self):
        """The existing rule stands: a bare request failure re-runs at the same
        rung rather than buying a bigger window."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scores_y.json")
            sec.save(path, {"scene_single": {"error": "connection reset"}})
            self.assertEqual(sec.capped_arms(path), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
