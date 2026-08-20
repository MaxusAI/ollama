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


if __name__ == "__main__":
    unittest.main(verbosity=2)
