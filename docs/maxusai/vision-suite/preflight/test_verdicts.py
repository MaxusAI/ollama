#!/usr/bin/env python3
"""Offline tests for the harness's own verdict logic. No GPU, no server.

The per-arch ladder verdict is the thing most worth pinning: a flat ladder means
an UNPATCHED payload for nemotron_h_omni but is the CORRECT result for gemma4
under 004. A shared heuristic gets this backwards, and has, twice. These tests
fail if the diagnosis is ever wired to the shape alone instead of the arch.

    python3 test_verdicts.py
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import pathlib
import re
import unittest
from unittest import mock

sys.path.insert(0, ".")

import checks  # noqa: E402
import measure_ladder  # noqa: E402
import probes  # noqa: E402
import release_matrix  # noqa: E402
from checks import FAIL, PASS, SKIP  # noqa: E402

SIZES = ["256x144", "512x288", "1024x576", "2048x1152", "3072x1728"]


class StubClient:
    """Returns canned token counts; records nothing else."""

    def __init__(self, values, think=None):
        self.values = list(values)
        self.think = think or {}
        self.host = "http://stub"
        self.queue_waits = []

    def visual_tokens(self, model, size, baseline, **kw):
        if "image_max_tokens" in kw and kw["image_max_tokens"] is not None:
            return self.values.pop(0), {"_queue_wait_s": 0.0}
        return self.values.pop(0), {"_queue_wait_s": 0.0}

    def generate(self, model, prompt, **kw):
        return dict(self.think, _queue_wait_s=0.0)

    def ps(self):
        return []


DYNAMIC = {"model": "m", "scaling": "dynamic", "ladder_tolerance": 2,
           "ladder": [266, 266, 578, 2306, 3270], "budget_max_tokens": 3328}
FLAT = {"model": "m", "scaling": "flat", "ladder_tolerance": 2,
        "ladder": [1102] * 5, "budget_max_tokens": 1120}


class TestLadderVerdict(unittest.TestCase):

    def test_dynamic_arch_matching_ladder_passes(self):
        r = checks.check_ladder(StubClient(DYNAMIC["ladder"]), DYNAMIC,
                                "nemotron_h_omni", SIZES, 0)
        self.assertEqual(r["status"], PASS)

    def test_flat_arch_matching_flat_ladder_passes(self):
        """The regression that matters: flat is CORRECT for gemma4 under 004."""
        r = checks.check_ladder(StubClient(FLAT["ladder"]), FLAT, "gemma4", SIZES, 0)
        self.assertEqual(r["status"], PASS)
        self.assertEqual(r["shape"], "flat")

    def test_flat_result_on_dynamic_arch_diagnoses_unpatched_payload(self):
        r = checks.check_ladder(StubClient([258] * 5), DYNAMIC,
                                "nemotron_h_omni", SIZES, 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("FLAT", r["diagnosis"])
        self.assertIn("unpatched", r["diagnosis"].lower())

    def test_scaling_result_on_flat_arch_does_not_say_unpatched(self):
        """The inverse must NOT reuse the unpatched-payload wording."""
        r = checks.check_ladder(StubClient([132, 363, 922, 1091, 1082]), FLAT,
                                "gemma4", SIZES, 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("VARIES", r["diagnosis"])
        self.assertNotIn("unpatched", r["diagnosis"].lower())

    def test_shifted_but_same_shape_is_not_diagnosed_as_unpatched(self):
        """A uniform offset is a behaviour change, not a missing patch."""
        r = checks.check_ladder(StubClient([v + 40 for v in DYNAMIC["ladder"]]),
                                DYNAMIC, "nemotron_h_omni", SIZES, 0)
        self.assertEqual(r["status"], FAIL)
        self.assertNotIn("unpatched", r["diagnosis"].lower())


class TestPinnedBudget(unittest.TestCase):

    EXPECT = dict(DYNAMIC, pinned={
        "size": "2048x1152", "pin_tokens": 3328, "expect_tokens": 3270,
        "tolerance": 4, "enforce_ceiling_invariant": True,
        "control_expect_tokens": 2306, "control_tolerance": 4})

    def test_post_005_values_pass(self):
        r = checks.check_pinned_image_token_budget(StubClient([3270, 2306]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], PASS)

    def test_pre_005_overshoot_fails_the_ceiling_invariant(self):
        """3390 delivered against a 3328 ceiling — the 005 defect class."""
        r = checks.check_pinned_image_token_budget(StubClient([3390, 2306]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("OVERSHOOT", r["diagnosis"])
        invariant = [a for a in r["arms"] if a["arm"] == "ceiling_invariant"][0]
        self.assertFalse(invariant["ok"])

    def test_unmeasured_overshoot_still_caught_by_the_invariant(self):
        """A value nobody has recorded must still fail if it breaks the ceiling."""
        r = checks.check_pinned_image_token_budget(StubClient([4001, 2306]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("OVERSHOOT", r["diagnosis"])

    def test_control_drift_is_reported_separately(self):
        r = checks.check_pinned_image_token_budget(StubClient([3270, 2500]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("control", r["diagnosis"])

    def test_missing_pinned_block_skips_rather_than_passing_silently(self):
        r = checks.check_pinned_image_token_budget(StubClient([]), DYNAMIC, "gemma4", 0)
        self.assertEqual(r["status"], SKIP)


class TestThinkFormat(unittest.TestCase):

    EXPECT = dict(DYNAMIC, think_format={
        "num_predict": 4000, "require_nonempty_response": True,
        "require_valid_json": True, "require_nonempty_thinking": True})

    def test_valid_json_after_thinking_passes(self):
        c = StubClient([], think={"response": '{"facts": ["a"]}',
                                  "thinking": "hmm", "eval_count": 1248})
        r = checks.check_think_format(c, self.EXPECT, "nemotron_h_omni", 600)
        self.assertEqual(r["status"], PASS)

    def test_num_predict_trap_is_named_as_such(self):
        """eval_count == num_predict with thinking present is the trap, not a
        vision failure — the distinction that cost real time."""
        c = StubClient([], think={"response": "", "thinking": "x" * 3306,
                                  "eval_count": 4000})
        r = checks.check_think_format(c, self.EXPECT, "nemotron_h_omni", 600)
        self.assertEqual(r["status"], FAIL)
        # The diagnosis was rewritten when think-mode sampling was re-measured
        # (runaway-reasoning-under-think.md). Assert the SEMANTICS that matter —
        # that it names the cap and denies a vision failure — not a fixed phrase.
        self.assertIn("eval_count", r["diagnosis"])
        self.assertIn("num_predict", r["diagnosis"])
        self.assertIn("NOT a vision failure", r["diagnosis"])

    def test_stock_signature_is_distinguished_from_the_trap(self):
        """Empty response well under budget is the stock think+format bug."""
        c = StubClient([], think={"response": "", "thinking": "short",
                                  "eval_count": 562})
        r = checks.check_think_format(c, self.EXPECT, "nemotron_h_omni", 600)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("stock", r["diagnosis"])
        self.assertNotIn("num_predict trap", r["diagnosis"])

    def test_num_predict_below_floor_refuses_to_run(self):
        low = dict(DYNAMIC, think_format={"num_predict": 120})
        r = checks.check_think_format(StubClient([]), low, "nemotron_h_omni", 600)
        self.assertEqual(r["status"], checks.ERROR)
        self.assertIn("floor", r["summary"])


# The quality arm delegates capped-arm detection to the vision suite's
# summarize_engine_compare, which is NOT on every lineage: release/0.32.1-dynres
# carries preflight/ without the suite, and CI asserts these tests still pass
# there. checks.check_quality already skips gracefully in that case, but these
# tests stub the vision_suite.py existence check to True precisely so they can
# exercise the scoring path -- so they, and only they, need the sibling module.
# Skip rather than fail: the harness is fine standalone, the test simply has
# nothing to assert about a scorer that cannot run.
try:
    import summarize_engine_compare as _sec  # noqa: F401
    _HAVE_SUITE = True
except ImportError:
    _HAVE_SUITE = False


@unittest.skipUnless(_HAVE_SUITE,
                     "summarize_engine_compare is not on this tree; the quality "
                     "arm cannot be scored here and check_quality skips it")
class TestQualityThresholds(unittest.TestCase):
    """check_quality turns vision_suite.py's scores into a verdict. The score
    field names below are the real ones vision_suite.py writes — if it ever
    renames them, this fails rather than silently scoring 0."""

    QUALITY = {"status": "measured", "tests": ["scene_single", "document_single"],
               "min_json_valid": 1.0, "min_label_recall": 0.70,
               "min_qty_price_exact": 0.70}
    GOOD = {
        "scene_single": {"json_valid": True, "labels_found": 5, "labels_total": 6,
                         "bbox_hits": 4, "prompt_eval_count": 2305},
        "document_single": {"json_valid": True, "items_found": 5, "items_total": 5,
                            "qty_price_right": 5, "total_right": True},
    }

    def _verdict(self, scores, tag="unittest-quality"):
        path = os.path.join(checks.SUITE_DIR, f"scores_{tag}.json")
        with open(path, "w") as fh:
            json.dump(scores, fh)
        real_exists = os.path.exists

        # Both the suite script and its ground truth must read as present. These
        # tests exercise the SCORING logic; the absence path has its own test.
        # Without the vision_suite.py leg they pass only where the rest of the
        # suite happens to exist, and fail on any tree carrying preflight/ alone
        # — release/0.32.1-dynres is exactly that.
        def present(p):
            return (True if p.endswith(("ground_truth.json", "vision_suite.py"))
                    else real_exists(p))

        try:
            with mock.patch.object(checks.subprocess, "run",
                                   return_value=mock.Mock(stdout="", stderr="")), \
                 mock.patch.object(checks.os.path, "exists", present):
                return checks.check_quality("http://stub", self.QUALITY,
                                            {"model": "m"}, "gemma4", tag)
        finally:
            os.remove(path)

    def test_scores_above_the_floors_pass(self):
        r = self._verdict(self.GOOD)
        self.assertEqual(r["status"], PASS)
        self.assertAlmostEqual(r["actual"]["label_recall"], 5 / 6, places=3)
        self.assertAlmostEqual(r["actual"]["qty_price_exact"], 1.0, places=3)

    def test_label_recall_below_floor_fails(self):
        bad = json.loads(json.dumps(self.GOOD))
        bad["scene_single"]["labels_found"] = 1        # 1/6 = 0.17
        r = self._verdict(bad)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("label_recall", r["summary"])

    def test_invalid_json_fails(self):
        bad = json.loads(json.dumps(self.GOOD))
        bad["scene_single"]["json_valid"] = False
        r = self._verdict(bad)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("json_valid", r["summary"])

    def test_an_errored_test_is_reported_not_ignored(self):
        bad = json.loads(json.dumps(self.GOOD))
        bad["document_single"] = {"error": "timed out"}
        r = self._verdict(bad)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("errored", r["summary"])

    def test_no_thresholds_recorded_skips(self):
        r = checks.check_quality("http://stub", None, {"model": "m"}, "gemma4", "t")
        self.assertEqual(r["status"], SKIP)

    def test_absent_vision_suite_skips_rather_than_erroring(self):
        """release/0.32.1-dynres carries preflight/ without the vision suite. A
        missing scorer is not a build defect and must not read as one."""
        real_exists = os.path.exists
        with mock.patch.object(checks.os.path, "exists",
                               lambda p: False if p.endswith("vision_suite.py")
                               else real_exists(p)):
            r = checks.check_quality("http://stub", self.QUALITY, {"model": "m"},
                                     "gemma4", "t")
        self.assertEqual(r["status"], SKIP)
        self.assertIn("vision_suite.py", r["summary"])


class TestContention(unittest.TestCase):
    """Queue starvation is invisible: a saturated single slot times requests out
    while the server reports perfectly healthy. A run that hits it must say so
    rather than emit a false failure."""

    def test_quiet_endpoint_passes(self):
        c = StubClient([])
        c.queue_waits = [("baseline", 0.0), ("1024x576", 0.4)]
        self.assertEqual(checks.check_exclusivity(c, 10.0)["status"], PASS)

    def test_large_queue_wait_reports_contention_not_failure(self):
        c = StubClient([])
        c.queue_waits = [("baseline", 0.1), ("2048x1152", 412.7)]
        r = checks.check_exclusivity(c, 10.0)
        self.assertEqual(r["status"], checks.CONTENTION)
        self.assertNotEqual(r["status"], FAIL)
        self.assertIn("2048x1152", r["summary"])

    def test_no_probes_recorded_does_not_crash(self):
        self.assertEqual(checks.check_exclusivity(StubClient([]), 10.0)["status"], PASS)


class TestPayloadProofCustomBounds(unittest.TestCase):
    """Which bounds carry '(custom value)' is per-arch, not universal.

    visionServerArgs passes both --image-min-tokens and --image-max-tokens for
    gemma4 and nemotron_h_omni, but only the min for the qwen VL family, whose
    max is llama.cpp's structural set_limit_image_tokens(8, 4096) ceiling. A
    check that demands "(custom value)" on both fails a correct qwen build.
    """

    QWEN = {"image_min_pixels": 1048576, "image_max_pixels": 4194304,
            "patch_stride": 32, "budget_min_tokens": 1024,
            "budget_max_tokens": 4096, "custom_bounds": ["min"]}
    BOTH = {"image_min_pixels": 262144, "image_max_pixels": 3407872,
            "patch_stride": 32, "budget_min_tokens": 256,
            "budget_max_tokens": 3328}

    @staticmethod
    def log(minv, maxv, min_custom=True, max_custom=True):
        c = " (custom value)"
        return (f"load_hparams: image_min_pixels:   {minv}"
                f"{c if min_custom else ''}\n"
                f"load_hparams: image_max_pixels:   {maxv}"
                f"{c if max_custom else ''}\n")

    def proof(self, expect, log):
        with mock.patch.object(checks, "container_logs", return_value=log):
            return checks.check_payload_proof(expect, "arch", "container", 0)

    def test_min_only_custom_passes_when_declared(self):
        r = self.proof(self.QWEN,
                       self.log(1048576, 4194304, max_custom=False))
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_min_only_custom_fails_when_both_are_declared(self):
        """The pre-existing behaviour, unchanged for arches that set both."""
        r = self.proof(self.BOTH, self.log(262144, 3407872, max_custom=False))
        self.assertEqual(r["status"], FAIL)
        self.assertIn("not marked", r["summary"])

    def test_unexpected_custom_marking_fails(self):
        """A bound declared untunable that starts logging custom means the arch
        gate changed. As much a finding as a flag going missing."""
        r = self.proof(self.QWEN, self.log(1048576, 4194304, max_custom=True))
        self.assertEqual(r["status"], FAIL)
        self.assertIn("arch gate changed", r["summary"])

    def test_wrong_value_still_fails_on_an_uncustomised_bound(self):
        """Relaxing the marking must not relax the value."""
        r = self.proof(self.QWEN,
                       self.log(1048576, 9999999, max_custom=False))
        self.assertEqual(r["status"], FAIL)
        self.assertIn("expected 4194304", r["summary"])

    def test_default_is_both_bounds(self):
        """An arch that says nothing keeps the strict behaviour."""
        self.assertNotIn("custom_bounds", self.BOTH)
        r = self.proof(self.BOTH, self.log(262144, 3407872))
        self.assertEqual(r["status"], PASS, r["summary"])


class TestPayloadProofAttribution(unittest.TestCase):
    """A pixel block is graded only against the arch whose load emitted it.

    The 2026-09-02 deploy smoke on a server WITHOUT OLLAMA_MAX_LOADED_MODELS=1
    read qwen3.8's (correct) 1048576/4194304 block as gemma4's and failed a
    healthy deploy with "expected 161280, got 1048576" — the same log carried
    gemma4's own correct blocks four times. Same family as the warm-up
    Reserve() log-scraper trap: unattributed log parsing.

    Attribution keys on what the check itself verifies: the runner-launch
    line's --image-min/max-tokens flags, matched on exactly the bounds the
    arch declares it passes (custom_bounds), plus patch_size*n_merge ==
    patch_stride — the flags alone collide (qwen35 and qwen2.5vl both pass
    only --image-min-tokens 1024; strides 32 vs 28 split them).

    Windows with NO launch line keep the old last-block behaviour — that is
    the fallback the bare-log tests above exercise, and a forced fresh load
    always brings its launch line into the window.
    """

    GEMMA = {"image_min_pixels": 161280, "image_max_pixels": 2580480,
             "patch_stride": 48, "budget_min_tokens": 70,
             "budget_max_tokens": 1120}
    QWEN = {"image_min_pixels": 1048576, "image_max_pixels": 4194304,
            "patch_stride": 32, "budget_min_tokens": 1024,
            "budget_max_tokens": 4096, "custom_bounds": ["min"]}

    @staticmethod
    def segment(min_tok, max_tok, patch, merge, minv, maxv,
                min_custom=True, max_custom=True):
        flags = f"--image-min-tokens {min_tok}"
        if max_tok is not None:
            flags += f" --image-max-tokens {max_tok}"
        c = " (custom value)"
        return (
            f'time=x level=INFO source=llama_server.go:435 '
            f'msg="starting llama-server" cmd="/usr/lib/ollama/llama-server '
            f'--model /root/.ollama/models/blobs/sha256-x --port 1 {flags} '
            f'-b 1024 -ub 1024"\n'
            f"load_hparams: image_size:         224\n"
            f"load_hparams: patch_size:         {patch}\n"
            f"load_hparams: n_merge:            {merge}\n"
            f"load_hparams: image_min_pixels:   {minv}"
            f"{c if min_custom else ''}\n"
            f"load_hparams: image_max_pixels:   {maxv}"
            f"{c if max_custom else ''}\n")

    def gemma_seg(self):
        return self.segment(70, 1120, 16, 3, 161280, 2580480)

    def qwen_seg(self):
        return self.segment(1024, None, 16, 2, 1048576, 4194304,
                            max_custom=False)

    def proof(self, expect, log):
        with mock.patch.object(checks, "container_logs", return_value=log):
            return checks.check_payload_proof(expect, "arch", "container", 0)

    def test_anothers_block_after_ours_does_not_fail_us(self):
        """The deploy-smoke reproduction: gemma4's correct block, then
        qwen3.8's load lands in the same window. Last-block grading fails a
        healthy payload against another model's numbers."""
        r = self.proof(self.GEMMA, self.gemma_seg() + self.qwen_seg())
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_ours_after_anothers_still_grades_ours(self):
        r = self.proof(self.QWEN, self.gemma_seg() + self.qwen_seg())
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_reversed_order_both_ways(self):
        log = self.qwen_seg() + self.gemma_seg()
        r = self.proof(self.QWEN, log)
        self.assertEqual(r["status"], PASS, r["summary"])
        r = self.proof(self.GEMMA, log)
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_only_foreign_loads_is_a_fail_not_a_borrowed_pass(self):
        """Another arch's correct block must never satisfy this arch — and the
        failure must say what WAS seen, or the operator re-reads the log
        blind."""
        r = self.proof(self.GEMMA, self.qwen_seg())
        self.assertEqual(r["status"], FAIL)
        self.assertIn("no load of this arch", r["summary"])
        self.assertIn("1 load(s) of other models", r["diagnosis"])

    def test_flag_collision_is_split_by_stride(self):
        """qwen35 and qwen2.5vl both pass only --image-min-tokens 1024; only
        patch_size*n_merge tells their segments apart. A qwen2.5vl-shaped
        segment (stride 28) must not satisfy qwen35 (stride 32)."""
        imposter = self.segment(1024, None, 14, 2, 802816, 4194304,
                                max_custom=False)
        r = self.proof(self.QWEN, imposter)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("no load of this arch", r["summary"])

    def test_wrong_values_in_our_own_segment_still_fail(self):
        """Attribution narrows WHICH block is graded, never HOW. A segment
        launched with our flags whose payload logs the wrong pixels is
        exactly the defect payload_proof exists to catch."""
        broken = self.segment(70, 1120, 16, 3, 161280, 9999999)
        r = self.proof(self.GEMMA, broken)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("expected 2580480", r["summary"])


class TestContainerLogsWindow(unittest.TestCase):
    """The --since window must be unambiguous, or it is the wrong window.

    Docker assumes the client's LOCAL timezone for a timestamp carrying no
    zone, while container_logs formats UTC. Off UTC the two disagree by the
    offset: east of it the window opens early and admits load_hparams lines
    from previous loads of OTHER models, which is how a caller that should
    have reported "no data" instead gets a plausible wrong answer.

    These assert the emitted string, not docker's behaviour, because the seam
    we control is the timestamp.
    """

    def since_arg(self, epoch):
        """The timestamp container_logs actually emits, via the log_cmd seam."""
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(probes.subprocess, "run", fake_run):
            probes.container_logs("c", epoch, log_cmd="SINCE={since}")
        return seen["cmd"].split("=", 1)[1]

    def test_since_carries_an_explicit_zone(self):
        self.assertTrue(self.since_arg(1_755_000_000).endswith("Z"),
                        "a zoneless timestamp is read as docker's local time")

    def test_since_is_utc_not_local(self):
        """Belt and braces: the Z must be truthful, not decoration."""
        epoch = 1_755_000_000
        self.assertEqual(self.since_arg(epoch),
                         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch)))

    def test_default_docker_argv_carries_the_same_stamp(self):
        """The log_cmd path and the plain `docker logs` path must not drift."""
        seen = {}

        def fake_run(argv, **kw):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with mock.patch.object(probes.subprocess, "run", fake_run):
            probes.container_logs("c", 1_755_000_000)
        self.assertIn("--since", seen["argv"])
        self.assertTrue(seen["argv"][seen["argv"].index("--since") + 1].endswith("Z"))


class TestMeasureLadderBudgetFields(unittest.TestCase):
    """What measure_ladder emits for patch_stride and the budget fields.

    Two failure directions, and they pull opposite ways. Emitting a value the
    tool cannot stand behind puts a wrong row into expectations.toml, which is
    the worse one; but dropping a value the operator supplied sends them back to
    retype it, and hand-transcription into this file is what ADR 0012 rule 8 is
    about. So the rule these pin is: keep the stride when nothing contradicts
    it, drop it when something does.
    """

    QWEN35 = (1048576, 4194304, ["min"])   # stride 32, both divide exactly
    GEMMA4_PIXELS = (161280, 2580480, [])  # another model's line in the window

    def fields(self, stride, budgets):
        lines, notes = measure_ladder.budget_fields(stride, budgets)
        return "\n".join(lines), "\n".join(notes)

    def test_measured_budgets_are_converted(self):
        out, notes = self.fields(32, self.QWEN35)
        self.assertIn("patch_stride = 32", out)
        self.assertIn("budget_min_tokens = 1024", out)
        self.assertIn("budget_max_tokens = 4096", out)
        self.assertIn("image_min_pixels = 1048576", out)
        self.assertIn("custom_bounds = [\"min\"]", out)
        self.assertEqual(notes, "")

    def test_stride_survives_an_empty_budget_read(self):
        """THE BUG. --stride 32 and no readable log used to emit stride 0."""
        out, notes = self.fields(32, None)
        self.assertIn("patch_stride = 32", out)
        self.assertNotIn("patch_stride = 0", out)
        self.assertIn("budget_min_tokens = 0   # TODO", out)
        self.assertIn("cross-checked", notes)

    def test_a_contradicted_stride_is_not_emitted_as_fact(self):
        """A remainder cannot say which half is wrong, so neither is asserted."""
        out, notes = self.fields(32, self.GEMMA4_PIXELS)
        self.assertIn("patch_stride = 0", out)
        self.assertIn("--stride said 32", out, "the contested value is lost")
        self.assertIn("budget_min_tokens = 0   # TODO", out)
        self.assertNotIn("157", out)      # what the unguarded division produced
        self.assertIn("REFUSING", notes)

    def test_no_stride_is_never_inferred(self):
        """Both 16 and 32 divide qwen35's counts; a guess would be invisible."""
        out, notes = self.fields(None, self.QWEN35)
        self.assertIn("patch_stride = 0", out)
        self.assertNotIn("budget_min_tokens = 4096", out)
        self.assertNotIn("budget_min_tokens = 1024", out)
        self.assertIn("will not guess", notes)

    def test_every_path_emits_the_same_five_keys(self):
        """A row missing a key reads as "not applicable" rather than TODO."""
        keys = ["patch_stride", "budget_min_tokens", "budget_max_tokens",
                "image_min_pixels", "image_max_pixels"]
        for stride, budgets in [(32, self.QWEN35), (32, None), (None, None),
                                (32, self.GEMMA4_PIXELS), (None, self.QWEN35)]:
            out, _ = self.fields(stride, budgets)
            emitted = [l.split(" =")[0] for l in out.split("\n")]
            self.assertEqual(emitted[:5], keys, f"stride={stride} budgets={budgets}")

    def test_both_custom_bounds_are_left_implicit(self):
        """custom_bounds is the exception list; ["min","max"] is the default."""
        out, _ = self.fields(32, (1048576, 4194304, ["max", "min"]))
        self.assertNotIn("custom_bounds", out)


class TestLineageProfilesTrackOneVersionFamily(unittest.TestCase):
    """The two lineage profiles describe ONE host and must not drift apart.

    cuda-dynres-903 and mlx-cuda are both keyed on the containerised `-dynres`
    stamp of the same CUDA machine; which one applies is chosen by --platform,
    not by the version. Every other profile is pinned to an exact baseline on
    purpose (mlx-metal: "tightened at first baseline") and a new version there
    wants a new profile with re-measured expectations, per ADR 0011.

    This exists because widening one and not the other has already happened
    twice. #175 fixed mlx-cuda's pattern at creation -- it required "-maxusai-",
    the NATIVE macOS stamp, and so could not match the container it describes.
    #218 then widened cuda-dynres-903 to 0.33.x for the v0.33.0 sync and left
    mlx-cuda at 0.32.

    The failure is silent in the worst way: --platform mlx-cuda does not fail
    red, it fails to RESOLVE A PROFILE, so the gate becomes unreachable rather
    than failing. A version pattern is the one field that breaks by standing
    still.
    """

    LINEAGE = ("cuda-dynres-903", "mlx-cuda")

    @classmethod
    def setUpClass(cls):
        with open(pathlib.Path(__file__).parent / "expectations.toml", "rb") as fh:
            cls.exp = tomllib.load(fh)

    def test_lineage_profiles_share_a_version_pattern(self):
        pats = {p: self.exp["profiles"][p]["version_pattern"] for p in self.LINEAGE}
        self.assertEqual(len(set(pats.values())), 1,
                         f"lineage profiles drifted: {pats}")

    def test_pinned_profiles_are_not_swept_along(self):
        """Widening must not be applied to a profile pinned at a baseline."""
        for pid in ("mlx-metal", "metal-0-32-14", "cpu", "rocm-0-32-1-dynres"):
            pat = self.exp["profiles"][pid]["version_pattern"]
            self.assertNotIn("[23]", pat,
                             f"{pid} is baseline-pinned; a new version needs a "
                             f"new profile with re-measured expectations (ADR 0011)")

    # ADR 0032 amendment (2026-09-04): a point tag `v<release>-dynres.N` names a
    # deployed commit between folds. `scripts/env.sh` stamps builds from
    # `git describe --tags --first-parent`, so once v0.33.2-dynres.1 existed at
    # the deployed commit, every later build on main described as
    # `0.33.2-dynres.1-<n>-g<sha>` -- a string the lineage patterns rejected,
    # which would have failed the version gate (exit 2) for the same payload.
    POINT_TAG_STAMPS = (
        "0.33.2-dynres.1-0-g2b95b4a",   # the deployed build, re-stamped from the point tag
        "0.33.2-dynres.1-27-gb54d4d0",  # main after the point tag
        "0.33.2-dynres-5-g2b95b4a",     # the same deployed build, pre-point-tag stamp
        "0.33.0-dynres-0-g5171887",     # a fold tag stamp
        "0.33.2-dynres-0f3a71be1",      # bare-sha form
    )
    FOREIGN_STAMPS = (
        "0.34.0-dynres-0-gabcdef0",     # next family: needs its own fold + widening
        "0.33.2-maxusai-2b95b4a5",      # the native Metal stamp is not this lineage
        "0.33.2-dynres.1",              # a tag name is not a build stamp
        "0.33.2-dynres.x-0-g2b95b4a",   # point tags are numeric
    )

    def test_lineage_patterns_admit_point_tags_and_reject_foreign_families(self):
        for pid in self.LINEAGE:
            pat = re.compile(self.exp["profiles"][pid]["version_pattern"])
            for stamp in self.POINT_TAG_STAMPS:
                self.assertRegex(stamp, pat, f"{pid} must admit {stamp}")
            for stamp in self.FOREIGN_STAMPS:
                self.assertNotRegex(stamp, pat, f"{pid} must reject {stamp}")

    def test_point_tags_do_not_leak_into_pinned_profiles(self):
        """A `.N` stamp is the same payload as its fold on the LINEAGE profiles
        only; a baseline-pinned profile keeps refusing to guess."""
        for pid in ("mlx-metal-0-33-2", "metal-0-32-14", "cpu", "rocm-0-32-1-dynres"):
            pat = re.compile(self.exp["profiles"][pid]["version_pattern"])
            self.assertNotRegex("0.33.2-dynres.1-0-g2b95b4a", pat, pid)


class TestExpectationsFile(unittest.TestCase):
    """The data file is the contract; keep it internally consistent."""

    @classmethod
    def setUpClass(cls):
        with open(pathlib.Path(__file__).parent / "expectations.toml", "rb") as fh:
            cls.exp = tomllib.load(fh)

    def test_every_profile_arch_has_an_expectation_block(self):
        for pid, prof in self.exp["profiles"].items():
            for arch in prof["arches"]:
                self.assertIn(arch, self.exp["expect"].get(pid, {}),
                              f"profile {pid} lists arch {arch} with no "
                              f"[expect.{pid}.{arch}] block")

    # mlx-cuda is declared but has never been measured (its qwen rows are
    # status="unmeasured" and it exits 4), so it has no pin to assert. Naming it
    # here keeps the hole VISIBLE: the previous version of this test filtered on
    # platform == "mlx-metal" and a second test asserted non-MLX profiles must
    # NOT carry a pin, which together left mlx-cuda — the widest version_pattern
    # in the file — silently exempt from both.
    KNOWN_UNPINNED = {"mlx-cuda"}

    def test_every_measured_mlx_profile_pins_its_mlx_build(self):
        """A profile serving the MLX payload must record which MLX it was
        measured on, or mlx_payload_pin has nothing to assert and a future MLX
        bump inherits its ladders silently."""
        for pid, prof in self.exp["profiles"].items():
            if not str(prof.get("platform", "")).startswith("mlx"):
                continue
            if pid in self.KNOWN_UNPINNED:
                self.assertIsNone(prof.get("mlx_build"),
                                  f"{pid} now has a pin; drop it from "
                                  f"KNOWN_UNPINNED so it is enforced")
                continue
            sha = prof.get("mlx_build")
            self.assertTrue(sha, f"profile {pid} serves the MLX payload but "
                                 f"records no mlx_build; mlx_payload_pin would "
                                 f"skip and the pin would go unasserted")
            self.assertRegex(sha, r"^[0-9a-f]{40}$",
                             f"{pid}: mlx_build should be the full MLX_VERSION "
                             f"commit")

    def test_pixel_budgets_equal_tokens_times_stride_squared(self):
        for pid, arches in self.exp["expect"].items():
            for arch, e in arches.items():
                if e.get("status") != "measured" or "image_max_pixels" not in e:
                    continue
                s = e["patch_stride"]
                self.assertEqual(e["image_max_pixels"], e["budget_max_tokens"] * s * s,
                                 f"{pid}/{arch}: max pixels != max_tokens * S^2")
                self.assertEqual(e["image_min_pixels"], e["budget_min_tokens"] * s * s,
                                 f"{pid}/{arch}: min pixels != min_tokens * S^2")

    def test_ladder_length_matches_the_declared_geometries(self):
        n = len(self.exp["ladder_sizes"])
        for pid, arches in self.exp["expect"].items():
            for arch, e in arches.items():
                if "ladder" in e:
                    self.assertEqual(len(e["ladder"]), n,
                                     f"{pid}/{arch}: ladder has {len(e['ladder'])} "
                                     f"entries for {n} geometries")

    def test_declared_scaling_matches_the_recorded_ladder(self):
        """Catches a row edited without its scaling field, which is exactly how
        the verdict logic would silently invert."""
        for pid, arches in self.exp["expect"].items():
            for arch, e in arches.items():
                if "ladder" not in e:
                    continue
                flat = len(set(e["ladder"])) == 1
                self.assertEqual(flat, e["scaling"] == "flat",
                                 f"{pid}/{arch}: scaling={e['scaling']} but the "
                                 f"recorded ladder is {'flat' if flat else 'varying'}")

    def test_think_format_num_predict_respects_the_global_floor(self):
        floor = self.exp["min_num_predict"]
        for pid, arches in self.exp["expect"].items():
            for arch, e in arches.items():
                tf = e.get("think_format")
                if tf and "num_predict" in tf:
                    self.assertGreaterEqual(tf["num_predict"], floor,
                                            f"{pid}/{arch}: num_predict below floor")

    def test_unmeasured_blocks_carry_a_reason(self):
        for pid, arches in self.exp["expect"].items():
            for arch, e in arches.items():
                if e.get("status") == "unmeasured":
                    self.assertTrue(e.get("reason", "").strip(),
                                    f"{pid}/{arch}: unmeasured with no reason")


class _PoisonStub:
    """tags/unload/generate for check_poison_probe; responses are canned."""

    def __init__(self, responses, models=("qwen2.5vl:3b-q4_K_M",)):
        self.responses = list(responses)
        self.models = list(models)
        self.queue_waits = []
        self.unloads = 0

    def tags(self):
        return self.models

    def unload(self, model):
        self.unloads += 1

    def generate(self, model, prompt, **kw):
        return self.responses.pop(0)


class TestPoisonProbe(unittest.TestCase):
    """The #214 degenerate-decode fingerprint and the probe's verdict wiring.

    The classifier must key on the actual poison signature — done_reason null
    or a single repeated glyph — and NOT on short/empty responses, which the
    num_predict trap and think-budget exhaustion produce for other reasons.
    """

    EXPECT = {"model": "qwen2.5vl:3b-q4_K_M"}

    def test_degenerate_fingerprints(self):
        deg = checks.is_degenerate_decode
        self.assertTrue(deg("?" * 31, None))          # clip-path poison
        self.assertTrue(deg("!" * 31, None))          # 0.7.1 Go-engine poison
        self.assertTrue(deg("?" * 31, "stop"))        # glyph run, however finished
        self.assertTrue(deg("anything", None))        # done_reason null alone
        self.assertFalse(deg("The photo shows a checkerboard.", "stop"))
        self.assertFalse(deg("", "stop"))             # empty is a different defect
        self.assertFalse(deg("OK", "stop"))           # short is not degenerate
        self.assertFalse(deg("???", "stop"))          # under the run-length floor

    def test_skip_without_expectation(self):
        r = checks.check_poison_probe(_PoisonStub([]), None, "cuda-dynres-903")
        self.assertEqual(r["status"], SKIP)

    def test_pass_on_healthy_decode_and_clean_slot(self):
        stub = _PoisonStub([
            {"response": "A black and white checkerboard.", "done_reason": "stop"},
            {"response": "OK", "done_reason": "stop"},
        ])
        r = checks.check_poison_probe(stub, self.EXPECT, "cuda-dynres-903")
        self.assertEqual(r["status"], PASS)
        self.assertEqual(stub.unloads, 2)  # fresh slot in, clean server out

    def test_fail_on_degenerate_trigger(self):
        stub = _PoisonStub([
            {"response": "?" * 31, "done_reason": None},
            {"response": "?" * 31, "done_reason": None},
        ])
        r = checks.check_poison_probe(stub, self.EXPECT, "cuda-dynres-903")
        self.assertEqual(r["status"], FAIL)
        self.assertIn("trigger request", r["summary"])

    def test_fail_on_slot_residue(self):
        stub = _PoisonStub([
            {"response": "A checkerboard pattern.", "done_reason": "stop"},
            {"response": "?" * 31, "done_reason": None},
        ])
        r = checks.check_poison_probe(stub, self.EXPECT, "cuda-dynres-903")
        self.assertEqual(r["status"], FAIL)
        self.assertIn("slot residue", r["summary"])

    def test_fail_when_model_missing(self):
        stub = _PoisonStub([], models=["something-else:latest"])
        r = checks.check_poison_probe(stub, self.EXPECT, "cuda-dynres-903")
        self.assertEqual(r["status"], FAIL)

    def test_poison_tables_reference_real_profiles(self):
        with open(pathlib.Path(__file__).parent / "expectations.toml", "rb") as fh:
            exp = tomllib.load(fh)
        for pid, entry in exp.get("poison", {}).items():
            self.assertIn(pid, exp["profiles"],
                          f"[poison.{pid}] names a profile that does not exist")
            self.assertTrue(entry.get("model", "").strip(),
                            f"[poison.{pid}] has no model")



class PoisonNodeCorroboration(unittest.TestCase):
    """The meter (llama/compat/801) is optional; absence must never fail, and
    node-level overflow must fail even when the decode reads healthily."""

    HEALTHY = {"response": "A black and white checkered pattern.", "done_reason": "stop"}
    EXPECT = {"model": "qwen2.5vl:3b-q4_K_M"}

    def _stub(self):
        return _PoisonStub([self.HEALTHY, {"response": "OK", "done_reason": "stop"}])

    def test_no_container_is_not_a_failure(self):
        r = checks.check_poison_probe(self._stub(), self.EXPECT, "cuda-dynres-903",
                                      container=None)
        self.assertEqual(r["status"], checks.PASS)
        self.assertIn("no container resolved", r["summary"])

    def test_meter_off_is_not_a_failure(self):
        with mock.patch.object(checks, "container_logs", return_value="no meter here"):
            r = checks.check_poison_probe(self._stub(), self.EXPECT,
                                          "cuda-dynres-903", container="c")
        self.assertEqual(r["status"], checks.PASS)
        self.assertIn("meter off", r["summary"])

    def test_log_read_failure_is_not_a_failure(self):
        with mock.patch.object(checks, "container_logs", side_effect=OSError("nope")):
            r = checks.check_poison_probe(self._stub(), self.EXPECT,
                                          "cuda-dynres-903", container="c")
        self.assertEqual(r["status"], checks.PASS)
        self.assertIn("log read failed", r["summary"])

    def test_clean_meter_is_reported(self):
        line = ("CLIP_NODE_STATS name=ffn_down-31 op=MUL_MAT type=f32 n=15728640 "
                "max_abs=49031.0 hr=0.7485 n_gt32k=36 n_gt49k=0 n_gt60k=0 "
                "n_inf=0 n_nan=0")
        with mock.patch.object(checks, "container_logs", return_value=line):
            r = checks.check_poison_probe(self._stub(), self.EXPECT,
                                          "cuda-dynres-903", container="c")
        self.assertEqual(r["status"], checks.PASS)
        self.assertIn("node meter clean", r["summary"])

    def test_overflow_fails_even_when_the_decode_looks_healthy(self):
        """The case the text check cannot see: inf at the node, readable text."""
        line = ("CLIP_NODE_STATS name=ffn_down-31 op=MUL_MAT type=f32 n=15728640 "
                "max_abs=53471.0 hr=0.8163 n_gt32k=15 n_gt49k=2 n_gt60k=0 "
                "n_inf=3 n_nan=0")
        with mock.patch.object(checks, "container_logs", return_value=line):
            r = checks.check_poison_probe(self._stub(), self.EXPECT,
                                          "cuda-dynres-903", container="c")
        self.assertEqual(r["status"], checks.FAIL)
        self.assertIn("non-finite", r["summary"])
        self.assertIn("ffn_down-31", r["summary"])

    def test_nan_counts_too(self):
        line = "CLIP_NODE_STATS name=ffn_down-31 n_inf=0 n_nan=2"
        with mock.patch.object(checks, "container_logs", return_value=line):
            r = checks.check_poison_probe(self._stub(), self.EXPECT,
                                          "cuda-dynres-903", container="c")
        self.assertEqual(r["status"], checks.FAIL)


@unittest.skipUnless(
    os.path.exists(os.path.join(checks.SUITE_DIR, "summarize_engine_compare.py")),
    "release lineages ship preflight/ without the suite; quality scoring "
    "SKIPs there and so must its tests")
class TestQualityCappedExcluded(unittest.TestCase):
    """A capped arm scores json_valid: False as a side effect of truncation;
    counting it as a QUALITY failure misattributes a harness setting to the
    model (ADR 0012 conv 9 — the same rule the summarizers enforce)."""

    def test_scoped_to_this_runs_tests_not_the_shared_file(self):
        # The scores file is shared per (platform, arch) tag across profiles;
        # a capped arm left by ANOTHER profile's test list must not fail a
        # run that never requested it.
        scores = {"bbox_contract": {"json_valid": False, "eval_count": 2200,
                                    "num_predict": 2200,
                                    "done_reason": "length"},
                  "scene_single": {"json_valid": True, "eval_count": 100,
                                   "num_predict": 2200,
                                   "done_reason": "stop"}}
        eligible, capped = checks.quality_eligible(
            scores, tests=["scene_single"])
        self.assertEqual(list(eligible), ["scene_single"])
        self.assertEqual(capped, [])

    def test_capped_blocks_leave_the_quality_denominator(self):
        scores = {"scene_single": {"json_valid": False, "eval_count": 2200,
                                   "num_predict": 2200,
                                   "done_reason": "length"},
                  "document_single": {"json_valid": True, "eval_count": 100,
                                      "num_predict": 2200,
                                      "done_reason": "stop"}}
        eligible, capped = checks.quality_eligible(scores)
        self.assertEqual(list(eligible), ["document_single"])
        self.assertEqual(capped, ["scene_single"])

    def test_errored_blocks_are_neither_eligible_nor_capped(self):
        scores = {"scene_single": {"error": "boom"}}
        eligible, capped = checks.quality_eligible(scores)
        self.assertEqual(eligible, {})
        self.assertEqual(capped, [])




class TestMlxPayloadPin(unittest.TestCase):
    """No mlx-metal profile asserted which MLX it was actually running.

    (The original wording here said these profiles "exist BECAUSE the MLX pin
    moved". That is false for two of them — a5d65906 and c82b0464 both carry
    adf21dea, so 0-32-14 was cut for the llama.cpp move — and the pins landed by
    this same change are what refute it. The rule stands on its own: a profile
    serving the MLX payload must record which MLX it was measured on.)

    mlx-metal-0-33-0 says so about itself: "The MLX runner is the payload under
    test here, so the MLX bump alone requires this new profile". But
    `llama_cpp_build` is the only payload identity the harness checked, and it
    describes the wrong payload on this platform (these profiles correctly set
    `patchset = []`). So another MLX bump under the same base version would
    match `version_pattern`, resolve the profile, and inherit its measured
    ladders and budgets with nothing failing loudly — the exact failure
    payload_pin exists to prevent, on the platform whose payload IS MLX.

    The runner already reports the value: x/mlxrunner/server.go logs
    "MLX engine initialized" with a `git describe` string whose g-suffix is the
    MLX_VERSION commit, e.g. 0.32.1-37-gc793734 for c793734eb715…
    """

    PIN = "c793734eb715dbcfdb1ced58e348ec53c2d7ed85"
    LINE = ('time=2026-08-30T09:20:01.893+10:00 level=INFO source=server.go:47 '
            'msg="MLX engine initialized" "MLX version"=%s device=gpu\n')

    def pin(self, profile, log, container="native", log_cmd="cat x"):
        with mock.patch.object(probes, "container_logs", return_value=log):
            return checks.check_mlx_payload_pin(profile, container, 0,
                                                log_cmd=log_cmd)

    def test_matching_pin_passes(self):
        r = self.pin({"mlx_build": self.PIN}, self.LINE % "0.32.1-37-gc793734")
        self.assertEqual(r["status"], PASS, r["summary"])
        self.assertIn("c793734", r["summary"])

    def test_a_different_mlx_build_fails(self):
        """The whole point: the previous pin under the same base version."""
        r = self.pin({"mlx_build": self.PIN}, self.LINE % "0.32.0-12-g27fec90")
        self.assertEqual(r["status"], FAIL)
        self.assertIn("27fec90", r.get("actual", ""))

    def test_no_declared_pin_skips_loudly(self):
        """Platforms with no MLX payload must not be failed, and must not be
        silently passed either."""
        r = self.pin({}, self.LINE % "0.32.1-37-gc793734")
        self.assertEqual(r["status"], SKIP)
        self.assertTrue(r.get("diagnosis"), "a skip must say how to close it")

    def test_no_log_access_skips_rather_than_passing(self):
        r = checks.check_mlx_payload_pin({"mlx_build": self.PIN}, None, 0,
                                         log_cmd=None)
        self.assertEqual(r["status"], SKIP)

    def test_an_unparseable_version_never_passes(self):
        """An exact-tag build reports no g-suffix. We cannot confirm the pin
        from that, and 'cannot confirm' must not read as 'confirmed'."""
        r = self.pin({"mlx_build": self.PIN}, self.LINE % "0.32.1")
        self.assertNotEqual(r["status"], PASS)
        self.assertIn("0.32.1", r["summary"] + str(r.get("actual", "")))

    def test_absent_line_never_passes(self):
        """No engine-init line in the window means nothing was verified."""
        r = self.pin({"mlx_build": self.PIN}, "some unrelated log\n")
        self.assertNotEqual(r["status"], PASS)




class TestMlxPayloadPinWindow(unittest.TestCase):
    """The window must be enforced by the CHECK, not by the fetch command.

    container_logs does `log_cmd.format(container=..., since=...)`, so a
    template with no {since} placeholder silently drops the window — and
    `cat <serve log>` is exactly what this check's own diagnosis prescribes on
    the native macOS path, which is the ONLY path where mlx_build is declared.
    Before this fix a month-old engine-init line, in a run that loaded no model
    at all, returned PASS "matches the measured payload".
    """

    PIN = "27fec909a3df9e572f5195607a453e273e7d80d0"

    @staticmethod
    def line(ts, ver):
        return (f'time={ts} level=INFO source=server.go:47 '
                f'msg="MLX engine initialized" "MLX version"={ver} device=gpu\n')

    def pin(self, log, since, profile=None):
        with mock.patch.object(probes, "container_logs", return_value=log):
            return checks.check_mlx_payload_pin(
                profile or {"mlx_build": self.PIN}, "native", since,
                log_cmd="cat serve.log")

    def test_a_line_older_than_the_window_never_passes(self):
        stale = self.line("2026-07-31T09:20:01.893+10:00", "0.32.0-12-g27fec90")
        r = self.pin(stale, time.time() - 5)
        self.assertNotEqual(r["status"], PASS,
                            "a month-old line satisfied a five-second window")

    def test_a_line_inside_the_window_still_passes(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + "+00:00"
        with mock.patch.object(probes, "container_logs",
                               return_value=self.line(now, "0.32.1-3-g27fec90")):
            r = checks.check_mlx_payload_pin({"mlx_build": self.PIN}, "native",
                                             0, log_cmd="cat serve.log")
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_nothing_in_the_window_is_a_failure_not_a_skip(self):
        """check_payload_proof FAILs for the same condition. A SKIP leaves the
        run's exit code at 0, so a deploy gated on it goes green with the pin
        unasserted."""
        r = self.pin("unrelated log line\n", time.time() - 5)
        self.assertEqual(r["status"], FAIL)

    def test_a_dirty_mlx_tree_fails_rather_than_skipping(self):
        """`git describe --dirty` (x/mlxrunner/mlx/CMakeLists.txt) marks a
        modified MLX source tree — precisely the different-payload case this
        check exists to catch. It must not degrade to 'cannot parse'."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + "+00:00"
        with mock.patch.object(
                probes, "container_logs",
                return_value=self.line(now, "0.32.1-3-g27fec90-dirty")):
            r = checks.check_mlx_payload_pin({"mlx_build": self.PIN}, "native",
                                             0, log_cmd="cat serve.log")
        self.assertEqual(r["status"], FAIL)
        self.assertIn("dirty", r["summary"] + str(r.get("diagnosis", "")))

    def test_a_pin_too_short_to_identify_a_commit_never_passes(self):
        """`short.startswith(expected)` accepted a 3-char pin against a commit
        it does not name."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + "+00:00"
        with mock.patch.object(probes, "container_logs",
                               return_value=self.line(now, "0.32.1-3-gc790000")):
            r = checks.check_mlx_payload_pin({"mlx_build": "c79"}, "native", 0,
                                             log_cmd="cat serve.log")
        self.assertNotEqual(r["status"], PASS)


class TestSchemaConstrainedFormat(unittest.TestCase):
    """ADR 0033 replaced the MLX runner's constrained sampling with upstream's
    grammar engine, and the harness gated it with `json.loads()` succeeding.

    fmt="json" was the only format usage anywhere in preflight, so a regression
    that emits syntactically valid but SCHEMA-VIOLATING JSON passed clean —
    which is precisely what constrained decoding exists to prevent. These pin a
    schema the response must actually conform to.
    """

    SCHEMA = {"type": "object",
              "properties": {"count": {"type": "integer"},
                             "colors": {"type": "array",
                                        "items": {"type": "string"}}},
              "required": ["count", "colors"]}

    def run_check(self, body):
        expect = {"model": "m",
                  "think_format": {"num_predict": 4000, "schema": self.SCHEMA,
                                   "require_nonempty_thinking": False}}
        client = StubClient([], think={"response": body, "thinking": "hm",
                                       "eval_count": 40})
        return checks.check_think_format(client, expect, "arch", 600)

    def test_a_conforming_response_passes(self):
        r = self.run_check('{"count": 3, "colors": ["red", "blue"]}')
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_valid_json_that_violates_the_schema_fails(self):
        """The exact regression the old gate could not see: parses fine, wrong
        shape entirely."""
        r = self.run_check('{"facts": ["a", "b", "c"]}')
        self.assertEqual(r["status"], FAIL)
        self.assertIn("count", r["summary"])

    def test_a_wrong_scalar_type_fails(self):
        """Constrained decoding is supposed to make this unreachable; if it
        stops doing so the harness must say it, not shrug at valid JSON."""
        r = self.run_check('{"count": "three", "colors": ["red"]}')
        self.assertEqual(r["status"], FAIL)
        self.assertIn("integer", r["summary"])

    def test_a_wrong_array_item_type_fails(self):
        r = self.run_check('{"count": 1, "colors": [7]}')
        self.assertEqual(r["status"], FAIL)

    def test_the_schema_is_sent_as_the_format(self):
        """A schema that is not actually transmitted gates nothing."""
        sent = {}

        class Recorder(StubClient):
            def generate(self, model, prompt, **kw):
                sent.update(kw)
                return dict(self.think, _queue_wait_s=0.0)

        expect = {"model": "m",
                  "think_format": {"num_predict": 4000, "schema": self.SCHEMA}}
        client = Recorder([], think={"response": '{"count": 1, "colors": []}',
                                     "thinking": "hm", "eval_count": 40})
        checks.check_think_format(client, expect, "arch", 600)
        self.assertEqual(sent.get("fmt"), self.SCHEMA)

    def test_no_schema_keeps_the_old_json_only_behaviour(self):
        expect = {"model": "m", "think_format": {"num_predict": 4000}}
        client = StubClient([], think={"response": '{"facts": []}',
                                       "thinking": "hm", "eval_count": 40})
        r = checks.check_think_format(client, expect, "arch", 600)
        self.assertEqual(r["status"], PASS, r["summary"])


class TestBudgetProvenance(unittest.TestCase):
    """`status = "measured"` covered two kinds of value, and only one was.

    In an mlx-metal block the LADDER is measured on the build under test, but
    budget_min/max_tokens and image_min/max_pixels are not: the native MLX path
    emits no load_hparams line, so check_payload_proof — the only consumer that
    could observe them — SKIPs, and check_pinned_image_token_budget SKIPs too
    for want of a `pinned` block. Sixteen values across four profiles were
    declared measured while nothing on the platform could see them.

    They are not deleted: they are true, they document the budget, and the
    arithmetic test still catches a later hand-edit typo. What changes is that
    the file now says which half was observed.
    """

    @classmethod
    def setUpClass(cls):
        with open(pathlib.Path(__file__).parent / "expectations.toml", "rb") as fh:
            cls.exp = tomllib.load(fh)

    def test_unobservable_budgets_say_so(self):
        for pid, arches in self.exp["expect"].items():
            prof = self.exp["profiles"].get(pid, {})
            if prof.get("platform") != "mlx-metal":
                continue
            for arch, e in arches.items():
                if not isinstance(e, dict) or "budget_max_tokens" not in e:
                    continue
                self.assertIs(e.get("budgets_observed"), False,
                              f"{pid}/{arch} declares budgets that nothing on "
                              f"this platform can observe, without saying so")

    def test_the_flag_cannot_dodge_a_check_that_could_have_run(self):
        """The abuse guard. `budgets_observed = false` is a statement about the
        PLATFORM, not a way to excuse a block on one that has a load log."""
        for pid, arches in self.exp["expect"].items():
            for arch, e in arches.items():
                if not isinstance(e, dict) or e.get("budgets_observed") is not False:
                    continue
                ref = self.exp["profiles"].get(pid, {}).get("reference_image", "")
                self.assertIn("no container", ref,
                              f"{pid}/{arch} claims budgets are unobservable, but "
                              f"its profile has a container and payload_proof "
                              f"could read them")

    def test_payload_proof_says_which_kind_of_skip_it_is(self):
        r = checks.check_payload_proof(
            {"budgets_observed": False, "image_min_pixels": 1, "patch_stride": 1,
             "budget_min_tokens": 1, "image_max_pixels": 1,
             "budget_max_tokens": 1}, "arch", None, 0)
        self.assertEqual(r["status"], SKIP)
        self.assertIn("not observable", r["summary"])

    def test_a_normal_block_keeps_the_old_skip(self):
        r = checks.check_payload_proof({"image_min_pixels": 1}, "arch", None, 0)
        self.assertEqual(r["status"], SKIP)
        self.assertIn("no container resolved", r["summary"])


class TestAspectLadder(unittest.TestCase):
    """The token ladder varies image SIZE at a fixed 16:9, which is one axis.

    ladder_sizes is 256x144 .. 3072x1728 — every rung 16:9. A budget-fill arch
    scales any input to the same grid, so for gemma4 all five rungs are
    mathematically forced to the same number and "5/5 geometries within +/-2"
    reports one degree of freedom as five. Measured on the deployed build,
    gemma4:12b spans 1058-1102 once the ASPECT RATIO moves (1:1 -> 1091,
    4:3 -> 1066, 4:1 -> 1058), so the axis exists and the ladder never touched
    it.

    This runs as a separate expectation rather than by widening ladder_sizes:
    that list is global, and every measured ladder in the file — CUDA, ROCm,
    metal, mlx — was taken at those five geometries. Adding a rung would
    invalidate ~29 blocks on hosts that cannot be re-measured from here, which
    ADR 0011 forbids doing to make anything line up.
    """

    def stub(self, values):
        return StubClient(list(values))

    def test_matching_counts_pass(self):
        e = {"model": "m", "ladder_tolerance": 2,
             "aspect_ladder": {"768x768": 1091, "1024x768": 1066}}
        r = checks.check_aspect_ladder(self.stub([1066, 1091]), e, "arch")
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_a_drifted_count_fails_and_names_the_geometry(self):
        e = {"model": "m", "ladder_tolerance": 2,
             "aspect_ladder": {"768x768": 1091, "1024x768": 1066}}
        r = checks.check_aspect_ladder(self.stub([1200, 1091]), e, "arch")
        self.assertEqual(r["status"], FAIL)
        self.assertIn("1024x768", r["summary"])

    def test_tolerance_is_honoured(self):
        e = {"model": "m", "ladder_tolerance": 2,
             "aspect_ladder": {"768x768": 1091}}
        r = checks.check_aspect_ladder(self.stub([1093]), e, "arch")
        self.assertEqual(r["status"], PASS, r["summary"])

    def test_no_expectation_skips(self):
        r = checks.check_aspect_ladder(self.stub([]), {"model": "m"}, "arch")
        self.assertEqual(r["status"], SKIP)

    def test_a_declared_probe_must_add_a_degree_of_freedom(self):
        """Its whole purpose is to move a value the 16:9 ladder cannot. A probe
        whose geometries all predict the SAME number would pass while proving
        nothing, which is the defect this check exists to fix."""
        with open(pathlib.Path(__file__).parent / "expectations.toml", "rb") as fh:
            exp = tomllib.load(fh)
        for pid, arches in exp["expect"].items():
            for arch, e in arches.items():
                if not isinstance(e, dict) or "aspect_ladder" not in e:
                    continue
                vals = list(e["aspect_ladder"].values())
                self.assertGreater(len(vals), 1, f"{pid}/{arch}: one geometry")
                flat = e.get("ladder", [None])[0]
                distinct = set(vals) | ({flat} if flat is not None else set())
                self.assertGreater(
                    len(distinct), 1,
                    f"{pid}/{arch}: every aspect geometry predicts the same "
                    f"count as the 16:9 ladder, so it adds no coverage")


class TestReleaseMatrixColumns(unittest.TestCase):
    """The generated matrix is the fold's headline artifact — it is embedded in
    README.md and attached to the release — so a cell that reads green for a
    check nobody ran is the exact claim release_matrix.py's docstring exists to
    forbid ("a hand-written green badge is worse than none").

    "Output quality" was mapped to {"text_baseline", "quality"} and read green
    on every run ever recorded. Neither name is the quality verdict:
    check_quality records "extraction_quality", nothing has ever emitted
    "quality", and "text_baseline" is preflight.py's prefix calibration, PASS
    for every arch that gets past the probe. A missed recall floor rendered
    green; so did a run that never asked for the quality arm.
    """

    COLUMN = "Output quality"
    HERE = pathlib.Path(__file__).parent

    # Names no longer emitted by the harness but carried by artifacts recorded
    # before a rename. Kept explicit so this test still holds on a lineage
    # cherry-picked without runs/, and so adding one is a deliberate act.
    RECORDED_ALIASES = {"pinned_budget"}

    def render(self, results, platform="cuda",
               version="0.33.2-dynres-5-gsynthet"):
        """Render one synthetic artifact and return {column: cell} for its row.

        Goes through main() rather than the group logic directly: the bug was
        in the mapping the renderer reads, which a unit test of worst() cannot
        see.
        """
        run = {"meta": {"platform": platform, "version": version,
                        "started_utc": "20260904T120000"},
               "results": results}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.json")
            with open(path, "w") as fh:
                json.dump(run, fh)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                release_matrix.main([path])
        columns = [g for g, _ in release_matrix.GROUPS] + ["measured on"]
        for line in out.getvalue().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] == f"**{platform}**":
                self.assertEqual(len(cells) - 1, len(columns), line)
                return dict(zip(columns, cells[1:]))
        self.fail(f"no row rendered for {platform}:\n{out.getvalue()}")

    @staticmethod
    def rec(check, status, arch="gemma4", summary="synthetic"):
        return {"check": check, "arch": arch, "status": status,
                "summary": summary}

    # A run always carries these; they are what made the false green invisible
    # — every other column was legitimately green beside it.
    def with_baseline(self, *extra):
        return [self.rec("version", PASS, arch=None),
                self.rec("text_baseline", PASS),
                self.rec("token_ladder", PASS),
                self.rec("endpoint_exclusive", PASS, arch=None)] + list(extra)

    def test_a_failed_quality_arm_is_not_green(self):
        """The regression. A missed recall floor is a FAIL an operator reads
        the matrix to find; it rendered green because the column was watching
        two names the check does not use."""
        row = self.render(self.with_baseline(
            self.rec("extraction_quality", FAIL,
                     summary="label_recall 0.42 < 0.7")))
        self.assertEqual(row[self.COLUMN], "**FAIL**")
        self.assertEqual(row["Image size ladder"], "green")   # unaffected

    def test_no_quality_result_reads_not_run(self):
        """--quality is opt-in (preflight.py), so most runs record no quality
        verdict at all. Absence is shown, never assumed green — the same rule
        the generator applies to a surface with no run."""
        row = self.render(self.with_baseline())
        self.assertEqual(row[self.COLUMN], "not run")

    def test_a_skipped_quality_arm_reads_skipped(self):
        """check_quality SKIPs on a lineage carrying preflight/ without
        vision_suite.py, and where no thresholds are recorded. Not measured is
        not the same as measured and passed."""
        row = self.render(self.with_baseline(
            self.rec("extraction_quality", SKIP,
                     summary="vision_suite.py is not present in this tree")))
        self.assertEqual(row[self.COLUMN], "skipped")

    def test_a_passing_quality_arm_is_green(self):
        """The column must still be able to say green — from the check that
        actually measured it."""
        row = self.render(self.with_baseline(
            self.rec("extraction_quality", PASS,
                     summary="json_valid=1.00 label_recall=0.83")))
        self.assertEqual(row[self.COLUMN], "green")

    def test_the_prefix_calibration_cannot_carry_a_column(self):
        """text_baseline is not a verdict: preflight.py records it PASS for
        every arch that reaches the ladder, and its only other status (ERROR)
        returns before any further check runs. A check that cannot fail can
        only inflate the group it is mapped into."""
        for column, names in release_matrix.GROUPS:
            self.assertNotIn("text_baseline", names, column)

    def test_every_mapped_name_is_one_something_records(self):
        """The bug in one line: "quality" was a name nothing had ever emitted,
        and a column watching only names nobody records reports on nothing.
        Rather than re-checking two names by hand, every name in GROUPS must be
        one the harness emits today or one a recorded artifact carries."""
        src = "\n".join((self.HERE / f).read_text()
                        for f in ("checks.py", "preflight.py"))
        known = set(re.findall(r'result\(\s*"([a-z0-9_]+)"', src))
        known |= set(re.findall(r'^\s*name = "([a-z0-9_]+)"', src, re.M))
        known |= self.RECORDED_ALIASES
        for artifact in sorted((self.HERE / "runs").glob("*.json")):
            try:
                recorded = json.loads(artifact.read_text())
            except (ValueError, OSError):
                continue          # a half-written artifact proves nothing here
            known |= {r.get("check") for r in recorded.get("results", [])}
        self.assertIn("extraction_quality", known,
                      "checks.py no longer emits the name this test relies on")
        for column, names in release_matrix.GROUPS:
            for name in sorted(names):
                self.assertIn(name, known,
                              f'the "{column}" column watches for "{name}", '
                              f"which neither checks.py/preflight.py emits nor "
                              f"any recorded run carries")


# The main block must stay at the END of the file: unittest.main() runs the
# classes defined ABOVE it, so a class appended after it silently never runs
# as a script — which is exactly what happened to PoisonNodeCorroboration's
# six tests between #230 and this line (58 defined, 52 collected).
if __name__ == "__main__":
    unittest.main(verbosity=2)
