#!/usr/bin/env python3
"""Offline tests for the harness's own verdict logic. No GPU, no server.

The per-arch ladder verdict is the thing most worth pinning: a flat ladder means
an UNPATCHED payload for nemotron_h_omni but is the CORRECT result for gemma4
under 004. A shared heuristic gets this backwards, and has, twice. These tests
fail if the diagnosis is ever wired to the shape alone instead of the arch.

    python3 test_verdicts.py
"""
import json
import os
import subprocess
import sys
import time
import tomllib
import pathlib
import unittest
from unittest import mock

sys.path.insert(0, ".")

import checks  # noqa: E402
import measure_ladder  # noqa: E402
import probes  # noqa: E402
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
        r = checks.check_pinned_budget(StubClient([3270, 2306]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], PASS)

    def test_pre_005_overshoot_fails_the_ceiling_invariant(self):
        """3390 delivered against a 3328 ceiling — the 005 defect class."""
        r = checks.check_pinned_budget(StubClient([3390, 2306]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("OVERSHOOT", r["diagnosis"])
        invariant = [a for a in r["arms"] if a["arm"] == "ceiling_invariant"][0]
        self.assertFalse(invariant["ok"])

    def test_unmeasured_overshoot_still_caught_by_the_invariant(self):
        """A value nobody has recorded must still fail if it breaks the ceiling."""
        r = checks.check_pinned_budget(StubClient([4001, 2306]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("OVERSHOOT", r["diagnosis"])

    def test_control_drift_is_reported_separately(self):
        r = checks.check_pinned_budget(StubClient([3270, 2500]), self.EXPECT,
                                       "nemotron_h_omni", 0)
        self.assertEqual(r["status"], FAIL)
        self.assertIn("control", r["diagnosis"])

    def test_missing_pinned_block_skips_rather_than_passing_silently(self):
        r = checks.check_pinned_budget(StubClient([]), DYNAMIC, "gemma4", 0)
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



if __name__ == "__main__":
    unittest.main(verbosity=2)


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
