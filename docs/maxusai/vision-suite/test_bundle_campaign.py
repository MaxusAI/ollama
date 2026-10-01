#!/usr/bin/env python3
"""Offline tests for bundle_campaign.py, and for the runner's tag rule it shares.

A campaign document's tables re-render from its bundle. So these tests pin the
three ways a bundle could misstate a run:
- a cell under the wrong tag;
- a cell that is not the file it came from;
- a meta the cells contradict.

The values here are fixtures, not measurements.

    python3 test_bundle_campaign.py
"""
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bundle_campaign as bc  # noqa: E402
import summarize_engine_compare as sec  # noqa: E402

HOST, BUILD = "http://127.0.0.1:11434", "0.34.4-dynres-0-gtest"
COMMITTED = os.path.join(HERE, "bench-runs", "vision-campaign-2026-09-18-mlx8a7ba949.json")


def scores(iou=0.9, host=HOST, build=BUILD):
    stamp = {"host": host, "server_version": build}
    return {"scene_single": dict(stamp, bbox_mean_iou=iou, eval_count=500),
            "finetext": dict(stamp, recall_22px=4)}


def write(rundir, tag, data, kind="scores"):
    with open(os.path.join(rundir, f"{kind}_{tag}.json"), "w") as fh:
        json.dump(data, fh)


def snapshot(rundir):
    """Every file in a run directory, as bytes."""
    files = {}
    for name in os.listdir(rundir):
        with open(os.path.join(rundir, name), "rb") as fh:
            files[name] = fh.read()
    return files


def run(*argv):
    """bundle_campaign.main(argv) -> (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = bc.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class TestArmPrefixIsTheRunners(unittest.TestCase):
    """arm_prefix restates run_engine_compare.sh's tag rule, which is shell.

    This runs the runner's own lines against it for every kind of arm. If either
    side changes, this fails; the bundler would otherwise look for cells under
    names the runner no longer writes."""

    def runner_lines(self):
        with open(os.path.join(HERE, "run_engine_compare.sh")) as fh:
            src = fh.read()
        base = re.search(r"^\s*(base=\$\(printf '%s' \"\$m\" \| tr ':\.' '__'\))\s*$", src, re.M)
        block = re.search(r"^(\s*if \[ \"\$REPEATS\" -gt 1 \] \|\| \[ -n \"\$TAG_PREFIX\" \]; then\n"
                          r".*?\n\s*fi)\n", src, re.M | re.S)
        self.assertTrue(base and block, "run_engine_compare.sh no longer builds tags with the "
                        "lines arm_prefix restates: change the two together")
        return base.group(1) + "\n" + block.group(1)

    def runner_tag(self, model, think, rep, repeats, tag_prefix):
        script = 'm="$1"; think="$2"; rep="$3"\n' + self.runner_lines() + '\nprintf "%s" "$tag"\n'
        env = dict(os.environ, REPEATS=str(repeats), TAG_PREFIX=tag_prefix)
        return subprocess.run(["sh", "-c", script, "sh", model, think, str(rep)], env=env,
                              capture_output=True, text=True, check=True).stdout

    def test_every_kind_of_arm_gets_the_runners_tag(self):
        arms = ((1, "", 1),                  # a plain campaign
                (1, "lt", 1),                # a prefix alone is still numbered
                (3, "rerun_", 2),            # repeats
                (2, "", 2),                  # repeats with no prefix
                (1, "mlx8a7ba9v2nv", 1))     # the 2026-09-18 campaign
        for model in ("gemma4:12b-nvfp4", "qwen3.8:27b-nvfp4", "nemotron3:33b-q8"):
            for think in ("false", "on"):
                for repeats, tag_prefix, rep in arms:
                    with self.subTest(model=model, think=think, arm=(repeats, tag_prefix, rep)):
                        self.assertEqual(
                            sec.arm_prefix(tag_prefix, rep, repeats) + sec.tag_for(model, think),
                            self.runner_tag(model, think, rep, repeats, tag_prefix))

    def test_a_plain_campaign_has_no_prefix(self):
        """H4: with neither knob set, tags are what they always were."""
        self.assertEqual(sec.arm_prefix(), "")


class TestBundle(unittest.TestCase):

    def setUp(self):
        self.rundir = tempfile.mkdtemp()
        self.outdir = tempfile.mkdtemp()
        self.out = os.path.join(self.outdir, "vision-campaign-test.json")
        self.addCleanup(shutil.rmtree, self.rundir)
        self.addCleanup(shutil.rmtree, self.outdir)

    def bundle(self, *argv):
        return run("--dir", self.rundir, "--out", self.out, *argv)

    def written(self):
        with open(self.out) as fh:
            return json.load(fh)

    def tags(self):
        return [c["tag"] for c in self.written()["cells"]]

    def test_cells_follow_the_runners_order(self):
        """Set by set, then model, think mode and repeat, as the runner loops."""
        for tag in ("qwen3_8_27b-nvfp4_thinkfalse", "qwen3_8_27b-nvfp4_thinkon",
                    "gemma4_31b-nvfp4_thinkfalse", "gemma4_31b-nvfp4_thinkon",
                    "nemotron3_33b-q8_thinkfalse"):
            write(self.rundir, tag, scores())
        rc, _, err = self.bundle(
            "--set", "MODELS=qwen3.8:27b-nvfp4 gemma4:31b-nvfp4",
            "--set", "MODELS=nemotron3:33b-q8", "THINK_MODES=false")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.tags(), ["qwen3_8_27b-nvfp4_thinkfalse", "qwen3_8_27b-nvfp4_thinkon",
                                       "gemma4_31b-nvfp4_thinkfalse", "gemma4_31b-nvfp4_thinkon",
                                       "nemotron3_33b-q8_thinkfalse"])

    def test_a_cell_is_its_files_unchanged(self):
        probe = {"host": HOST, "server_version": BUILD, "recall_9px": 2}
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores(0.965))
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", probe, kind="ft")
        write(self.rundir, "qwen3_8_27b-nvfp4_thinkfalse", scores(0.999))
        before = snapshot(self.rundir)
        rc, _, err = self.bundle("--set", "MODELS=gemma4:31b-nvfp4 qwen3.8:27b-nvfp4",
                                 "THINK_MODES=false")
        self.assertEqual(rc, 0, err)
        gemma, qwen = self.written()["cells"]
        self.assertEqual(gemma["scores"], scores(0.965))
        self.assertEqual(gemma["finetext_probe"], probe)
        self.assertEqual(qwen["scores"], scores(0.999))
        self.assertIsNone(qwen["finetext_probe"], "no ft file is null, never {}")
        self.assertEqual(before, snapshot(self.rundir), "the run's files were touched")

    def test_an_arm_numbers_its_runs(self):
        for rep in (1, 2, 3):
            write(self.rundir, f"rerun_{rep}_gemma4_12b-nvfp4_thinkfalse", scores())
        write(self.rundir, "lt1_gemma4_12b-nvfp4_thinkfalse", scores())
        rc, _, err = self.bundle(
            "--set", "MODELS=gemma4:12b-nvfp4", "THINK_MODES=false", "TAG_PREFIX=rerun_", "REPEATS=3",
            "--set", "MODELS=gemma4:12b-nvfp4", "THINK_MODES=false", "TAG_PREFIX=lt")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.tags(), ["rerun_1_gemma4_12b-nvfp4_thinkfalse",
                                       "rerun_2_gemma4_12b-nvfp4_thinkfalse",
                                       "rerun_3_gemma4_12b-nvfp4_thinkfalse",
                                       "lt1_gemma4_12b-nvfp4_thinkfalse"])

    def test_a_missing_scores_file_is_an_error_and_nothing_is_written(self):
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        rc, _, err = self.bundle("--set", "MODELS=gemma4:31b-nvfp4")      # THINK_MODES "false on"
        self.assertEqual(rc, 1)
        self.assertIn("scores_gemma4_31b-nvfp4_thinkon.json", err)
        self.assertFalse(os.path.exists(self.out))

    def test_an_unreadable_scores_file_is_an_error_not_an_empty_cell(self):
        with open(os.path.join(self.rundir, "scores_gemma4_31b-nvfp4_thinkfalse.json"), "w") as fh:
            fh.write('{"scene_single": {')                          # a write cut short
        rc, _, err = self.bundle("--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 1)
        self.assertIn("unreadable", err)
        self.assertFalse(os.path.exists(self.out))

    def test_a_descoped_cell_is_skipped_and_said(self):
        """The runner never measures it, so its absence is not a missing file."""
        self.assertTrue(sec.is_descoped("gemma4:12b-nvfp4", "on"))
        write(self.rundir, "gemma4_12b-nvfp4_thinkfalse", scores())
        rc, _, err = self.bundle("--set", "MODELS=gemma4:12b-nvfp4", "THINK_MODES=false on")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.tags(), ["gemma4_12b-nvfp4_thinkfalse"])
        self.assertIn("descoped", err)

    def test_a_think_off_tag_from_before_the_mode_suffix_is_found(self):
        """resolve_tag's fallback: runs before 2026-08-09 wrote the bare tag."""
        write(self.rundir, "qwen3_8_27b-nvfp4", scores())
        rc, _, err = self.bundle("--set", "MODELS=qwen3.8:27b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.tags(), ["qwen3_8_27b-nvfp4"])

    def test_two_sets_naming_one_cell_are_refused(self):
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        rc, _, err = self.bundle("--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false",
                                 "--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 1)
        self.assertIn("two sets", err)

    def test_a_set_takes_only_the_knobs_that_name_cells(self):
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        for bad, says in ((["MODELS=gemma4:31b-nvfp4", "ONLY_TESTS=scene_single"], "ONLY_TESTS"),
                          (["THINK_MODES=false"], "needs MODELS"),
                          (["MODELS=gemma4:31b-nvfp4", "REPEATS=three"], "REPEATS"),
                          (["MODELS=gemma4:31b-nvfp4", "REPEATS=0"], "at least 1"),
                          (["MODELS=gemma4:31b-nvfp4", "gemma4:26b-nvfp4"], "KNOB=VALUE")):
            with self.subTest(bad=bad):
                rc, _, err = self.bundle("--set", *bad)
                self.assertEqual(rc, 1)
                self.assertIn(says, err)

    def test_written_with_indent_1_and_a_trailing_newline(self):
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        self.bundle("--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        with open(self.out) as fh:
            raw = fh.read()
        self.assertEqual(raw, json.dumps(json.loads(raw), indent=1) + "\n")

    def test_meta_file_first_then_pairs_in_order(self):
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        meta = os.path.join(self.outdir, "meta.json")
        with open(meta, "w") as fh:
            json.dump({"purpose": "p", "scope": "s"}, fh)
        rc, _, err = self.bundle("--meta-file", meta, "--meta", "scope=s2",
                                 "--meta", "rendered_tables=docs/x.md",
                                 "--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 0, err)
        self.assertEqual(list(self.written()["meta"].items()),
                         [("purpose", "p"), ("scope", "s2"), ("rendered_tables", "docs/x.md")])

    def test_a_stamp_the_cells_contradict_is_refused(self):
        """H11: the cells say where they ran. A typed meta must agree."""
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        rc, _, err = self.bundle("--meta", "host=http://127.0.0.1:11436",
                                 "--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 1)
        self.assertIn(HOST, err)
        self.assertFalse(os.path.exists(self.out))
        rc, _, err = self.bundle("--meta", f"host={HOST}", "--meta", f"server_version={BUILD}",
                                 "--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 0, err)

    def test_a_campaign_over_two_builds_leaves_the_stamp_out(self):
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores(build="0.34.3-test"))
        write(self.rundir, "qwen3_8_27b-nvfp4_thinkfalse", scores(build="0.34.4-test"))
        sets = ("--set", "MODELS=gemma4:31b-nvfp4 qwen3.8:27b-nvfp4", "THINK_MODES=false")
        self.assertEqual(self.bundle("--meta", "server_version=0.34.4-test", *sets)[0], 1)
        self.assertEqual(self.bundle(*sets)[0], 0)

    def test_the_host_profile_sits_beside_the_bundle(self):
        """H26: meta.host_profile names a file next to the bundle, or nothing is written."""
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        sets = ("--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        name = "vision-campaign-test.host-profile.json"
        rc, _, err = self.bundle("--meta", f"host_profile={name}", *sets)
        self.assertEqual(rc, 1)
        self.assertIn("H26", err)
        self.assertFalse(os.path.exists(self.out))
        with open(os.path.join(self.outdir, name), "w") as fh:
            json.dump({"schema": "host-profile/1"}, fh)
        rc, _, err = self.bundle("--meta", f"host_profile={name}", *sets)
        self.assertEqual(rc, 0, err)
        self.assertNotIn("warning", err)

    def test_no_host_profile_warns_and_still_bundles(self):
        """A campaign from before 2026-09-30 has none, so this cannot be a refusal."""
        write(self.rundir, "gemma4_31b-nvfp4_thinkfalse", scores())
        rc, _, err = self.bundle("--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false")
        self.assertEqual(rc, 0)
        self.assertIn("SPEC H26", err)

    def test_a_set_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bc.main(["--dir", self.rundir, "--out", self.out])


class TestSharedHelpers(unittest.TestCase):

    def test_imported_never_redefined(self):
        """SPEC H5. A second tag rule is how a bundle and a table disagree."""
        for name in ("arm_prefix", "is_descoped", "load", "resolve_tag", "save"):
            self.assertIs(getattr(bc, name), getattr(sec, name), name)


class TestCommittedBundle(unittest.TestCase):
    """The 2026-09-18 bundle, rebuilt by this tool from its own cells.

    That bundle was written by a one-off script in a compact layout, one arm per line,
    so the rebuild is compared as parsed JSON, not as bytes. What must match is every
    cell, the meta and the order."""

    def test_the_2026_09_18_bundle_rebuilds_cell_for_cell(self):
        with open(COMMITTED) as fh:
            committed = json.load(fh)
        rundir, outdir = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, rundir)
        self.addCleanup(shutil.rmtree, outdir)
        for cell in committed["cells"]:
            write(rundir, cell["tag"], cell["scores"])
            if cell["finetext_probe"] is not None:
                write(rundir, cell["tag"], cell["finetext_probe"], kind="ft")
        meta = os.path.join(outdir, "meta.json")
        with open(meta, "w") as fh:
            json.dump(committed["meta"], fh)
        out = os.path.join(outdir, "bundle.json")
        p = "TAG_PREFIX=mlx8a7ba9v2nv"
        rc, _, err = run("--dir", rundir, "--out", out, "--meta-file", meta,
                         "--set", "MODELS=gemma4:12b-nvfp4 gemma4:26b-nvfp4", "THINK_MODES=false", p,
                         "--set", "MODELS=gemma4:31b-nvfp4", "THINK_MODES=false on", p,
                         "--set", "MODELS=qwen3.6:35b-a3b-nvfp4", "THINK_MODES=false", p,
                         "--set", "MODELS=qwen3.8:27b-nvfp4", "THINK_MODES=false on", p)
        self.assertEqual(rc, 0, err)
        with open(out) as fh:
            self.assertEqual(json.load(fh), committed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
