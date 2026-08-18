#!/usr/bin/env python3
"""Rescore regression: the bbox contract scorer must not move old numbers.

`9c4416e5` taught `score_bbox_contract` per-object declarations, anchor
metrics and named coordinate keys, and `5081fcbb` added the adversarial arms.
Both were claimed to be behaviour-preserving: every response scored before the
change must score the same after it, or the historical numbers in the campaign
docs silently stop meaning what they say.

That claim was never checkable in the repo. Run outputs are gitignored
(`resp_*.json`), so the corpus it was verified against does not survive the
machine that produced it, and nothing here would catch a regression.

This test makes it checkable. It loads the pre-change scorer straight out of
git, runs both versions over a committed corpus, and asserts that every field
the old scorer produced is unchanged. The change is purely additive — 16 fields
before, 25 after, none removed — so identity on the old 16 is exactly the
guarantee, and the 9 new fields are free to appear.

    python3 test_rescore.py

The committed corpus is DERIVED, not captured (see fixtures/.../README.md). To
run the same assertion over real captured responses, point at a directory of
them — the historical corpus, if it is ever recovered:

    RESCORE_CORPUS=/path/to/responses python3 test_rescore.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest

DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(DIR, "fixtures", "bbox_contract")

# The commit immediately before 9c4416e5, which introduced declaration_scope,
# the anchor metrics and named coordinates. This is the "old scorer".
BASELINE = "c52bc00a"

REL = "docs/maxusai/vision-suite/vision_suite.py"

# Fields the baseline scorer emitted. Identity on these IS the guarantee.
OLD_FIELDS = [
    "json_valid", "declared_type", "declared_order", "declared_ref",
    "field_name", "labels_found", "labels_total", "hits_declared",
    "iou_declared", "hits_bestfit", "bestfit_dialect", "implied_scale",
    "iou_at_implied_scale", "declaration_valid", "declaration_matches_boxes",
    "contract_followed",
]

# Fields added since the baseline that every score must still carry. Not an
# exhaustive list of what the scorer may emit -- later additions are allowed and
# do not belong here unless something depends on their presence.
NEW_FIELDS = [
    "declaration_scope", "anchor_present", "anchor_implied_type",
    "anchor_implied_ref", "hits_anchor", "iou_anchor",
    "anchor_beats_declared", "self_check", "self_check_reason",
]


def load_module(path, name):
    """Import a vision_suite.py by path, with the suite dir on sys.path so its
    `from sampling import ...` resolves."""
    if DIR not in sys.path:
        sys.path.insert(0, DIR)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_baseline(tmpdir):
    """Materialise the pre-change scorer from git.

    The module resolves ground truth relative to its own file, so it is written
    into a directory carrying a visimgs symlink rather than imported in place.
    """
    src = subprocess.run(
        ["git", "-C", DIR, "show", f"{BASELINE}:{REL}"],
        capture_output=True, text=True, check=True).stdout
    path = os.path.join(tmpdir, "vision_suite_baseline.py")
    with open(path, "w") as f:
        f.write(src)
    link = os.path.join(tmpdir, "visimgs")
    if not os.path.exists(link):
        os.symlink(os.path.join(DIR, "visimgs"), link)
    return load_module(path, "vision_suite_baseline")


def read_text(path):
    with open(path) as f:
        return f.read()


def read_corpus(d):
    if not os.path.isdir(d):
        return []
    return [(n, read_text(os.path.join(d, n)))
            for n in sorted(os.listdir(d)) if not n.startswith(".")]


import tempfile

_TMP = tempfile.mkdtemp(prefix="rescore-")
NEW = load_module(os.path.join(DIR, "vision_suite.py"), "vision_suite_current")
try:
    OLD = load_baseline(_TMP)
    BASELINE_ERROR = None
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    OLD, BASELINE_ERROR = None, e


class TestSchemaIsAdditive(unittest.TestCase):
    """No field the old scorer emitted may disappear."""

    def test_no_field_removed(self):
        """Removal breaks consumers; addition cannot. Asserted asymmetrically.

        This originally required `new - old` to equal NEW_FIELDS exactly, which
        made every later scorer addition a failure. It duly failed when
        `degenerate_boxes` / `degenerate_labels` landed — a purely additive
        change that left all 14 pre-existing fixtures scoring identically. The
        guarantee is that nothing DISAPPEARS and that recorded values do not
        move; new columns are the feature, not a regression.
        """
        if OLD is None:
            self.skipTest(f"baseline {BASELINE} unavailable: {BASELINE_ERROR}")
        old = set(OLD.score_bbox_contract("{}").keys())
        new = set(NEW.score_bbox_contract("{}").keys())
        self.assertEqual(old - new, set(), "fields dropped by the change")
        self.assertEqual(set(NEW_FIELDS) - new, set(),
                         "a documented field stopped being emitted")

    def test_every_score_carries_the_new_fields(self):
        for name, text in read_corpus(os.path.join(FIXTURES, "preexisting")):
            with self.subTest(name):
                s = NEW.score_bbox_contract(text)
                for f in NEW_FIELDS:
                    self.assertIn(f, s, f"{name}: {f} missing")


class TestBehaviourPreserved(unittest.TestCase):
    """The load-bearing assertion: old responses score identically."""

    def _compare(self, corpus, label):
        if OLD is None:
            self.skipTest(f"baseline {BASELINE} unavailable: {BASELINE_ERROR}")
        cases = read_corpus(corpus)
        self.assertTrue(cases, f"no fixtures found in {corpus}")
        for name, text in cases:
            with self.subTest(f"{label}:{name}"):
                before = OLD.score_bbox_contract(text)
                after = NEW.score_bbox_contract(text)
                for f in OLD_FIELDS:
                    self.assertEqual(
                        before[f], after[f],
                        f"{name}: {f} moved {before[f]!r} -> {after[f]!r}")

    def test_committed_corpus(self):
        self._compare(os.path.join(FIXTURES, "preexisting"), "fixture")

    def test_external_corpus(self):
        """Same assertion over real captured responses, when supplied."""
        d = os.environ.get("RESCORE_CORPUS")
        if not d:
            self.skipTest("set RESCORE_CORPUS to a directory of raw responses")
        self._compare(d, "external")


class TestNewFeatures(unittest.TestCase):
    """Goldens for what the change added.

    Not compared against the baseline: it predates the syntax, so it scores
    these differently by design. That divergence is the feature.
    """

    def score(self, name):
        return NEW.score_bbox_contract(
            read_text(os.path.join(FIXTURES, "new_features", f"{name}.txt")))

    def preexisting(self, name):
        return NEW.score_bbox_contract(
            read_text(os.path.join(FIXTURES, "preexisting", f"{name}.txt")))

    def test_perobject_declaration_scope(self):
        s = self.score("perobject_declarations")
        self.assertEqual(s["declaration_scope"], "perobject")
        self.assertEqual(s["hits_declared"], 6)

    def test_toplevel_declaration_scope(self):
        self.assertEqual(self.preexisting("clean_real_xyxy")["declaration_scope"],
                         "toplevel")

    def test_absent_declaration_scope(self):
        self.assertEqual(self.preexisting("no_declaration")["declaration_scope"],
                         "none")

    def test_anchor_calibrates(self):
        s = self.score("anchored_real")
        self.assertTrue(s["anchor_present"])
        self.assertEqual(s["anchor_implied_type"], "real")
        self.assertEqual(s["hits_anchor"], 6)

    def test_anchor_recovers_a_frame_the_declaration_gets_wrong(self):
        """ref_size says 1024x768, the anchor says 1920x1080. The anchor wins."""
        s = self.score("anchor_disagrees_with_declaration")
        self.assertEqual(s["anchor_implied_ref"], [1920, 1080])
        self.assertTrue(s["anchor_beats_declared"])
        self.assertGreater(s["hits_anchor"], s["hits_declared"])

    def test_named_coords_are_extracted(self):
        """get_bbox reads x1/y1/x2/y2: every shape is located, in the right space."""
        s = self.score("named_coords")
        self.assertEqual(s["labels_found"], 6)
        self.assertEqual(s["hits_bestfit"], 6)
        self.assertEqual(s["bestfit_dialect"], "real/xyxy")

    def test_named_coords_satisfy_a_toplevel_declaration(self):
        """The schema the prompt asks for: one declaration, named coords per object.

        `read_decl` only sees the root, where the coordinate keys are not, so
        the order is inferred from the boxes instead. Before that inference this
        response read every box correctly and still scored the declaration
        invalid — hits_declared 0 against hits_bestfit 6.
        """
        s = self.score("named_coords")
        self.assertEqual(s["declared_order"], "xyxy")
        self.assertTrue(s["declaration_valid"])
        self.assertEqual(s["hits_declared"], 6)
        self.assertTrue(s["contract_followed"])

    def test_mixed_named_and_positional_infers_nothing(self):
        """One named object and one array: no single order to infer.

        Assuming xyxy here would score the array against a convention it never
        claimed — the unearned trust this probe exists to catch.
        """
        mixed = json.dumps({"bbox_type": "real", "ref_size": [1920, 1080],
                            "objects": [
                                {"label": "ANCHOR",
                                 "x1": 140, "y1": 160, "x2": 420, "y2": 360},
                                {"label": "BEACON", "box_2d": [620, 120, 900, 330]}]})
        s = NEW.score_bbox_contract(mixed)
        self.assertIsNone(s["declared_order"])
        self.assertFalse(s["declaration_valid"])

    def test_named_coords_infer_order_where_the_keys_share_the_dict(self):
        perobject = json.dumps({"objects": [
            {"label": "ANCHOR", "bbox_type": "real", "ref_size": [1920, 1080],
             "x1": 140, "y1": 160, "x2": 420, "y2": 360}]})
        self.assertEqual(NEW.score_bbox_contract(perobject)["hits_declared"], 1)

        at_root = json.dumps({
            "bbox_type": "real", "ref_size": [1920, 1080],
            "x1": 140, "y1": 160, "x2": 420, "y2": 360,
            "objects": [{"label": "ANCHOR",
                         "x1": 140, "y1": 160, "x2": 420, "y2": 360}]})
        s = NEW.score_bbox_contract(at_root)
        self.assertEqual(s["declared_order"], "xyxy")
        self.assertEqual(s["hits_declared"], 1)

    def test_self_check_only_runs_with_an_anchor(self):
        """Documented limit: no anchor, no verdict — the field stays None."""
        self.assertIsNotNone(self.score("anchored_real")["self_check"])
        self.assertIsNone(self.preexisting("clean_real_xyxy")["self_check"])


class TestMultiQ4Anchor(unittest.TestCase):
    """q4 is scored in the frame the response declares, not one assumed for it.

    The prompt asks for "image 1 pixel coordinates" and image 1 is 1920x1080, but
    a model that resizes internally answers in ITS frame. qwen3.8 returns DYNAMO
    scaled by ~1.33 on both axes — right object, right extent, wrong frame — so
    trying only 1920x1080 and norm-1000 scored five consecutive correct answers
    as misses, and they were published as a think-on grounding loss.
    """

    TRUTH_FRAME = [0, 0, 2560, 1440]          # what qwen3.8 actually declares
    Q4_IN_THAT_FRAME = [290, 795, 645, 1140]  # rescales to (351, 726); truth (350, 730)

    def resp(self, q4, anchor=None, extra_objects=()):
        objs = list(extra_objects)
        if anchor is not None:
            objs.insert(0, {"label": "__IMAGE__", "bbox": anchor})
        return json.dumps({
            "images": [{"index": 1, "type": "shapes_scene", "key_objects": objs},
                       {"index": 2}, {"index": 3}],
            "answers": {"q1": 2, "q4": q4},
        })

    def test_a_declared_frame_is_used(self):
        s = NEW.score_multi(self.resp(self.Q4_IN_THAT_FRAME, self.TRUTH_FRAME))
        self.assertTrue(s["q4_bbox_hit"])
        self.assertEqual(s["q4_bbox_space"], "anchor/xyxy")
        self.assertEqual(s["q4_anchor_frame"], [2560, 1440])

    def test_without_a_calibration_entry_nothing_changes(self):
        """Every historical response lacks one; none of their numbers may move."""
        s = NEW.score_multi(self.resp(self.Q4_IN_THAT_FRAME))
        self.assertFalse(s["q4_bbox_hit"])
        self.assertNotIn("q4_anchor_frame", s)

    def test_the_native_norm1000_answer_still_scores(self):
        """The think-off path, which was already correct, must be untouched."""
        s = NEW.score_multi(self.resp([113, 552, 251, 795]))
        self.assertTrue(s["q4_bbox_hit"])
        self.assertEqual(s["q4_bbox_space"], "norm1000/xyxy")

    def test_a_norm1000_anchor_adds_no_new_space(self):
        """[0,0,1000,1000] carries no information — that space is tried anyway,
        and treating it as a frame would divide by 1000 and land nowhere."""
        s = NEW.score_multi(self.resp([113, 552, 251, 795], [0, 0, 1000, 1000]))
        self.assertTrue(s["q4_bbox_hit"])
        self.assertEqual(s["q4_bbox_space"], "norm1000/xyxy")
        self.assertNotIn("q4_anchor_frame", s)

    def test_a_wrong_box_is_not_rescued(self):
        """The frame comes from the response, but it is not a licence to search
        for whatever scale makes the answer land inside the target — that is
        hits_bestfit, which SPEC C9 keeps out of every consumer path."""
        s = NEW.score_multi(self.resp([2000, 100, 2400, 400], self.TRUTH_FRAME))
        self.assertFalse(s["q4_bbox_hit"])

    def test_a_fabricated_frame_is_recorded_even_when_it_misleads(self):
        """A model can lie about its frame, as gemma4 does in the adversarial
        arms. The scorer cannot detect that here, so it records the frame it
        used and the reader can check it."""
        s = NEW.score_multi(self.resp(self.Q4_IN_THAT_FRAME, [0, 0, 4000, 2250]))
        self.assertEqual(s["q4_anchor_frame"], [4000, 2250])
        self.assertFalse(s["q4_bbox_hit"])


class TestFixtureCorpusCoversTheDialects(unittest.TestCase):
    """A corpus that only holds clean responses would pass while proving little."""

    def test_corpus_exercises_both_outcomes(self):
        scores = [NEW.score_bbox_contract(t)
                  for _, t in read_corpus(os.path.join(FIXTURES, "preexisting"))]
        self.assertTrue(any(s["contract_followed"] for s in scores),
                        "no fixture where the contract is followed")
        self.assertTrue(any(not s["contract_followed"] for s in scores),
                        "no fixture where the contract is broken")
        self.assertTrue(any(not s["json_valid"] for s in scores),
                        "no unparseable fixture")
        self.assertTrue(any(s["implied_scale"] for s in scores),
                        "no fixture exercising implied_scale")
        self.assertEqual(
            {"real", "norm1", "norm1000", None},
            {s["declared_type"] for s in scores},
            "corpus does not cover every declared bbox_type")


if __name__ == "__main__":
    unittest.main(verbosity=2)
