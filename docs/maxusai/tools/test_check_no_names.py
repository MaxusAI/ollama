#!/usr/bin/env python3
"""Tests for the personal-name guard. No repo, no git: a synthetic tree.

The guard exists because the repo is public and every host posts as one
GitHub account, so a person's name written into a doc, a commit message or a
pull request is published under that account. Its deny-list is NOT in the
tree: it arrives through the NAME_DENYLIST secret or a file outside the
checkout, because a guard that spells the name out publishes it too. For the
same reason nothing it prints may contain a match -- CI logs of a public repo
are public.

    python3 test_check_no_names.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_no_names as guard  # noqa: E402

# A stand-in deny-list. Any distinctive word does; the real one is a secret.
PATTERN = guard.load_denylist("zanzibar|quuxley", None)


class Tree:
    """A synthetic checkout: tracked files with content (text or bytes)."""

    def __init__(self, files):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name
        for rel, content in files.items():
            path = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(content if isinstance(content, bytes) else content.encode())
        self.tracked = sorted(files)

    def scan(self, allow=()):
        return guard.scan(self.root, self.tracked, PATTERN, list(allow))


class TreeTest(unittest.TestCase):

    def tree(self, files):
        t = Tree(files)
        self.addCleanup(t.dir.cleanup)
        return t


class TestFindsWhatMatters(TreeTest):

    def test_a_name_in_a_tracked_file_is_reported_with_its_line(self):
        t = self.tree({"docs/adr/0001.md": "# Title\n\n- **Deciders:** Zanzibar\n"})
        hits = t.scan()
        self.assertEqual([(h.where, h.line) for h in hits], [("docs/adr/0001.md", 3)])

    def test_case_does_not_hide_a_name(self):
        """The hostname that survived #380 was lower case: a case-sensitive
        grep reported the tree clean while three docs still named the host's
        owner."""
        t = self.tree({"docs/campaign.md": "| host | `zanzibar-NucBox` (10.8.0.4) |\n"})
        self.assertEqual(len(t.scan()), 1)

    def test_every_match_on_a_line_is_one_hit_not_many(self):
        t = self.tree({"a.md": "Zanzibar asked; ZANZIBAR agreed.\n"})
        self.assertEqual(len(t.scan()), 1)

    def test_pull_request_text_and_commit_messages_are_checked(self):
        """Most of the 2026-09-25 cleanup was pull request descriptions, which
        are not in the tree at all."""
        hits = guard.scan_texts({"PR body": "Merging is Quuxley's call.\n",
                                 "PR title": "docs: nothing personal"}, PATTERN)
        self.assertEqual([(h.where, h.line) for h in hits], [("PR body", 1)])


class TestNeverPublishesWhatItFinds(TreeTest):
    """A public CI log that quotes the match republishes the name."""

    def test_the_report_masks_the_match(self):
        t = self.tree({"a.md": "the call is Zanzibar's, not the merger's\n"})
        (hit,) = t.scan()
        self.assertNotIn("zanzibar", hit.context.lower())
        self.assertIn("<name>", hit.context)
        self.assertIn("not the merger's", hit.context)

    def test_the_rendered_line_masks_the_match(self):
        t = self.tree({"a.md": "per Quuxley\n"})
        (hit,) = t.scan()
        self.assertNotIn("quuxley", guard.render(hit).lower())


class TestConfiguration(unittest.TestCase):

    def test_no_denylist_means_not_configured(self):
        self.assertIsNone(guard.load_denylist(None, None))

    def test_an_empty_or_blank_denylist_is_not_configured(self):
        """An empty regex matches every line of every file: treated as a
        pattern it would fail the whole tree, so it must read as unset."""
        self.assertIsNone(guard.load_denylist("", None))
        self.assertIsNone(guard.load_denylist("  \n ", None))

    def test_a_denylist_file_is_one_pattern_per_line_with_comments(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# names that must not appear\nzanzibar\n\nquuxley\n")
        try:
            p = guard.load_denylist(None, fh.name)
            self.assertTrue(p.search("ZANZIBAR") and p.search("Quuxley"))
            self.assertIsNone(p.search("names that must not appear"))
        finally:
            os.unlink(fh.name)

    def test_a_pattern_that_matches_the_empty_string_is_an_error(self):
        """`x?` matches the empty string, so it would flag every line of every
        file -- a typo that turns the guard into noise must not pass as config."""
        with self.assertRaises(guard.ConfigError):
            guard.load_denylist("zanzibar|x?", None)

    def test_an_invalid_pattern_is_an_error_not_a_pass(self):
        with self.assertRaises(guard.ConfigError):
            guard.load_denylist("zanzibar(", None)


class TestReadsTheEventNotTheEnvironment(unittest.TestCase):
    """Actions prints every step-level env: value in the public run log, so a
    description passed that way is republished in a log that outlives any
    later edit of the description. The event file on the runner is not printed."""

    def event(self, payload):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_title_and_body_come_from_the_event_file(self):
        path = self.event({"pull_request": {"title": "docs: tidy", "body": "Merging is Zanzibar's call."}})
        self.assertEqual(guard.event_texts(path),
                         {"PR title": "docs: tidy", "PR body": "Merging is Zanzibar's call."})

    def test_a_push_event_has_no_pull_request_text(self):
        self.assertEqual(guard.event_texts(self.event({"ref": "refs/heads/main"})), {})

    def test_an_empty_description_is_empty_text_not_an_error(self):
        path = self.event({"pull_request": {"title": "t", "body": None}})
        self.assertEqual(guard.event_texts(path), {"PR title": "t", "PR body": ""})


class TestDoesNotCryWolf(TreeTest):

    def test_an_allowlisted_file_is_skipped(self):
        """The invoice fixture renders a customer line into document.png;
        changing it moves every recorded score, so it is allowlisted by path."""
        t = self.tree({"docs/maxusai/vision-suite/gen_scenes.py": 'd.text("Customer: Zanzibar Pty Ltd")\n',
                       "docs/other.md": "clean\n"})
        self.assertEqual(t.scan(allow=["docs/maxusai/vision-suite/gen_scenes.py"]), [])

    def test_an_allowlisted_directory_covers_its_files_only(self):
        t = self.tree({"mlxrunner/tokenizer/testdata/vocab.json": '"Zanzibar": 40208\n',
                       "mlxrunner/tokenizer/testdata_notes.md": "Zanzibar\n"})
        hits = t.scan(allow=["mlxrunner/tokenizer/testdata/"])
        self.assertEqual([h.where for h in hits], ["mlxrunner/tokenizer/testdata_notes.md"])

    def test_binary_files_are_skipped(self):
        t = self.tree({"visimgs/document.png": b"\x89PNG\r\n\x1a\n\x00\x00zanzibar\x00"})
        self.assertEqual(t.scan(), [])

    def test_the_allowlist_file_holds_paths_not_names(self):
        """The allowlist is tracked, so it must never need the name itself."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# fixtures\ndocs/maxusai/vision-suite/gen_scenes.py\n\nmlxrunner/tokenizer/testdata/\n")
        try:
            self.assertEqual(guard.load_allowlist(fh.name),
                             ["docs/maxusai/vision-suite/gen_scenes.py", "mlxrunner/tokenizer/testdata/"])
        finally:
            os.unlink(fh.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
