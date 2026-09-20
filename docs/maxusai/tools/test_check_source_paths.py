#!/usr/bin/env python3
"""Tests for the source-path guard. No repo, no git: a synthetic tree.

Every rule here was learned by measuring the real tree, and each one exists
because a simpler checker got that case wrong. A guard whose false alarms
outnumber its findings teaches people to ignore it (SPEC H20), so the
false-positive rules are pinned as hard as the true-positive one.

    python3 test_check_source_paths.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_source_paths as guard  # noqa: E402


class Tree:
    """A synthetic repo: paths that exist, and files with content."""

    def __init__(self, files):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name
        for rel, text in files.items():
            path = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(text)
        self.tracked = sorted(files)

    def scan(self):
        return guard.scan(self.root, self.tracked)


class TestFindsWhatMatters(unittest.TestCase):

    def test_a_missing_owned_path_is_reported(self):
        """The whole point."""
        t = Tree({"mlxrunner/client.go": "// see mlxrunner/gone.go for why\n",
                  "mlxrunner/other.go": "package mlxrunner\n"})
        hits = t.scan()
        self.assertEqual([h.path for h in hits], ["mlxrunner/gone.go"])
        self.assertEqual(hits[0].line, 1)

    def test_a_confidently_wrong_rewrite_is_reported(self):
        """The case that justifies the guard over review alone. A mechanical
        rename pass turned x/models/base/media.go into
        mlxrunner/model/base/media.go while the reorganisation removed the
        base/ segment -- a plausible path no reviewer checks by hand."""
        t = Tree({"docs/maxusai/note.md": "reached via mlxrunner/model/base/media.go\n",
                  "mlxrunner/model/media.go": "package model\n"})
        self.assertEqual([h.path for h in t.scan()], ["mlxrunner/model/base/media.go"])


class TestScopedToTheChange(unittest.TestCase):
    """The guard runs on the files a change TOUCHES, not the whole tree.

    Measured on main 2026-09-20: eleven references name files that were
    DELETED, not moved -- `x/mlxrunner/constrain.go` after ADR 0033 retired it,
    `integration/imagegen_test.go` after imagegen went, upstream's
    `model/models/gemma4/model_vision.go`. Those are living documents
    discussing dead code, which is legitimate prose. Gating the whole tree
    would need an eleven-entry allowlist on day one, and an allowlist that
    grows is how a guard becomes decoration.

    Scoped to the diff it still catches what it was built for: every stale
    reference the v0.34.2 fold left, and both paths it invented, were in files
    that fold touched.
    """

    def test_only_the_named_files_are_reported(self):
        t = Tree({"mlxrunner/a.go": "// see mlxrunner/gone.go\n",
                  "mlxrunner/b.go": "// see mlxrunner/alsogone.go\n",
                  "mlxrunner/real.go": "package mlxrunner\n"})
        scoped = guard.scan(t.root, t.tracked, limit_to=["mlxrunner/b.go"])
        self.assertEqual([h.file for h in scoped], ["mlxrunner/b.go"])
        self.assertEqual(len(t.scan()), 2, "unscoped still sees both")

    def test_an_untouched_file_with_a_dead_reference_is_not_the_changes_problem(self):
        t = Tree({"docs/maxusai/old.md": "x/mlxrunner/constrain.go did the thing\n",
                  "mlxrunner/touched.go": "package mlxrunner\n"})
        self.assertEqual(guard.scan(t.root, t.tracked,
                                    limit_to=["mlxrunner/touched.go"]), [])


class TestDoesNotCryWolf(unittest.TestCase):
    """Each of these produced a false alarm in a real measurement against main:
    99 findings naive, 26 with relative resolution, 10 once other projects were
    excluded -- and only that last set was worth a human's attention."""

    def test_another_projects_path_is_not_ours_to_resolve(self):
        """mlx_vlm/..., vlmeval/config.py, upstream's ollama/model/model.go.
        The first segment is not a directory this repo owns."""
        t = Tree({"docs/maxusai/note.md":
                  "compare mlx_vlm/models/gemma4/processing_gemma4.py and "
                  "vlmeval/config.py\n",
                  "mlxrunner/client.go": "package mlxrunner\n"})
        self.assertEqual(t.scan(), [])

    def test_a_path_relative_to_the_document_resolves(self):
        """ADR 0011 writes preflight/test_verdicts.py, meaning the one beside
        it, not a top-level preflight/."""
        t = Tree({"docs/maxusai/vision-suite/README.md": "see preflight/probes.py\n",
                  "docs/maxusai/vision-suite/preflight/probes.py": "x = 1\n"})
        self.assertEqual(t.scan(), [])

    def test_a_suffix_of_a_real_path_resolves(self):
        """Docs name paths from a package root constantly: `cache/recurrent.go`
        for mlxrunner/cache/recurrent.go."""
        t = Tree({"docs/maxusai/note.md": "see cache/recurrent.go\n",
                  "mlxrunner/cache/recurrent.go": "package cache\n"})
        self.assertEqual(t.scan(), [])

    def test_an_import_line_is_not_a_repo_path(self):
        """golang.org/x/sync/errgroup contains x/sync/errgroup; github.com/...
        contains the rest. Neither is a file in this tree."""
        t = Tree({"server/x.go": '\t"golang.org/x/sync/errgroup"\n'
                                 '\t"github.com/ollama/ollama/mlx/quant"\n',
                  "mlxrunner/client.go": "package mlxrunner\n"})
        self.assertEqual(t.scan(), [])

    def test_a_substring_of_a_longer_path_is_not_a_match(self):
        """mlx/quant contains x/quant. A checker without a left boundary
        reports the live package as missing -- and a rewriter without one
        corrupts it."""
        t = Tree({"server/x.go": "// the table in mlx/quant/quant.go\n",
                  "mlx/quant/quant.go": "package quant\n"})
        self.assertEqual(t.scan(), [])

    def test_the_guard_does_not_check_itself(self):
        """Caught by running it on its own first commit: 30 findings, every one
        a path quoted in a docstring or built as a fixture. A guard that
        documents the patterns it catches will always flag its own
        documentation, so docs/maxusai/tools/ is out of scope."""
        t = Tree({"docs/maxusai/tools/check_source_paths.py":
                  "# e.g. mlxrunner/model/base/media.go is an invented path\n",
                  "docs/maxusai/tools/test_check_source_paths.py":
                  "fixture = 'mlxrunner/gone.go'\n",
                  "mlxrunner/real.go": "package mlxrunner\n"})
        self.assertEqual(t.scan(), [])

    def test_a_historical_record_may_name_a_deleted_path(self):
        """An ADR records what was true when it was decided, and a dated task
        doc records what was run. Re-pointing those would make the history
        wrong, so they are out of scope -- deliberately, not incidentally."""
        t = Tree({"docs/maxusai/adr/0018-imagegen.md": "x/imagegen/mlx/mlx.go did\n",
                  "docs/maxusai/tasks/sync.md": "patched x/models/gemma4/audio.go\n",
                  "docs/design/rebase.md": "upstream's model/model.go\n",
                  "mlxrunner/client.go": "package mlxrunner\n"})
        self.assertEqual(t.scan(), [])

    def test_a_deliberately_retired_package_is_still_history(self):
        """`the fork's retired pure-Go sampler (x/structured, deleted
        2026-09-17)` is a true sentence about a deleted package. Only file
        references are checked, so a bare package name never trips it."""
        t = Tree({"mlxrunner/xgrammar/engine_behaviour_test.go":
                  "// the retired sampler (x/structured, deleted 2026-09-17)\n",
                  "mlxrunner/client.go": "package mlxrunner\n"})
        self.assertEqual(t.scan(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
