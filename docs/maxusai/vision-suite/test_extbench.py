#!/usr/bin/env python3
"""extbench.py's row cache, retries and scorers — no network.

    python3 test_extbench.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extbench  # noqa: E402


class TestRowCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(extbench, "DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("REFRESH_ROWS", None)

    def test_slice_is_fetched_once_then_read_from_disk(self):
        page = {"rows": [{"row": {"question": f"q{i}", "answer": ["a"], "image": "u"}}
                         for i in range(3)]}
        with mock.patch.object(extbench, "http_json", return_value=page) as get:
            first = extbench.fetch_rows("ocrbench", 0, 3)
            self.assertEqual(get.call_count, 1)
        with mock.patch.object(extbench, "http_json", side_effect=AssertionError("refetched")) as get:
            second = extbench.fetch_rows("ocrbench", 0, 3)
            self.assertEqual(get.call_count, 0, "a cached slice must not hit the network")
        self.assertEqual(first, second)
        self.assertTrue(os.path.exists(extbench.rows_cache_path("ocrbench", 0, 3)))

    def test_refresh_rows_bypasses_the_cache(self):
        page = {"rows": [{"row": {"question": "q", "answer": ["a"], "image": "u"}}]}
        with mock.patch.object(extbench, "http_json", return_value=page):
            extbench.fetch_rows("ocrbench", 0, 1)
        os.environ["REFRESH_ROWS"] = "1"
        self.addCleanup(os.environ.pop, "REFRESH_ROWS", None)
        with mock.patch.object(extbench, "http_json", return_value=page) as get:
            extbench.fetch_rows("ocrbench", 0, 1)
            self.assertEqual(get.call_count, 1)

    def test_short_slice_is_not_cached(self):
        """A truncated fetch must not poison the cache with a partial arm."""
        page = {"rows": [{"row": {"question": "q", "answer": ["a"], "image": "u"}}]}
        with mock.patch.object(extbench, "http_json", side_effect=[page, {"rows": []}]):
            rows = extbench.fetch_rows("ocrbench", 0, 5)
        self.assertEqual(len(rows), 1)
        self.assertFalse(os.path.exists(extbench.rows_cache_path("ocrbench", 0, 5)))

    def test_cache_is_keyed_by_offset_and_limit(self):
        self.assertNotEqual(extbench.rows_cache_path("ocrbench", 0, 200),
                            extbench.rows_cache_path("ocrbench", 200, 200))


class TestRetry(unittest.TestCase):
    def test_http_json_retries_then_succeeds(self):
        with mock.patch.object(extbench.urllib.request, "urlopen",
                               side_effect=[OSError("dns"), mock.MagicMock()]) as op, \
             mock.patch.object(extbench.json, "load", return_value={"ok": True}), \
             mock.patch.object(extbench.time, "sleep"):
            self.assertEqual(extbench.http_json("https://x", attempts=3), {"ok": True})
            self.assertEqual(op.call_count, 2)

    def test_http_json_raises_the_last_error_when_every_attempt_fails(self):
        with mock.patch.object(extbench.urllib.request, "urlopen", side_effect=OSError("dns")), \
             mock.patch.object(extbench.time, "sleep"):
            with self.assertRaises(OSError):
                extbench.http_json("https://x", attempts=2)


class TestOCRBenchScoring(unittest.TestCase):
    def test_contains_match_is_case_insensitive(self):
        self.assertTrue(extbench.score_ocrbench("Friend", {"answer": ["FRIEND"]}))
        self.assertTrue(extbench.score_ocrbench("the sign reads CHAIN.", {"answer": ["chain"]}))
        self.assertFalse(extbench.score_ocrbench("Centuries", {"answer": ["CENTRE"]}))

    def test_handwritten_maths_ignores_whitespace_on_both_sides(self):
        row = {"answer": ["x ^ 2 + 1"], "question_type": "Handwritten Mathematical Expression Recognition"}
        self.assertTrue(extbench.score_ocrbench("x^2+1", row))


if __name__ == "__main__":
    unittest.main(verbosity=2)
