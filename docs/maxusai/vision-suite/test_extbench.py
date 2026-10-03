#!/usr/bin/env python3
"""extbench.py's row cache, image fetching, retries, checkpoints and scorers — no network.

    python3 test_extbench.py
"""
import json
import os
import sys
import tempfile
import unittest
import urllib.error
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


# A datasets-server image link as /rows returns it: signed, with an Expires time.
EXPIRED = ("https://datasets-server.huggingface.co/cached-assets/echo840/OCRBench/--/rev/--/default/"
           "test/0/image/image.jpg?Expires=1000&Signature=old")
FRESH = EXPIRED.replace("Expires=1000", "Expires=9999999999").replace("Signature=old", "Signature=new")
REVOKED = FRESH.replace("Signature=new", "Signature=revoked")      # unexpired, but refused


def item(question, link):
    return {"question": question, "answer": ["a"], "question_type": "Scene Text-centric VQA",
            "image": {"src": link, "height": 10, "width": 10}}


def page(*rows):
    return {"rows": [{"row_idx": i, "row": r} for i, r in enumerate(rows)]}


def forbidden(url):
    return urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)


class ImageTest(unittest.TestCase):
    """A temporary extimgs/ tree, and a download() that fakes the network."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(extbench, "DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.downloads = []

    def fake_download(self, src, path):
        self.downloads.append(src)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\xff\xd8")


class TestSignedLinks(unittest.TestCase):
    def test_expiry_is_read_from_the_link(self):
        self.assertEqual(extbench.link_expiry(EXPIRED), 1000)
        self.assertIsNone(extbench.link_expiry("https://example.org/image.jpg"))

    def test_a_link_about_to_expire_counts_as_expired(self):
        self.assertTrue(extbench.link_expired(EXPIRED, now=990))         # inside the margin
        self.assertFalse(extbench.link_expired(EXPIRED, now=1000 - 3600))
        self.assertFalse(extbench.link_expired("https://example.org/image.jpg", now=10 ** 10))


class TestEnsureImage(ImageTest):
    def test_an_expired_link_is_refreshed_before_it_is_used(self):
        rows = [item("q0", EXPIRED), item("q1", EXPIRED)]
        extbench.save_rows(extbench.rows_cache_path("ocrbench", 0, 2), rows)
        with mock.patch.object(extbench, "http_json", return_value=page(item("q0", FRESH), item("q1", FRESH))) as get, \
             mock.patch.object(extbench, "download", side_effect=self.fake_download):
            path, err = extbench.ensure_image("ocrbench", 0, rows, 1)
        self.assertIsNone(err)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(self.downloads, [FRESH], "the expired link must never be requested")
        self.assertEqual(get.call_count, 1)
        with open(extbench.rows_cache_path("ocrbench", 0, 2)) as f:
            cached = json.load(f)
        self.assertEqual([extbench.image_src(r) for r in cached], [FRESH, FRESH],
                         "the next arm starts from the fresh links")

    def test_a_403_is_retried_once_on_a_refreshed_link(self):
        rows = [item("q0", REVOKED)]
        refusals = [forbidden(REVOKED)]

        def download(src, path):
            if refusals:
                raise refusals.pop()
            self.fake_download(src, path)
        with mock.patch.object(extbench, "http_json", return_value=page(item("q0", FRESH))), \
             mock.patch.object(extbench, "download", side_effect=download):
            path, err = extbench.ensure_image("ocrbench", 0, rows, 0)
        self.assertIsNone(err)
        self.assertEqual(self.downloads, [FRESH])

    def test_an_image_that_cannot_be_fetched_is_an_error_not_a_crash(self):
        rows = [item("q0", EXPIRED)]
        with mock.patch.object(extbench, "http_json", return_value=page(item("q0", FRESH))), \
             mock.patch.object(extbench, "download", side_effect=forbidden(FRESH)):
            path, err = extbench.ensure_image("ocrbench", 0, rows, 0)
        self.assertIsNone(path)
        self.assertIn("HTTP Error 403", err)

    def test_a_page_that_now_holds_other_items_is_not_taken(self):
        rows = [item("q0", EXPIRED)]
        with mock.patch.object(extbench, "http_json", return_value=page(item("another question", FRESH))), \
             mock.patch.object(extbench, "download", side_effect=self.fake_download):
            path, err = extbench.ensure_image("ocrbench", 0, rows, 0)
        self.assertIsNone(path)
        self.assertIn("changed under the pinned offset", err)
        self.assertEqual(rows[0], item("q0", EXPIRED), "the slice must not change")
        self.assertEqual(self.downloads, [])

    def test_only_the_page_holding_the_item_is_refreshed(self):
        rows = [item(f"q{i}", EXPIRED) for i in range(150)]
        fresh = page(*[item(f"q{i}", FRESH) for i in range(100, 150)])
        with mock.patch.object(extbench, "http_json", return_value=fresh) as get, \
             mock.patch.object(extbench, "download", side_effect=self.fake_download):
            _, err = extbench.ensure_image("ocrbench", 200, rows, 120)
        self.assertIsNone(err)
        url = get.call_args[0][0]
        self.assertIn("offset=300", url)             # the slice starts at 200; item 120 is on its second page
        self.assertIn("length=50", url)
        self.assertEqual(extbench.image_src(rows[99]), EXPIRED)
        self.assertEqual(extbench.image_src(rows[120]), FRESH)

    def test_a_cached_image_needs_no_link(self):
        rows = [item("q0", EXPIRED)]
        path = extbench.image_path("ocrbench", 0, EXPIRED)
        os.makedirs(os.path.dirname(path))
        open(path, "wb").close()
        with mock.patch.object(extbench, "http_json", side_effect=AssertionError("network")), \
             mock.patch.object(extbench, "download", side_effect=AssertionError("network")):
            self.assertEqual(extbench.ensure_image("ocrbench", 0, rows, 0), (path, None))

    def test_a_download_that_fails_part_way_leaves_no_file(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = OSError("connection reset")
        path = extbench.image_path("ocrbench", 0, FRESH)
        with mock.patch.object(extbench.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(OSError):
                extbench.download(FRESH, path)
        self.assertEqual(os.listdir(os.path.dirname(path)), [])

    def test_prefetch_reports_what_it_could_not_fetch(self):
        rows = [item(f"q{i}", FRESH) for i in range(3)]
        ensure = lambda bench, offset, rows, i: (None, "gone") if i == 2 else ("x.jpg", None)
        with mock.patch.object(extbench, "ensure_image", side_effect=ensure):
            self.assertEqual(extbench.prefetch_images("ocrbench", 500, rows), {502: "gone"})


class TestMainLoop(ImageTest):
    """main() end to end, with the rows, the images and the model faked."""

    def run_main(self, rows, gen, ensure):
        env = {"LIMIT": str(len(rows)), "OFFSET": "0", "SLEEP": "0"}
        argv = ["extbench.py", "http://127.0.0.1:1", "t", "m", "ocrbench"]
        with mock.patch.dict(os.environ, env), mock.patch.object(sys, "argv", argv), \
             mock.patch.object(extbench, "fetch_rows", return_value=rows), \
             mock.patch.object(extbench, "ensure_image", side_effect=ensure), \
             mock.patch.object(extbench, "gen", side_effect=gen), \
             mock.patch("builtins.print"):
            extbench.main()

    def results(self, name):
        with open(os.path.join(self.tmp.name, name)) as f:
            return json.load(f)

    def image(self, bench, offset, rows, i):
        path = os.path.join(self.tmp.name, f"{i}.jpg")
        open(path, "wb").close()
        return path, None

    def test_an_image_it_cannot_fetch_is_one_error_and_the_arm_finishes(self):
        rows = [item(f"q{i}", FRESH) for i in range(3)]
        missing = "image not fetched: HTTP Error 403: Forbidden"
        ensure = lambda b, o, r, i: (None, missing) if i == 1 else self.image(b, o, r, i)
        self.run_main(rows, lambda *a, **k: {"response": "a"}, ensure)
        out = self.results("ext_t_ocrbench.json")
        s = out["summary"]
        self.assertEqual((s["scored"], s["errors"], s["correct"]), (2, 1, 2))
        self.assertEqual(out["records"][1], {"i": 1, "error": missing})
        self.assertNotIn("complete", s, "a finished arm's summary is unchanged")
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "ext_t_ocrbench.partial.json")))

    def test_a_stopped_arm_keeps_what_it_measured(self):
        rows = [item(f"q{i}", FRESH) for i in range(4)]
        answers = [{"response": "a"}, KeyboardInterrupt()]

        def gen(*args, **kwargs):
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(rows, gen, self.image)
        s = self.results("ext_t_ocrbench.partial.json")["summary"]
        self.assertEqual((s["complete"], s["stopped_at"], s["scored"]), (False, 1, 1))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "ext_t_ocrbench.json")),
                         "a stopped arm must not look finished to summarize_extbench.py")

    def test_a_running_arm_checkpoints(self):
        rows = [item(f"q{i}", FRESH) for i in range(3)]
        writes, real = [], extbench.write_json

        def spy(path, obj):
            writes.append((os.path.basename(path), obj["summary"].get("stopped_at")))
            real(path, obj)
        with mock.patch.object(extbench, "CHECKPOINT_EVERY", 2), \
             mock.patch.object(extbench, "write_json", side_effect=spy):
            self.run_main(rows, lambda *a, **k: {"response": "a"}, self.image)
        self.assertEqual(writes, [("ext_t_ocrbench.partial.json", 2), ("ext_t_ocrbench.json", None)])

    def test_the_summary_records_the_prompt_cache_setting(self):
        # It moves seconds per item, which summarize_extbench --timing compares.
        rows = [item("q0", FRESH)]
        for env, want in ((None, 0), ("server", None), ("4096", 4096)):
            with self.subTest(env=env), mock.patch.dict(os.environ):
                os.environ.pop("PROMPT_CACHE_RAM", None)
                if env:
                    os.environ["PROMPT_CACHE_RAM"] = env
                self.run_main(rows, lambda *a, **k: {"response": "a"}, self.image)
                self.assertEqual(self.results("ext_t_ocrbench.json")["summary"]["prompt_cache_ram"], want)

    def test_a_bad_prompt_cache_setting_stops_the_arm_before_a_request(self):
        sent = []
        with mock.patch.dict(os.environ, {"PROMPT_CACHE_RAM": "off"}), self.assertRaises(ValueError):
            self.run_main([item("q0", FRESH)], lambda *a, **k: sent.append(a), self.image)
        self.assertEqual(sent, [])

    def test_sigterm_ends_the_arm_like_an_interrupt(self):
        with self.assertRaises(SystemExit) as stop:
            extbench.stop_on_sigterm(15, None)
        self.assertEqual(stop.exception.code, 143)


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
