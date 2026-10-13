#!/usr/bin/env python3
"""Tests for loop_detect.py. No network, stdlib only: python3 test_loop_detect.py

The two fixtures under fixtures/loops/ are real outputs from the muse-glimmer campaign (2026-10-11, #463), both
think-off with format "json" on the release build:
  glimmer-mxfp8-bbox-object-repeat.txt     30b-mxfp8, bboxm_pin_noanc_named: one box object's keys, repeated to
                                           num_predict (2200 tokens)
  glimmer-nvfp4-multi3img-degenerate.txt   30b-nvfp4, multi_3img: the prompt's words, then `, "  1, "  :"}` repeated
"""
import os
import random
import time
import unittest

import loop_detect as ld

HERE = os.path.dirname(os.path.abspath(__file__))


def fixture(name):
    with open(os.path.join(HERE, "fixtures", "loops", name)) as fh:
        return fh.read()


def prose(n, seed=0):
    """Non-repeating text: a seeded word stream, so no 400-char window recurs."""
    rng = random.Random(seed)
    words, size = [], 0
    while size < n:
        words.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 11))))
        size += len(words[-1]) + 1
    return " ".join(words)[:n]


def feed_all(det, text, kind="content", chunk=7):
    """Feed text in fixed chunks; return the first Loop or None."""
    for i in range(0, len(text), chunk):
        piece = text[i:i + chunk]
        got = det.feed(content=piece) if kind == "content" else det.feed(thinking=piece)
        if got:
            return got
    return None


class TestBlockLoops(unittest.TestCase):
    def test_short_text_never_triggers(self):
        self.assertIsNone(feed_all(ld.LoopDetector(), "abc " * 150))  # 600 chars < min_length

    def test_long_prose_does_not_trigger(self):
        for kind in ("thinking", "content"):
            self.assertIsNone(feed_all(ld.LoopDetector(), prose(30000), kind=kind), kind)

    def test_a_block_restated_once_is_not_a_loop(self):
        # One earlier occurrence (a table restated, a quote repeated) must not trip it: that was the one-copy
        # false positive of ai-invoice-to-xero's thinking_is_looping.
        block = prose(600, seed=1)
        text = prose(1200, seed=2) + block + prose(300, seed=3) + block
        self.assertIsNone(feed_all(ld.LoopDetector(), text))

    def test_a_draft_restated_between_checks_is_not_a_loop(self):
        # Three copies that are NOT back to back: a thinking trace that restates its draft answer after each round of
        # checking. The CUDA host's replay of 1,505 finished think-on traces (#469 review) found 17-18 of these called
        # loops by a copy count alone -- 13 of them multi_3img cells with q1, q2 and q4 all right.
        draft = prose(500, seed=20)
        text = prose(1000, seed=21) + draft + prose(300, seed=22) + draft + prose(300, seed=23) + draft
        self.assertIsNone(feed_all(ld.LoopDetector(), text, kind="thinking"))

    def test_a_restated_draft_then_a_real_loop_still_fires(self):
        # The periodicity test must not hide a loop that starts after earlier restatements.
        draft, block = prose(500, seed=24), prose(430, seed=25)
        text = prose(1000, seed=26) + draft + prose(300, seed=27) + draft + block * 4
        got = feed_all(ld.LoopDetector(), text, kind="thinking")
        self.assertIsNotNone(got)
        self.assertEqual((got.kind, got.period), (ld.THINKING, 430))

    def test_a_block_repeated_is_a_loop(self):
        block = prose(450, seed=4)
        got = feed_all(ld.LoopDetector(), prose(900, seed=5) + block * 4, kind="thinking")
        self.assertIsNotNone(got)
        self.assertEqual(got.kind, ld.THINKING)
        self.assertEqual(got.failure_mode(), "loop detected:thinking")
        self.assertEqual(got.period, 450)
        self.assertGreaterEqual(got.repeats, 3)

    def test_kind_follows_the_stream_that_loops(self):
        block = prose(300, seed=6)
        det = ld.LoopDetector()
        self.assertIsNone(det.feed(thinking=prose(2000, seed=7)))
        got = feed_all(det, block * 6, kind="content")
        self.assertEqual(got.kind, ld.CONTENT)

    def test_short_period_loop(self):
        got = feed_all(ld.LoopDetector(), prose(900, seed=8) + '{"a": 1, "b": 2}, ' * 200)
        self.assertIsNotNone(got)
        self.assertEqual(got.period, len('{"a": 1, "b": 2}, '))


class TestRealLoops(unittest.TestCase):
    def test_glimmer_bbox_object_repeat(self):
        got = feed_all(ld.LoopDetector(), fixture("glimmer-mxfp8-bbox-object-repeat.txt"))
        self.assertIsNotNone(got)
        self.assertEqual(got.kind, ld.CONTENT)
        self.assertLess(got.at_chars, 2000, "should fire well before the 4350 chars num_predict produced")

    def test_glimmer_multi3img_degeneration(self):
        got = feed_all(ld.LoopDetector(), fixture("glimmer-nvfp4-multi3img-degenerate.txt"))
        self.assertIsNotNone(got)
        self.assertEqual(got.kind, ld.CONTENT)


class TestTokenRun(unittest.TestCase):
    def test_a_run_of_one_token_is_a_loop(self):
        det = ld.LoopDetector(token_run=64)
        got = None
        for _ in range(200):
            got = det.feed(content="  ") or got
            if got:
                break
        self.assertIsNotNone(got)
        self.assertEqual(got.kind, ld.TOKEN_RUN)
        self.assertEqual(got.failure_mode(), "loop detected:token-run")
        self.assertEqual(got.repeats, 64)

    def test_a_short_run_is_not(self):
        det = ld.LoopDetector(token_run=64)
        self.assertIsNone(next((g for g in (det.feed(content=" ") for _ in range(63)) if g), None))
        self.assertIsNone(det.feed(content="x"))

    def test_token_run_is_counted_per_stream(self):
        # Interleaving does not break a stream's run, and one stream's chunks never extend the other's:
        # thinking alternates (never runs) while content repeats, so the run found is content's.
        det = ld.LoopDetector(token_run=10)
        got = None
        for i in range(30):
            got = got or det.feed(thinking="ab"[i % 2])
            got = got or det.feed(content="z")
        self.assertEqual((got.kind, got.stream, got.repeats), (ld.TOKEN_RUN, ld.CONTENT, 10))


class TestIncremental(unittest.TestCase):
    def test_chunking_does_not_change_the_verdict(self):
        text = prose(1500, seed=9) + prose(420, seed=10) * 5
        verdicts = set()
        for chunk in (1, 3, 17, 64, 500):
            got = feed_all(ld.LoopDetector(), text, chunk=chunk)
            verdicts.add((got.kind, got.period) if got else None)
        self.assertEqual(verdicts, {(ld.CONTENT, 420)})

    def test_linear_enough(self):
        det = ld.LoopDetector()
        text = prose(300000, seed=11)
        t0 = time.perf_counter()     # the detector alone: generating the text is not timed
        self.assertIsNone(feed_all(det, text, chunk=4))
        self.assertLess(time.perf_counter() - t0, 3.0, "a 300k-char stream must not cost quadratic time")

    def test_record_is_self_describing(self):
        got = feed_all(ld.LoopDetector(), prose(900, seed=12) + prose(500, seed=13) * 4)
        rec = got.record()
        for key in ("loop", "loop_kind", "loop_stream", "loop_at_chars", "loop_period", "loop_repeats",
                    "loop_snippet", "loop_params"):
            self.assertIn(key, rec)
        self.assertEqual(rec["loop"], "loop detected:content")
        self.assertLessEqual(len(rec["loop_snippet"]), 160)


class TestCompatibility(unittest.TestCase):
    """The functions ai-invoice-to-xero's ollama_toolkit exports keep their names and meaning (one earlier copy)."""

    def test_thinking_is_looping(self):
        self.assertFalse(ld.thinking_is_looping("x" * 799))
        block = prose(450, seed=14)
        self.assertTrue(ld.thinking_is_looping(prose(500, seed=15) + block + block))
        self.assertFalse(ld.thinking_is_looping(prose(5000, seed=16)))

    def test_dedup_thinking(self):
        head, block = prose(500, seed=17), prose(450, seed=18)
        out = ld.dedup_thinking(head + block + block + block)
        self.assertTrue(out.startswith(head))
        self.assertLess(len(out), len(head) + 2 * len(block))
        self.assertEqual(ld.dedup_thinking("  short  "), "  short")


if __name__ == "__main__":
    unittest.main()
