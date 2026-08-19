#!/usr/bin/env python3
"""Payload regression tests for client.generate() — the shared request path.

Written after a refactor consolidated five hand-built request paths into
client.py and shipped two blockers that every existing test passed straight
over, because test_rescore.py and test_summarizers.py only ever look at SCORES.
Nothing asserted what actually goes on the wire, which is the part a caller's
published numbers depend on.

The two that got through, both now covered here:

  * client.generate() could never emit `"think": true`. variants.py's thinkon
    arm -- whose only job is to PIN thinking on -- silently became
    "whatever the model defaults to".
  * finetext_probe.run() raised NameError AFTER its inference completed, so the
    model was paid for and the result discarded, and `set -eu` in
    run_engine_compare.sh aborted the campaign.

No network. urlopen is patched and the payload captured.
"""
import json
import os
import unittest
import urllib.request

import client


class Capture:
    """Patch urlopen, record the request, return a canned response."""

    def __init__(self, body=None):
        self.body = body or {"response": "{}", "thinking": "",
                             "eval_count": 3, "prompt_eval_count": 7}
        self.url = self.payload = self.timeout = None

    def __enter__(self):
        self._orig = urllib.request.urlopen
        def fake(req, timeout=None):
            self.url, self.timeout = req.full_url, timeout
            self.payload = json.loads(req.data)
            canned = json.dumps(self.body).encode()
            return type("R", (), {"read": lambda s: canned})()
        urllib.request.urlopen = fake
        return self

    def __exit__(self, *a):
        urllib.request.urlopen = self._orig


def call(**kw):
    with Capture() as cap:
        client.generate("http://h", "M", "p", ["IMG"], **kw)
    return cap


class TestThinkField(unittest.TestCase):
    """send_think is tri-state and each state has a caller depending on it."""

    def test_auto_sends_false_when_off(self):
        self.assertIs(call(think=False).payload.get("think"), False)

    def test_auto_omits_when_on(self):
        # vision_suite has always omitted it under think-on; preserving that is
        # what keeps campaign payloads unchanged across the refactor.
        self.assertNotIn("think", call(think=True).payload)

    def test_true_pins_thinking_on(self):
        # THE REGRESSION. variants.py thinkon passes send_think=True and needs
        # "think": true on the wire, not an omission that defers to a server
        # default which measure.py measured to be model- and build-specific.
        self.assertIs(call(think=True, send_think=True).payload.get("think"), True)

    def test_true_pins_thinking_off(self):
        self.assertIs(call(think=False, send_think=True).payload.get("think"), False)

    def test_false_omits_entirely(self):
        # measure.py's calibrated count probes were measured with no such field.
        for t in (True, False):
            self.assertNotIn("think", call(think=t, send_think=False).payload)


class TestNumCtxSentinel(unittest.TestCase):
    """False omits, 0 is a real value. `is False` matters: 0 == False in python."""

    def test_false_omits(self):
        self.assertNotIn("num_ctx", call(num_ctx=False).payload["options"])

    def test_zero_is_sent(self):
        self.assertEqual(call(num_ctx=0).payload["options"].get("num_ctx"), 0)

    def test_none_uses_default(self):
        self.assertEqual(call().payload["options"]["num_ctx"],
                         client.default_num_ctx())


class TestEndpoint(unittest.TestCase):

    def test_override_beats_env(self):
        os.environ["ENDPOINT"] = "chat"
        try:
            self.assertTrue(call(endpoint_override="generate").url.endswith("/api/generate"))
        finally:
            os.environ.pop("ENDPOINT", None)

    def test_chat_is_the_default(self):
        os.environ.pop("ENDPOINT", None)
        self.assertTrue(call().url.endswith("/api/chat"))

    def test_generate_carries_prompt_and_images(self):
        p = call(endpoint_override="generate").payload
        self.assertEqual(p["prompt"], "p")
        self.assertEqual(p["images"], ["IMG"])

    def test_chat_nests_into_messages(self):
        p = call(endpoint_override="chat").payload
        self.assertEqual(p["messages"][0]["content"], "p")
        self.assertEqual(p["messages"][0]["images"], ["IMG"])


class TestTextOnly(unittest.TestCase):
    """An empty image list is a text-only request and must omit the key.

    measure.py's text-prefix calibration compares an image-bearing request
    against a genuinely image-free one; "images": [] is neither."""

    def test_generate_omits_images(self):
        with Capture() as cap:
            client.generate("http://h", "M", "p", [], endpoint_override="generate")
        self.assertNotIn("images", cap.payload)

    def test_chat_omits_images(self):
        with Capture() as cap:
            client.generate("http://h", "M", "p", [], endpoint_override="chat")
        self.assertNotIn("images", cap.payload["messages"][0])


class TestSamplingAndOpts(unittest.TestCase):

    def test_apply_sampling_false_adds_nothing(self):
        opts = call(apply_sampling=False, num_ctx=False).payload["options"]
        self.assertEqual(set(opts), {"num_predict"})

    def test_extra_opts_win(self):
        self.assertEqual(call(num_predict=99, extra_opts={"num_predict": 7})
                         .payload["options"]["num_predict"], 7)

    def test_fmt_none_omits_format(self):
        self.assertNotIn("format", call(fmt=None).payload)


class TestNormalisation(unittest.TestCase):
    """Both envelopes must come back the same shape, or every caller needs to
    know which endpoint ran — which is the duplication this module removed."""

    def test_chat_envelope_normalised(self):
        with Capture({"message": {"content": "A", "thinking": "T"},
                      "eval_count": 5}) as cap:
            r = client.generate("http://h", "M", "p", ["IMG"],
                                endpoint_override="chat")
        self.assertEqual((r["response"], r["thinking"]), ("A", "T"))

    def test_generate_envelope_normalised(self):
        with Capture({"response": "A", "thinking": "T", "eval_count": 5}) as cap:
            r = client.generate("http://h", "M", "p", ["IMG"],
                                endpoint_override="generate")
        self.assertEqual((r["response"], r["thinking"]), ("A", "T"))


class TestCallersComplete(unittest.TestCase):
    """Every entry point must survive a full call, not merely import.

    finetext_probe.run() imported fine and raised NameError on the line AFTER
    the response came back. Import-only checks cannot see that."""

    def test_finetext_run_completes(self):
        import finetext_probe
        with Capture({"response": '{"codes": []}', "thinking": "",
                      "eval_count": 5, "prompt_eval_count": 9}):
            finetext_probe.run("http://h", "UNITTEST", "M")
        for f in (f"{client.DIR}/ft_UNITTEST.json",
                  f"{client.DIR}/resp_UNITTEST_finetext.json"):
            self.assertTrue(os.path.exists(f), f)
            os.remove(f)




class TestTransportRetry(unittest.TestCase):
    """Retry what is transient; never retry what is deterministic.

    A container restart killed rep 3 of a 3-repeat sweep after ~23 minutes of
    generation, with reps 1 and 2 already byte-identical -- an hour lost for no
    information. A 400, by contrast, will fail identically forever."""

    def setUp(self):
        self._orig = urllib.request.urlopen
        self._sleep = client.time.sleep
        client.time.sleep = lambda s: None      # no real waiting in tests

    def tearDown(self):
        urllib.request.urlopen = self._orig
        client.time.sleep = self._sleep

    def _flaky(self, fail_times, exc):
        state = {"n": 0}
        def fake(req, timeout=None):
            if state["n"] < fail_times:
                state["n"] += 1
                raise exc
            return type("R", (), {"read": lambda s: b'{"response":"ok","eval_count":1}'})()
        urllib.request.urlopen = fake
        return state

    def test_recovers_from_dropped_connection(self):
        self._flaky(2, ConnectionResetError("Remote end closed connection"))
        r = client.generate("http://h", "M", "p", ["IMG"])
        self.assertEqual(r["response"], "ok")
        self.assertEqual(len(r["_retries"]), 2)
        self.assertEqual([x["waited_s"] for x in r["_retries"]], [5, 15])

    def test_gives_up_after_the_backoff_is_exhausted(self):
        self._flaky(99, ConnectionResetError("host down"))
        with self.assertRaises(RuntimeError) as cm:
            client.generate("http://h", "M", "p", ["IMG"])
        self.assertIn("after 3 retries", str(cm.exception))

    def test_clean_call_records_no_retries(self):
        self._flaky(0, ConnectionResetError("never raised"))
        self.assertNotIn("_retries", client.generate("http://h", "M", "p", ["IMG"]))

    def test_400_is_NOT_retried(self):
        """A context-overflow 400 is deterministic. Retrying it three times only
        delays an actionable error by 50 seconds."""
        calls = {"n": 0}
        def fake(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                "u", 400, "Bad Request", {},
                __import__("io").BytesIO(b'{"error":"exceed_context_size"}'))
        urllib.request.urlopen = fake
        with self.assertRaises(RuntimeError):
            client.generate("http://h", "M", "p", ["IMG"])
        self.assertEqual(calls["n"], 1, "a 400 must be raised on the first attempt")

    def test_503_IS_retried(self):
        """A restarting container answers 503 before it answers properly."""
        state = {"n": 0}
        def fake(req, timeout=None):
            if state["n"] < 1:
                state["n"] += 1
                raise urllib.error.HTTPError("u", 503, "Unavailable", {},
                                             __import__("io").BytesIO(b""))
            return type("R", (), {"read": lambda s: b'{"response":"ok","eval_count":1}'})()
        urllib.request.urlopen = fake
        self.assertEqual(client.generate("http://h", "M", "p", ["IMG"])["response"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
