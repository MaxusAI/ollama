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
import urllib.error
import urllib.request
from unittest import mock

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


class TestRawField(unittest.TestCase):
    """raw skips the chat template; token_split.py --server counts
    prompt_eval_count of bare text, and a template wrapper would add tokens
    that belong to the template, not the text. Inert unless set (H4)."""

    def test_omitted_by_default(self):
        self.assertNotIn("raw", call().payload)

    def test_sent_when_requested(self):
        p = call(endpoint_override="generate", raw=True).payload
        self.assertIs(p["raw"], True)


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
        # use_env_opts=False: the Runner options the harness sends by default
        # (prompt_cache_ram) are TestPromptCacheRAM's subject, not this one's.
        opts = call(apply_sampling=False, num_ctx=False,
                    use_env_opts=False).payload["options"]
        self.assertEqual(set(opts), {"num_predict"})

    def test_extra_opts_win(self):
        self.assertEqual(call(num_predict=99, extra_opts={"num_predict": 7})
                         .payload["options"]["num_predict"], 7)

    def test_fmt_none_omits_format(self):
        self.assertNotIn("format", call(fmt=None).payload)


class TestPromptCacheRAM(unittest.TestCase):
    """Every harness request turns llama-server's prompt cache off unless
    PROMPT_CACHE_RAM says otherwise (ADR 0047), with one value for every tool:
    a Runner option that differs between vision_suite.py and finetext_probe.py
    would relaunch the model between the two halves of a cell."""

    def opts(self, env=None, **kw):
        with mock.patch.dict(os.environ):
            os.environ.pop("PROMPT_CACHE_RAM", None)
            os.environ.update(env or {})
            with Capture() as cap:
                r = client.generate("http://h", "M", "p", ["IMG"], **kw)
        return cap.payload["options"], r

    def test_off_by_default(self):
        opts, r = self.opts()
        self.assertEqual(opts["prompt_cache_ram"], 0)
        self.assertEqual(r["_prompt_cache_ram"], 0)

    def test_a_size_or_no_limit_is_sent_as_given(self):
        self.assertEqual(self.opts({"PROMPT_CACHE_RAM": "4096"})[0]["prompt_cache_ram"], 4096)
        self.assertEqual(self.opts({"PROMPT_CACHE_RAM": "-1"})[0]["prompt_cache_ram"], -1)

    def test_server_sends_none(self):
        opts, r = self.opts({"PROMPT_CACHE_RAM": "server"})
        self.assertNotIn("prompt_cache_ram", opts)
        self.assertNotIn("_prompt_cache_ram", r)

    def test_calibrated_callers_send_none(self):
        # measure.py, token_split.py and prompt_cache_probe.py: exact payloads.
        opts, _ = self.opts(use_env_opts=False)
        self.assertNotIn("prompt_cache_ram", opts)

    def test_a_value_the_server_would_ignore_fails_before_sending(self):
        # The server keeps its default for -5, so sending it would stamp a
        # setting that never ran; "off" is a typo for 0.
        for bad in ("-5", "off", "1.5"):
            with self.subTest(bad=bad), mock.patch.dict(os.environ, {"PROMPT_CACHE_RAM": bad}):
                with Capture() as cap, self.assertRaises(ValueError):
                    client.generate("http://h", "M", "p", ["IMG"])
                self.assertIsNone(cap.payload)

    def test_the_block_records_what_was_sent(self):
        _, r = self.opts()
        self.assertEqual(client.capture_stamps(r, "M", "T")["req_prompt_cache_ram"], 0)
        _, r = self.opts({"PROMPT_CACHE_RAM": "server"})
        self.assertNotIn("req_prompt_cache_ram", client.capture_stamps(r, "M", "T"))


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
        # resp_*_finetext_probe.json, NOT resp_*_finetext.json: the probe
        # persists under its own name so it can never overwrite the suite's
        # folded finetext text (SPEC H14 — one producer per persist name).
        for f in (f"{client.DIR}/ft_UNITTEST.json",
                  f"{client.DIR}/resp_UNITTEST_finetext_probe.json"):
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


class TestTimeoutPolicy(unittest.TestCase):
    """A timeout is the one transport-shaped failure that is deterministic for
    the request (blueprint P0-6): budget scales with num_predict, operator env
    wins verbatim, and a blown timeout is never retried."""

    def test_timeout_scales_with_num_predict(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("HTTP_TIMEOUT", None)
            cap = call(num_predict=122880)
        self.assertGreaterEqual(cap.timeout, 122880 / 20)

    def test_operator_http_timeout_wins_verbatim(self):
        with mock.patch.dict(os.environ, {"HTTP_TIMEOUT": "900"}):
            cap = call(num_predict=122880)
        self.assertEqual(cap.timeout, 900)

    def test_http_timeout_zero_means_unset(self):
        # HTTP_TIMEOUT="0" would otherwise reach urlopen(timeout=0) and fail
        # every request instantly — and terminally, per the branch below.
        with mock.patch.dict(os.environ, {"HTTP_TIMEOUT": "0"}):
            cap = call(num_predict=122880)
        self.assertGreaterEqual(cap.timeout, 122880 / 20)

    def test_read_timeout_is_terminal_not_retried(self):
        # A blown READ budget surfaces as a BARE TimeoutError (urllib does
        # not wrap getresponse) and is deterministic: escalation or a larger
        # HTTP_TIMEOUT is the cure, not a retry.
        calls = {"n": 0}
        orig, origsleep = urllib.request.urlopen, client.time.sleep

        def fake(req, timeout=None):
            calls["n"] += 1
            raise TimeoutError("timed out")

        urllib.request.urlopen = fake
        client.time.sleep = (lambda s:
                             (_ for _ in ()).throw(AssertionError("slept")))
        try:
            with self.assertRaises(RuntimeError) as cm:
                client.generate("http://h", "M", "p", [])
            self.assertIn("blown generation budget", str(cm.exception))
            self.assertEqual(calls["n"], 1)
        finally:
            urllib.request.urlopen, client.time.sleep = orig, origsleep

    def test_connect_timeout_is_retried(self):
        # A CONNECT timeout arrives wrapped as URLError(reason=TimeoutError)
        # — the server-restart window the backoff exists for. socket.timeout
        # IS TimeoutError on py3.10+, so a naive reason check made this
        # terminal and the documented retry-on-restart path unreachable.
        calls = {"n": 0}
        orig, origsleep = urllib.request.urlopen, client.time.sleep
        canned = json.dumps({"response": "ok", "eval_count": 1}).encode()

        def fake(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError(TimeoutError("connect timed out"))
            return type("R", (), {"read": lambda s: canned})()

        urllib.request.urlopen = fake
        client.time.sleep = lambda s: None
        try:
            r = client.generate("http://h", "M", "p", [])
            self.assertEqual(r["response"], "ok")
            self.assertEqual(calls["n"], 2)
            self.assertEqual(len(r["_retries"]), 1)
        finally:
            urllib.request.urlopen, client.time.sleep = orig, origsleep


class TestCaptureStamps(unittest.TestCase):
    """Identity must live in the record, not the filename (blueprint P0-2):
    the request path stamps endpoint and think-as-sent, and capture_stamps
    turns them plus env provenance into block fields."""

    def test_endpoint_and_think_stamped_on_response(self):
        # Pinned against ambient campaign env: THINK/ENDPOINT are documented
        # driver exports and must not steer these assertions.
        with mock.patch.dict(os.environ):
            os.environ.pop("THINK", None)
            os.environ.pop("ENDPOINT", None)
            with Capture() as cap:
                r = client.generate("http://h", "M", "p", ["IMG"])
        self.assertEqual(r["_endpoint"], "chat")
        # send_think defaults to "auto", which with think OFF sends
        # "think": false — _think_sent is the wire truth (False here, None
        # only when the field was omitted) while _think records the RESOLVED
        # mode, which under auto+on is otherwise invisible in the payload.
        self.assertIs(r["_think_sent"], False)
        self.assertIs(r["_think"], False)

    def test_think_on_auto_resolved_mode_still_recorded(self):
        with Capture():
            r = client.generate("http://h", "M", "p", ["IMG"], think=True)
        self.assertIsNone(r["_think_sent"])   # auto omits the field when on
        self.assertIs(r["_think"], True)      # but the mode is not lost

    def test_think_omitted_stamps_none(self):
        with Capture():
            r = client.generate("http://h", "M", "p", ["IMG"], send_think=False)
        self.assertIsNone(r["_think_sent"])

    def test_generate_override_and_think_true(self):
        with Capture():
            r = client.generate("http://h", "M", "p", ["IMG"],
                                endpoint_override="generate",
                                think=True, send_think=True)
        self.assertEqual(r["_endpoint"], "generate")
        self.assertIs(r["_think_sent"], True)

    def test_capture_stamps_block_fields(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("THINK", None)
            os.environ.pop("ENDPOINT", None)
            with Capture():
                r = client.generate("http://h", "M", "p", ["IMG"])
        # Driver facts arrive as PARAMETERS: the producers read POWERMODE /
        # COLD_START_MECH from the env at the campaign layer; client takes
        # data, not ambient env (blueprint rule 2).
        st = client.capture_stamps(r, model="M", tag="T1", powermode="2")
        self.assertEqual(st["capture_schema"], 1)
        self.assertEqual(st["model"], "M")
        self.assertEqual(st["tag"], "T1")
        self.assertEqual(st["endpoint"], "chat")
        self.assertIs(st["think_sent"], False)
        self.assertEqual(st["retries"], 0)
        self.assertEqual(st["powermode"], "2")

    def test_retried_cell_records_count(self):
        with Capture():
            r = client.generate("http://h", "M", "p", ["IMG"])
        r["_retries"] = [{"attempt": 1, "waited_s": 5, "error": "x"}]
        st = client.capture_stamps(r, model="M", tag="T1")
        self.assertEqual(st["retries"], 1)




class StreamCapture:
    """Patch urlopen with a streaming server: one JSON line per chunk, then a final done chunk."""

    def __init__(self, chunks, final=None, endpoint="chat"):
        self.lines = []
        for th, co in chunks:
            if endpoint == "chat":
                self.lines.append({"message": {"role": "assistant", "content": co, "thinking": th}, "done": False})
            else:
                self.lines.append({"response": co, "thinking": th, "done": False})
        last = {"done": True, "done_reason": "stop", "eval_count": len(chunks), "prompt_eval_count": 7}
        last.update(final or {})
        if endpoint == "chat":
            last["message"] = {"role": "assistant", "content": ""}
        else:
            last["response"] = ""
        self.lines.append(last)
        self.payload, self.closed, self.read_lines = None, False, 0

    def __enter__(self):
        self._orig = urllib.request.urlopen
        cap = self

        class Resp:
            def __iter__(self_inner):
                for obj in cap.lines:
                    if cap.closed:
                        return
                    cap.read_lines += 1
                    yield (json.dumps(obj) + "\n").encode()

            def close(self_inner):
                cap.closed = True

        def fake(req, timeout=None):
            cap.payload = json.loads(req.data)
            return Resp()
        urllib.request.urlopen = fake
        return self

    def __exit__(self, *a):
        urllib.request.urlopen = self._orig


class TestLoopDetect(unittest.TestCase):
    """LOOP_DETECT=1 streams the request and cancels a verbatim loop (loop_detect.py, ADR 0049)."""

    def setUp(self):
        self._env = os.environ.get("LOOP_DETECT")
        os.environ["LOOP_DETECT"] = "1"

    def tearDown(self):
        if self._env is None:
            os.environ.pop("LOOP_DETECT", None)
        else:
            os.environ["LOOP_DETECT"] = self._env

    def test_off_by_default_sends_no_stream(self):
        os.environ.pop("LOOP_DETECT", None)
        self.assertFalse(call().payload["stream"])

    def test_a_clean_stream_returns_the_same_shape(self):
        chunks = [("thinking ", ""), ("done.", ""), ("", '{"a"'), ("", ": 1}")]
        with StreamCapture(chunks) as cap:
            r = client.generate("http://h", "M", "p", ["IMG"], think=True)
        self.assertTrue(cap.payload["stream"])
        self.assertEqual(r["response"], '{"a": 1}')
        self.assertEqual(r["thinking"], "thinking done.")
        self.assertEqual(r["done_reason"], "stop")
        self.assertEqual(r["eval_count"], 4)
        self.assertTrue(r["_stream"])
        self.assertEqual(r["_loop_detect"]["min_repeats"], 3)

    def test_generate_endpoint_streams_too(self):
        with StreamCapture([("", "hel"), ("", "lo")], endpoint="generate") as cap:
            r = client.generate("http://h", "M", "p", ["IMG"], endpoint_override="generate")
        self.assertTrue(cap.payload["stream"])
        self.assertEqual(r["response"], "hello")

    def test_a_content_loop_is_cancelled_and_returned_as_a_verdict(self):
        obj = '{"x1": 70, "y1": 147, "x2": 210, "y2": 336, "label": "ANCHOR", "bbox_type": "norm1000", '
        chunks = [("", '{"boxes": [')] + [("", obj[i:i + 5]) for _ in range(40) for i in range(0, len(obj), 5)]
        with StreamCapture(chunks) as cap:
            r = client.generate("http://h", "M", "p", ["IMG"])
        self.assertTrue(cap.closed, "the stream must be closed so the server stops generating")
        self.assertLess(cap.read_lines, len(cap.lines), "it must stop reading at the loop, not drain the stream")
        self.assertEqual(r["done_reason"], "loop")
        self.assertEqual(r["failure_mode"], "loop detected:content")
        self.assertEqual(r["_loop"]["loop_period"], len(obj))
        self.assertTrue(r["response"].startswith('{"boxes": [{"x1": 70'), "the partial answer is kept for scoring")
        self.assertIsNone(r["eval_duration"])

    def test_a_token_run_is_cancelled(self):
        with StreamCapture([("", '{"a": ')] + [("", "  ")] * 200) as cap:
            r = client.generate("http://h", "M", "p", ["IMG"])
        self.assertEqual(r["failure_mode"], "loop detected:token-run")
        self.assertTrue(cap.closed)

    def test_a_thinking_loop_keeps_the_trace(self):
        block = "Let me reconsider the coordinates of the second box once more carefully. " * 7
        with StreamCapture([(block[i:i + 9], "") for _ in range(4) for i in range(0, len(block), 9)]):
            r = client.generate("http://h", "M", "p", ["IMG"], think=True)
        self.assertEqual(r["failure_mode"], "loop detected:thinking")
        self.assertEqual(r["response"], "")
        self.assertTrue(r["thinking"].startswith("Let me reconsider"))

    def test_a_loop_is_finished_not_capped(self):
        # A loop is a verdict, not a cap: it must neither climb the context ladder (capped_arms) nor be re-run on
        # resume (arm_done / ft_done), because the same greedy decode loops identically at every rung.
        import summarize_engine_compare as sec
        import vision_suite
        blk = {"done_reason": "loop", "failure_mode": "loop detected:content", "eval_count": 312,
               "num_predict": 2200, "loop": "loop detected:content"}
        self.assertFalse(sec.was_capped(blk))
        self.assertTrue(vision_suite.arm_done(blk))
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"case_a": blk, "case_b": {"done_reason": "length", "eval_count": 2200, "num_predict": 2200}},
                      fh)
        self.assertEqual(sec.capped_arms(fh.name), ["case_b"])


class TestLoopFields(unittest.TestCase):
    """client.loop_fields is the one derivation both producers stamp onto a block (as metrics_block is)."""

    def test_a_loop_result_stamps_its_verdict(self):
        r = {"done_reason": "loop", "failure_mode": "loop detected:content", "_stream": True,
             "_loop_detect": {"window": 400}, "_loop": {"loop": "loop detected:content", "loop_period": 87}}
        f = client.loop_fields(r)
        self.assertEqual(f["failure_mode"], "loop detected:content")
        self.assertEqual(f["loop_period"], 87)
        self.assertEqual(f["stream"], True)
        self.assertEqual(f["loop_detect"], {"window": 400})

    def test_a_clean_streamed_result_says_detection_ran(self):
        f = client.loop_fields({"done_reason": "stop", "_stream": True, "_loop_detect": {"window": 400}})
        self.assertEqual(f, {"stream": True, "loop_detect": {"window": 400}})

    def test_a_non_streamed_result_adds_nothing(self):
        self.assertEqual(client.loop_fields({"done_reason": "stop"}), {})

if __name__ == "__main__":
    unittest.main(verbosity=2)
