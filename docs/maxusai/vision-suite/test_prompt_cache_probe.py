#!/usr/bin/env python3
"""Tests for prompt_cache_probe.py, against a fake ollama server."""
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import prompt_cache_probe as probe


class FakeOllama(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"version": "test-build"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append((self.path, body))
        self.reply({"response": f"answer {len(FakeOllama.seen)}", "prompt_eval_count": 100,
                    "prompt_eval_duration": 250_000_000, "eval_count": 3, "total_duration": 400_000_000})


class Run(unittest.TestCase):
    def setUp(self):
        FakeOllama.seen = []
        self.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def args(self, **kw):
        base = dict(host=self.host, model="m:1", label="cache on", image=[], conversations=2, turns=3,
                    doc_words=200, num_ctx=8192, num_predict=48, timeout=30)
        return SimpleNamespace(**dict(base, **kw))

    def test_conversations_take_turns_and_grow_a_shared_prefix(self):
        d = probe.run(self.args())
        self.assertEqual([r["context"] for r in d["records"]], [0, 1, 0, 1, 0, 1])
        prompts = [b["prompt"] for _, b in FakeOllama.seen]
        self.assertNotEqual(prompts[0][:300], prompts[1][:300])       # each conversation has its own document
        for earlier, later in ((0, 2), (2, 4), (1, 3), (3, 5)):          # and each request extends its last one
            self.assertTrue(prompts[later].startswith(prompts[earlier].rsplit("Question", 1)[0]))
            self.assertIn(f"answer {earlier + 1}", prompts[later])         # with the answer it got
        self.assertTrue(all(p == "/api/generate" for p, _ in FakeOllama.seen))

    def test_every_request_goes_through_the_shared_client(self):
        probe.run(self.args(turns=1))
        _, body = FakeOllama.seen[0]
        self.assertEqual(body["options"], {"num_predict": 48, "num_ctx": 8192, "temperature": 0})
        self.assertIs(body["think"], False)
        self.assertNotIn("format", body)

    def test_a_record_carries_the_server_timings_and_build(self):
        d = probe.run(self.args(turns=1))
        r = d["records"][0]
        self.assertEqual((r["prompt_eval_ms"], r["total_ms"], r["prompt_eval_count"]), (250.0, 400.0, 100))
        self.assertGreater(r["wall_ms"], 0)
        self.assertEqual((d["server_version"], d["workload"], d["settings"]["contexts"]),
                         (["test-build"], "conversations", 2))

    def test_images_take_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, data in (("a.png", b"image-a"), ("b.png", b"image-b")):
                paths.append(os.path.join(tmp, name))
                with open(paths[-1], "wb") as f:
                    f.write(data)
            d = probe.run(self.args(image=paths, turns=2))
        sent = [b["images"][0] for _, b in FakeOllama.seen]
        self.assertEqual(len(sent), 4)
        self.assertEqual((sent[0], sent[1]), (sent[2], sent[3]))          # a, b, a, b
        self.assertNotEqual(sent[0], sent[1])
        self.assertEqual([r["context"] for r in d["records"]], [0, 1, 0, 1])
        self.assertEqual(d["workload"], "images")

    def test_documents_are_deterministic_and_about_the_asked_length(self):
        text, facts = probe.document(1000, 2200)
        self.assertEqual(text, probe.document(1000, 2200)[0])
        self.assertTrue(2200 <= len(text.split()) < 2220 and facts > 100)


class Compare(unittest.TestCase):
    def test_one_row_per_run(self):
        def run(label, back_ms):
            recs = [{"context": c, "turn": t, "prompt_eval_count": 1000 + 10 * t,
                     "prompt_eval_ms": 2000.0 if t == 0 else back_ms, "wall_ms": 2500.0 if t == 0 else back_ms + 400}
                    for t in range(3) for c in range(2)]
            return {"label": label, "model": "m:1", "server_version": ["test-build"], "workload": "conversations",
                    "settings": {"contexts": 2, "turns": 3}, "records": recs}
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for label, back in (("cache on", 100.0), ("cache off", 2100.0)):
                paths.append(os.path.join(tmp, label.replace(" ", "-") + ".json"))
                with open(paths[-1], "w") as f:
                    json.dump(run(label, back), f)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                probe.compare(paths)
        text = out.getvalue()
        self.assertIn("| cache on | `m:1` | conversations | 2 × 3 | 2.00 s, 2.50 s | 0.10 s, 0.50 s | 1020 |", text)
        self.assertIn("| cache off | `m:1` | conversations | 2 × 3 | 2.00 s, 2.50 s | 2.10 s, 2.50 s | 1020 |", text)
        self.assertIn("build(s): test-build", text)


if __name__ == "__main__":
    unittest.main()
