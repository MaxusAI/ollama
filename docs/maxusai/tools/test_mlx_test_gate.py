#!/usr/bin/env python3
"""Tests for mlx_test_gate.py, on output captured from real runs."""
import io
import unittest

from mlx_test_gate import parse, verdict

# Captured 2026-09-20 from `go test -v ./mlxrunner/model/` in a container with
# NO MLX library on the search path. Note the last two lines: the package
# reports ok while nothing ran. This is the case the gate exists for.
NO_PAYLOAD = """=== RUN   TestMakeEmbeddingLayerQuantizedGlobalScale
2026/09/20 06:36:37 ERROR failed to load MLX dynamic library: path=/build/lib/ollama/mlx_cuda_v13/libmlxc.so
    embedding_test.go:78: MLX not available: failed to load MLX dynamic library (searched: [/tmp/go-build/lib/ollama])
--- SKIP: TestMakeEmbeddingLayerQuantizedGlobalScale (0.00s)
=== RUN   TestReadGlobalScale
    globalscale_test.go:175: MLX not available: failed to load MLX dynamic library (searched: [/tmp/go-build/lib/ollama])
--- SKIP: TestReadGlobalScale (0.00s)
PASS
ok  \tgithub.com/ollama/ollama/mlxrunner/model\t0.021s
"""

# Same day, payload wired, on CUDA: two real failures and one intentional skip.
WIRED = """=== RUN   TestSetWiredLimitRejectsOversizeWithoutChangingLimit
    memory_test.go:19: mlx: no max_recommended_working_set_size in device info
--- FAIL: TestSetWiredLimitRejectsOversizeWithoutChangingLimit (0.10s)
=== RUN   TestDequantizeGlobalScale
    ops_extra_test.go:152: scalar[1] = 1344, want 0.5
--- FAIL: TestDequantizeGlobalScale (0.30s)
=== RUN   TestMulGatherQMMGlobalScale
    ops_extra_test.go:36: building the unscaled gather requires a GPU backend
--- SKIP: TestMulGatherQMMGlobalScale (0.00s)
FAIL
FAIL\tgithub.com/ollama/ollama/mlx\t0.774s
"""

HEALTHY = """=== RUN   TestDequantizeGlobalScale
--- PASS: TestDequantizeGlobalScale (0.39s)
=== RUN   TestSetWiredLimitRejectsOversizeWithoutChangingLimit
    memory_test.go:19: wired limits are Metal-only
--- SKIP: TestSetWiredLimitRejectsOversizeWithoutChangingLimit (0.00s)
PASS
ok  \tgithub.com/ollama/ollama/mlx\t13.617s
"""


def check(text):
    out = io.StringIO()
    return verdict(parse(text.splitlines(keepends=True)), out), out.getvalue()


class TestGate(unittest.TestCase):
    def test_no_payload_fails_despite_ok(self):
        """The package printed ok; the gate must not."""
        rc, out = check(NO_PAYLOAD)
        self.assertEqual(rc, 1)
        self.assertIn("SKIPPED BECAUSE THE RUNTIME WAS NOT THERE", out)
        self.assertIn("MLX was not loadable", out)

    def test_no_payload_names_every_hollow_test(self):
        res = parse(NO_PAYLOAD.splitlines(keepends=True))
        self.assertEqual(len(res.hollow), 2)
        self.assertEqual(res.intentional, [])
        self.assertEqual(res.passed, [])

    def test_failures_fail(self):
        rc, out = check(WIRED)
        self.assertEqual(rc, 1)
        self.assertIn("tests failed", out)
        self.assertIn("TestDequantizeGlobalScale", out)

    def test_intentional_skip_is_reported_not_failed(self):
        res = parse(WIRED.splitlines(keepends=True))
        self.assertEqual([t for t, _ in res.intentional], ["TestMulGatherQMMGlobalScale"])
        self.assertEqual(res.hollow, [])

    def test_healthy_run_passes_with_the_skip_visible(self):
        rc, out = check(HEALTHY)
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", out)
        self.assertIn("skipped for a stated reason", out)
        self.assertIn("wired limits are Metal-only", out)

    def test_reason_is_taken_from_the_right_test(self):
        """Interleaved tests must not borrow each other's skip reason."""
        res = parse(WIRED.splitlines(keepends=True))
        reasons = dict(res.skipped)
        self.assertIn("GPU backend", reasons["TestMulGatherQMMGlobalScale"])

    def test_empty_run_fails(self):
        rc, out = check("PASS\nok  \tgithub.com/ollama/ollama/mlx\t0.001s\n")
        self.assertEqual(rc, 1)
        self.assertIn("no test ran", out)


if __name__ == "__main__":
    unittest.main()
