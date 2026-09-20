#!/usr/bin/env python3
"""Did the MLX tests actually run, or did they skip?

    python3 docs/maxusai/tools/mlx_test_gate.py --parse gotest.log
    go test -v ./mlx/... ./mlxrunner/... | python3 docs/maxusai/tools/mlx_test_gate.py --parse -

Exit 0 when every MLX test either passed or skipped for a stated, intentional
reason. Exit 1 on a failure, or on a skip that means the payload was not there.

WHY. `go test` prints the same word for a package whose tests passed and one
whose tests never ran:

    ok  github.com/ollama/ollama/mlxrunner/model  0.021s

That line is from a run in which both tests skipped with "MLX not available",
because the container had no MLX dynamic library on the search path. The v0.34.2
fold's Go gate read exactly that line as a pass and merged three failing ADR 0039
assertions onto `main`; the same trap hid a vision golden behind a 0640 manifest
the same day. The tell is the duration -- a package that touches the GPU does not
finish in 21 ms -- but nobody reads durations, so this reads them instead.

WHAT COUNTS AS A REAL SKIP. A test that says *why* it is not applicable here is
doing its job: wired limits are Metal-only, an e2e test wants a model and an env
flag. Those are reported, never failed on -- a gate that cries wolf gets skipped
itself (SPEC H20). What fails the gate is a skip that says the runtime was not
available at all, because such a run carries no information about the code:

    MLX not available: failed to load MLX dynamic library
    MLX GPU not available

The distinction is the point. An intentional skip narrows what was measured; an
unavailable-runtime skip means nothing was measured, while the package still
prints `ok`.

ON THIS HOST it means pointing the run at the CUDA payload -- LD_LIBRARY_PATH,
CUDA_PATH and CUDA_HOME at a `mlx_cuda_v13` directory, or a symlink to one at
build/lib/ollama. --run does that for you; --parse takes a log from anywhere,
including an Apple host, so both sides of a fold can be checked the same way.
"""
import argparse
import os
import re
import subprocess
import sys

# `go test -v` frames every test with these. The reason for a skip is printed by
# t.Skipf on the preceding indented `file.go:NN: ...` line.
RUN = re.compile(r"^=== RUN\s+(\S+)")
RESULT = re.compile(r"^\s*--- (PASS|FAIL|SKIP): (\S+)")
DETAIL = re.compile(r"^\s+\S+\.go:\d+:\s*(.*)$")
PKG = re.compile(r"^(ok|FAIL|---)\s+(\S+)")

# A skip carrying one of these did not measure anything: the runtime the test
# needs was missing, so the package's `ok` is empty. Keep this list short and
# literal -- every entry must name a runtime, never a capability a backend
# legitimately lacks.
UNAVAILABLE = (
    "MLX not available",
    "MLX GPU not available",
    "failed to load MLX dynamic library",
)


class Result:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []          # (test, reason)

    @property
    def hollow(self):
        """Skips that mean the runtime was absent."""
        return [(t, r) for t, r in self.skipped
                if any(u in r for u in UNAVAILABLE)]

    @property
    def intentional(self):
        return [(t, r) for t, r in self.skipped
                if not any(u in r for u in UNAVAILABLE)]


def parse(lines):
    """Classify `go test -v` output.

    The reason for a skip is the last detail line seen since the test started,
    which is where t.Skipf writes it. Tests interleave under -parallel, so the
    detail is tracked per test name rather than globally.
    """
    res = Result()
    current = None
    details = {}
    for raw in lines:
        line = raw.rstrip("\n")
        m = RUN.match(line)
        if m:
            current = m.group(1)
            details.setdefault(current, "")
            continue
        m = RESULT.match(line)
        if m:
            status, name = m.group(1), m.group(2)
            if status == "PASS":
                res.passed.append(name)
            elif status == "FAIL":
                res.failed.append(name)
            else:
                res.skipped.append((name, details.get(name, "").strip()))
            continue
        m = DETAIL.match(line)
        if m and current:
            details[current] = m.group(1)
    return res


def render(res, out=sys.stdout):
    print(f"  passed {len(res.passed)}   skipped {len(res.skipped)}   failed {len(res.failed)}", file=out)
    if res.intentional:
        print("\n  skipped for a stated reason:", file=out)
        for test, reason in res.intentional:
            print(f"    {test}: {reason or '(no reason given)'}", file=out)
    if res.hollow:
        print("\n  SKIPPED BECAUSE THE RUNTIME WAS NOT THERE:", file=out)
        for test, reason in res.hollow:
            print(f"    {test}: {reason}", file=out)
    if res.failed:
        print("\n  FAILED:", file=out)
        for test in res.failed:
            print(f"    {test}", file=out)


def verdict(res, out=sys.stdout):
    render(res, out)
    if res.failed:
        print("\nVERDICT: FAIL — tests failed.", file=out)
        return 1
    if res.hollow:
        print(f"\nVERDICT: FAIL — {len(res.hollow)} test(s) skipped because MLX was not "
              "loadable. The package still printed 'ok'; that result means nothing. "
              "Wire the payload (LD_LIBRARY_PATH/CUDA_PATH/CUDA_HOME at mlx_cuda_v13) "
              "and run again.", file=out)
        return 1
    if not res.passed:
        print("\nVERDICT: FAIL — no test ran.", file=out)
        return 1
    print("\nVERDICT: PASS", file=out)
    return 0


def run(payload, packages, cache=None, ptx_cache=None, image="golang:1.26.0"):
    """Run the packages in a container with the payload wired, return its output.

    cache is a directory holding gocache/ and gomodcache/. Without it every run
    recompiles cgo from scratch, which takes minutes and is why people stop
    using a gate.
    """
    # Resolve the repo from the working directory, not from this file: the tool
    # gets copied into a scratch dir to run against a branch that does not carry
    # it yet, and a script-relative root then points at the wrong tree -- `go`
    # reports "go.mod file not found" and the gate calls it "no test ran".
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip()
    if not repo:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    payload = os.path.abspath(payload)
    if not os.path.isdir(payload):
        sys.exit(f"payload directory not found: {payload}")
    # Mount the payload's PARENT, not the payload. MLX's JIT resolves its
    # support headers from the backend include dir and, on Linux, also probes
    # current_binary_dir().parent_path()/"include" -- which in a shipped layout
    # is a sibling symlink next to mlx_cuda_v13 (see cmake/mlx/CMakeLists.txt).
    # Mounting only mlx_cuda_v13 hides that sibling, and the failure is silent
    # until a kernel misses the PTX cache and has to compile:
    #   catastrophic error: cannot open source file "cuda/std/tuple"
    parent = os.path.dirname(payload)
    cmd = [
        "docker", "run", "--rm", "--gpus", '"device=0"',
        "-v", f"{repo}:{repo}", "-v", f"{parent}:{parent}",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "GOFLAGS=-buildvcs=false", "-e", "CGO_ENABLED=1",
        "-e", f"LD_LIBRARY_PATH={payload}", "-e", f"CUDA_PATH={payload}",
        "-e", f"CUDA_HOME={payload}",
    ]
    if ptx_cache:
        # One directory per GPU architecture. A shared cache once aborted every
        # gemma4 request with a Turing kernel on a Blackwell card.
        ptx_cache = os.path.abspath(ptx_cache)
        os.makedirs(ptx_cache, exist_ok=True)
        cmd += ["-v", f"{ptx_cache}:{ptx_cache}", "-e", f"MLX_PTX_CACHE_DIR={ptx_cache}"]
    if cache:
        cache = os.path.abspath(cache)
        for sub, env in (("gocache", "GOCACHE"), ("gomodcache", "GOMODCACHE")):
            os.makedirs(os.path.join(cache, sub), exist_ok=True)
            cmd += ["-v", f"{os.path.join(cache, sub)}:/{sub}", "-e", f"{env}=/{sub}"]
    cmd += ["-w", repo, image, "go", "test", "-v", "-count=1", *packages]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).splitlines(keepends=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parse", metavar="FILE",
                    help="classify an existing `go test -v` log ('-' for stdin)")
    ap.add_argument("--payload", metavar="DIR",
                    help="mlx_cuda_v13 directory to wire into the run")
    ap.add_argument("--cache", metavar="DIR",
                    help="directory holding gocache/ and gomodcache/ to reuse")
    ap.add_argument("--ptx-cache", metavar="DIR",
                    help="MLX_PTX_CACHE_DIR for this GPU architecture (one dir per arch)")
    ap.add_argument("packages", nargs="*", default=None,
                    help="packages to test (default ./mlx/... ./mlxrunner/...)")
    args = ap.parse_args()

    if args.parse:
        src = sys.stdin if args.parse == "-" else open(args.parse)
        lines = src.readlines()
    elif args.payload:
        pkgs = args.packages or ["./mlx/...", "./mlxrunner/..."]
        lines = run(args.payload, pkgs, cache=args.cache, ptx_cache=args.ptx_cache)
        sys.stdout.writelines(lines)
    else:
        ap.error("pass --parse FILE or --payload DIR")

    sys.exit(verdict(parse(lines)))


if __name__ == "__main__":
    main()
