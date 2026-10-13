#!/usr/bin/env python3
"""mmq-rules-check-native.py [--llama-src DIR] [--keep DIR]

The CPU padding sweep (mmq-rules-check.cu) built and run with an ordinary C++ compiler -- no CUDA toolkit, so it
runs on the Metal host too. By default it reads build/_deps/llama_cpp-src, the tree the last build compiled, with
the compat series applied.

mmq-native-shim.py copies the host-only definitions the sweep uses out of that tree, verbatim, so the unmodified
sweep source compiles. On llama.cpp dd266785c the output is byte-identical to an nvcc build's
(mmq-successor-results/check-rules-wide.txt), and b11351's ten config tables are byte-identical to dd266785c's.
About 45 s on one core.

A TOOL, NOT A GATE. The two rules the fork has shipped -- "#448 amended (widest padded tile)" and "#29953 head
(launched padded tile)" -- are sufficient by construction in the sweep's own model: both are computed from the
same tables the requirement is. No table change can make them short (tested: an ampere q4_0 config at 192 threads
still reads "covered"). What can regress in a build -- the allocation using another rule, or a non-power-of-two
thread count breaking GGML_PAD -- is what 903's compile-time y-tile guard catches, in every CUDA and HIP build.
Use this to score a NEW candidate rule (e.g. upstream's, at a pin move) against every shape on ten architectures.

Exit: the sweep's own (0 when both shipped rules are covered); 2 for no source tree or a build failure.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TASKS = HERE
SWEEP = os.path.join(TASKS, "mmq-rules-check.cu")
SHIM = os.path.join(TASKS, "mmq-native-shim.py")
SHIPPED = ("#448 amended (widest padded tile)", "#29953 head (launched padded tile)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llama-src", default=os.environ.get("LLAMA_CPP_SRC",
                                                          os.path.join(REPO, "build", "_deps", "llama_cpp-src")),
                    help="llama.cpp source tree (default: the last build's patched tree)")
    ap.add_argument("--keep", help="keep the shim, binary and full output in this directory")
    args = ap.parse_args()

    src = os.path.abspath(args.llama_src)
    if not os.path.isfile(os.path.join(src, "ggml", "src", "ggml-cuda", "mmq.cuh")):
        print(f"mmq-rules-check-native: no llama.cpp tree at {src}; build first, or pass --llama-src", file=sys.stderr)
        return 2
    rev = subprocess.run(["git", "-C", src, "describe", "--tags", "--always"], capture_output=True, text=True)
    with open(os.path.join(src, "ggml", "src", "ggml-cuda", "mmq.cu")) as fh:
        mmq_cu = fh.read()
    guarded = "GGML_CUDA_MMQ_Y_TILE_GUARD" in mmq_cu or "GGML_CUDA_MMQ_PADDING_GUARD" in mmq_cu
    print(f"llama.cpp tree: {src} ({rev.stdout.strip() or 'no git'}); 903's compile-time guard in mmq.cu: "
          f"{'present' if guarded else 'ABSENT'}")

    work = args.keep or tempfile.mkdtemp(prefix="mmq-padding-")
    os.makedirs(work, exist_ok=True)
    try:
        r = subprocess.run([sys.executable, SHIM, src, os.path.join(work, "shim")], capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr, file=sys.stderr)
            return 2
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or "c++"
        exe = os.path.join(work, "mmq-rules-check")
        inc = ["-I", os.path.join(work, "shim"), "-I", src, "-I", os.path.join(src, "ggml", "include"),
               "-I", os.path.join(src, "ggml", "src"), "-I", os.path.join(src, "ggml", "src", "ggml-cuda")]
        r = subprocess.run([cxx, "-std=c++17", "-O2", "-x", "c++", *inc, SWEEP, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr, file=sys.stderr)
            return 2
        r = subprocess.run([exe], capture_output=True, text=True)
        if args.keep:
            with open(os.path.join(work, "check-rules.txt"), "w") as fh:
                fh.write(r.stdout)
        # the section headers, and the rows of the two shipped rules (a SHORT row's detail lines follow it)
        show = False
        for line in r.stdout.splitlines():
            if not line.startswith("  "):
                show = not line.startswith("    ")
                print(line)
            elif not line.startswith("   "):
                show = any(line.strip().startswith(name) for name in SHIPPED)
                if show:
                    print(line)
            elif show:
                print(line)
        print("shipped rules:", "covered" if r.returncode == 0 else "SHORT -- rerun with --keep for the full table")
        return 0 if r.returncode == 0 else 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
