#!/usr/bin/env python3
"""What each request leaves behind, from a trace-level (OLLAMA_DEBUG=2) runner log.

    summarize_retained_memory.py <runner.log> [model] [--unnamed] [--shape 'F32 [1 48 128 128]'] [--top N]

At every request's teardown the runner sweeps, clears MLX's cache, then lists every array it still tracks and logs
MLX's own active figure. So each row is what survives BETWEEN requests: weights, live caches, prefix-trie snapshots.

The column that matters is `untracked`, MLX's active memory minus the arrays the runner tracks. It should sit near
zero. Memory that MLX holds but no tracked array accounts for is invisible to the admission headroom and to the
trie's byte cap, so a figure that grows request after request is a leak in everything that prices memory. In the
v0.34.0 fold, drafting under a grammar took it to +6.8 GiB by request 28 on qwen3.8 while the no-drafting control
stayed within 0.14 GiB.

**Runners from v0.34.1 log none of that.** Array lifetimes are scoped, the per-array listing and its totals are gone,
and the runner logs one `memory` line per request: its peak and `held`, MLX's active memory once the request's scope
ended and the cache was cleared — the same "what survives between requests", with no tracked side to subtract. So
on those logs this prints `held` and its step from the previous request of the same model; a step that keeps coming
is the signal `untracked` used to carry. The array-shape options have nothing to read there.

`--unnamed` groups only arrays without a dotted name — caches and trie snapshots, not the fixed weights — which is
how an accumulating shape shows itself. `--shape` prints one compact line per request for a single (dtype, dims)
group, for setting two runs side by side.

An exploratory table by ADR 0012 rule 7.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runnerlog  # noqa: E402  (ADR 0028 rule 3: one runner-log parser)


def flag(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--top", "--shape")]
    if not args:
        sys.exit(__doc__)
    path = args[0]
    model = args[1] if len(args) > 1 else None
    top, shape, unnamed = int(flag("--top", 4)), flag("--shape"), "--unnamed" in sys.argv
    last_held = {}
    last_resid = {}
    for r in runnerlog.iter_requests(path, model=model, shapes=bool(shape) or top > 0, named=not unnamed):
        peak = runnerlog.gib(r.peak)
        if r.held is not None:
            held = runnerlog.gib(r.held)
            prev = last_held.get(r.model)
            last_held[r.model] = held
            step = f", step {held - prev:+.2f} GiB" if prev is not None else ""
            trie = ""
            if r.trie_active is not None:
                # With the trie's own accounting line (trace level) the residual held − trie is the weights plus
                # whatever nothing tracks; its step between requests is the leak rate with the trie's growth removed.
                resid = held - runnerlog.gib(r.trie_active)
                rprev = last_resid.get(r.model)
                last_resid[r.model] = resid
                rstep = f", step {resid - rprev:+.2f} GiB" if rprev is not None else ""
                trie = f" | trie {runnerlog.gib(r.trie_active):.2f} GiB | held−trie {resid:.2f} GiB{rstep}"
            print(f"{r.model} req {r.index:2d}: peak {peak:.2f} GiB | held {held:.2f} GiB{step}{trie}")
            continue
        if shape:
            dt, _, dims = shape.partition(" ")
            count, size = r.shapes.get((dt, dims.strip("[]")), (0, 0.0))
            paged = runnerlog.gib(r.paged_out)
            print(f"{r.model} req {r.index:2d}: peak {peak:6.2f} GiB | {shape} ×{count:5d} = {size / (1 << 30):6.2f} GiB"
                  f" | trie paged_out {paged:.2f} GiB" if paged is not None else "")
            continue
        head = f"{r.model} req {r.index:2d}: peak {peak:.2f} GiB"
        if r.tracked is not None:
            head += (f" | live arrays {r.arrays}, tracked {runnerlog.gib(r.tracked):.2f} GiB, "
                     f"active {runnerlog.gib(r.active):.2f} GiB, untracked {runnerlog.gib(r.untracked):+.2f} GiB")
        print(head)
        if r.paged_out is not None:
            print(f"      trie paged_out {runnerlog.gib(r.paged_out):.2f} GiB, nodes {r.trie_nodes}, "
                  f"snapshots {r.trie_snapshots}, active_tokens {r.active_tokens}")
        for (dt, dims), (count, size) in sorted(r.shapes.items(), key=lambda kv: -kv[1][1])[:top]:
            print(f"      {count:5d} × {dt:8s} [{dims}] = {size / (1 << 30):.2f} GiB")


if __name__ == "__main__":
    main()
