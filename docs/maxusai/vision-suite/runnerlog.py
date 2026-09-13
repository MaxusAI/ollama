#!/usr/bin/env python3
"""One parser for the runner log that every drafting and memory summarizer reads.

The input is `docker logs <ollama container>`. Three summarizers here report on it, and a second parser would be a
second thing to get wrong (ADR 0028 rule 3), so the line shapes live in one place:

  * `MLX admission priced the context rung ... model=<m>` — the server writes it at every load, so it is what
    attributes the lines after it to a model. Nothing else in the log names the model per request.
  * `peak memory size="<n> GiB"` — the runner logs it at each request's teardown, AFTER the sweep and cache clear
    (`x/mlxrunner/pipeline.go`), and resets MLX's peak at the start of every request. So it is that request's own
    peak, and it ends the record here.
  * `speculative decode stats iterations=... drafted=... accepted=... avg_draft=... max_draft=...` — one per
    completion that could draft. Absent means the request never drafted.
  * `tensors total: N, size: S, active: A` and the `tensor <name> <DTYPE> <size> pinned=<n> [dims]` lines before it —
    trace level only (OLLAMA_DEBUG=2). `size` sums the arrays the runner tracks; `active` is MLX's own figure, so
    `active - size` is memory MLX holds that no tracked array accounts for.
  * `prefix cache active_tokens: ..., paged_out: ..., trie: nodes=..., snapshots=...` — trace level only.

A record is one request. Fields that the log did not carry are None, never zero: a missing draft-stats line means
"did not draft", and reporting that as 0 tokens per round would be a measurement the log does not support.
"""
import re

ADMISSION = re.compile(r'msg="MLX admission priced the context rung" model=(\S+)')
PEAK = re.compile(r'msg="peak memory" size="([^"]+)"')
SPEC = re.compile(r'speculative decode stats" iterations=(\d+) drafted=(\d+) accepted=(\d+) .*?avg_draft=([\d.]+) max_draft=(\d+)')
TOTALS = re.compile(r'msg="tensors total: (\d+), size: ([^,]+), active: ([^"]+)"')
TENSOR = re.compile(r'msg="tensor (?P<name>.*?)\s+(?P<dtype>\S+)\s+(?P<size>[\d.]+ ?[KMGT]?i?B)\s+pinned=(?P<pinned>\d+) \[(?P<dims>[^\]]*)\]')
TRIE = re.compile(r'msg="prefix cache active_tokens: (\d+), active_size: ([^,]+), paged_out: ([^,]+), trie: nodes=(\d+), snapshots=(\d+)"')
COMPLETION = "path=/v1/completions"

_UNITS = {"B": 1.0, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40,
          "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def bytes_of(text):
    """"3.0 MiB" -> 3145728.0. None for anything this does not recognise, so a changed log format reads as missing
    rather than as zero."""
    m = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)\s*$", text or "")
    return float(m.group(1)) * _UNITS[m.group(2)] if m and m.group(2) in _UNITS else None


def gib(value):
    return None if value is None else value / (1 << 30)


class Request:
    """One request's lines. `index` counts requests per model, from 1."""

    def __init__(self, model, index):
        self.model = model
        self.index = index
        self.peak = None            # bytes, the request's own peak (MLX peak is reset per request)
        self.rounds = None          # decode rounds, from the draft-stats line
        self.drafted = None
        self.accepted = None
        self.avg_draft = None
        self.max_draft = None
        self.arrays = None          # live tracked arrays at teardown (trace)
        self.tracked = None         # bytes summed over those arrays (trace)
        self.active = None          # bytes MLX reports as active (trace)
        self.paged_out = None       # bytes the prefix-cache trie holds (trace)
        self.trie_nodes = None
        self.trie_snapshots = None
        self.active_tokens = None
        self.shapes = {}            # (dtype, dims) -> [count, bytes], live arrays at teardown (trace)

    @property
    def drafted_per_round(self):
        return None if not self.rounds else self.drafted / self.rounds

    @property
    def acceptance(self):
        return None if not self.drafted else self.accepted / self.drafted

    @property
    def tokens_per_round(self):
        """Tokens each target forward commits: the round's own token plus the drafts it kept."""
        return None if not self.rounds else (self.rounds + self.accepted) / self.rounds

    @property
    def untracked(self):
        """MLX's active memory minus the arrays the runner tracks, in bytes. Trace level only."""
        if self.active is None or self.tracked is None:
            return None
        return self.active - self.tracked


def iter_requests(path, model=None, shapes=False, named=True):
    """Yield a Request per `peak memory` line, in log order.

    `model` keeps only that model's requests. `shapes` collects the per-(dtype, dims) live-array groups, which costs
    a dict per request and is only useful on a trace-level log. `named=False` drops arrays whose name contains a dot
    — the fixed weights — leaving the caches and prefix-trie snapshots that grow.
    """
    current, counts, req = None, {}, None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ADMISSION.search(line)
            if m:
                # A load ends whatever was in flight: a request cannot span one, and keeping the pending record would
                # attribute the next model's first request to the previous model (measured: it silently dropped
                # qwen3.8's first request from a two-model log and shifted every pairing after it).
                current, req = m.group(1), None
                continue
            if current is None or (model is not None and current != model):
                continue
            if req is None:
                counts[current] = counts.get(current, 0) + 1
                req = Request(current, counts[current])
            m = SPEC.search(line)
            if m:
                req.rounds, req.drafted, req.accepted = int(m.group(1)), int(m.group(2)), int(m.group(3))
                req.avg_draft, req.max_draft = float(m.group(4)), int(m.group(5))
                continue
            if shapes:
                m = TENSOR.search(line)
                if m:
                    if named or "." not in m.group("name").strip():
                        g = req.shapes.setdefault((m.group("dtype"), m.group("dims")), [0, 0.0])
                        g[0] += 1
                        g[1] += bytes_of(m.group("size")) or 0.0
                    continue
            m = TOTALS.search(line)
            if m:
                req.arrays = int(m.group(1))
                req.tracked = bytes_of(m.group(2))
                req.active = bytes_of(m.group(3))
                continue
            m = TRIE.search(line)
            if m:
                req.active_tokens = int(m.group(1))
                req.paged_out = bytes_of(m.group(3))
                req.trie_nodes, req.trie_snapshots = int(m.group(4)), int(m.group(5))
                continue
            m = PEAK.search(line)
            if m:
                req.peak = bytes_of(m.group(1))
                yield req
                req = None


def completions(path):
    """Completions per model, whether or not they drafted. The drafting summarizer reports both, because
    "no draft-stats lines" only means something next to the number of completions that could have had them."""
    out, current = {}, None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ADMISSION.search(line)
            if m:
                current = m.group(1)
            elif current and COMPLETION in line:
                out[current] = out.get(current, 0) + 1
    return out
