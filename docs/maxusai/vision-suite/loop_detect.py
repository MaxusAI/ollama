"""Loop detection for streamed model output: a repeating thinking trace, a repeating answer, or a run of one token.

Stdlib only, so the same file serves this vision suite, ai-invoice-to-xero's ollama_toolkit and the structured-docs
backend (one implementation, three copies kept identical; see ADR 0049).

A greedy (temperature 0) decode that starts repeating itself verbatim does not recover: the same context produces the
same continuation. Waiting for num_predict, or climbing a context ladder after the length stop, spends the rest of the
budget proving it. LoopDetector watches a stream as it arrives and says so the moment the loop is unambiguous, as one
of three failure modes:

  loop detected:thinking    the thinking trace's last WINDOW chars occur MIN_REPEATS times in it (period = distance)
  loop detected:content     the same, in the answer -- the mode ai-invoice-to-xero's detector could not see
                            ("Invoice own own own...", extraction-failure-modes.md)
  loop detected:token-run   one stream emitted the same chunk (whitespace-stripped) TOKEN_RUN times in a row -- the
                            whitespace runs a JSON grammar allows, which ollama's server aborts at 100 with an HTTP 500
                            and no output (llm/llama_server.go)

Why three copies, not two: a single earlier occurrence of 400 characters is how a model restates a table or quotes a
passage. ai-invoice-to-xero's thinking_is_looping fires on it; this detector does not. Its two functions are kept
below, unchanged in meaning, for callers that use them.

Cost: the block check runs every CHECK_EVERY new characters and searches only the last LOOKBACK characters, so a stream
of n characters costs O(n / CHECK_EVERY * LOOKBACK) in C-level string search -- linear in n, unlike re-scanning the
whole buffer on every chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

THINKING = "thinking"
CONTENT = "content"
TOKEN_RUN = "token-run"

LOOP_CHECK_WINDOW = 400   # chars of tail compared against the earlier stream (ai-invoice-to-xero's value)
LOOP_MIN_LENGTH = 800     # a stream shorter than this is never judged (ai-invoice-to-xero's value)
MIN_REPEATS = 3           # the tail window must occur this many times in all: itself plus two earlier copies
TOKEN_RUN_LEN = 64        # identical chunks in a row; below the server's 100, so the client keeps the partial output
CHECK_EVERY = 64          # chars between block checks
LOOKBACK = 16384          # chars searched for earlier copies; a period longer than LOOKBACK / MIN_REPEATS is not seen


@dataclass
class Loop:
    kind: str           # THINKING, CONTENT or TOKEN_RUN
    stream: str         # THINKING or CONTENT: which stream looped (a token-run names its stream here)
    at_chars: int       # length of that stream when the loop was called
    period: int         # chars per repetition (block loops) or 0 (token-run)
    repeats: int        # copies seen (block) or run length (token-run)
    snippet: str        # one period of the loop (or the repeated chunk), truncated for logs
    params: dict = field(default_factory=dict)

    def failure_mode(self) -> str:
        return f"loop detected:{self.kind}"

    def record(self) -> dict:
        """Flat fields for a result record: everything a reader needs to see the loop without the stream."""
        return {"loop": self.failure_mode(), "loop_kind": self.kind, "loop_stream": self.stream,
                "loop_at_chars": self.at_chars, "loop_period": self.period, "loop_repeats": self.repeats,
                "loop_snippet": self.snippet[:160], "loop_params": dict(self.params)}

    def __str__(self) -> str:
        what = (f"period {self.period} chars x{self.repeats}" if self.kind != TOKEN_RUN
                else f"{self.repeats} identical chunks")
        return f"{self.failure_mode()} ({what}, at {self.at_chars} chars of {self.stream}): {self.snippet[:80]!r}"


class _Stream:
    # buf holds only the last `keep` chars (the search region); total counts every char seen. Deltas wait in
    # `pending` and are joined at each check: appending to a str attribute copies the whole buffer per chunk.
    __slots__ = ("name", "buf", "pending", "total", "checked", "last_tok", "run")

    def __init__(self, name):
        self.name, self.buf, self.pending, self.total = name, "", [], 0
        self.checked, self.last_tok, self.run = 0, None, 0

    def text(self, keep):
        if self.pending:
            self.buf = (self.buf + "".join(self.pending))[-keep:]
            self.pending = []
        return self.buf


class LoopDetector:
    """Feed streamed deltas; feed() returns the first Loop found, then the same Loop on every later call."""

    def __init__(self, window=LOOP_CHECK_WINDOW, min_length=LOOP_MIN_LENGTH, min_repeats=MIN_REPEATS,
                 token_run=TOKEN_RUN_LEN, check_every=CHECK_EVERY, lookback=LOOKBACK):
        if min_length < window or min_repeats < 2 or token_run < 2 or lookback < window * min_repeats:
            raise ValueError("loop_detect: need min_length >= window, min_repeats >= 2, token_run >= 2 and "
                             "lookback >= window * min_repeats")
        self.window, self.min_length, self.min_repeats = window, min_length, min_repeats
        self.token_run, self.check_every, self.lookback = token_run, check_every, lookback
        self.params = {"window": window, "min_length": min_length, "min_repeats": min_repeats,
                       "token_run": token_run, "lookback": lookback}
        self._streams = {THINKING: _Stream(THINKING), CONTENT: _Stream(CONTENT)}
        self.loop = None

    def feed(self, thinking: str = "", content: str = ""):
        if self.loop is None and thinking:
            self.loop = self._add(self._streams[THINKING], thinking)
        if self.loop is None and content:
            self.loop = self._add(self._streams[CONTENT], content)
        return self.loop

    def _add(self, s: _Stream, delta: str):
        tok = delta.strip()
        if tok == s.last_tok:
            s.run += 1
        else:
            s.last_tok, s.run = tok, 1
        s.pending.append(delta)
        s.total += len(delta)
        if s.run >= self.token_run:
            return Loop(TOKEN_RUN, s.name, s.total, 0, s.run, delta, self.params)
        if s.total < self.min_length or s.total - s.checked < self.check_every:
            return None
        s.checked = s.total
        return self._block(s)

    def _block(self, s: _Stream):
        buf = s.text(self.lookback)
        n, w = len(buf), self.window
        tail = buf[n - w:]
        lo = 0
        # Earlier copies, nearest first. rfind's end bound keeps each copy strictly before the previous one's end,
        # so overlapping copies of a short period are all counted.
        end, copies, nearest = n - 1, 1, None
        while copies < self.min_repeats:
            pos = buf.rfind(tail, lo, end)
            if pos < 0:
                return None
            if nearest is None:
                nearest = pos
            copies += 1
            end = pos + w - 1
        period = (n - w) - nearest
        return Loop(s.name, s.name, s.total, period, copies, buf[n - period:] if period <= w else buf[n - w:],
                    self.params)


# ---- ai-invoice-to-xero's ollama_toolkit API, unchanged in meaning --------------------------------------------------

def thinking_is_looping(thinking: str) -> bool:
    """True if the last LOOP_CHECK_WINDOW chars of the trace occur verbatim earlier in it (once is enough).

    ai-invoice-to-xero's original predicate, kept for its callers. A one-copy repeat is also how a model restates a
    table, so new code should use LoopDetector, which needs MIN_REPEATS copies and also watches content and token runs.
    """
    if len(thinking) < LOOP_MIN_LENGTH:
        return False
    tail = thinking[-LOOP_CHECK_WINDOW:]
    return tail in thinking[:-LOOP_CHECK_WINDOW]


def dedup_thinking(thinking: str) -> str:
    """Trim a looping trace to its first unique pass: cut where a LOOP_CHECK_WINDOW-wide segment first recurs.

    Same result as ai-invoice-to-xero's original (which sliced the prefix for every index, O(n^2)); this keeps every
    complete earlier window in a hash table, so the scan is linear.
    """
    w = LOOP_CHECK_WINDOW
    seen = {}
    for i in range(w, len(thinking) - w + 1):
        j = i - w
        seen.setdefault(thinking[j:j + w], j)       # windows wholly inside thinking[:i]
        if thinking[i:i + w] in seen:
            return thinking[:i].rstrip()
    return thinking.rstrip()
