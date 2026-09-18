# #17459 / #17475 reproduction harness

Produced the results in [`../rocm-gate-issues-result.md`](../rocm-gate-issues-result.md).
Committed so the reproduction can be repeated or disputed, not because it is a permanent
part of the suite — it is scratch tooling, not the vision harness.

| file | what it does |
|---|---|
| `gatelib.py` | scratch-container plumbing: start/stop `ollama-gate` on **:11435**, capture the server log, record which models production held at the start of each run |
| `gen_forms.py` | the two synthetic claim forms. Everything on them is invented — #17475 is a PII leak and its reporter used real records. Asserts at generation time that no 6-character window of the donor's marker appears anywhere else on either form |
| `repro_17459.py` | the reporter's `/api/chat` request verbatim, cold and warm, `think` both ways |
| `repro_17475.py` | protocols V1/V2/V3, with donor aborts and the multi-model noise loop |
| `t17459.py`, `t17475.py`, `t_dio.py` | generate the three tables in the write-up from the saved results (ADR 0012 rule 8 — tables are generated, never transcribed) |

Two variables are host-specific and deliberately not committed:

```bash
GATE_STORE=/path/to/model/store GATE_GPU_GROUPS=<kfd-gid>,<dri-gid> python3 repro_17459.py <label> <image> prod
```

**Never point this at production.** It binds production's model store read-write so the
tests exercise the same blobs, and every container therefore runs with `OLLAMA_NOPRUNE=1` —
a server that cannot parse a newer manifest would otherwise delete blobs production needs.
The scratch container is removed as soon as a run finishes.
