# ADR 0046: a published result carries a profile of the machine it ran on

- **Status:** accepted 2026-10-04 by the maintainer; proposed 2026-10-01 in #428. Enforced, as far as
  anything enforces it, by [SPEC H26](../spec/vision-harness-reuse.md). The procedure that captures it
  is the `vision-campaign` skill.
- **Date:** 2026-10-01
- **Deciders:** MaxusAI fork maintainers

## Context

SPEC H11 stamps every score block with two things: the server that answered (`host`) and the build
it ran (`server_version`). Neither field says what the machine is. A run's machine has so far been
described in its document's prose, when at all. So results from the CUDA host, the gfx1151 APU, the
Metal hosts and now a cloud VM can be compared only by reading prose, and cannot be filtered.

The machine is rarely what its name suggests. The first cloud host was a GCP `a3-highgpu-1g` VM
(2026-09-30). Its H100 SXM5 supports PCIe Gen5 x16, but the port above it, as the VM sees it, tops
out at Gen4. So the link trains at Gen4 x16, half the host bandwidth, and neither the machine type
nor the GPU's name says so. Other machines hide other facts: an APU's GPU memory is whatever the
firmware carved out, plus GTT, and a Mac's GPU memory is its RAM.

Three constraints shape the record:

- **A cloud host is not a machine.** A flex-start VM is deleted when its run time ends. The next VM
  with the same machine type and label can land on different hardware, with a different image and
  driver. A record of "the H100 host" can go stale before anyone rereads it.
- **The client is often not the server.** The suite drives remote servers (SPEC H10). The machine
  that answered cannot be read from wherever `client.generate()` runs.
- **The fork is public** (AGENTS.md). A machine description attracts identifiers: hostnames, cloud
  project and instance names, serial numbers, GPU UUIDs and home paths.

## Decision

1. **`docs/maxusai/tools/host_profile.py` records the machine as `host-profile/1` JSON.** It covers:
   - the cloud machine: provider, zone, region, machine type, CPU platform;
   - the OS, CPU and memory;
   - each GPU and its PCIe link: the device's maximum, the upstream port's maximum, and the trained
     link;
   - the filesystems named with `--fs ROLE=PATH`;
   - the driver, toolkit and Ollama versions.
2. **Every field is atomic and optional.** Each key holds one fact of one type:
   - units go in the key (`memory_total_gib`, `power_limit_w`);
   - states are booleans;
   - a probe with nothing to report leaves `null`.
   
   So a Mac, an APU and an H100 share one schema and filter with `jq`.
3. **It never records an identifier.** That excludes a hostname, a user or account name, a cloud
   project, an instance name or id, a physical-host id, a serial number, a GPU UUID, a filesystem
   path and a device name. The host is named by its collab label, which exists to be written down;
   a filesystem is named by the role it plays.
4. **It is captured on the serving machine during the run, and committed beside the results it
   describes.** SPEC H26 sets the file names. A campaign bundle names its profile in
   `meta.host_profile`.
5. **A profile is evidence for one run, not a registry entry.** Every run captures its own. On
   2026-09-30, two captures 8.6 hours apart differed only in `collected_at`. A repeat costs a
   120-line file; a stale shared profile would cost a wrong comparison.
6. **The schema is versioned.** Adding a field keeps `host-profile/1`. Renaming, removing or
   re-typing a field makes it `host-profile/2`.

## Alternatives considered

- **Prose in the document's header, the status quo.** It cannot be filtered, it drifts from the
  machine, and it is where identifiers leak.
- **One profile per host, referenced by every run.** A label names a role, not hardware. The cloud
  case breaks this outright, and a driver update breaks it without anyone noticing.
- **A profile in every score block, written by `client.generate()` as H11's fields are.** Whenever
  the server is remote, the client runs on another machine, so it would describe the wrong one. It
  would also add 120 lines to every cell for facts that do not change during a run.

## Consequences

- A published run has one more file. The first two are the H100's: OCRBench (#430) and the vision
  campaign (#431).
- Two hosts compare field by field with `diff <(jq -S . A.json) <(jq -S . B.json)`. A set of hosts
  filters by fact, for example the ones whose GPU runs below Gen5:
  `jq -s 'map(select(any(.gpus[]; .pcie_link_gen < 5)) | .label)'`.
- Results from before 2026-09-30 have no profile. Their documents' headers stay their provenance, as
  H11 already says of cells with no `host`.
- Nothing checks yet that a published run carries a profile (SPEC H26's conformance row). What is
  checked is the tool itself. `test_host_profile.py`, run by `.github/workflows/host-profile.yaml`,
  asserts that no identifier is read or written, and the workflow profiles its own runner, a host it
  has never seen, where every probe with nothing to report must leave a field null rather than fail.
