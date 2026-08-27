# Benchmark binary archive

Every measurement in `docs/maxusai/` is attributable to one server binary, and
[ADR 0011](../adr/0011-preflight-expectations-are-versioned-code.md) keys
preflight expectations on the version string that binary reports. When the
payload moves, the old numbers do not become wrong — they become numbers about a
different build, and re-checking them needs the build back.

Binaries live on the benchmark host at `~/.ollama/binaries/`, **not in git** —
they are ~48 MB each. This file is the durable part: the identity and the recipe.
A host can be rebuilt from it; a binary sitting only in `/tmp` cannot.

## Archived

| version | git sha | payload | MLX | sha256 | notes |
|---|---|---|---|---|---|
| `0.32.5-maxusai-a5d65906` | [`a5d65906`](https://github.com/MaxusAI/ollama/commit/a5d65906) | llama.cpp **b10353** | pre-v0.32.14 pin | `d807360e94e0e17a…` | provenance for the 2026-08-16/17 vision work |
| `0.32.14-maxusai-9594f81e` | [`9594f81e`](https://github.com/MaxusAI/ollama/commit/9594f81e) | llama.cpp **b10434** | v0.32.14 pin | `711d4ad126773ddf…` | the `mlx-metal-0-32-14` preflight baseline |
| `0.32.14-maxusai-c82b0464` | [`c82b0464`](https://github.com/MaxusAI/ollama/commit/c82b0464) | llama.cpp **b10434** | v0.32.14 pin | `03f9f9289dbaba1b…` | last pre-0.33.0 deploy on :11435 (2026-08-22 → 27); preflight PASS 2026-08-22; rollback target for the 0.33.0 swap |
| `0.33.0-maxusai-21cfe88e` | [`21cfe88e`](https://github.com/MaxusAI/ollama/commit/21cfe88e) | llama.cpp **b10488** | 27fec909 pin | `6ab35025981be587…` | current; provenance for the `mlx-metal-0-33-0` profile measurement and the 27fec909 golden recalibration |

Full checksums:

```
d807360e94e0e17ac346df9bef198b6a182ef2f47bff78a0e772f6d1d67bad72  ~/.ollama/binaries/ollama-0.32.5-maxusai-a5d65906
711d4ad126773ddfadeac01c7fea1dc924c60bcfdaf071f9212909aa24ee7a61  ~/.ollama/binaries/ollama-0.32.14-maxusai-9594f81e
03f9f9289dbaba1bba2a5826a22e1aa85525e96c7e78942bf41828ff90908a71  ~/.ollama/binaries/ollama-0.32.14-maxusai-c82b0464
6ab35025981be587ff0a73f0b7ae007b608300defb46f176065c8f3f52d78139  ~/.ollama/binaries/ollama-0.33.0-maxusai-21cfe88e
```

## What b10353 is the provenance for

Everything measured 2026-08-16/17 and cited in these documents:

- [18-model campaign](../vision-campaign-2026-08-17-eighteen-model.md) — 36
  model-modes, both think modes
- [SPEC C1–C12](../spec/vision-bbox-response-contract.md) — every rate behind the
  bbox contract, including C2's 11-of-72 vs 0-of-36 and C7's 107-cell separation
- [low-temperature negative result](../vision-lowtemp-thinkon-negative-result.md)
- [seven-model campaign](../vision-campaign-2026-08-16-seven-model.md)

**None of those may be cited against b10434 without re-measurement.** The
preflight ladders reproduce exactly across the bump
([PR #166](https://github.com/MaxusAI/ollama/pull/166)), which is evidence the
payload move is inert for *token accounting* — it says nothing about generative
quality, which is what those documents measure.

## Rebuilding one

`build-macos.sh` derives the stamp, so a rebuild is a checkout away:

```sh
git checkout a5d65906
CLEAN_DEPS=1 OUT=/tmp/ollama-rebuild sh docs/maxusai/vision-suite/build-macos.sh
```

`CLEAN_DEPS=1` matters when moving between payload pins: the compat patches are
applied to the vendored llama.cpp checkout as working-tree edits, and CMake's
stash/fetch/unstash cycle fails across a version change.

**Do not expect a byte-identical binary.** `-trimpath` removes absolute paths,
but Go and the native payload are not reproducible builds here. The checksums
above identify the archived artefacts; they are not a target a rebuild must hit.
The version string is the identity that matters, because it is what preflight
gates on.

## Adding one

Archive the binary whenever a payload pin moves, before the new build replaces
it on the host:

```sh
cp -p /tmp/ollama-vs ~/.ollama/binaries/ollama-$(OLLAMA_HOST=127.0.0.1:1 /tmp/ollama-vs --version 2>&1 \
  | sed -n 's/^Warning: client version is //p')
```

Then add a row here with its sha256 and what it is the provenance for.
