# ADR 0038: a measured model is identified by its manifest digest, not its tag

- **Status:** proposed 2026-09-19. Awaiting Glenn. Enforces SPEC `vision-harness-reuse` H17,
  which the mlx-metal campaign wrote from the other direction on the same day.
- **Date:** 2026-09-19
- **Deciders:** MaxusAI fork maintainers

## Context

`gemma4:31b-nvfp4` in this store and `gemma4:31b-nvfp4` in the library are not the same
model. Measured 2026-09-18 by diffing the manifests: **194 of 1248 layers differ, every
one of them on the vision path**, each about 3.5× larger upstream — `nvfp4` (4 bits plus
scales) widened to `bf16`. The library re-published the tag with an unquantised vision
tower. Our copy was pulled 2026-08-17; the re-publish is after that date.

The same move hit `gemma4:26b-nvfp4` (190 of 1072 layers) and `gemma4:12b-nvfp4` (its six
vision and audio projections), and `qwen3.5:0.8b-mlx` was re-published with its **tensors
renamed** (473 of 481 layers). 39 of the store's 54 tags are byte-identical to the
registry; 10 are local builds the registry never served.

Three properties make this invisible:

- the **config blob digest is unchanged** in every case, so `ollama show` reports the
  same parameters, template and licence for both artifacts;
- the pull is idempotent by tag, not by content, so `ollama pull` on a host that already
  has the old copy *replaces* it without a word;
- nothing in the store records where a blob came from or when, beyond file mtimes.

The consequence is not hypothetical. Every gemma4 vision result in this repo — the #312
encoder investigation, `mlxrunner/testdata`'s vision goldens, the fine-text tiers, the
OCRBench ladder's first row — was measured against the **quantised** tower. A host that
pulls that tag today gets the bf16 one and will not reproduce any of them, and the old
artifact **cannot be fetched back**: the registry no longer serves it under any name.

This also decides what a cross-host comparison means. The Metal host's OCRBench number
and the CUDA host's are comparable only if both ran the same tower precision, and neither
number carried anything that could answer that.

## Decision

**A record that names a model names its manifest digest.** The tag is how a human finds
the model; the digest is what the number describes.

1. Any document, issue comment or score file that reports a measurement states the
   manifest digest of each model measured. `vision-suite/store_audit.py --digests` prints
   them.
2. Before trusting a comparison between two hosts, run `store_audit.py` on both. A tag
   present on both hosts is not evidence they hold the same weights.
3. Benchmark and campaign pulls go to a **separate store** (a directory mounted at
   `/root/.ollama`), never the production store, so a pull cannot overwrite an artifact
   an existing measurement was taken against.
4. Do not `ollama pull` a tag this repo has measured into the production store. If a
   newer artifact is wanted, pull it beside the old one and record both digests.
5. The store is the archive. A measured artifact that is deleted locally is gone, because
   the registry serves only the current content of a tag.

## Options considered

- **Cite tags, as before.** Cheapest, and what every record did until now. Rejected: the
  failure is silent, the artifact is unrecoverable, and the first symptom is an
  irreproducible result on someone else's host.
- **Pin by the model's config digest.** Rejected on the evidence: the config digest was
  identical across the re-publish in all four cases, so it identifies the model's
  definition, not its weights.
- **Snapshot every measured model to the array.** Complete, and about 19 GB per gemma4
  nvfp4 tag. Worth doing for artifacts a published result depends on; not required by
  this ADR, which only makes the identity recordable.
- **Ask upstream to publish immutable tags.** Out of our hands, and nothing goes to
  `ollama/ollama` by standing instruction.

## Consequences

- A number without a digest is now an incomplete record. Existing documents are not
  retro-fitted wholesale; the digest is added when a document is next touched, and
  `docs/maxusai/ocrbench-quantisation-ladder.md` carries them from the start.
- The vision suite and preflight do not yet stamp digests into their score files. They
  should, and that is the natural follow-up — the harness already has the store path.
- Re-pulling to "get the latest" becomes a deliberate act with a recorded before and
  after, not routine hygiene.
- This says nothing about which artifact is *better*. The bf16 tower may well be the
  better model; see the OCRBench ladder, where it is 39 % faster per image on CUDA and
  scores within noise.
