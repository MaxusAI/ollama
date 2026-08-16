# bbox_contract fixture corpus

Input for `test_rescore.py`. Read this before citing anything here as a result.

## These are not measurements

Every fixture is **derived from ground truth by `generate.py`**, not captured
from a model. They reproduce response *shapes* — dialects, transpositions,
missing fields — so the scorer can be exercised deterministically and offline.
No number in this directory is evidence about how any model behaves.

The failure modes they imitate are real and were measured; the imitations are
not. Where a fixture reconstructs a documented behaviour, `generate.py` names
the model and date in a comment, and the measurement itself lives in
`docs/maxusai/vision-campaign-2026-08-16-seven-model.md`. Cite the campaign,
never the fixture.

## Why they exist

`9c4416e5` and `5081fcbb` changed `score_bbox_contract` and were claimed to be
behaviour-preserving: responses scored before the change score the same after
it, so the historical numbers in the campaign docs still mean what they say.

That claim was verified at the time against ~69 run outputs, and then became
uncheckable — `resp_*.json` is gitignored, so the corpus did not outlive the
machine that produced it. Nothing in the repo would have caught a regression.

This corpus is the durable stand-in. It cannot re-prove the original result
(different responses), but it holds the guarantee going forward, and the test
accepts a real corpus if the captured responses are ever recovered:

```bash
RESCORE_CORPUS=/path/to/raw/responses python3 ../../test_rescore.py
```

A directory of raw response text, one file per response, any extension.

## Layout

- `preexisting/` — top-level declaration, positional arrays, no anchor. The
  shapes that predate the change, so **old and new must score them identically**
  on all 16 baseline fields. This is the guarantee.
- `new_features/` — per-object declarations, anchors, named coordinates. The
  pre-change scorer predates the syntax and scores these differently *by
  design*, so they are pinned as goldens rather than compared.

`test_rescore.py` also asserts the corpus stays adversarial: it must contain a
followed contract and a broken one, an unparseable response, a case exercising
`implied_scale`, and every `bbox_type` including none. A corpus of clean
responses would pass while proving very little.

## Regenerating

```bash
python3 generate.py
```

Deterministic — same ground truth in, same bytes out. Regenerate only when
adding a case. Changing an existing fixture changes what the guarantee covers,
so treat it as a deliberate edit, not a refresh.

## What `named_coords.txt` covers

A top-level declaration with `x1`/`y1`/`x2`/`y2` on each object — the schema the
prompt asks for. This fixture originally pinned a defect: the boxes were read
correctly but the declaration never validated (`hits_declared` 0 against
`hits_bestfit` 6), because `read_decl` looks for the coordinate keys on the dict
carrying the declaration, and for a top-level declaration that is the root while
the keys are on the objects.

Fixed by inferring the order from the boxes when every boxed object uses named
keys. The fixture now asserts the working behaviour, and a sibling case pins the
mixed response — named keys on one object, a positional array on another — where
there is no single order to infer and none is assumed.

Nothing in `preexisting/` moved: those responses use positional arrays, so the
inference never fires for them. That is the guarantee this corpus exists to hold,
and it was confirmed by running the rescore comparison across the fix.
