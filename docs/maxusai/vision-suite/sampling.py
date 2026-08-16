#!/usr/bin/env python3
"""Per-model sampling parameters, so think-on cells are not measured off-policy.

WHY THIS EXISTS. The suite used to generate at `temperature: 0` everywhere. For
think-off extraction that is the right call — it is the closest thing to a
repeatable score. For think-ON it manufactures a failure: reasoning never
terminates, every token lands in `thinking`, `eval_count` hits `num_predict`,
and `response` comes back empty. Measured on b10353, `multi_3img`,
`num_predict=24000` (raw_sampling_test.json, raw_qwen_q8_sampling.json,
raw_gemma_owncard.json):

    qwen3.6:35b-a3b-q4_K_M   temp 0 -> 24000 tok, EMPTY
                             card   -> 10810 / 8023 tok, valid JSON
    gemma4:12b-nvfp4         temp 0 -> 24000 tok, EMPTY
                             card   ->  4316 / 5014 tok, valid JSON

The lever is leaving greedy decoding, NOT `presence_penalty` specifically:
gemma4 converges under its own card values, which carry no penalty at all.
Full analysis in ../runaway-reasoning-under-think.md.

THINK-OFF IS DELIBERATELY UNCHANGED. Every published think-off number, every
preflight expectation and every release record was measured at `temperature: 0`.
Changing that would invalidate all of them for no benefit — the failure this
module addresses only occurs with thinking on.

VALUES ARE CARD-SOURCED OR ABSENT. A model whose card we cannot read gets NO
sampling overrides at all — its packaged parameters apply — plus a warning.
Never an invented configuration, and never greedy, which is the very thing that
breaks think-on. Adding a row here means reading that model's card and citing it.

The table below is corroborated by what the models themselves ship: `ollama show`
reports gemma4 declaring temperature 1 / top_k 64 / top_p 0.95 and qwen3.6
declaring temperature 1 / top_k 20 / top_p 0.95 / min_p 0 / presence_penalty 1.5
— i.e. exactly these values. nemotron3 declares none, which is why it has no row.
Pinning temperature 0 was therefore overriding each model's own packaged default.
"""
import os
import sys

# Think-off: unchanged from the original hardcoded value, on purpose (see above).
GREEDY = {"temperature": 0}

# Think-on, keyed by the model-name prefix before the first ':'.  Each entry
# must cite the card it came from — these are not tuned values.
CARD_THINKING = {
    # https://huggingface.co/Qwen/Qwen3.6-35B-A3B — thinking mode, general tasks.
    # The card names presence_penalty (0-2) as the anti-repetition lever.
    "qwen3.6": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5},
    "qwen3.5": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5},
    # https://huggingface.co/google/gemma-4-12B-it — "Use the following
    # standardized sampling configuration across all use cases". The card draws
    # no distinction between thinking and non-thinking, and specifies no
    # penalty; measured convergence needs none.
    "gemma4": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    # nemotron3: the NVIDIA card is gated (HTTP 401), so no values are recorded
    # here. It is the one family that converges under greedy decoding anyway, so
    # the fallback below is not currently costing us a measurement.
}


def family(model):
    """'qwen3.6:35b-a3b-q4_K_M' -> 'qwen3.6'."""
    return (model or "").split(":", 1)[0].strip().lower()


# The one-off probe overrides. Kept as a module constant rather than inline in
# sampling_for, because provenance() has to report exactly the set that
# sampling_for applies — two hand-maintained copies would drift, and a drifted
# copy is worse than no label at all: it would assert an override that was
# never applied, or hide one that was.
OVERRIDE_ENV = (("TEMPERATURE", "temperature", float),
                ("TOP_P", "top_p", float),
                ("TOP_K", "top_k", int),
                ("MIN_P", "min_p", float),
                ("PRESENCE_PENALTY", "presence_penalty", float))


def active_overrides():
    """The overrides currently set in the environment, {key: raw string}.

    Presence here does NOT mean they were applied — sampling_for only reaches
    the override loop on the think-on, non-legacy, has-a-card path. Callers
    must gate on that themselves; provenance() does.
    """
    return {key: os.environ[env] for env, key, _ in OVERRIDE_ENV
            if os.environ.get(env)}


def sampling_for(model, think, warn=True):
    """Options dict for this model in this think mode.

    `SAMPLING=legacy` forces greedy everywhere, which is how you reproduce a
    pre-2026-08-13 run. Individual parameters can be overridden from the
    environment (TEMPERATURE, TOP_P, TOP_K, MIN_P, PRESENCE_PENALTY) for
    one-off probes; anything set that way is recorded in the scores alongside
    the resolved values, so an off-card run stays identifiable afterwards.
    """
    if os.environ.get("SAMPLING", "").lower() == "legacy":
        return dict(GREEDY)
    if not think:
        return dict(GREEDY)

    fam = family(model)
    opts = CARD_THINKING.get(fam)
    if opts is None:
        # No card entry: send NO sampling keys, so the model's own packaged
        # parameters apply — or, if it declares none, the server defaults
        # (temperature 0.8 / top_k 40 / top_p 0.9, api/types.go DefaultOptions).
        # Falling back to greedy here would be actively wrong twice over: it
        # overrides whatever the packager shipped, and temperature 0 is the
        # configuration that causes non-termination in the first place.
        if warn:
            print(f"[sampling] no card-sourced thinking parameters for '{fam}'; "
                  f"sending no sampling overrides, so the model's packaged "
                  f"defaults apply. Check `ollama show` to see what those are.",
                  file=sys.stderr)
        return {}
    opts = dict(opts)

    for env, key, cast in OVERRIDE_ENV:
        if os.environ.get(env):
            opts[key] = cast(os.environ[env])
    return opts


def provenance(model, think):
    """What to record in the scores so a run can be attributed after the fact.

    ADR 0005 asks benchmark runs to record their runtime configuration; the
    suite did not, which is why the b10353 caps could not be attributed to a
    sampling mode without re-running them.

    An env-overridden run is NOT card-sourced and must not claim to be. Until
    2026-08-16 this returned a bare `card:<fam>` whichever values were actually
    sent, so a `TEMPERATURE=0.01` probe was archived as though it carried the
    card's temperature 1.0 — the resolved `sampling` dict told the truth, but
    the field everyone filters on did not. The suffix below closes that.

    The gate mirrors sampling_for's control flow exactly. Overrides reach the
    resolved options ONLY on the think-on, non-legacy, has-a-card path: legacy
    and think-off return GREEDY before the override loop, and a family with no
    card entry returns {} before it. Labelling any of those as overridden would
    be the same class of lie in the opposite direction.
    """
    fam = family(model)
    legacy = os.environ.get("SAMPLING", "").lower() == "legacy"
    if legacy:
        source = "legacy-greedy"
    elif not think:
        source = "greedy-think-off"
    elif fam in CARD_THINKING:
        source = f"card:{fam}"
    else:
        source = "packaged-defaults-no-card"

    if think and not legacy and fam in CARD_THINKING:
        ov = active_overrides()
        if ov:
            source += "+override(" + ",".join(
                f"{k}={v}" for k, v in sorted(ov.items())) + ")"
    return {"sampling_source": source, "sampling": sampling_for(model, think, warn=False)}
