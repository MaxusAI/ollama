#!/usr/bin/env python3
"""Split eval_count into thinking / answer / control tokens, exactly.

    python3 token_split.py --hf <repo-or-path> <tag> [<tag> ...]
    python3 token_split.py --server <host> <model> <tag> [<tag> ...]
    python3 token_split.py --gguf <path-to.gguf> <tag> [<tag> ...]
    python3 token_split.py --check <tag> ...          # chars only, no tokenizer

`eval_count` is the TOTAL generated tokens; it conflates reasoning with answer.
For req/hour planning the reasoning half is the number that matters — "how many
tokens does this model need to finish thinking" — and nothing recorded it.

Three buckets, summing to eval_count exactly:

    thinking_tokens  tokenize(think_<tag>_<probe>.txt)
    answer_tokens    tokenize(resp_<tag>_<probe>.json)
    control_tokens   eval_count - the other two

control_tokens is real generated output that carries no visible text: the think
delimiters, EOS, and the `format:json` grammar tokens. It is REPORTED, never
distributed into the other two. Measured on this host, it is not a rounding
error — a 62-token qwen3.8 think-on response emitted only 49 chunks of visible
text, so 21% of the tokens were control. A character-proportional estimate
would have silently pushed that 21% into the visible buckets and misstated both.

THE ACCEPTANCE GATE. A wrong vocab is not a small error, it is a confidently
wrong number. But it is self-detecting: if the tokenizer disagrees with the one
the server used, `control_tokens` goes negative or implausibly large. So the
tokenizer proves itself against eval_count and needs no separate trust argument.
This refuses to emit a split unless the residue is non-negative and small across
the sample. Failing that, it reports characters only -- option 3 in the plan --
rather than publishing something that looks exact and is not.
"""
import glob
import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
GATE_MAX = 64           # per-cell control tokens above this is implausible
GATE_MIN_SAMPLE = 20    # below this the gate cannot conclude, so it says so


def load_tokenizer(kind, ref):
    """Return encode(str)->int, or exit with why it could not."""
    if kind == "hf":
        try:
            from tokenizers import Tokenizer
            tok = (Tokenizer.from_file(ref) if os.path.exists(ref)
                   else Tokenizer.from_pretrained(ref))
            return lambda s: len(tok.encode(s, add_special_tokens=False).ids)
        except ImportError:
            sys.exit("--hf needs the `tokenizers` package (pip install tokenizers)")
        except Exception as e:
            sys.exit(f"could not load tokenizer {ref!r}: {type(e).__name__}: {e}")
    if kind == "gguf":
        # The vocab the server actually uses. /api/show names the tokenizer but
        # nulls tokenizer.ggml.tokens and .merges, so the arrays have to come
        # from the blob itself.
        try:
            from gguf import GGUFReader
        except ImportError:
            sys.exit("--gguf needs the `gguf` package (pip install gguf)")
        try:
            r = GGUFReader(ref)
            fields = {f.name: f for f in r.fields.values()}
            if "tokenizer.ggml.tokens" not in fields:
                sys.exit(f"{ref} carries no tokenizer.ggml.tokens")
        except Exception as e:
            sys.exit(f"could not read {ref!r}: {type(e).__name__}: {e}")
        sys.exit("gguf vocab loading is not implemented; use --hf with the "
                 "matching repo, and rely on the acceptance gate to catch a "
                 "mismatch. (Reading tokens+merges out of the blob and rebuilding "
                 "the BPE merge table is the exact path, but a half-correct "
                 "reimplementation would defeat the gate's purpose. Tried "
                 "2026-08-20 for nemotron3 via the gguf package + tokenizers "
                 "BPE: refused at 54/54 cells — use --server instead, which "
                 "cannot have a vocab mismatch.)")
    if kind == "server":
        # THE SERVER IS THE REFERENCE TOKENIZER. prompt_eval_count of a
        # raw-mode /api/generate request IS the server counting the text with
        # the exact vocab that produced eval_count — no reconstruction, no
        # repo-matching, nothing for the gate to catch except this function's
        # own overhead handling. raw:true skips the chat template (a wrapper
        # would add tokens that belong to the template, not the text), and
        # the per-request constant the runner still prepends (BOS, when the
        # model declares one) is MEASURED, not assumed: "\n" is a single base
        # token in every BPE and SPM vocab this suite has met, so
        # count("\n") - 1 is the overhead. If that ever goes wrong the error
        # is constant per request, shifts every control residue equally, and
        # the acceptance gate refuses — the same self-detection argument as
        # --hf. Cost: one prefill-only request per text on a live host.
        host, model = ref
        import client

        def count(text):
            r = client.generate(host, model, text, [], num_predict=1,
                                num_ctx=32768, fmt=None,
                                endpoint_override="generate", think=False,
                                apply_sampling=False, send_think=False,
                                use_env_opts=False, raw=True)
            return r["prompt_eval_count"]

        overhead = count("\n") - 1

        def enc(s):
            # An empty text is zero tokens by definition; an empty raw prompt
            # would instead be a bare load request (done_reason "load").
            return count(s) - overhead if s else 0
        return enc
    return None


def cells(tag):
    """(probe, eval_count, thinking_text, answer_text) per scored probe."""
    path = os.path.join(DIR, f"scores_{tag}.json")
    if not os.path.exists(path):
        return
    for probe, sc in (json.load(open(path)) or {}).items():
        if not isinstance(sc, dict) or sc.get("eval_count") is None:
            continue
        tp = os.path.join(DIR, f"think_{tag}_{probe}.txt")
        rp = os.path.join(DIR, f"resp_{tag}_{probe}.json")
        think = open(tp).read() if os.path.exists(tp) else ""
        ans = open(rp).read() if os.path.exists(rp) else ""
        yield probe, sc, think, ans


def main():
    args = sys.argv[1:]
    kind = ref = None
    if args and args[0] in ("--hf", "--gguf"):
        kind, ref, args = args[0][2:], args[1], args[2:]
    elif args and args[0] == "--server":
        kind, ref, args = "server", (args[1], args[2]), args[3:]
    elif args and args[0] == "--check":
        args = args[1:]
    write = "--write" in args
    tags = [a for a in args if a != "--write"]
    if not tags:
        sys.exit(__doc__)

    enc = load_tokenizer(kind, ref) if kind else None
    rows, residues = [], []
    for tag in tags:
        for probe, sc, think, ans in cells(tag):
            ev = sc["eval_count"]
            row = {"tag": tag, "probe": probe, "eval_count": ev,
                   "thinking_chars": len(think), "answer_chars": len(ans)}
            if enc:
                tt, at = enc(think), enc(ans)
                row.update(thinking_tokens=tt, answer_tokens=at,
                           control_tokens=ev - tt - at)
                residues.append(ev - tt - at)
            rows.append(row)

    if not rows:
        sys.exit("no scored cells found for those tags")

    ok = True
    if enc:
        bad = [r for r in residues if r < 0 or r > GATE_MAX]
        # A FEW outliers in a large clean sample are not a vocab mismatch — a
        # wrong tokenizer skews EVERY cell, so ≥90% of cells landing in
        # [0, GATE_MAX] proves the vocab and indicts the outliers' DATA
        # instead: text on disk that is not the generation eval_count counted.
        # Known cause, measured 2026-08-20 on cudafull1: the finetext
        # double-persist (fixed in finetext_probe.py the same day) overwrote
        # the suite's text with the probe's own run, and think-on sampling is
        # non-greedy, so the generations differ — control -114 (gemma4
        # 26b-a4b) and +444 (qwen3.8) on exactly those cells while 53/54 and
        # 35/36 sat in [0, 6]. Such cells get NO split — named, excluded from
        # the table and from --write — because a split of someone else's text
        # would be exact-looking and wrong.
        if bad and len(residues) - len(bad) >= GATE_MIN_SAMPLE \
                and len(bad) * 10 <= len(residues):
            bad_rows = [r for r in rows
                        if not 0 <= r["control_tokens"] <= GATE_MAX]
            print(f"⚠ SKIPPED {len(bad_rows)} cell(s) whose persisted text "
                  f"does not reconcile with eval_count:")
            for r in bad_rows:
                print(f"    {r['tag']} {r['probe']} "
                      f"(control {r['control_tokens']})")
            print()
            rows = [r for r in rows if 0 <= r["control_tokens"] <= GATE_MAX]
            residues = [r for r in residues if 0 <= r <= GATE_MAX]
            bad = []
        if bad and len(residues) < GATE_MIN_SAMPLE:
            # A NEGATIVE or absurd residue is disqualifying at ANY sample size:
            # it means the tokenizer counted more tokens than the server
            # generated, which no amount of extra sampling makes acceptable.
            # Previously `bad` was computed and then ignored on this branch, so a
            # mismatched vocab printed a split and --write would have persisted it.
            ok = False
            print(f"✗ GATE FAILED on {len(bad)}/{len(residues)} cells "
                  f"(min {min(residues)}, max {max(residues)}) — the tokenizer "
                  f"does not match the server's. Reporting characters only.\n")
        elif len(residues) < GATE_MIN_SAMPLE:
            print(f"⚠ GATE INCONCLUSIVE: {len(residues)} cells, need "
                  f"{GATE_MIN_SAMPLE}. Numbers shown, do not publish yet.\n")
        elif bad:
            ok = False
            print(f"✗ GATE FAILED: {len(bad)}/{len(residues)} cells have "
                  f"control_tokens outside [0, {GATE_MAX}] "
                  f"(min {min(residues)}, max {max(residues)}).\n"
                  f"  The tokenizer does not match the one the server used. "
                  f"Reporting characters only.\n")
        else:
            print(f"✓ gate passed: control_tokens in "
                  f"[{min(residues)}, {max(residues)}] over {len(residues)} cells\n")

    show = enc and ok
    hdr = "| tag | probe | eval | think tok | answer tok | control |" if show \
        else "| tag | probe | eval | think chars | answer chars |"
    print(hdr)
    print("|---|---|---|---|---|---|" if show else "|---|---|---|---|---|")
    for r in rows:
        if show:
            print(f"| {r['tag']} | {r['probe']} | {r['eval_count']} | "
                  f"{r['thinking_tokens']} | {r['answer_tokens']} | "
                  f"{r['control_tokens']} |")
        else:
            print(f"| {r['tag']} | {r['probe']} | {r['eval_count']} | "
                  f"{r['thinking_chars']} | {r['answer_chars']} |")

    if write and show:
        for tag in tags:
            path = os.path.join(DIR, f"scores_{tag}.json")
            data = json.load(open(path))
            for r in (x for x in rows if x["tag"] == tag):
                data[r["probe"]].update(
                    thinking_tokens=r["thinking_tokens"],
                    answer_tokens=r["answer_tokens"],
                    control_tokens=r["control_tokens"])
            json.dump(data, open(path, "w"), indent=1)
        print(f"\nwrote splits into {len(tags)} scores file(s)")
    elif write:
        print("\nNOT written: the gate did not pass.")


if __name__ == "__main__":
    main()
