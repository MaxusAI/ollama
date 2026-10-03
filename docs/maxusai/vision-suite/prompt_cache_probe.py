#!/usr/bin/env python3
"""Requests that take turns between contexts: the case llama-server's prompt cache is for.

Usage:
    prompt_cache_probe.py run --host URL --model M --label L [--image A --image B ...] > run.json
    prompt_cache_probe.py compare run.json [run.json ...]

llama.cpp's server keeps a prompt cache in host RAM: 8192 MiB unless `--cache-ram` says
otherwise, and ollama passes nothing unless the `prompt_cache_ram` option or
OLLAMA_LLAMA_SERVER_CACHE_RAM sets it. With one slot, a request that cannot reuse the slot's prompt first saves the slot's state into that
cache, and a later request can restore it instead of evaluating its prompt again. This probe
sends requests that could each resume a state saved two requests earlier:

- **Without `--image`, two conversations take turns.** Each opens with its own document of
  about `--doc-words` words and asks one question a turn. The conversation so far is the
  prompt's prefix, so it grows turn by turn and is shared with the conversation's previous
  request.
- **With two or more `--image`, the images take turns,** one question each.

Every request goes through `client.generate()` (SPEC H9): greedy, think off, `/api/generate`,
no ambient Runner options. Each records ollama's `prompt_eval_duration` (the prefill
llama-server actually ran), `total_duration` and the client's wall clock. `compare` prints one
row per run. Run it once with the cache and once with OLLAMA_LLAMA_SERVER_CACHE_RAM=0, on the
same build, and read the cache's own record of the runs with
`docs/maxusai/tools/prompt_cache_stats.py`.
"""
import argparse
import json
import random
import statistics
import sys
import time

import client

WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lagoon meadow "
         "nickel orchid pewter quartz river slate tundra umber violet willow xenon yarrow zinc").split()

QUESTIONS = ["What is written in this image?", "Describe this image in one sentence.",
             "Which colours dominate this image?", "How many distinct objects are in this image?",
             "Is there any text in this image? Answer yes or no.", "What is in the top left of this image?"]


def document(seed, words):
    """A deterministic document of numbered facts, about `words` words long."""
    rng, out, n = random.Random(seed), [], 0
    while sum(len(s.split()) for s in out) < words:
        n += 1
        out.append(f"Fact {n}: the {rng.choice(WORDS)} {rng.choice(WORDS)} weighs {rng.randint(2, 999)} units "
                   f"and sits beside the {rng.choice(WORDS)} {rng.choice(WORDS)}.")
    return " ".join(out), n


def ask(args, prompt, images):
    t0 = time.time()
    r = client.generate(args.host, args.model, prompt, images, num_predict=args.num_predict,
                        num_ctx=args.num_ctx, fmt=None, extra_opts={"temperature": 0},
                        endpoint_override="generate", think=False, apply_sampling=False,
                        timeout=args.timeout, use_env_opts=False)
    return r, (time.time() - t0) * 1000


def run(args):
    records, builds = [], set()

    def record(kind, ctx, turn, r, wall):
        builds.add(r.get("_server_version"))
        records.append({"kind": kind, "context": ctx, "turn": turn,
                        "prompt_eval_count": r.get("prompt_eval_count"),
                        "prompt_eval_ms": round((r.get("prompt_eval_duration") or 0) / 1e6, 1),
                        "eval_count": r.get("eval_count"),
                        "total_ms": round((r.get("total_duration") or 0) / 1e6, 1),
                        "wall_ms": round(wall, 1)})

    if args.image:
        imgs = [client.b64_file(p) for p in args.image]
        for turn, q in enumerate(QUESTIONS[:args.turns]):
            for i, img in enumerate(imgs):
                r, wall = ask(args, q, [img])
                record("image", i, turn, r, wall)
    else:
        docs = [document(1000 + c, args.doc_words) for c in range(args.conversations)]
        so_far = [f"Read this document, then answer my questions about it in one short sentence each.\n\n"
                  f"{text}\n\n" for text, _ in docs]
        for turn in range(args.turns):
            for c, (_, n_facts) in enumerate(docs):
                question = f"Question {turn + 1}: what is fact {1 + (turn * 7) % n_facts}?\nAnswer:"
                r, wall = ask(args, so_far[c] + question, [])
                so_far[c] += f"{question} {r.get('response', '').strip()}\n"
                record("conversation", c, turn, r, wall)
    return {"label": args.label, "model": args.model,
            "server_version": sorted(b for b in builds if b) or None,
            "workload": "images" if args.image else "conversations",
            "settings": {"num_ctx": args.num_ctx, "num_predict": args.num_predict, "turns": args.turns,
                         "contexts": len(args.image) if args.image else args.conversations,
                         "doc_words": None if args.image else args.doc_words},
            "records": records}


def compare(paths):
    """One row per run: the first request of each context, then every later one."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    print("| run | model | workload | contexts × turns | first visit: prefill, request | "
          "every return: prefill, request | prompt tokens, last turn |")
    print("|---|---|---|---|---|---|---|")
    for d in runs:
        recs = d["records"]
        first = [r for r in recs if r["turn"] == 0]
        back = [r for r in recs if r["turn"] > 0]
        last = max(r["turn"] for r in recs)
        mean = lambda rs, k: statistics.mean(r[k] for r in rs) / 1000 if rs else float("nan")  # noqa: E731
        s = d["settings"]
        print(f"| {d['label']} | `{d['model']}` | {d['workload']} | {s['contexts']} × {s['turns']} "
              f"| {mean(first, 'prompt_eval_ms'):.2f} s, {mean(first, 'wall_ms'):.2f} s "
              f"| {mean(back, 'prompt_eval_ms'):.2f} s, {mean(back, 'wall_ms'):.2f} s "
              f"| {round(statistics.mean(r['prompt_eval_count'] or 0 for r in recs if r['turn'] == last))} |")
    builds = sorted({b for d in runs for b in (d.get("server_version") or ["not recorded"])})
    print(f"\nbuild(s): {', '.join(builds)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True, help="names the run in compare's table, e.g. 'cache on'")
    r.add_argument("--image", action="append", default=[], help="two or more: the images take turns")
    r.add_argument("--conversations", type=int, default=2)
    r.add_argument("--turns", type=int, default=6)
    r.add_argument("--doc-words", type=int, default=2200)
    r.add_argument("--num-ctx", type=int, default=8192)
    r.add_argument("--num-predict", type=int, default=48)
    r.add_argument("--timeout", type=int, default=600)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "compare":
        compare(args.runs)
        return
    if len(args.image) == 1:
        ap.error("--image takes two or more images, so that they take turns")
    json.dump(run(args), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
