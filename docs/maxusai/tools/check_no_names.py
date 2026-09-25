#!/usr/bin/env python3
"""Does a tracked file, a commit message or a pull request's text name a person?

    NAME_DENYLIST='<regex>' python3 docs/maxusai/tools/check_no_names.py      # repo root
    python3 docs/maxusai/tools/check_no_names.py --denylist-file ~/.config/collab/denylist

Exit 0 when nothing matches, 1 with a list when something does, 2 on a
configuration error (and, with --require-denylist, when there is no deny-list).

WHY. The repo and every issue, pull request and comment on it are public, and
every host's agent posts as the same GitHub account, so a name written into a
doc, a commit message or a PR description is published under that account.
Agents pick the name up from the conversation and the git author field and
write it as attribution -- "X's call", "Deciders: X", "(X, 2026-09-12)". On
2026-09-25 it had to be scrubbed from 24 tracked files (#380) and from 32 pull
request, issue and comment texts. AGENTS.md now says "the maintainer"; this
guard fails a change that brings a name back.

THE DENY-LIST IS NOT IN THE TREE. A guard that spells the name out publishes
it. The patterns arrive through the NAME_DENYLIST environment variable (the CI
secret of the same name) or --denylist-file, a file outside the checkout such
as ~/.config/collab/denylist: one Python regex per line, `#` comments allowed,
all matched case-insensitively. Without either the guard says it did not run
and exits 0, so a fork or a fresh clone is not blocked; CI passes
--require-denylist, so a missing secret fails loudly instead of passing
silently.

NOTHING IT PRINTS CONTAINS A MATCH. CI logs of a public repo are public, so a
finding is reported as path:line with every match replaced by <name>. For the
same reason CI hands it the pull request through --github-event (the event file
on the runner), never through step env: Actions prints env values in the log.

CASE-INSENSITIVE, deliberately. The hostname that survived #380 was lower
case, and the case-sensitive grep that verified #380 reported the tree clean.

THE ALLOWLIST (no_names_allowlist.txt, beside this script) holds paths, never
names. A path ending in / covers every file under that directory. Binary files
and symlinks are skipped.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

Hit = collections.namedtuple("Hit", "where line context")

MASK = "<name>"
DEFAULT_ALLOWLIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "no_names_allowlist.txt")


class ConfigError(Exception):
    pass


def _entries(text):
    """Non-blank, non-comment lines, stripped."""
    return [s for s in (ln.strip() for ln in (text or "").splitlines())
            if s and not s.startswith("#")]


def load_denylist(env_value, file_path):
    """One compiled, case-insensitive pattern, or None when nothing is set.

    A blank value counts as unset: an empty regex matches every line, so taken
    literally it would fail the whole tree.
    """
    patterns = _entries(env_value)
    if file_path:
        with open(file_path) as fh:
            patterns += _entries(fh.read())
    if not patterns:
        return None
    try:
        # re.error's message describes the syntax fault without echoing the
        # pattern, so it is safe to print.
        pattern = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    except re.error as exc:
        raise ConfigError(f"the deny-list is not a valid regex: {exc.msg}") from None
    if pattern.search(""):
        raise ConfigError("a deny-list pattern matches the empty string, so it would flag every line")
    return pattern


def load_allowlist(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return _entries(fh.read())


def allowed(rel, allow):
    return any(rel == a or (a.endswith("/") and rel.startswith(a)) for a in allow)


def _hits_in(where, text, pattern):
    return [Hit(where, n, pattern.sub(MASK, line.strip())[:160])
            for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]


def scan(root, tracked, pattern, allow):
    """[Hit] for every line of a tracked text file that matches the deny-list."""
    hits = []
    for rel in tracked:
        path = os.path.join(root, rel)
        if allowed(rel, allow) or os.path.islink(path) or not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        if b"\0" in data:        # binary: an image or a compiled blob
            continue
        hits += _hits_in(rel, data.decode("utf-8", errors="replace"), pattern)
    return hits


def scan_texts(texts, pattern):
    """[Hit] for labelled free text: a PR title or body, commit messages."""
    hits = []
    for label, text in texts.items():
        hits += _hits_in(label, text or "", pattern)
    return hits


def event_texts(path):
    """A pull request's title and body, from the Actions event file.

    Read from $GITHUB_EVENT_PATH, not passed through step env: Actions prints
    every step-level env value in the public run log, and that log outlives
    any later edit of the description.
    """
    with open(path) as fh:
        pr = json.load(fh).get("pull_request")
    if not isinstance(pr, dict):
        return {}
    return {"PR title": pr.get("title") or "", "PR body": pr.get("body") or ""}


def render(hit):
    return f"  {hit.where}:{hit.line}  {hit.context}"


def tracked_files(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def commit_messages(root, revision_range):
    """The branch's OWN commit messages. --first-parent keeps a fold's merged
    upstream history out: those messages are upstream's, not ours to police."""
    out = subprocess.run(["git", "-C", root, "log", "--first-parent",
                          "--format=%H%x00%B%x00", revision_range],
                         capture_output=True, text=True, check=True)
    fields = out.stdout.split("\0")
    return {f"commit {fields[i].strip()[:9]}": fields[i + 1]
            for i in range(0, len(fields) - 1, 2) if fields[i].strip()}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--denylist-file")
    ap.add_argument("--allowlist", default=DEFAULT_ALLOWLIST)
    ap.add_argument("--commits", metavar="RANGE",
                    help="also check these commits' messages, e.g. BASE..HEAD")
    ap.add_argument("--github-event", metavar="FILE",
                    help="also check the pull request title and body in this Actions event file")
    ap.add_argument("--text-env", metavar="VAR", action="append", default=[],
                    help="also check the text in this environment variable (repeatable)")
    ap.add_argument("--require-denylist", action="store_true",
                    help="exit 2 instead of 0 when no deny-list is configured")
    args = ap.parse_args(argv[1:])

    try:
        pattern = load_denylist(os.environ.get("NAME_DENYLIST"), args.denylist_file)
    except (ConfigError, OSError) as exc:
        print(f"names: {exc}")
        return 2
    if pattern is None:
        print("names: no deny-list configured (NAME_DENYLIST or --denylist-file); nothing was checked")
        return 2 if args.require_denylist else 0

    tracked = tracked_files(args.root)
    hits = scan(args.root, tracked, pattern, load_allowlist(args.allowlist))
    texts = {var: os.environ.get(var, "") for var in args.text_env}
    if args.github_event:
        texts.update(event_texts(args.github_event))
    if args.commits:
        texts.update(commit_messages(args.root, args.commits))
    hits += scan_texts(texts, pattern)

    if not hits:
        print(f"names: none of {len(tracked)} tracked files"
              + (f" or {len(texts)} texts" if texts else "")
              + " names a person on the deny-list")
        return 0
    print(f"names: {len(hits)} line(s) name a person on the deny-list\n")
    for hit in hits:
        print(render(hit))
    print("\nWrite \"the maintainer\" for the person who decides and merges, and a host's"
          "\ncollab label (e.g. amd-server/rocm-gfx1151) for a machine. See AGENTS.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
