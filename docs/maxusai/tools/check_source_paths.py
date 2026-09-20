#!/usr/bin/env python3
"""Does every source file this repo's prose points at still exist?

    python3 docs/maxusai/tools/check_source_paths.py        # repo root

Exit 0 when every reference resolves, 1 with a list when one does not.

WHY. A rename pass that is 99% right is the dangerous kind. The v0.34.2 fold
moved the MLX engine out of `x/`, re-pointed most references, and left 31
behind -- including AGENTS.md, which every agent is told to follow, pointing at
directories the fold deleted. Worse, the mechanical substitution CREATED two
paths that look right and are not: `x/models/base/media.go` became
`mlxrunner/model/base/media.go` when the reorganisation had removed the `base/`
segment, and `x/mlxrunner/cache.go` became `mlxrunner/cache.go` when the file
lives in the `cache/` subpackage. A missing reference is obvious to a reader; a
confidently wrong one is not, and only an existence check finds it.

SCOPE. With `--changed-since <ref>` it reports only on files the change TOUCHED,
resolving against the whole tree. That is the CI mode, and it is deliberate:
eleven references on main name files that were DELETED rather than moved --
`x/mlxrunner/constrain.go` after ADR 0033 retired it, `integration/imagegen_test.go`
after imagegen went -- which is living prose about dead code, not an error.
Gating the whole tree would need an eleven-entry allowlist on day one, and an
allowlist that grows is how a guard becomes decoration. Scoped to the diff it
still catches what it exists for: every stale reference the v0.34.2 fold left,
and both paths it invented, were in files that fold touched. Run it with no
flags for a full-tree audit.

WHAT IT DOES NOT DO, and why each exclusion is load-bearing. Measured against
main: naive matching finds 99 references across 29 files, of which the large
majority are noise. Resolving document-relative and package-root-relative forms
brings it to 26. Ignoring paths whose first segment is not a directory this
repo owns brings it to 10, and those are real. A guard that cries wolf is worse
than no guard -- people learn to skip it (SPEC H20) -- so:

  * Only FILE references (.go/.py/.sh). A bare directory mention is ambiguous
    and a package name in prose ("the retired x/structured sampler") is not a
    path at all.
  * Only paths whose first segment is a top-level directory this repo owns.
    `mlx_vlm/models/...`, `vlmeval/config.py` and upstream's
    `ollama/model/model.go` belong to other trees.
  * Lines carrying `golang.org/` or `github.com/` are import lines, not paths.
  * A reference resolves if it matches from the repo root, from the directory
    of the file citing it, or as the suffix of any tracked path -- docs name
    paths from a package root constantly.
  * HISTORICAL_PREFIXES are skipped entirely. An ADR records what was true when
    it was decided and a dated task doc records what was run; re-pointing those
    would make the history wrong. That is a policy, not an oversight.
"""
import collections
import os
import re
import subprocess
import sys

# A path-looking token ending in a source extension. The left boundary excludes
# a dot, slash or word character, so `mlx/quant/quant.go` is never read as the
# `x/quant/quant.go` hiding inside it -- the substring bug that would also
# corrupt the file if a rewriter shared this regex.
REFERENCE = re.compile(
    r"(?<![\w./-])((?:[a-z0-9_]+/)+[a-z0-9_.-]+\.(?:go|py|sh))(?![\w])")

SCANNED_SUFFIXES = (".go", ".py", ".md", ".sh", ".toml")

# The guard cannot check itself. Its docstrings quote the very paths it exists
# to catch -- `mlxrunner/model/base/media.go` is the worked example of an
# invented path -- and its tests build fixtures out of names like
# `mlxrunner/gone.go` that are supposed not to exist. Measured: without this it
# reports 30 findings against its own diff, every one of them a quotation.
SELF = ("docs/maxusai/tools/",)

HISTORICAL_PREFIXES = (
    "docs/maxusai/adr/",                 # decided then, true then
    "docs/maxusai/tasks/",               # what a dated sync actually ran
    "docs/superpowers/plans/",           # a plan as it was written
    "docs/design/",                      # against upstream's tree, not ours
    "docs/maxusai/vision-campaign-",     # a campaign's record
    "docs/maxusai/retirement-register",  # names things precisely because they are gone
)

Finding = collections.namedtuple("Finding", "file line path")


def tracked_files(root):
    out = subprocess.run(["git", "-C", root, "ls-files"],
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def scan(root, tracked, limit_to=None):
    """[Finding] for every source-file reference that does not resolve.

    `limit_to` restricts REPORTING to those files while still resolving against
    the whole tree -- the diff-scoped mode CI uses. Resolution must see every
    tracked path or a reference to an untouched file would look missing.
    """
    tracked = list(tracked)
    known = set(tracked)
    owned = {p.split("/")[0] for p in tracked if "/" in p}
    # Docs name paths from a package root, not the repo root: `cache/kvcache.go`
    # for mlxrunner/cache/kvcache.go. Every suffix of every tracked path counts.
    suffixes = set()
    for path in tracked:
        parts = path.split("/")
        for i in range(len(parts)):
            suffixes.add("/".join(parts[i:]))

    scope = None if limit_to is None else set(limit_to)
    findings = []
    for rel in tracked:
        if scope is not None and rel not in scope:
            continue
        if not rel.endswith(SCANNED_SUFFIXES) or rel.startswith(HISTORICAL_PREFIXES):
            continue
        if rel.startswith(SELF):
            continue
        full = os.path.join(root, rel)
        try:
            with open(full, errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        here = os.path.dirname(rel)
        for number, line in enumerate(text.splitlines(), 1):
            if "golang.org/" in line or "github.com/" in line:
                continue
            for match in REFERENCE.finditer(line):
                path = match.group(1)
                if path.split("/")[0] not in owned:
                    continue
                if path in known or path in suffixes:
                    continue
                if os.path.exists(os.path.join(root, here, path)):
                    continue
                findings.append(Finding(rel, number, path))
    return findings


def changed_files(root, base):
    """Tracked files this branch changed against `base`, for the CI mode."""
    out = subprocess.run(["git", "-C", root, "diff", "--name-only", base, "HEAD"],
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split() if p]


def main(argv):
    args = argv[1:]
    base = None
    if "--changed-since" in args:
        i = args.index("--changed-since")
        base = args[i + 1]
        args = args[:i] + args[i + 2:]
    root = args[0] if args else "."
    tracked = tracked_files(root)
    limit_to = None
    if base:
        limit_to = changed_files(root, base)
        if not limit_to:
            print("source paths: this change touches no tracked file")
            return 0
        print(f"source paths: checking {len(limit_to)} changed file(s) against {base}")
    findings = scan(root, tracked, limit_to=limit_to)
    if not findings:
        print("source paths: every referenced file resolves")
        return 0
    print(f"source paths: {len(findings)} reference(s) name a file that does not "
          f"exist\n")
    for f in findings:
        print(f"  {f.file}:{f.line}  ->  {f.path}")
    print("\nEither the file moved and the reference needs re-pointing, or the "
          "reference\nbelongs in a historical record. See this script's "
          "docstring for the policy.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
