"""flatten_cleanslate.py — move a clean-slate tree onto the flat ``output/autolens/`` layout.

autolens_inference#35 dropped the dated ``cleanslate_2026_10_17`` level. The code writes the
new layout from that PR on; this one-off script moves what the campaign already wrote::

    output/cleanslate_2026_10_17/slam/<dataset_class>/<instrument>/<variant>/<config>/
        -> output/autolens/<dataset_class>/<instrument>/slam/<variant>/<config>/
    output/prepared/cleanslate_2026_10_17/<config>/  -> output/autolens/prepared/<config>/
    output/experiments/<run_id>/                     -> output/autolens/experiments/<run_id>/
    output/experiment_sources/<run_id>/              -> output/autolens/experiment_sources/<run_id>/
    results/slam/.../<config>__cleanslate_2026_10_17/ -> results/slam/.../<config>/

and rewrites the paths recorded inside the result rows (``output_path``, prepared
``record_id`` anchors; ``results_tag`` becomes ``null``), the experiment rows, the prepared
manifests (``output/autolens/prepared/**/manifest.json`` and the published
``prepared/<id>/manifest.json``) and ``.cortex/pull.json`` (its checkpoint keys). Each file is
rewritten as text, so its formatting is kept, and the result must parse as JSON. PyAutoFit
identifiers hash search + model + unique_tag, never the path_prefix, so a moved stage
directory keeps its hash name and still resumes.

Dry run by default: prints every move and rewrite and touches nothing. ``--apply`` acts.
It is idempotent (a second ``--apply`` finds nothing to do) and refuses, before moving
anything, if any target already exists. Moves are ``os.rename`` only, so a cross-filesystem
move fails loudly instead of half-copying a tree. Directories left empty by the moves are
removed; nothing else is deleted.

WHEN AND WHERE. Run only after RAL arrays 412541 and 412565 have completed (they write into
the dated tree; never move a tree under a live job). Order:

1. ``hpc/sync pull`` while RAL is still dated, and check the laptop mirror is complete.
2. RAL first, from the RAL checkout:
   ``python3 scripts/misc/flatten_cleanslate.py`` (read the plan), then ``--apply``.
3. The laptop mirror: ``--root`` the checkout, ``--output-root`` the ``output/`` target
   (``/mnt/c/Users/Jammy/Science/inference/autolens_inference/output``).
4. ``hpc/sync pull`` again: it should move next to nothing, which verifies the two trees
   agree, and must not re-create the dated folder.
5. Regenerate the dashboard (``build_readme.py``, ``export_inference_summary.py``).

The RAL stage directories are named like ``source_lp[1]``: never shell-glob them; this
script walks directories with ``os.scandir`` and never globs.

Run from the repo root::

    python3 scripts/misc/flatten_cleanslate.py                       # plan only
    python3 scripts/misc/flatten_cleanslate.py --apply
    python3 scripts/misc/flatten_cleanslate.py --output-root /path/to/output --apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

TAG = "cleanslate_2026_10_17"

#: Text rewrites applied to every recorded path, in order. The SLaM one reorders
#: ``slam/<dataset_class>/<instrument>`` to ``<dataset_class>/<instrument>/slam``.
REWRITES = (
    (re.compile(rf"output/{TAG}/slam/([^/\"]+)/([^/\"]+)/"), r"output/autolens/\1/\2/slam/"),
    (re.compile(rf"output/prepared/{TAG}/"), "output/autolens/prepared/"),
    (re.compile(r"(?<![^/\"])output/experiments/"), "output/autolens/experiments/"),
    (re.compile(r"(?<![^/\"])output/experiment_sources/"), "output/autolens/experiment_sources/"),
    (re.compile(rf"__{TAG}/"), "/"),
)
RESULTS_TAG = re.compile(rf'("results_tag"\s*:\s*)"{TAG}"')


def rewrite_text(text: str, *, row: bool = False) -> str:
    for pattern, replacement in REWRITES:
        text = pattern.sub(replacement, text)
    if row:
        text = RESULTS_TAG.sub(r"\1null", text)
    return text


def _subdirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    with os.scandir(path) as entries:
        return sorted(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))


def plan_moves(root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    """Every (source, target) directory rename, in the order they will be made."""
    moves = []
    new = output_root / "autolens"
    for dataset_class in _subdirs(output_root / TAG / "slam"):
        for instrument in _subdirs(dataset_class):
            for variant in _subdirs(instrument):
                for config in _subdirs(variant):
                    target = new / dataset_class.name / instrument.name / "slam"
                    moves.append((config, target / variant.name / config.name))
    for source, target in (
        (output_root / "prepared" / TAG, new / "prepared"),
        (output_root / "experiments", new / "experiments"),
        (output_root / "experiment_sources", new / "experiment_sources"),
    ):
        for child in _subdirs(source):
            moves.append((child, target / child.name))
    for directory, names, _ in os.walk(root / "results" / "slam"):
        for name in sorted(names):
            if name.endswith(f"__{TAG}"):
                old = Path(directory) / name
                moves.append((old, old.with_name(name[: -len(f"__{TAG}")])))
    return moves


def _json_files(path: Path, name: str | None = None) -> list[Path]:
    if not path.is_dir():
        return []
    found = []
    for directory, _, files in os.walk(path):
        for file in files:
            if file.endswith(".json") and (name is None or file == name):
                found.append(Path(directory) / file)
    return sorted(found)


def plan_rewrites(root: Path, output_root: Path) -> list[tuple[Path, str]]:
    """Every (file, new text) whose recorded paths change. Read at the files' current places."""
    candidates = [(p, True) for p in _json_files(root / "results" / "slam")]
    candidates += [(p, False) for p in _json_files(root / "results" / "searches")]
    for prepared in (
        root / "prepared",
        output_root / "prepared" / TAG,
        output_root / "autolens" / "prepared",
    ):
        candidates += [(p, False) for p in _json_files(prepared, "manifest.json")]
    pull = root / ".cortex" / "pull.json"
    if pull.is_file():
        candidates.append((pull, False))
    rewrites = []
    for path, row in candidates:
        text = path.read_text()
        new = rewrite_text(text, row=row)
        if new != text:
            json.loads(new)  # a rewrite must never leave a file that does not parse
            rewrites.append((path, new))
    return rewrites


def _moved(path: Path, moves: list[tuple[Path, Path]]) -> Path:
    """Where ``path`` lives once ``moves`` are made."""
    for source, target in moves:
        if path == source or source in path.parents:
            return target / path.relative_to(source)
    return path


def _prune(directory: Path, stop: Path) -> list[Path]:
    """Remove ``directory`` and its parents up to (and including) ``stop`` while empty."""
    removed = []
    while directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        removed.append(directory)
        if directory == stop:
            break
        directory = directory.parent
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout (results/, .cortex/)")
    parser.add_argument(
        "--output-root", type=Path, default=None, help="the PyAutoFit output root (<root>/output)"
    )
    parser.add_argument("--apply", action="store_true", help="act; without it, plan only")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_root = (args.output_root or root / "output").resolve()

    moves = plan_moves(root, output_root)
    rewrites = plan_rewrites(root, output_root)
    targets = [target for _, target in moves]
    conflicts = [t for t in targets if t.exists()]
    conflicts += sorted({t for t in targets if targets.count(t) > 1})

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"flatten_cleanslate [{mode}]  root={root}  output-root={output_root}")
    for source, target in moves:
        print(f"  move     {source}\n        -> {target}")
    for path, _ in rewrites:
        print(f"  rewrite  {_moved(path, moves)}")
    if conflicts:
        for target in conflicts:
            print(f"  REFUSE   target already exists: {target}", file=sys.stderr)
        print("Nothing changed: resolve the targets above first.", file=sys.stderr)
        return 2
    if not moves and not rewrites:
        print("Nothing to do: the tree is already flat.")
        return 0
    if not args.apply:
        print(f"{len(moves)} move(s), {len(rewrites)} rewrite(s) planned; --apply to act.")
        return 0

    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
    for path, text in rewrites:
        _moved(path, moves).write_text(text)
    # (emptied tree, the highest directory pruning may climb to): never above these.
    removed = []
    for directory, stop in (
        (output_root / TAG, output_root / TAG),
        (output_root / "prepared" / TAG, output_root / "prepared"),
        (output_root / "experiments", output_root / "experiments"),
        (output_root / "experiment_sources", output_root / "experiment_sources"),
    ):
        for leaf, _, _ in sorted(os.walk(directory), reverse=True):
            removed += _prune(Path(leaf), stop)
    print(
        f"Done: {len(moves)} move(s), {len(rewrites)} rewrite(s), "
        f"{len(removed)} empty dir(s) removed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
