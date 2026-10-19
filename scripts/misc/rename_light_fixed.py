"""rename_light_fixed.py — rename the ``_lightfixed`` variant to ``_light_fixed``, and set the
superseded numba legs aside.

autolens_inference#38 renamed the fixed-lens-light run variant suffix from ``_lightfixed`` to
``_light_fixed`` (``slam_base_lightfixed`` -> ``slam_base_light_fixed``). The code writes the
new name from that PR on; this one-off script moves what the clean-slate campaign already
wrote, so old runs carry the new name too::

    results/**/<variant>_lightfixed/                       -> .../<variant>_light_fixed/
    output/autolens/<class>/<instrument>/slam/<variant>_lightfixed/
                                                           -> .../<variant>_light_fixed/
    output/autolens/prepared/<config>/seed_<n>/<variant>_lightfixed/
                                                           -> .../<variant>_light_fixed/

and rewrites what is recorded inside:

- **Result rows** (every ``*.json`` under ``results/``) and ``.cortex/pull.json``: every string
  (and key) containing ``_lightfixed`` gets ``_light_fixed`` (``variant``, ``target``,
  ``output_path``, record anchors), and every prepared-problem id that this run re-identifies
  (``problem_id``) gets its new id. Each file is loaded as JSON, rewritten structurally and
  dumped back in the producer's own format (``indent=2``, its trailing newline kept); only
  files that change are written. PNGs are left alone.
- **Prepared manifests** (``output/autolens/prepared/**/manifest.json`` whose text names
  ``_lightfixed``): ``controls.variant``, artifact ``path`` / ``record_id`` and
  ``baseline_record_id`` are rewritten, so the scientific identity changes (``controls`` is in
  it) and the manifest ``id`` is recomputed with ``experiments.prepared.manifest_problem_id``
  — the same function ``export_problem`` and ``verify_manifest`` use. The artifacts' sha256s
  do not change (their bytes are not touched; only their paths move). The new manifest is
  re-published (``prepared.publish_manifest``: ``prepared/<new id>/manifest.json`` at the
  root), and the old published copy ``prepared/<old id>/`` is removed ONLY when it holds
  nothing but a ``manifest.json`` equal to the manifest this run rewrote — a stale copy of
  it; anything else is left in place and reported. Every rewritten manifest is then verified,
  and the run exits 1 loudly if any fails.

PyAutoFit identifiers hash search + model + unique_tag, never the path_prefix, so a renamed
variant directory keeps its stage hashes and still resumes under the new name.

``--aside`` instead sets the superseded ``hpc_a100_numba_cpu_{dense,sparse}_fp64`` legs aside
(their 2026-10-17 rows ran with unpinned BLAS pools, ~35x slow; #38 reruns them as
``hpc_ral_numba_cpu_sparse_fp64``). Nothing is deleted:

    output/autolens/**/slam/<variant>/<config>/   -> <aside-dest>/autolens/**/slam/<variant>/<config>/
    output/autolens/prepared/<config>/            -> <aside-dest>/autolens/prepared/<config>/
    results/slam/**/<variant>/<config>[__<tag>]/  -> results/archive/<aside-label>/slam/**/...
    prepared/<id>/ (a manifest of those configs)  -> results/archive/<aside-label>/prepared/<id>/

The rows go to ``results/archive/`` (the repo's convention for superseded rows: the exporter
marks them ``archived``), so they stay on the checkout's filesystem; ``--aside-dest`` takes
only the bulk output trees.

Dry run by default: prints every move and rewrite and touches nothing. ``--apply`` acts. It is
idempotent (a second ``--apply`` finds nothing to do) and refuses, before moving anything, if
any target already exists or if a move would cross filesystems. Moves are ``os.rename`` only,
so a cross-filesystem move can never half-copy a tree.

WHEN AND WHERE. Never under a live job writing into these trees. RAL first, from the RAL
checkout (plan, then ``--apply``), then the laptop mirror (``--root`` the checkout,
``--output-root`` the ``output/`` target), then ``hpc/sync pull`` (it should move next to
nothing), then regenerate the dashboard. Rename first, then ``--aside``.

The RAL stage directories are named like ``source_lp[1]``: never shell-glob them; this script
walks directories with ``os.scandir`` and never globs.

Run from the repo root::

    python3 scripts/misc/rename_light_fixed.py                       # plan only
    python3 scripts/misc/rename_light_fixed.py --apply
    python3 scripts/misc/rename_light_fixed.py --output-root /path/to/output --apply
    python3 scripts/misc/rename_light_fixed.py --aside --aside-dest /path/to/aside   # plan
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from experiments import prepared  # noqa: E402

OLD, NEW = "_lightfixed", "_light_fixed"

#: The superseded numba legs ``--aside`` moves (autolens_inference#38).
ASIDE_CONFIGS = ("hpc_a100_numba_cpu_dense_fp64", "hpc_a100_numba_cpu_sparse_fp64")
ASIDE_LABEL = "2026-10-19_numba_thread_oversubscription"

#: How deep the variant directories sit below each walked base. Bounded so the walk never
#: descends into PyAutoFit's per-stage trees (``<variant>/<config>/seed_<n>/source_lp[1]/...``).
RESULTS_DEPTH = 7  # results/[archive/<label>/]slam/<class>/<instrument>/<variant>
OUTPUT_DEPTH = 5  # autolens/<class>/<instrument>/slam/<variant>, autolens/prepared/<c>/seed/<v>


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------


def _subdirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    with os.scandir(path) as entries:
        return sorted(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))


def _walk_dirs(base: Path, max_depth: int):
    """Every directory below ``base`` down to ``max_depth`` levels, parents first."""
    stack = [(base, 0)]
    while stack:
        path, depth = stack.pop()
        if depth >= max_depth:
            continue
        children = _subdirs(path)
        for child in children:
            yield child, depth + 1
        stack.extend((child, depth + 1) for child in reversed(children))


def _json_files(path: Path, name: str | None = None) -> list[Path]:
    if not path.is_dir():
        return []
    found = []
    for directory, _, files in os.walk(path):
        for file in files:
            if file.endswith(".json") and (name is None or file == name):
                found.append(Path(directory) / file)
    return sorted(found)


def _moved(path: Path, moves: list[tuple[Path, Path]]) -> Path:
    """Where ``path`` lives once ``moves`` are made."""
    for source, target in moves:
        if path == source or source in path.parents:
            return target / path.relative_to(source)
    return path


# ---------------------------------------------------------------------------
# JSON rewriting
# ---------------------------------------------------------------------------


def _rewrite_string(value: str, ids: dict[str, str]) -> str:
    value = value.replace(OLD, NEW)
    for old_id, new_id in ids.items():
        value = value.replace(old_id, new_id)
    return value


def rewrite_value(value, ids: dict[str, str] | None = None):
    """``value`` with every string (and key) renamed, recursively."""
    ids = ids or {}
    if isinstance(value, str):
        return _rewrite_string(value, ids)
    if isinstance(value, list):
        return [rewrite_value(v, ids) for v in value]
    if isinstance(value, dict):
        return {_rewrite_string(k, ids): rewrite_value(v, ids) for k, v in value.items()}
    return value


def _dump_like(text: str, doc) -> str:
    """``doc`` in the format of ``text``: the producers write ``indent=2``; manifests also
    sort their keys (``prepared.json_bytes``). The trailing newline is the file's own."""
    original = json.loads(text)
    for sort_keys in (False, True):
        if json.dumps(original, indent=2, sort_keys=sort_keys) == text.rstrip("\n"):
            new = json.dumps(doc, indent=2, sort_keys=sort_keys)
            return new + text[len(text.rstrip("\n")) :]
    raise ValueError("unrecognised JSON formatting (not indent=2); refusing to reformat it")


def rewrite_json_text(text: str, ids: dict[str, str] | None = None) -> str:
    """The file's new text, or the same text when nothing in it changes."""
    if OLD not in text and not any(old in text for old in (ids or {})):
        return text
    doc = json.loads(text)
    new_doc = rewrite_value(doc, ids)
    if new_doc == doc:
        return text
    return _dump_like(text, new_doc)


# ---------------------------------------------------------------------------
# The rename plan
# ---------------------------------------------------------------------------


def plan_renames(root: Path, output_root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """``(moves, odd)``: every ``*_lightfixed`` directory rename, plus directories that name
    ``lightfixed`` without ending in it (reported, never touched)."""
    moves, odd = [], []
    for base, depth in (
        (root / "results", RESULTS_DEPTH),
        (output_root / "autolens", OUTPUT_DEPTH),
    ):
        renamed: list[Path] = []
        for path, _ in _walk_dirs(base, depth):
            if any(r in path.parents for r in renamed):
                continue  # inside a directory already being renamed: it moves with it
            if path.name.endswith(OLD):
                moves.append((path, path.with_name(path.name[: -len(OLD)] + NEW)))
                renamed.append(path)
            elif "lightfixed" in path.name:
                odd.append(path)
    return moves, odd


def plan_manifests(output_root: Path, moves) -> list[dict]:
    """Every prepared manifest whose text names ``_lightfixed``, with its rewritten doc."""
    plans = []
    for path in _json_files(output_root / "autolens" / "prepared", "manifest.json"):
        text = path.read_text()
        if OLD not in text:
            continue
        old_doc = json.loads(text)
        new_doc = rewrite_value(old_doc)
        new_doc["id"] = prepared.manifest_problem_id(new_doc)
        plans.append(
            dict(
                path=path,
                target=_moved(path, moves),
                old_doc=old_doc,
                new_doc=new_doc,
                text=_dump_like(text, new_doc),
            )
        )
    return plans


def _published_dir(root: Path, problem_id: str) -> Path:
    return root / "prepared" / problem_id


def plan_publish(root: Path, manifests: list[dict]):
    """``(publish, stale, kept, clashes)`` for the published ``prepared/<id>/`` copies."""
    publish, stale, kept, clashes = [], [], [], []
    for plan in manifests:
        new_dir = _published_dir(root, plan["new_doc"]["id"])
        new_file = new_dir / "manifest.json"
        if new_file.exists() and json.loads(new_file.read_text()) != plan["new_doc"]:
            clashes.append(new_file)
        publish.append(plan["new_doc"])
        old_dir = _published_dir(root, plan["old_doc"]["id"])
        if not old_dir.is_dir():
            continue
        entries = sorted(p.name for p in old_dir.iterdir())
        old_file = old_dir / "manifest.json"
        if entries == ["manifest.json"] and json.loads(old_file.read_text()) == plan["old_doc"]:
            stale.append(old_dir)
        else:
            kept.append(old_dir)
    return publish, stale, kept, clashes


def plan_rewrites(root: Path, ids: dict[str, str], moves) -> list[tuple[Path, str]]:
    """Every (file at its post-move place, new text) under ``results/`` and ``.cortex/``."""
    candidates = _json_files(root / "results")
    pull = root / ".cortex" / "pull.json"
    if pull.is_file():
        candidates.append(pull)
    rewrites = []
    for path in candidates:
        text = path.read_text()
        new = rewrite_json_text(text, ids)
        if new != text:
            json.loads(new)  # a rewrite must never leave a file that does not parse
            rewrites.append((_moved(path, moves), new))
    return rewrites


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _output_is_contained(root: Path, output_root: Path) -> bool:
    """True when ``<root>/output`` IS ``output_root`` and resolves inside the checkout — the
    only layout ``prepared.verify_manifest``'s containment check accepts."""
    linked = root / "output"
    return (
        not linked.is_symlink()
        and linked.resolve() == output_root.resolve()
        and linked.resolve().is_relative_to(root.resolve())
    )


def verify(root: Path, output_root: Path, manifest_path: Path) -> None:
    """Fail closed on a rewritten manifest.

    Where the output tree sits inside the checkout (RAL; the tests) this IS
    ``prepared.verify_manifest``. Where ``output/`` is a symlink out of the checkout (the
    laptop mirror) ``verify_manifest`` refuses every artifact as escaping project storage
    whatever its content, so the same checks run with the ``output/`` prefix mapped onto
    ``output_root``: each artifact's sha256 via ``prepared.digest``, the id via
    ``prepared.manifest_problem_id``, and the three derived ids.
    """
    rel = manifest_path.relative_to(output_root)
    if _output_is_contained(root, output_root):
        prepared.verify_manifest(root, Path("output") / rel)
        return
    doc = json.loads(manifest_path.read_text())
    hashes = {}
    for item in doc["artifacts"]:
        parts = Path(item["path"]).parts
        if not parts or parts[0] != "output" or ".." in parts:
            raise ValueError(f"Artifact outside output/: {item['path']}")
        data = output_root.joinpath(*parts[1:]).read_bytes()
        if prepared.digest(data) != item["sha256"]:
            raise ValueError(f"Artifact digest mismatch: {item['path']}")
        if item["kind"] in hashes:
            raise ValueError("Duplicate prepared artifact kind")
        hashes[item["kind"]] = item["sha256"]
    if doc["id"] != prepared.manifest_problem_id(doc, hashes):
        raise ValueError("Prepared scientific identity mismatch")
    for key, kind in (
        ("dataset_id", "dataset_snapshot"),
        ("model_id", "model_definition"),
        ("priors_id", "priors_definition"),
    ):
        if doc[key] != "sha256:" + hashes[kind]:
            raise ValueError(f"Prepared {key} mismatch")


# ---------------------------------------------------------------------------
# Aside
# ---------------------------------------------------------------------------


def _is_aside_config(name: str) -> bool:
    return any(name == c or name.startswith(c + "__") for c in ASIDE_CONFIGS)


def plan_aside(root: Path, output_root: Path, dest: Path, label: str) -> list[tuple[Path, Path]]:
    moves = []
    autolens = output_root / "autolens"
    for path, _ in _walk_dirs(autolens, OUTPUT_DEPTH):
        rel = path.relative_to(autolens).parts
        slam_config = len(rel) == 5 and rel[2] == "slam"
        prepared_config = len(rel) == 2 and rel[0] == "prepared"
        if (slam_config or prepared_config) and _is_aside_config(path.name):
            moves.append((path, dest / "autolens" / Path(*rel)))
    archive = root / "results" / "archive" / label
    slam = root / "results" / "slam"
    for path, _ in _walk_dirs(slam, 4):
        rel = path.relative_to(slam).parts
        if len(rel) == 4 and _is_aside_config(path.name):
            moves.append((path, archive / "slam" / Path(*rel)))
    for problem in _subdirs(root / "prepared"):
        manifest = problem / "manifest.json"
        if manifest.is_file():
            controls = json.loads(manifest.read_text()).get("controls") or {}
            if _is_aside_config(str(controls.get("config_name"))):
                moves.append((problem, archive / "prepared" / problem.name))
    return moves


def _device(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return path.stat().st_dev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _refuse(conflicts: list[str]) -> int:
    for line in conflicts:
        print(f"  REFUSE   {line}", file=sys.stderr)
    print("Nothing changed: resolve the above first.", file=sys.stderr)
    return 2


def _move_all(moves) -> None:
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)


def _target_conflicts(moves) -> list[str]:
    targets = [target for _, target in moves]
    conflicts = [f"target already exists: {t}" for t in targets if t.exists()]
    conflicts += [
        f"two moves share a target: {t}" for t in sorted(set(targets)) if targets.count(t) > 1
    ]
    return conflicts


def run_aside(args, root: Path, output_root: Path) -> int:
    if args.aside_dest is None:
        print("ERROR: --aside needs --aside-dest", file=sys.stderr)
        return 2
    dest = args.aside_dest.resolve()
    moves = plan_aside(root, output_root, dest, args.aside_label)
    conflicts = _target_conflicts(moves)
    conflicts += [
        f"cross-filesystem move (os.rename would fail): {s} -> {t}"
        for s, t in moves
        if _device(s) != _device(t.parent)
    ]
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"rename_light_fixed --aside [{mode}]  root={root}  output-root={output_root}")
    print(f"  aside-dest={dest}  rows -> results/archive/{args.aside_label}/")
    for source, target in moves:
        print(f"  aside    {source}\n        -> {target}")
    if conflicts:
        return _refuse(conflicts)
    if not moves:
        print("Nothing to do: no superseded numba legs left in place.")
        return 0
    if not args.apply:
        print(f"{len(moves)} move(s) planned; --apply to act.")
        return 0
    _move_all(moves)
    print(f"Done: {len(moves)} move(s) aside.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="the checkout (results/, prepared/)"
    )
    parser.add_argument(
        "--output-root", type=Path, default=None, help="the PyAutoFit output root (<root>/output)"
    )
    parser.add_argument("--apply", action="store_true", help="act; without it, plan only")
    parser.add_argument(
        "--aside",
        action="store_true",
        help=f"set the superseded {', '.join(ASIDE_CONFIGS)} legs aside instead of renaming",
    )
    parser.add_argument(
        "--aside-dest", type=Path, default=None, help="--aside: where the output trees go"
    )
    parser.add_argument(
        "--aside-label",
        default=ASIDE_LABEL,
        help=f"--aside: the results/archive/<label>/ the rows go to (default {ASIDE_LABEL})",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_root = (args.output_root or root / "output").resolve()

    if args.aside:
        return run_aside(args, root, output_root)

    moves, odd = plan_renames(root, output_root)
    manifests = plan_manifests(output_root, moves)
    ids = {m["old_doc"]["id"]: m["new_doc"]["id"] for m in manifests}
    publish, stale, kept, clashes = plan_publish(root, manifests)
    rewrites = plan_rewrites(root, ids, moves)

    conflicts = _target_conflicts(moves)
    conflicts += [f"published manifest differs: {c}" for c in clashes]

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"rename_light_fixed [{mode}]  root={root}  output-root={output_root}")
    for source, target in moves:
        print(f"  move     {source}\n        -> {target}")
    for plan in manifests:
        print(
            f"  manifest {plan['target']}\n"
            f"           id {plan['old_doc']['id']}\n"
            f"           -> {plan['new_doc']['id']}"
        )
    for doc in publish:
        print(f"  publish  {_published_dir(root, doc['id']) / 'manifest.json'}")
    for directory in stale:
        print(f"  remove   {directory}  (stale published copy of a rewritten manifest)")
    for directory in kept:
        print(f"  KEEP     {directory}  (old published id, not an exact stale copy: left alone)")
    for path, _ in rewrites:
        print(f"  rewrite  {path}")
    for path in odd:
        print(f"  NOTE     names 'lightfixed' mid-name, not renamed: {path}")
    if conflicts:
        return _refuse(conflicts)
    if not moves and not manifests and not rewrites:
        print("Nothing to do: no _lightfixed names left.")
        return 0
    if not args.apply:
        print(
            f"{len(moves)} move(s), {len(manifests)} manifest(s), {len(rewrites)} rewrite(s), "
            f"{len(stale)} stale published copy(ies) planned; --apply to act."
        )
        return 0

    _move_all(moves)
    for plan in manifests:
        plan["target"].write_text(plan["text"])
    for doc in publish:
        prepared.publish_manifest(root, doc)
    for directory in stale:
        shutil.rmtree(directory)
    for path, text in rewrites:
        path.write_text(text)

    failed = []
    for plan in manifests:
        try:
            verify(root, output_root, plan["target"])
        except Exception as exc:  # report every failure, then fail
            failed.append(f"{plan['target']}: {exc}")
    print(
        f"Done: {len(moves)} move(s), {len(manifests)} manifest(s) re-identified and "
        f"published, {len(stale)} stale published copy(ies) removed, "
        f"{len(rewrites)} rewrite(s)."
    )
    if failed:
        for line in failed:
            print(f"  VERIFY FAILED {line}", file=sys.stderr)
        return 1
    print(f"Verified {len(manifests)} rewritten manifest(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
