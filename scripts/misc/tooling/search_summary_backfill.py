"""Backfill the search.summary timings of committed SLaM validation stage rows.

Rows written before ``_stage_row`` recorded ``search_summary`` carry only the
absolute ``output_path`` of each stage's identifier directory on the machine that
ran it. This tool maps each one into a local copy of the run output tree
(``--output-root``, the directory that holds ``autolens/...``), reads the stage's
``search.summary`` with the same parser the runner uses, and writes the committed
sidecar ``catalogue/search_summary_backfill.json``, keyed by the row's repo path and
the stage name (the identity ``export_inference_summary.py`` uses).

Resolution, per stage:

1. the recorded identifier directory, when it exists locally with a summary;
2. else the stage directory's identifier directories that hold a summary: one is
   used; several are used only when every candidate agrees on both timings, and
   are otherwise refused as ambiguous;
3. else the stage is listed with null timings and a reason.

Stdlib only; it never touches the output tree. ``--check`` regenerates against
``--output-root`` and fails when the committed sidecar differs (a local-only
check: CI has no output tree).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from _inference_cli import SEARCH_SUMMARY_LINES, search_summary_from  # noqa: E402

SIDECAR = "catalogue/search_summary_backfill.json"

#: The committed rows this tool backfills: the SLaM validation chains, which are
#: the rows ``walltime_validation.json`` is built from.
ROW_GLOB = "results/**/validation/**/stages.json"


def output_relative(output_path) -> PurePosixPath | None:
    """The part of a recorded ``output_path`` below its ``output/`` directory."""
    if not isinstance(output_path, str) or "/output/" not in f"/{output_path}":
        return None
    tail = f"/{output_path}".rsplit("/output/", 1)[1]
    parts = PurePosixPath(tail).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        return None
    return PurePosixPath(*parts)


def _timings(block):
    return tuple(block[field] for field in SEARCH_SUMMARY_LINES)


def resolve_stage(stage: dict, output_root: Path) -> dict:
    """The sidecar entry of one stage row (see the module docstring)."""
    rel = output_relative(stage.get("output_path"))
    entry = {field: None for field in SEARCH_SUMMARY_LINES}
    entry.update(path=None, resolution=None)
    if rel is None:
        entry["reason"] = "Stage row records no output_path below an output/ directory"
        return entry
    recorded = output_root / rel
    label = f"output/{rel.as_posix()}/search.summary"
    if (recorded / "search.summary").is_file():
        block = search_summary_from(recorded, path_label=label)
        entry.update(block, resolution="recorded_identifier")
        return entry
    stage_dir = recorded.parent
    candidates = (
        sorted(p for p in stage_dir.iterdir() if (p / "search.summary").is_file())
        if stage_dir.is_dir()
        else []
    )
    if not candidates:
        entry["reason"] = (
            "No search.summary under the stage directory in the output tree"
            if stage_dir.is_dir()
            else "Stage directory absent from the output tree"
        )
        return entry
    blocks = [
        search_summary_from(c, path_label=f"output/{rel.parent.as_posix()}/{c.name}/search.summary")
        for c in candidates
    ]
    if len({_timings(b) for b in blocks}) > 1:
        entry["reason"] = (
            f"Ambiguous: {len(blocks)} identifier directories under the stage hold a "
            "search.summary with different timings and none is the recorded one"
        )
        entry["candidates"] = [b["path"] for b in blocks]
        return entry
    entry.update(blocks[0])
    entry["resolution"] = (
        "sole_candidate" if len(blocks) == 1 else f"agreeing_candidates:{len(blocks)}"
    )
    return entry


def build(output_root: Path, root: Path = ROOT) -> dict:
    rows = {}
    for path in sorted(root.glob(ROW_GLOB)):
        payload = json.loads(path.read_text())
        stages = {}
        for stage in payload.get("stages") or []:
            if isinstance(stage, dict) and isinstance(stage.get("name"), str):
                stages[stage["name"]] = resolve_stage(stage, output_root)
        rows[path.relative_to(root).as_posix()] = stages
    entries = [e for stages in rows.values() for e in stages.values()]
    found = sum(1 for e in entries if any(e[f] is not None for f in SEARCH_SUMMARY_LINES))
    return {
        "schema": "search-summary-backfill",
        "version": 1,
        "description": (
            "search.summary timings (seconds) of committed SLaM validation stage rows "
            "that predate the row's own search_summary field, read from a local copy "
            "of the run output tree; keyed by row path then stage name"
        ),
        "fields": dict(SEARCH_SUMMARY_LINES),
        "coverage": {"stages": len(entries), "with_search_summary": found},
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", required=True, help="the run output tree (holds autolens/...)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help=f"write {SIDECAR}")
    mode.add_argument("--check", action="store_true", help=f"fail when {SIDECAR} is stale")
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    if not (output_root / "autolens").is_dir():
        raise SystemExit(f"{output_root} holds no autolens/ run tree")
    doc = build(output_root)
    content = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = ROOT / SIDECAR
    cov = doc["coverage"]
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit(f"{SIDECAR} is stale; rerun with --write")
        print(f"{SIDECAR} current: {cov['with_search_summary']}/{cov['stages']} stages")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    print(f"wrote {SIDECAR}: {cov['with_search_summary']}/{cov['stages']} stages")
    for row, stages in doc["rows"].items():
        for name, entry in stages.items():
            if entry.get("reason"):
                print(f"  {row} {name}: {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
