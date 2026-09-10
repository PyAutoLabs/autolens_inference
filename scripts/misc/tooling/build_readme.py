"""build_readme.py — refresh the auto-generated tables in every README from the
result rows under `results/`.

Run from the repo root::

    python scripts/misc/tooling/build_readme.py         # rewrite README tables in place
    python scripts/misc/tooling/build_readme.py --check # exit non-zero if rewriting
                                                        # would change any file (CI gate)

Each table region in a README is delimited by sentinel comments::

    <!-- BEGIN auto-table:slam -->
    | ... |
    <!-- END auto-table:slam -->

Only the content between BEGIN and END is rewritten; the prose around it is left
exactly as it is. An unknown sentinel is left intact and warned about rather than
blanked — a renderer that has not been written yet must not eat someone's table.

Regions covered today
---------------------

``slam``
    Pipeline runs, scanned from ``results/slam/**/*.json``.

``searches``
    Single-search runs, scanned from ``results/searches/**/*.json``.

Both render a one-line empty state when nothing has been run yet, which is the
state this repo was born in — the renderer is deliberately correct on an empty
tree, so `--check` is a real gate from the first commit rather than something
that only starts working once results exist.

Row shape
---------

A result row is a JSON object. These keys are read when present; anything else in
the payload is ignored, and a row missing a key renders an em dash rather than
failing the build:

    target          the thing being fitted — dataset + model + pipeline
    config_name     {local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}
    instrument      hst / euclid / sma / ...
    sampler         nautilus / prodigy / nuts / ... (searches only)
    stage           pipeline stage name (slam only)
    wall_s          total wall-clock seconds
    log_evidence    the search's log evidence, when it reports one
    version         the PyAutoLens version that produced the row

Rows sharing a ``target`` and differing only in ``config_name`` are one **parity
row**, and they are rendered as adjacent lines of one table on purpose. The
backend is a column, never a reason to split a table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

RESULTS_ROOT = REPO_ROOT / "results"
SLAM_ROOT = RESULTS_ROOT / "slam"
SEARCHES_ROOT = RESULTS_ROOT / "searches"

# Sentinel block: keeps surrounding hand-written prose intact, only the
# content between BEGIN and END is rewritten.
SENTINEL_RE = re.compile(
    r"(<!-- BEGIN auto-table:(?P<name>[a-z0-9_\-]+) -->)"
    r".*?"
    r"(<!-- END auto-table:(?P=name) -->)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _scan_rows(root: Path) -> list[dict]:
    """Every result row under ``root``, sorted for a stable render.

    A file that is not JSON, or whose top level is not an object, is skipped
    with a warning rather than failing the build: a half-written row pulled off
    RAL mid-transfer must not break the dashboard for every other row.
    """
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARNING: skipping {path.relative_to(REPO_ROOT)} — {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, dict):
            print(
                f"  WARNING: skipping {path.relative_to(REPO_ROOT)} — top level is not an object",
                file=sys.stderr,
            )
            continue
        payload = dict(payload)
        payload.setdefault("_path", str(path.relative_to(RESULTS_ROOT)))
        rows.append(payload)
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _empty(message: str) -> str:
    return f"\n_{message}_\n"


def _cell(value) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _format_time(seconds) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return "—"
    if math.isnan(seconds):
        return "—"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 3600:
        return f"{seconds:.1f} s"
    return f"{seconds / 3600:.2f} h"


def _format_evidence(value) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    if math.isnan(value):
        return "—"
    return f"{value:,.2f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n" + "\n".join(lines) + "\n"


def _sort_key(row: dict, *extra: str) -> tuple:
    return tuple(str(row.get(key, "")) for key in ("target", *extra, "config_name"))


def render_slam() -> str:
    """Pipeline runs — one line per (target, stage, config)."""
    rows = _scan_rows(SLAM_ROOT)
    if not rows:
        return _empty(
            "No pipeline runs yet — the first is the HST SLaM base run "
            "(`results/slam/`, phase 4 of the autolens-inference epic)."
        )
    body = [
        [
            f"`{_cell(row.get('target'))}`",
            _cell(row.get("stage")),
            _cell(row.get("instrument")),
            f"`{_cell(row.get('config_name'))}`",
            _format_time(row.get("wall_s")),
            _format_evidence(row.get("log_evidence")),
            _cell(row.get("version")),
        ]
        for row in sorted(rows, key=lambda r: _sort_key(r, "stage"))
    ]
    return _render_table(
        ["Target", "Stage", "Instrument", "Config", "Wall", "Log evidence", "Version"],
        body,
    )


def render_searches() -> str:
    """Single-search runs — one line per (target, sampler, config)."""
    rows = _scan_rows(SEARCHES_ROOT)
    if not rows:
        return _empty("No search runs yet — results land under `results/searches/`.")
    body = [
        [
            f"`{_cell(row.get('target'))}`",
            _cell(row.get("sampler")),
            _cell(row.get("instrument")),
            f"`{_cell(row.get('config_name'))}`",
            _format_time(row.get("wall_s")),
            _format_evidence(row.get("log_evidence")),
            _cell(row.get("version")),
        ]
        for row in sorted(rows, key=lambda r: _sort_key(r, "sampler"))
    ]
    return _render_table(
        ["Target", "Sampler", "Instrument", "Config", "Wall", "Log evidence", "Version"],
        body,
    )


RENDERERS = {
    "slam": render_slam,
    "searches": render_searches,
}


# Files that may contain auto-table regions. Listed explicitly rather than
# walked, so the script's surface stays obvious.
TARGET_READMES = [
    REPO_ROOT / "README.md",
]


def _rewrite_file(path: Path, renderers: dict) -> tuple[str, str, list[str]]:
    """Return (original_text, rewritten_text, unknown_sentinels)."""
    original = path.read_text()
    unknown: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group("name")
        begin = match.group(1)
        end = match.group(3)
        renderer = renderers.get(name)
        if renderer is None:
            unknown.append(name)
            return match.group(0)  # leave intact
        return f"{begin}{renderer()}{end}"

    rewritten = SENTINEL_RE.sub(replace, original)
    return original, rewritten, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any target file would be rewritten (CI gate).",
    )
    args = parser.parse_args(argv)

    n_slam = len(_scan_rows(SLAM_ROOT))
    n_searches = len(_scan_rows(SEARCHES_ROOT))
    print(f"Scanned {n_slam} pipeline row(s) and {n_searches} search row(s) under {RESULTS_ROOT}")

    any_changed = False
    all_unknown: list[tuple[Path, str]] = []
    for target in TARGET_READMES:
        if not target.exists():
            print(f"  skip      {target.relative_to(REPO_ROOT)} — not present", flush=True)
            continue
        original, rewritten, unknown = _rewrite_file(target, RENDERERS)
        for name in unknown:
            all_unknown.append((target, name))
        if rewritten == original:
            print(f"  unchanged {target.relative_to(REPO_ROOT)}", flush=True)
            continue
        any_changed = True
        if args.check:
            print(f"  WOULD rewrite {target.relative_to(REPO_ROOT)}", flush=True)
        else:
            target.write_text(rewritten)
            print(f"  rewrote   {target.relative_to(REPO_ROOT)}", flush=True)

    for path, name in all_unknown:
        print(
            f"WARNING: unknown sentinel '{name}' in {path.relative_to(REPO_ROOT)} — left intact",
            file=sys.stderr,
        )

    if args.check and any_changed:
        print(
            "ERROR: `build_readme.py --check` found pending changes. Run "
            "`python scripts/misc/tooling/build_readme.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
