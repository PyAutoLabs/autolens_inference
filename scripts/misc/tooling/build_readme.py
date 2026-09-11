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

Multi-stage payloads
--------------------

A pipeline result JSON is **one file per (config_name, seed) holding a ``stages``
list**, not one file per stage — a SLaM leg is a chain, and splitting it across
five files loses the fact that they share one dataset, one seed and one process.
:func:`_scan_rows` flattens such a payload into one row per stage, each carrying
the payload's header fields (``target``, ``config_name``, ``backend``, ``seed``,
``version``, …) plus that stage's own (``stage``, ``wall_s``, ``log_evidence``,
``likelihood_evals``, ``completed``, ``posterior``, …). Every renderer therefore
keeps seeing flat rows.

On top of the flat table :func:`render_slam` renders a **parity view** per
(target, seed): the ``mass_total[1]`` posterior of every config side by side,
each cell the difference from the alphabetically-first config in units of that
reference leg's 1σ. The two legs fit the same data with the same seed, so their
errors are heavily correlated and adding them in quadrature would overstate the
denominator; the reference σ is the honest scale for "do these two backends
agree".
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


#: Stage fields lifted onto a flattened row. A stage key that is absent stays
#: absent, so the renderer's em-dash fallback still applies.
_STAGE_FIELDS = (
    "wall_s",
    "total_wall_s",
    "compile_s",
    "likelihood_evals",
    "log_evidence",
    "log_evidence_err",
    "max_log_likelihood",
    "free_parameters",
    "n_live",
    "n_batch",
    "completed",
    "resumed",
    "positions_info_present",
    "inversion_applies",
    "posterior",
    "truth_delta_sigma",
)


def _flatten_stages(payload: dict) -> list[dict]:
    """One row per entry of a payload's ``stages`` list.

    Each row is the payload header (everything but ``stages``) plus that stage's
    fields, with the stage's ``name`` promoted to ``stage`` — the key every
    renderer already reads. The raw stage dict is kept on ``_stage`` for the
    parity view, which needs the nested ``posterior``.
    """
    header = {key: value for key, value in payload.items() if key != "stages"}
    rows = []
    for stage in payload["stages"]:
        if not isinstance(stage, dict):
            continue
        row = dict(header)
        row["stage"] = stage.get("name")
        for key in _STAGE_FIELDS:
            if key in stage:
                row[key] = stage[key]
        row["_stage"] = stage
        rows.append(row)
    return rows


def _scan_rows(root: Path) -> list[dict]:
    """Every result row under ``root``, sorted for a stable render.

    A payload carrying a ``stages`` list is flattened into one row per stage
    (see :func:`_flatten_stages`); anything else is one row as it stands.

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
        relative = str(path.relative_to(RESULTS_ROOT))
        if isinstance(payload.get("stages"), list):
            for row in _flatten_stages(payload):
                row.setdefault("_path", relative)
                rows.append(row)
            continue
        payload.setdefault("_path", relative)
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


#: The stage the parity view reads: the headline `PowerLaw` mass model, which is
#: the only stage whose posterior every leg of a parity row must agree on.
PARITY_STAGE = "mass_total[1]"

#: Parameters the parity view leads with, in this order. Everything else follows
#: alphabetically. These three are what a lensing result is quoted as, so they go
#: at the top rather than wherever the alphabet puts them.
PARITY_LEADING_KEYS = ("einstein_radius", "slope", "shear_magnitude")

#: Chain order of the SLaM stages. A table sorted alphabetically would put
#: `light[1]` before `source_lp[1]` and read as if the pipeline ran backwards.
#: A stage not listed here sorts after every listed one, by name.
SLAM_STAGE_ORDER = (
    "source_lp[1]",
    "source_pix[1]",
    "source_pix[2]",
    "light[1]",
    "mass_total[1]",
)


def _stage_sort_key(row: dict) -> tuple:
    stage = str(row.get("stage", ""))
    try:
        index = SLAM_STAGE_ORDER.index(stage)
    except ValueError:
        index = len(SLAM_STAGE_ORDER)
    return (
        str(row.get("target", "")),
        str(row.get("seed", "")),
        index,
        stage,
        str(row.get("config_name", "")),
    )


def _format_evals(value) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    return f"{int(value):,}"


def _format_sigma(value) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    if math.isnan(value) or math.isinf(value):
        return "—"
    return f"{value:+.2f}σ"


def _format_median(entry) -> str:
    if not isinstance(entry, dict):
        return "—"
    median = entry.get("median")
    sigma = entry.get("sigma")
    if not isinstance(median, (int, float)):
        return "—"
    if not isinstance(sigma, (int, float)):
        return f"{median:.4g}"
    return f"{median:.4g} ± {sigma:.2g}"


def _parity_key_order(keys) -> list[str]:
    leading = [key for key in PARITY_LEADING_KEYS if key in keys]
    rest = sorted(key for key in keys if key not in PARITY_LEADING_KEYS)
    return leading + rest


def _parity_delta_sigma(entry, reference) -> float | None:
    """``(median - reference median) / reference sigma``.

    The reference leg's σ is the denominator, not the two added in quadrature:
    the legs fit the same data at the same seed, so their errors are heavily
    correlated and quadrature would flatter every disagreement.
    """
    if not isinstance(entry, dict) or not isinstance(reference, dict):
        return None
    median = entry.get("median")
    reference_median = reference.get("median")
    sigma = reference.get("sigma")
    if not isinstance(median, (int, float)) or not isinstance(reference_median, (int, float)):
        return None
    if not isinstance(sigma, (int, float)) or not sigma:
        return None
    return (median - reference_median) / sigma


def _render_parity(rows: list[dict]) -> str:
    """The parity view: one block per (target, seed), configs as columns."""
    groups: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("stage") != PARITY_STAGE:
            continue
        posterior = (row.get("_stage") or {}).get("posterior") or row.get("posterior") or {}
        if not posterior:
            continue
        key = (str(row.get("target", "")), str(row.get("seed", "")))
        groups.setdefault(key, {})[str(row.get("config_name"))] = posterior

    if not groups:
        return ""

    blocks: list[str] = []
    for (target, seed), by_config in sorted(groups.items()):
        configs = sorted(by_config)
        heading = f"\n**Parity — `{target}` (seed {seed}), `{PARITY_STAGE}`**\n"
        if len(configs) < 2:
            blocks.append(
                heading + "\n_Only one config has run for this target and seed — "
                "a parity view needs at least two legs._\n"
            )
            continue

        reference = configs[0]
        reference_posterior = by_config[reference]
        keys = _parity_key_order({k for posterior in by_config.values() for k in posterior})

        headers = ["Parameter", f"`{reference}` (ref)"] + [f"`{c}`" for c in configs[1:]]
        body = []
        for key in keys:
            line = [f"`{key}`", _format_median(reference_posterior.get(key))]
            for config in configs[1:]:
                line.append(
                    _format_sigma(
                        _parity_delta_sigma(
                            by_config[config].get(key), reference_posterior.get(key)
                        )
                    )
                )
            body.append(line)
        blocks.append(heading + _render_table(headers, body))

    return "".join(blocks)


def render_slam() -> str:
    """Pipeline runs — one line per (target, stage, config), plus the parity view."""
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
            f"`{_cell(row.get('config_name'))}`",
            _cell(row.get("seed")),
            _format_time(row.get("wall_s")),
            _format_evals(row.get("likelihood_evals")),
            _format_evidence(row.get("log_evidence")),
            _cell(row.get("version")),
        ]
        for row in sorted(rows, key=_stage_sort_key)
    ]
    table = _render_table(
        ["Target", "Stage", "Config", "Seed", "Wall", "Evals", "log Z", "Version"],
        body,
    )
    return table + _render_parity(rows)


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
