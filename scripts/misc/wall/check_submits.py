"""Gate: every HPC submit's `--time` must be justified per cell, from measured data.

Run from the repo root::

    python scripts/misc/wall/check_submits.py           # report on every submit
    python scripts/misc/wall/check_submits.py --check   # exit non-zero on any violation

Why
---

A submit that justified ``--time=0:30:00`` with an **MGE** step rate, for an
array whose arms were mostly pixelized, had 35 of its 39 arms killed at ~12% of
budget — an overnight A100 block lost. See `wall/rates.py` for the rule and
`wall/README.md` for the authoring contract.

This checker's central rule is the one that would have caught it:

    **Every cell a submit actually runs must have its own basis row.**

A parametric row cannot cover a pixelized arm, however plausible the prose
around it.

The WALL-BASIS block
--------------------

A comment block anywhere in the submit's header, one row per cell::

    # WALL-BASIS:
    #   cell: imaging/slam_hst_base/hst  device: a100  precision: fp64
    #   lanes: 1  steps: 3000  source: unmeasured  probe-first: yes

A row starts at its ``cell:`` key and runs to the next ``cell:`` or the end of
the block. Three kinds of basis are accepted, and each says plainly how much
the author actually knows:

``source: rates``
    The honest case. ``rate`` must match `wall.rates.STEP_RATE` for this exact
    (cell, instrument, device, precision, lanes, batch_size) within 5%.
    Estimated wall = ``rate * steps + compile``.

``source: measured-wall``
    A directly observed total for this cell — a sampler fork row, or previous
    runs of this same arm. Requires ``wall:`` (seconds) and ``ref:`` naming
    where it was observed.

``source: unmeasured``
    Nothing has been measured on this cell. Permitted — inventing a number to
    satisfy a linter is the disease, not the cure — but it requires
    ``probe-first: yes``, and any ``wall:`` it does offer carries a 3x floor.
    The row earns its place by forcing the author to state, per cell, that this
    cell's wall clock rests on nothing. **This is the normal case in this repo
    today**: `wall/rates.py` starts empty and inherits nothing, so until a rate
    is measured here every submit says so out loud. The honest next step is to
    run one short arm: a truncated arm still measures s/step.

Which submits must carry one
----------------------------

Every ``submit_*`` under ``hpc/batch_{gpu,cpu}/`` must carry one, and the
``template`` in each of those directories is checked too — the template is what
every future submit is copied from, so a template without a block seeds the
defect into every child. This is a path predicate, not a hand-maintained
allowlist: no submit is individually exempted, because an exemption list would
hide exactly the class of leak this gate exists to close.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()


def _inference_root() -> Path:
    for parent in _HERE.parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_inference root (ruff.toml) not found")


ROOT = _inference_root()
if str(ROOT / "scripts" / "misc") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from wall.rates import STEP_RATE, UnmeasuredCellError, step_rate_for, wall_estimate  # noqa: E402

# Files under hpc/batch_*/ required to carry a WALL-BASIS block. Every submit,
# plus the `template` each new submit is copied from — a template without a
# block seeds the defect into every child.
REQUIRED_PREFIXES = ("submit_", "template")

# Tolerance on a `source: rates` row against the table.
RATE_TOLERANCE = 0.05

# Minimum headroom multiplier per basis kind — the less direct the evidence,
# the wider the margin the submit must leave.
HEADROOM_FLOOR = {
    "measured-wall": 1.25,  # a directly observed total for this same cell
    "rates": 1.5,  # a measured s/step for this cell, times a step count
    "unmeasured": 3.0,  # nothing measured; only a declared guess
}
DEFAULT_HEADROOM = HEADROOM_FLOOR["rates"]

_KV = re.compile(r"([A-Za-z][\w-]*):\s*(\S+)")
_PYTHON_CALL = re.compile(r"python3?\s+(scripts/[\w./${}\[\]-]+\.py)")
_INSTRUMENT = re.compile(r"--instrument\s+(\S+)")
_VAR_REF = re.compile(r"^\$\{?(\w+)(?:\[[^\]]*\])?\}?$")


class Problem(str):
    """One human-readable violation, rendered under its file."""


# --------------------------------------------------------------------------
# SLURM + bash parsing
# --------------------------------------------------------------------------


def parse_slurm_time(value: str) -> float:
    """SLURM ``--time`` to seconds.

    Accepts the formats SLURM documents: ``M``, ``M:S``, ``H:M:S``, ``D-H``,
    ``D-H:M`` and ``D-H:M:S``.
    """
    days = 0
    rest = value.strip()
    if "-" in rest:
        head, rest = rest.split("-", 1)
        days = int(head)
    parts = [int(p) for p in rest.split(":")]
    if days:
        # D-H, D-H:M, D-H:M:S
        h, m, s = (parts + [0, 0])[:3]
    elif len(parts) == 1:
        h, m, s = 0, parts[0], 0  # bare minutes
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]  # M:S
    else:
        h, m, s = parts[:3]
    return days * 86400 + h * 3600 + m * 60 + s


def resolve_var(text: str, name: str, _seen: frozenset[str] = frozenset()) -> set[str]:
    """Every literal value a bash variable is ever assigned in this script.

    Handles the three shapes the submits use: a ``VAR=(a b c)`` array, a plain
    ``VAR=value`` (including the ``case`` arms), and the ``VAR=${OTHER[$I]}``
    indirection that array submits use to index the arm tables. Returns an
    empty set when nothing resolves, which callers treat as "unknown", never as
    "no cells".
    """
    if name in _seen:
        return set()
    seen = _seen | {name}
    values: set[str] = set()

    for match in re.finditer(rf"^\s*(?:export\s+)?{re.escape(name)}=\((.*?)\)", text, re.M | re.S):
        values.update(tok for tok in match.group(1).split() if tok and not tok.startswith("#"))

    for match in re.finditer(rf"(?:^|\s)(?:export\s+)?{re.escape(name)}=(\S+)", text, re.M):
        raw = match.group(1)
        if raw.startswith("("):
            continue
        ref = _VAR_REF.match(raw)
        if ref:
            values.update(resolve_var(text, ref.group(1), seen))
        elif re.fullmatch(r"[\w.\-/]+", raw):
            values.add(raw)

    return values


def _expand(text: str, token: str) -> set[str]:
    """A literal token, or every value the variable it names can take."""
    ref = _VAR_REF.match(token)
    if ref:
        return resolve_var(text, ref.group(1))
    return {token} if re.fullmatch(r"[\w.\-]+", token) else set()


def strip_comments(text: str) -> str:
    """The script with whole-line comments blanked out.

    Submit headers routinely *mention* commands the job does not run — a header
    citing the scoring step it feeds, or a template whose only "invocation" is a
    commented-out placeholder. Reading cells out of prose would credit the
    submit with a cell it never executes, so only executable lines are scanned.
    """
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.splitlines())


def cells_run(text: str) -> tuple[set[tuple[str, str]], set[str]]:
    """The ``(dataset, cell)`` pairs a submit actually runs, and its instruments.

    Read from the ``python3 scripts/<dataset>/.../<cell>.py`` invocations,
    resolving ``${CELL}``-style paths through `resolve_var`. This is the half
    the author cannot fudge: it is what the job will really execute, and it is
    what each WALL-BASIS row is checked against.
    """
    text = strip_comments(text)
    cells: set[tuple[str, str]] = set()
    for match in _PYTHON_CALL.finditer(text):
        parts = match.group(1).split("/")
        if len(parts) < 3:
            continue
        dataset = parts[1]
        stem = parts[-1][: -len(".py")]
        for cell in _expand(text, stem) or {stem}:
            if not _VAR_REF.match(cell):
                cells.add((dataset, cell))

    instruments: set[str] = set()
    for match in _INSTRUMENT.finditer(text):
        instruments.update(_expand(text, match.group(1)))

    return cells, instruments


# --------------------------------------------------------------------------
# WALL-BASIS block
# --------------------------------------------------------------------------


def parse_basis_rows(text: str) -> list[dict[str, str]]:
    """Rows from the ``# WALL-BASIS:`` block, each a flat key->value dict."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("# WALL-BASIS:"))
    except StopIteration:
        return []

    rows: list[dict[str, str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("#") or stripped in {"#", "#!"}:
            break
        if stripped.startswith("#SBATCH"):
            break
        pairs = _KV.findall(stripped)
        if not pairs:
            continue  # a prose line inside the block — explanation, not data
        if pairs[0][0] == "cell":
            rows.append({})
        if not rows:
            continue
        rows[-1].update(dict(pairs))
    return rows


def _row_wall(row: dict[str, str], where: str, problems: list[Problem]) -> float | None:
    """Estimated wall seconds for one row, or None if the row is malformed."""
    source = row.get("source")
    compile_s = float(row.get("compile", 0))

    if source == "rates":
        missing = [k for k in ("lanes", "steps", "rate", "device", "precision") if k not in row]
        if missing:
            problems.append(Problem(f"{where}: `source: rates` row missing {', '.join(missing)}"))
            return None
        dataset, cell, instrument = row["_cell_parts"]
        batch = row.get("batch_size")
        batch_size = None if batch in (None, "none", "None") else int(batch)
        try:
            table_rate = step_rate_for(
                dataset,
                cell,
                instrument,
                row["device"],
                row["precision"],
                int(row["lanes"]),
                batch_size,
            )
        except UnmeasuredCellError:
            problems.append(
                Problem(
                    f"{where}: `source: rates` but wall/rates.py has no measured row for "
                    f"{dataset}/{cell}/{instrument} at lanes={row['lanes']} "
                    f"batch_size={batch}. Measure this cell or declare `source: unmeasured` "
                    f"with `probe-first: yes` — do NOT cite another cell's rate."
                )
            )
            return None
        cited = float(row["rate"])
        if abs(cited - table_rate) > RATE_TOLERANCE * table_rate:
            problems.append(
                Problem(
                    f"{where}: cited rate {cited} s/step disagrees with wall/rates.py "
                    f"({table_rate} s/step) for {dataset}/{cell}/{instrument}"
                )
            )
            return None
        return wall_estimate(table_rate, int(row["steps"]), compile_s)

    if source == "measured-wall":
        if "wall" not in row or "ref" not in row:
            problems.append(
                Problem(f"{where}: `source: measured-wall` needs both `wall:` and `ref:`")
            )
            return None
        return float(row["wall"]) + compile_s

    if source == "unmeasured":
        if row.get("probe-first") != "yes":
            problems.append(
                Problem(
                    f"{where}: `source: unmeasured` needs `probe-first: yes` — run one short "
                    f"arm on this cell first; a truncated arm still measures s/step"
                )
            )
            return None
        if "wall" not in row:
            # Nothing measured and no guess offered. The row still did its job:
            # it forced the author to state, per cell, that this cell's wall
            # clock rests on nothing — which is what the confident prose concealed.
            # There is no estimate to check `--time` against, so don't invent one.
            return None
        return float(row["wall"]) + compile_s

    problems.append(
        Problem(f"{where}: unknown or missing `source:` (want rates | measured-wall | unmeasured)")
    )
    return None


def check_text(text: str, name: str) -> list[Problem]:
    """Every violation in one submit script."""
    problems: list[Problem] = []
    required = name.startswith(REQUIRED_PREFIXES)
    rows = parse_basis_rows(text)

    if not rows:
        if required:
            problems.append(
                Problem(
                    "no `# WALL-BASIS:` block. Every cell this submit runs needs its own "
                    "basis row — see wall/README.md"
                )
            )
        return problems

    time_match = re.search(r"^#SBATCH\s+--time=(\S+)", text, re.M)
    if not time_match:
        problems.append(Problem("WALL-BASIS block but no `#SBATCH --time=`"))
        return problems
    budget = parse_slurm_time(time_match.group(1))

    run_cells, run_instruments = cells_run(text)
    declared: set[tuple[str, str]] = set()

    for row in rows:
        cell_field = row.get("cell", "")
        parts = cell_field.split("/")
        if len(parts) != 3:
            problems.append(
                Problem(f"row `cell: {cell_field}` is not `<dataset>/<cell>/<instrument>`")
            )
            continue
        row["_cell_parts"] = parts  # type: ignore[assignment]
        dataset, cell, instrument = parts
        declared.add((dataset, cell))
        where = f"row {cell_field}"

        if run_cells and (dataset, cell) not in run_cells:
            problems.append(
                Problem(f"{where}: declared, but this submit never runs {dataset}/{cell}")
            )
        if run_instruments and instrument not in run_instruments:
            problems.append(
                Problem(
                    f"{where}: declared instrument `{instrument}` is not one this submit runs "
                    f"({', '.join(sorted(run_instruments))})"
                )
            )

        wall = _row_wall(row, where, problems)
        if wall is None:
            continue

        floor = HEADROOM_FLOOR.get(row.get("source", ""), DEFAULT_HEADROOM)
        headroom = float(row.get("headroom", floor))
        if headroom < floor:
            problems.append(
                Problem(f"{where}: headroom {headroom} is below the {floor} floor for this source")
            )
            headroom = floor
        needed = wall * headroom
        if budget < needed:
            problems.append(
                Problem(
                    f"{where}: --time={time_match.group(1)} ({budget:.0f} s) is below "
                    f"{headroom}x the estimated {wall:.0f} s wall ({needed:.0f} s needed) — "
                    f"this cell will be killed mid-run"
                )
            )

    # The rule that would have caught it: a cell the job runs with no row
    # of its own is a cell whose wall clock was justified by some other cell.
    for dataset, cell in sorted(run_cells - declared):
        problems.append(
            Problem(
                f"cell {dataset}/{cell} is RUN by this submit but has no WALL-BASIS row — "
                f"its --time is being justified by another cell's rate. This is the defect "
                f"that killed 35 of 39 arms in one overnight A100 block."
            )
        )

    return problems


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def submit_paths(root: Path) -> list[Path]:
    """Every file under hpc/batch_*/ that the contract applies to.

    Both the real submits and the `template` beside them: a template is the
    shape every future submit inherits, so it is checked exactly like one.
    """
    paths: list[Path] = []
    for sub in ("batch_gpu", "batch_cpu"):
        directory = root / "hpc" / sub
        if not directory.is_dir():
            continue
        paths.extend(sorted(directory.glob("submit_*")))
        template = directory / "template"
        if template.is_file():
            paths.append(template)
    return [p for p in paths if p.is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if any submit violates the contract"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root (default: autodetected)")
    args = parser.parse_args(argv)

    paths = submit_paths(args.root)
    failures = 0
    checked = 0

    for path in paths:
        text = path.read_text()
        problems = check_text(text, path.name)
        if path.name.startswith(REQUIRED_PREFIXES) or parse_basis_rows(text):
            checked += 1
        if problems:
            failures += 1
            print(f"\n{path.relative_to(args.root)}")
            for problem in problems:
                print(f"  - {problem}")

    print(
        f"\nwall check: {checked} submit(s) with a wall-basis contract, "
        f"{failures} failing, {len(paths)} scanned "
        f"({len(STEP_RATE)} measured rates in wall/rates.py)"
    )
    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
