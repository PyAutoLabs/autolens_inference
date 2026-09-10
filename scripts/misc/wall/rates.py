"""Per-cell measured step rates for HPC submit `--time` estimates.

Populated **only** from rates measured in this repo, on the cell the row names.
Nothing in this table is interpolated, extrapolated, carried over from a
neighbouring cell, or imported from another project — every one of those is the
defect this module exists to prevent.

The table starts EMPTY
----------------------

This repo was born 2026-09-10 and has measured nothing yet. It does not inherit
the retired `inference_programme`'s rate table, and it must not: those numbers
were measured on a different tree, by runs this project cannot name. Until a row
is measured here, every submit declares ``source: unmeasured`` with
``probe-first: yes`` — which is honest, and which the gate accepts.

The rule
--------

Never carry a ``--time`` justification across cells. Derive it from a measured
rate for *that* cell, and when no measurement exists, run one short arm first —
a 30-minute truncated arm still measures s/step.

Why the rule exists, in one sentence: a submit that justified ``--time=0:30:00``
with an MGE step rate, for an array whose arms were mostly pixelized, lost 35 of
its 39 arms at ~12% of budget and an overnight A100 block with them. Pixelized
cells are per-eval-inversion bound and parametric cells are not, and nothing
about a step rate is portable across that boundary however confident the prose
around the number.

`wall/check_submits.py` enforces this on every submit; `wall/README.md` states
the authoring contract.

Table shape
-----------

Keys are ``(dataset, cell, instrument, device, precision, n_lanes,
batch_size)``. ``batch_size`` is part of the key — not a detail — because an
unbatched lane row and a ``batch_size=4`` row are genuinely different
configurations of the same cell, and their rates can differ by more than 2x at
the same lane count. ``None`` means the arm ran unbatched.

Adding a row
------------

1. Measure it on the cell itself — a full arm, or a truncated one (steps
   completed / wall elapsed, from the arm's own log).
2. Add the row below with an inline comment naming the RAL job and the date.
3. Add the matching ``PROVENANCE`` entry: what ran, what configuration, and
   what the row must NOT be used for.
4. Point the submit's ``# WALL-BASIS:`` block at it with ``source: rates``.

A row that quotes a cell whose rate varies must quote its **slowest** measured
arm. An array submit is sized by its slowest cell, never its fastest.
"""

from __future__ import annotations

# Key: (dataset, cell, instrument, device, precision, n_lanes, batch_size)
# Value: seconds per step, measured on that exact configuration, in this repo.
STEP_RATE: dict[tuple[str, str, str, str, str, int, int | None], float] = {
    # Empty on purpose — see the module docstring. Nothing has been measured
    # here yet, and no rate is inherited from anywhere else.
}

#: One entry per key prefix in ``STEP_RATE``: what ran, on which job, and the
#: configuration the row is and is not valid for.
PROVENANCE: dict[str, str] = {}


class UnmeasuredCellError(KeyError):
    """No measured step rate exists for the requested configuration.

    Raised instead of returning a nearby cell's rate. Falling back across cells
    is the bug — see this module's docstring.
    """


def step_rate_for(
    dataset: str,
    cell: str,
    instrument: str,
    device: str,
    precision: str,
    n_lanes: int,
    batch_size: int | None = None,
) -> float:
    """Seconds per step for exactly this configuration.

    There is **no** nearest-neighbour fallback: an unmeasured configuration
    raises `UnmeasuredCellError` rather than silently answering with a rate
    measured on a different cell, lane count or batching.
    """
    key = (dataset, cell, instrument, device, precision, n_lanes, batch_size)
    try:
        return STEP_RATE[key]
    except KeyError:
        raise UnmeasuredCellError(
            f"no measured step rate for {key!r}. Do not substitute another cell's rate — "
            f"run one short arm on this cell and add the row to wall/rates.py, or declare "
            f"`source: unmeasured` with `probe-first: yes` in the submit's WALL-BASIS block."
        ) from None


def wall_estimate(rate_s_per_step: float, n_steps: int, compile_s: float = 0.0) -> float:
    """Estimated wall seconds for `n_steps` at `rate_s_per_step`, plus compile."""
    return rate_s_per_step * n_steps + compile_s
