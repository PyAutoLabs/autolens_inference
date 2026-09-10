"""Wall-clock estimation for HPC submit scripts.

`wall.rates` holds the curated per-cell step-rate table that a submit's
``# WALL-BASIS:`` block cites; `wall.check_submits` is the CI gate that checks
every cell a submit actually runs has its own basis row, that cited rates match
the table, and that ``--time`` clears the declared headroom.

See ``wall/README.md`` for the methodology and the authoring contract.
"""

from wall.rates import (
    PROVENANCE,
    STEP_RATE,
    UnmeasuredCellError,
    step_rate_for,
    wall_estimate,
)

__all__ = [
    "PROVENANCE",
    "STEP_RATE",
    "UnmeasuredCellError",
    "step_rate_for",
    "wall_estimate",
]
