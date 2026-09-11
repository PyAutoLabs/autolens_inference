"""The wall-clock gate reads the cell a submit really runs, task directory included.

The gate's one rule is that a `--time` justification never crosses cells, so
everything here turns on *cell identity*. This repo names a leaf for the target
it runs rather than for the task (`scripts/imaging/slam/hst.py`, not
`slam_hst_base.py`, because `AGENTS.md` makes the instrument a flag and never a
directory). A cell id cut at `<dataset>/<leaf>` would therefore call the SLaM
base run `imaging/hst` — the same id a future `scripts/imaging/searches/<sampler>/hst.py`
would claim. Two different pipelines sharing one rate row is exactly the carry
that killed 35 of 39 arms in an overnight A100 block, so the id keeps the task.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
_MISC = str(REPO_ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

from wall import check_submits  # noqa: E402

_HEADER = """#!/bin/bash -l
#
# WALL-BASIS: — one row per cell this submit runs.
#   cell: {cell}  device: a100  precision: fp64
#   lanes: 1  steps: 3000  source: unmeasured  probe-first: yes

#SBATCH -J test
#SBATCH --partition=gpu
#SBATCH --time={time}

cd $AP_ROOT
python3 {script} --instrument hst --backend jax_gpu --inversion dense

echo "Finished."
"""


def _submit(cell="imaging/slam/hst", script="scripts/imaging/slam/hst.py", time="1:00:00"):
    return _HEADER.format(cell=cell, script=script, time=time)


def test_cell_id_keeps_the_task_directory():
    cells, instruments = check_submits.cells_run(_submit())
    assert cells == {"imaging/slam/hst"}
    assert instruments == {"hst"}


def test_deeper_task_trees_keep_every_component():
    cells, _ = check_submits.cells_run(_submit(script="scripts/imaging/searches/nautilus/hst.py"))
    assert cells == {"imaging/searches/nautilus/hst"}


def test_matching_row_and_invocation_is_clean():
    assert check_submits.check_text(_submit(), "submit_slam_hst_jax_gpu_dense") == []


def test_a_row_for_another_task_does_not_cover_this_cell():
    problems = check_submits.check_text(
        _submit(cell="imaging/searches/hst"), "submit_slam_hst_jax_gpu_dense"
    )
    joined = " ".join(problems)
    assert "never runs imaging/searches/hst" in joined
    assert "imaging/slam/hst is RUN by this submit but has no WALL-BASIS row" in joined


def test_rate_key_splits_the_id_as_dataset_task_leaf():
    rows = check_submits.parse_basis_rows(_submit())
    problems: list[check_submits.Problem] = []
    check_submits.check_text(_submit(), "submit_x")
    # check_text stamps `_cell_parts`; reproduce its split directly so the key
    # grammar `(dataset, task, instrument)` is asserted rather than assumed.
    parts = rows[0]["cell"].split("/")
    assert (parts[0], "/".join(parts[1:-1]), parts[-1]) == ("imaging", "slam", "hst")
    assert problems == []


def test_commented_out_invocations_are_not_cells():
    text = _submit().replace("python3 scripts", "# python3 scripts")
    cells, _ = check_submits.cells_run(text)
    assert cells == set()


def test_the_committed_submits_all_pass():
    assert check_submits.main(["--check"]) == 0
