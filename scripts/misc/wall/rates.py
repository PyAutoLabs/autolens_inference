"""Per-cell measured step rates for HPC submit `--time` estimates.

Populated **only** from rates measured in this repo, on the cell the row names.
Nothing in this table is interpolated, extrapolated, carried over from a
neighbouring cell, or imported from another project — every one of those is the
defect this module exists to prevent.

What is in the table, and what it may not be used for
-----------------------------------------------------

This repo was born 2026-09-10 and inherits nothing from the retired
`inference_programme`: those numbers were measured on a different tree, by runs
this project cannot name. The first rows landed on **2026-09-11**, and they are
narrower than they look. Every one of them is ``source_lp[1]`` only — the
*parametric* MGE stage that opens the five-stage SLaM chain — measured with
``--stages source_lp``. The other four stages of that same cell are pixelized and
per-eval-inversion bound at many times the cost, so **no row here may size a
five-stage submit**. That is not a caution, it is the module's founding rule:
carrying a parametric rate onto pixelized arms is what lost 35 of 39 arms of an
overnight A100 block.

Consequently every production submit in `hpc/batch_*` still declares
``source: unmeasured`` with ``probe-first: yes``. The rows below are cited by
exactly one submit — ``batch_gpu/submit_slam_hst_rate_jax_gpu``, the
``--stages source_lp`` probe they describe.

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
    # ------------------------------------------------------------------ 2026-09-11
    # The first rates measured in this repo. Both are `source_lp[1]` of the cell
    # `imaging/slam/hst`, at production settings (no PYAUTO_TEST_MODE), seed 0,
    # dense inversion, fp64, on the HST cell's 15,361 masked pixels, 17 free
    # parameters, nautilus n_live=200 / n_batch=50. The device token names the
    # HOST as well as the backend, so a laptop rate can never be cited by a RAL
    # submit: `ral_numba_cpu`, `ral_jax_cpu` and `a100` are deliberately absent.
    #
    # Both are PARTIAL runs — see PROVENANCE. Neither left nautilus's exploration
    # phase, so neither says how many evaluations the stage needs, only how long
    # one costs.
    ("imaging", "slam", "hst", "laptop_numba_cpu", "fp64", 1, 50): 0.06831,
    ("imaging", "slam", "hst", "laptop_jax_cpu", "fp64", 1, 50): 0.04925,
}

#: One entry per key prefix in ``STEP_RATE``: what ran, on which job, and the
#: configuration the row is and is not valid for.
PROVENANCE: dict[str, str] = {
    "imaging/slam/hst/laptop_numba_cpu": (
        "2026-09-11, DESKTOP-H143S82 (WSL2, 8 cores, 15 GB), autolens/autofit/autoarray "
        "2026.8.17.1. `python3 scripts/imaging/slam/hst.py --backend numba_cpu --inversion "
        "dense --config-name local_numba_cpu_dense_fp64 --stages source_lp --cores 8`, "
        "environment `env -u XLA_FLAGS OMP_NUM_THREADS=1` (the session's ambient "
        "XLA_FLAGS=--xla_disable_hlo_passes=constant_folding removed; OMP_NUM_THREADS=1 is "
        "correct here because nautilus takes the number_of_cores multiprocessing path when "
        "use_jax=False, and kernel threading under it costs rather than pays). "
        "PARTIAL RUN: 17,950 reject-inclusive likelihood evaluations over 1,226 s wall "
        "(nautilus checkpoint.hdf5 `n_like` against the sampler's own `.start_time`), "
        "= 0.06831 s/eval, 14.6 eval/s across 8 cores. The run was stopped there, still "
        "inside nautilus's exploration phase at shell 21 of 21, to free the machine for the "
        "jax_cpu measurement -- the two cannot share 8 cores without each measuring "
        "contention rather than its backend. "
        "NOT VALID FOR: any pixelized stage (source_pix[1], source_pix[2], light[1], "
        "mass_total[1]) or any submit that runs them; any RAL node (this is a laptop, and "
        "the device token says so); any core count other than 8; the sparse inversion, "
        "which changes the likelihood but not this stage, which builds no inversion at all. "
        "IT ALSO DOES NOT SAY HOW LONG THE STAGE TAKES: exploration had not finished, so "
        "the stage's total evaluation count is unmeasured."
    ),
    "imaging/slam/hst/laptop_jax_cpu": (
        "2026-09-11, DESKTOP-H143S82 (WSL2, 8 cores, 15 GB), autolens/autofit/autoarray "
        "2026.8.17.1, jax 0.10.2. `python3 scripts/imaging/slam/hst.py --backend jax_cpu "
        "--inversion dense --config-name local_jax_cpu_dense_fp64 --stages source_lp "
        "--cores 8`, environment `env -u XLA_FLAGS OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1` "
        "(the session's ambient XLA_FLAGS=--xla_disable_hlo_passes=constant_folding removed; "
        "the driver itself exports JAX_PLATFORMS=cpu, JAX_ENABLE_X64=True and NPROC=8, which "
        "is what actually sizes XLA's CPU pool). "
        "PARTIAL RUN: 25,500 reject-inclusive likelihood evaluations over 1,256 s wall "
        "(nautilus checkpoint.hdf5 `n_like` against the sampler's own `.start_time`), "
        "= 0.04925 s/eval, 20.3 eval/s across 8 cores. Stopped there, still inside nautilus's "
        "exploration phase at shell 31. Its window (1,256 s) deliberately matches the "
        "numba_cpu row's (1,226 s), and the two legs were run SEQUENTIALLY and never "
        "concurrently: eight cores shared between them would have measured contention rather "
        "than either backend. On that basis jax_cpu was 1.39x faster per evaluation than "
        "numba_cpu on this stage -- an observation about `source_lp[1]`, not about the chain. "
        "NOT VALID FOR: any pixelized stage or any submit that runs them; any RAL node; any "
        "core count other than 8; and it says nothing about how long the stage takes, because "
        "exploration had not finished when it was stopped."
    ),
}


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
