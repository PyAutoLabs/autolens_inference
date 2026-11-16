"""The calibrated wall model: how long a Nautilus SLaM job takes, before it runs.

autolens_inference#69. Phase 4 priced each stage as ``profiled per-call x evaluations``,
with the evaluations borrowed from another likelihood's rows. Against the SLURM walls of the
completed phase-4 legs that missed by 0.55-89.5x, for three reasons:

1. **No per-evaluation floor and no startup.** A Nautilus evaluation costs ~5-10 ms of sampler
   bookkeeping whatever the likelihood, and each stage then spends a further few ms per
   evaluation writing its samples and outputs. A likelihood profiled at microseconds still
   costs ~14 ms per evaluation in a fit (weak, point source, cluster, MGE on the A100).
2. **The evaluation count was a proxy.** multi_galaxy's ``source_lp[1]`` (24 free
   parameters, n_live 250) took 250k-272k evaluations against an imaging proxy of 92,550.
3. **The per-evaluation basis was wrong in places.** numba runs one process per Nautilus
   worker, at a parallel efficiency well below 1; a parametric ``source_lp`` stage of a
   pixelized chain was priced at the pixelized per-call; and multi_dataset's profiled
   per-call is a vmap batch-3 OOM fallback, slower than its own single jit.

The model
---------

For one stage with ``d`` free parameters and ``n_live`` live points::

    evals   = EVALS_A * n_live * d ** EVALS_B * EVALS_FACTOR[dataset_type]
    seconds = evals * (FLOOR_S + price)

and a job is ``STARTUP_S`` plus the sum of the stages it fits. ``price`` is the per-call the
stage actually pays: the likelihood's profiled per-call (``numba_cpu``: divided by
``cores * PARALLEL_EFFICIENCY[cores]``); for a parametric ``source_lp`` stage of a pixelized
chain, the dataset type's MGE per-call (``numba_cpu``: ``NUMBA_PARAMETRIC_FACTOR`` times the
JAX CPU one); and a declared override where the profiled per-call does not price what the fit
evaluates (:data:`PER_CALL_OVERRIDE`, :data:`PER_EVAL_OVERRIDE`).

Everything here is an input available before a run: the model's dimension and ``n_live``, the
dataset type, the setup, the core count and the profiled per-call. The constants are fitted to
the phase-4 rows on release pin 2026.10.25.1 (completed, non-resumed stages only: a stage that
straddled a kill and resume reads its ``samples_info`` time 3-4x high, 414362), and checked
against every completed phase-4 leg by ``validation_p4.py --validate``.

What this model is not valid for
--------------------------------

- Samplers other than Nautilus, or Nautilus settings other than the phase-4 ones (n_batch 50,
  the SLaM stage priors). The evaluation law is a fit to Nautilus on these models.
- A dataset type with no phase-4 row: it takes ``EVALS_FACTOR`` 1.0, an uncalibrated default
  (the per-type factors span 0.34-1.65).
- Per-stage accuracy. Single stages scatter by up to ~3x around the law (interferometer and
  ``source_pix[2]`` most); the model is calibrated on whole jobs, which average it out.
- A resumed or requeued job: its walls are not measurements, and its remaining time is not
  what this model prices.

Phase 5's run-time estimator (likelihood-lifecycle 5E, ``al.Setup.estimate``) reuses this
model rather than rebuilding one: same law, same floor, same efficiency curve.
"""

from __future__ import annotations

#: evals = EVALS_A * n_live * d ** EVALS_B (least squares in log space over 231 completed,
#: non-resumed phase-4 stages outside interferometer; 2026-11-16).
EVALS_A = 4.30
EVALS_B = 1.637

#: The median of measured / law evaluations per dataset type, over its phase-4 stages.
#: Interferometer's high-S/N parametric stages converge in ~3x fewer evaluations; multi_dataset
#: fits two datasets' worth of posterior structure.
EVALS_FACTOR: dict[str, float] = {
    "cluster": 0.77,
    "group": 0.79,
    "imaging": 1.13,
    "interferometer": 0.34,
    "multi_dataset": 1.65,
    "multi_galaxy": 1.06,
    "point_source_image": 0.88,
    "point_source_source": 0.81,
    "weak": 0.42,
}

#: Seconds per evaluation the fit spends outside the likelihood: Nautilus bookkeeping (~5-10 ms,
#: the implied cost of weak / point source / A100 MGE stages) plus the stage's own sample and
#: output writing (~4-12 ms per evaluation, ``total_wall_s - wall_s``). One value serves the
#: A100 and the RAL CPU node; the fit was flat across 12-16 ms.
FLOOR_S = 0.014

#: Seconds per job outside every stage: imports, dataset load, the first compile. SLURM
#: ElapsedRaw minus the summed stage walls of the single-stage legs (10-35 s).
STARTUP_S = 30

#: numba's parallel efficiency under Nautilus's multiprocessing, by core count: the phase-4
#: cores sweep (imaging/rectangular/hst, numba_cpu_sparse_fp64, source_pix[1], seed 0; RAL jobs
#: 414141, 413913, 414142, 413911, 414143, 414144).
PARALLEL_EFFICIENCY: dict[int, float] = {1: 1.0, 2: 0.68, 4: 0.62, 8: 0.55, 16: 0.36, 32: 0.28}

#: A parametric stage under numba costs about this many times the dataset type's JAX CPU MGE
#: per-call (phase-4 ``source_lp`` stages of the numba chains: 71-138 ms per evaluation at 8
#: cores, against 16-37 ms profiled for JAX CPU MGE).
NUMBA_PARAMETRIC_FACTOR = 3.0

#: Profiling records whose per-call does not price a fit's evaluation, by record id:
#: ``(seconds, why)``.
PER_CALL_OVERRIDE: dict[str, tuple[float, str]] = {
    "lifecycle/b5dea27fcaf8b14ec9bb": (
        0.110,
        "multi_dataset/delaunay A100: the record's vmap.per_call (0.176 s) timed a batch of 3, "
        "an OOM fallback, slower than the same record's single-jit median of 110 ms "
        "(wiki/project/state.md, 2026-11-05)",
    ),
}

#: Whole per-evaluation costs (floor included) for a likelihood whose fit evaluates something
#: its profiling does not, by ``(dataset_type, variant, host_class)``: ``(seconds, why)``.
PER_EVAL_OVERRIDE: dict[tuple[str, str, str], tuple[float, str]] = {
    ("cluster", "image_plane", "hpc_a100"): (
        0.060,
        "the image-plane PointSolver fit costs 15 ms per evaluation in the sampler and ~43 ms "
        "per evaluation in stage output on the A100, against a 0.2 ms vmap per-call "
        "(phase-4 413966: total_wall_s / likelihood_evals 0.056-0.060 s)",
    ),
    ("cluster", "image_plane", "hpc_ral_cpu"): (
        0.049,
        "as the A100 entry; RAL CPU 414145: total_wall_s / likelihood_evals 0.035-0.049 s",
    ),
}

#: Dataset types whose single search is the whole fit: no parametric opening stage to reprice.
SINGLE_SEARCH = {"point_source_source", "point_source_image", "cluster", "weak"}


def stage_evals(free_parameters: int, n_live: int, dataset_type: str) -> int:
    """Evaluations one Nautilus stage takes, from its dimension and live points."""
    law = EVALS_A * n_live * free_parameters**EVALS_B
    return round(law * EVALS_FACTOR.get(dataset_type, 1.0))


def parallel_efficiency(cores: int) -> float:
    """The measured efficiency at ``cores``; between sweep points, the next lower one."""
    known = [c for c in sorted(PARALLEL_EFFICIENCY) if c <= cores]
    return PARALLEL_EFFICIENCY[known[-1]] if known else 1.0


def effective_per_call(per_call_s: float, backend: str, cores: int) -> float:
    """What one evaluation costs the job: a numba per-call is one thread and Nautilus forks
    ``cores`` workers; a JAX per-call (vmap) already prices one evaluation on the device."""
    if backend == "numba_cpu":
        return per_call_s / (cores * parallel_efficiency(cores))
    return per_call_s


def stage_seconds(evals: int, price_s: float) -> float:
    """Wall seconds for one stage of ``evals`` evaluations at ``price_s`` likelihood cost."""
    return evals * (FLOOR_S + price_s)


def job_seconds(stage_walls: list[float]) -> float:
    """A job's wall: startup plus the stages it fits."""
    return STARTUP_S + sum(stage_walls)
