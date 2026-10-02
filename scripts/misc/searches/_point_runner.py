"""The single-search driver for point-source leaves — and the admission bar.

One function, :func:`run_point_search`, fits one point-source model with one
sampler at one seed, records the search as a result row, and — the reason this
module exists — measures how much of that fit's wall clock the likelihood
itself is. Every ``scripts/point_source/searches/<sampler>/<leaf>.py`` is a
~25-line shim over it.

The model and the likelihood
----------------------------

The model is the **L5 rung** of the autolens_profiling source-plane campaign
(``autolens_profiling/scripts/point_source_source/likelihood_breakdown/
gradient_mode_crossover.py``, ``_ladder_model("L5", solved=True)``): one
``Isothermal`` lens (5 free parameters: ``centre_0``, ``centre_1``,
``ell_comps_0``, ``ell_comps_1``, ``einstein_radius``) and a source galaxy whose
point is ``al.ps.PointSolved`` (0 free parameters — the source-plane centre is
solved analytically for every trial mass model), fitted with
``al.FitPositionsSourceSolved``, the tensor-weighted **source-plane** χ², on the
``simple`` dataset (4 images, 0.005" position noise).

**What is NOT mirrored: the priors.** The profiling rung uses truth-centred
Gaussians (σ = 0.005" on the centre, 0.05 on θ_E) because it times a call, and
the call's cost does not depend on where the prior sits. A search's evaluation
count does, and a σ = 0.005" prior around the answer would measure a fit nobody
runs. So this leaf takes the **library-default** ``Isothermal`` priors — the
ones ``autolens_workspace/scripts/point_source/modeling.py`` fits with — with a
single change: θ_E is bounded to ``U(0.5, 4.0)`` instead of the default
``U(0, 8)``. This records the chosen benchmark prior; it does not exclude the reported
no-ring basin near 2.4", which remains within these bounds. The truth, 1.6",
sits well inside the interval.

Nautilus runs at the workspace ``modeling.py`` settings, ``n_live=100`` /
``n_batch=50``.

The admission bar
-----------------

A likelihood-speed phase (in autolens_profiling) is only worth running if the
likelihood is a large enough share of a real fit that making it faster moves
the fit. Every row therefore carries, beside the sampler's own ``wall_s`` and
``likelihood_evals``:

``per_call_s``
    Steady-state cost of **one likelihood evaluation as the sampler makes it**,
    in seconds. Under ``use_jax=True`` Nautilus never calls the likelihood one
    point at a time: PyAutoFit hands it ``Fitness.call_wrap`` with
    ``vectorized=True``, which evaluates each batch of ``n_batch`` points
    through ``jax.jit(jax.vmap(Fitness.call))`` (``Fitness._vmap``). So the
    matching cost is the **batched** one: the median wall of one
    ``Fitness._vmap`` call on an ``(n_batch, n_dim)`` array, divided by
    ``n_batch``. ``per_call_basis`` records which basis was used
    (``"batched_vmap"``).

    Measured on a *separate* ``Fitness`` instance built exactly as
    ``NautilusSearch._fit`` builds its own (so it also pays the vector → instance
    mapping traced into the same XLA program), at one fixed parameter vector
    (the prior medians), **after** :data:`WARMUP_CALLS` warm-up calls and as the
    **median of** :data:`TIMED_CALLS` individually ``block_until_ready``-timed
    calls. The warm-up is not decoration: on 2026-09-28 a single-block timing
    taken straight after compile read 2.4x the steady-state cost on a GPU.

``per_call_single_s``
    The same measurement for ``jax.jit(Fitness.call)`` on one vector. Recorded
    for comparison with autolens_profiling's per-call numbers, which are
    single-point; it is **not** what the sampler pays and is not used in the
    share.

``per_batch_s``
    The median wall of one ``Fitness._vmap`` call on the ``n_batch`` array
    (``per_call_s = per_batch_s / n_batch``).

``compile_s``
    Wall of the **first** ``Fitness._vmap`` call on that array — trace, lower,
    compile and one execution. The search compiles its own copy inside
    ``search.fit``, and it recompiles once per distinct batch length Nautilus
    hands it; both are inside ``wall_s``.

``likelihood_share_single``
    The same ratio with ``per_call_single_s``, recorded as an alternative
    timing basis. It is not a proven upper bound across different batch lengths
    and parameter vectors.

``likelihood_share``
    ``per_call_s * likelihood_evals / wall_s``: an estimate of the sampler's
    wall clock spent evaluating the likelihood at steady state. It uses a fixed
    vector and batch size, rather than timing every evaluation in the fit. The rest —
    ``1 - likelihood_share`` — is sampler overhead (Nautilus's neural-network
    bound training and sampling, the Python per-batch wrapper, compiles,
    checkpointing, output). ``likelihood_s = per_call_s * likelihood_evals`` and
    ``overhead_s = wall_s - likelihood_s`` are written beside it.

    Under this cost model and at unchanged evaluation count, a speed-up
    of ``k`` on the likelihood would shorten the fit by approximately
    ``likelihood_share * (1 - 1/k)``.

``wall_s`` / ``likelihood_evals`` / ``log_evidence`` are read from
``files/samples_info.json`` by the SLaM runner's ``_stage_row`` (``time``,
``total_samples`` — Nautilus's reject-inclusive ``n_like`` — and
``log_evidence``), so a search row and a SLaM stage row mean the same thing by
the same word.

Hazards (inherited from the SLaM runner; see ``slam/_runner.py``)
-----------------------------------------------------------------

- The backend environment is exported **before** autolens is imported, so argv
  is parsed first and the modelling stack is imported inside the driver.
- ``config_name`` and ``seed_<n>`` live in the PyAutoFit ``path_prefix``:
  ``seed`` is an identifier field and ``config_name`` is not, so two legs that
  differ only in config would resume each other's fit.
- ``--inversion`` is meaningless for a point source (there is no inversion); the
  config-name grammar still carries the field, so this leaf accepts only
  ``dense`` and says so in the row (``inversion_applies: false``).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._runner import (  # noqa: E402
    _as_float,
    _device_info,
    _stage_row,
    checkpoint_exists,
    config_name_precision,
    posterior_from_result,
    posterior_keys,
    set_backend_env,
    stage_output_dir,
    truth_delta_sigma_from,
    validate_config_name,
)

from _inference_cli import auto_simulate_if_missing, parse_inference_cli  # noqa: E402

#: Version of the ``search_seed<n>.json`` schema written by :func:`run_point_search`.
SCHEMA_VERSION = 1

#: Nautilus settings, from ``autolens_workspace/scripts/point_source/modeling.py``.
N_LIVE = 100
N_BATCH = 50

#: The θ_E prior bound (see the module docstring — the default ``U(0, 8)``
#: reaches a no-ring basin near 2.4" on this lens).
EINSTEIN_RADIUS_LOWER = 0.5
EINSTEIN_RADIUS_UPPER = 4.0

#: Admission-bar timing: warm-up calls discarded, then this many timed calls.
WARMUP_CALLS = 10
TIMED_CALLS = 200

#: The run variant — *what was fitted*: the source-plane solved χ² on the L5 model.
VARIANT = "source_plane_solved"

#: The fit class name recorded in the row (resolved against ``al`` at run time).
FIT_POSITIONS_CLS = "FitPositionsSourceSolved"

#: The PyAuto* libraries whose revisions a row records.
LIBRARIES = ("autonerves", "autofit", "autoarray", "autogalaxy", "autolens")

LOG_EVIDENCE_ERR_NOTE = (
    "nautilus 1.0.5 exposes no log_z_err; samples_info carries only "
    "log_evidence, total_samples, time and number_live_points."
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in scripts/misc/test/test_point_runner.py)
# ---------------------------------------------------------------------------


def likelihood_share(per_call_s, likelihood_evals, wall_s) -> float | None:
    """``per_call_s * likelihood_evals / wall_s`` — or ``None`` when any input is
    missing or the wall clock is not positive.

    Never clipped to 1: a share above 1 means the steady-state per-call cost was
    measured on a slower host state than the fit ran in, and hiding that would
    make the admission bar look cleaner than the measurement it rests on.
    """
    per_call = _as_float(per_call_s)
    wall = _as_float(wall_s)
    if per_call is None or wall is None or wall <= 0 or likelihood_evals is None:
        return None
    try:
        evals = int(likelihood_evals)
    except (TypeError, ValueError):
        return None
    return per_call * evals / wall


def timing_summary(samples: list[float]) -> dict:
    """``{median_s, p16_s, p84_s, min_s, n}`` for a list of per-call walls."""
    if not samples:
        return {"median_s": None, "p16_s": None, "p84_s": None, "min_s": None, "n": 0}
    ordered = sorted(samples)
    n = len(ordered)

    def quantile(q: float) -> float:
        return ordered[min(n - 1, max(0, round(q * (n - 1))))]

    return {
        "median_s": float(statistics.median(ordered)),
        "p16_s": float(quantile(0.16)),
        "p84_s": float(quantile(0.84)),
        "min_s": float(ordered[0]),
        "n": n,
    }


def time_calls(fn, arg, *, warmup: int = WARMUP_CALLS, n: int = TIMED_CALLS, block=None) -> dict:
    """Steady-state timing of ``fn(arg)``: ``warmup`` discarded calls, then ``n``
    individually timed calls, each blocked on its result before the clock stops.

    ``block`` is the synchronisation applied to each result
    (``jax.block_until_ready`` for a jitted call); ``None`` means the call is
    synchronous already. Returns :func:`timing_summary` of the timed calls.
    """
    sync = block if block is not None else (lambda value: value)
    for _ in range(warmup):
        sync(fn(arg))
    walls = []
    for _ in range(n):
        start = time.perf_counter()
        sync(fn(arg))
        walls.append(time.perf_counter() - start)
    summary = timing_summary(walls)
    summary["warmup"] = warmup
    return summary


def point_truth_dict(truth_record: dict) -> dict:
    """The truths a posterior can be compared against, from the simulator's ``truth.json``.

    Keyed to agree with ``posterior_keys`` on this model's names (every leaf name
    is unique: ``centre_0``, ``centre_1``, ``ell_comps_0``, ``ell_comps_1``,
    ``einstein_radius``), so ``truth_delta_sigma`` is a plain key intersection.
    """
    truths: dict[str, float] = {}
    centre = truth_record.get("lens_centre")
    if centre is not None and len(centre) == 2:
        truths["centre_0"] = float(centre[0])
        truths["centre_1"] = float(centre[1])
    ell = truth_record.get("lens_ell_comps")
    if ell is not None and len(ell) == 2:
        truths["ell_comps_0"] = float(ell[0])
        truths["ell_comps_1"] = float(ell[1])
    einstein_radius = truth_record.get("lens_einstein_radius")
    if einstein_radius is not None:
        truths["einstein_radius"] = float(einstein_radius)
    return truths


def truth_vector(keys: list[str], truths: dict) -> list[float] | None:
    """The truth as a parameter vector in the model's own order, or ``None`` if
    any free parameter has no truth."""
    if any(key not in truths for key in keys):
        return None
    return [truths[key] for key in keys]


def results_paths(
    root: Path,
    dataset_class: str,
    sampler: str,
    instrument: str,
    variant: str,
    config_name: str,
    seed: int,
) -> tuple[Path, Path]:
    """``(json_path, png_path)`` for one ``(sampler, variant, config_name, seed)``.

    ``results/searches/<dataset_class>/<sampler>/<instrument>/<variant>/<config_name>/
    search_seed<n>.json`` — the seed in the filename so a seed spread never
    overwrites itself; the variant a level above the config so two models never
    read as two backends of one fit.
    """
    directory = (
        root / "results" / "searches" / dataset_class / sampler / instrument / variant / config_name
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"search_seed{seed}.json", directory / f"search_seed{seed}.png"


def output_path_prefix(
    dataset_class: str, sampler: str, instrument: str, variant: str, config_name: str, seed: int
) -> Path:
    """The PyAutoFit ``path_prefix`` — config name and seed in the PATH (see the
    module docstring)."""
    return (
        Path("searches")
        / dataset_class
        / sampler
        / instrument
        / variant
        / config_name
        / f"seed_{seed}"
    )


def target_id(instrument: str, variant: str, seed: int) -> str:
    """``<instrument>/<variant>/seed<n>`` — the key the dashboard groups legs by."""
    return f"{instrument}/{variant}/seed{seed}"


def library_revisions() -> dict:
    """``{package: short git revision}`` for each PyAuto* library as imported.

    Read from the checkout each package was imported from (``PYTHONPATH`` on RAL
    and on the laptop); ``None`` when that directory is not a git checkout.
    """
    import importlib

    revisions: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            directory = Path(module.__file__).resolve().parent
            revisions[name] = (
                subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                .decode()
                .strip()
            )
        except Exception:
            revisions[name] = None
    return revisions


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def point_model(af, al):
    """The L5 model: ``Isothermal`` lens (library-default priors, θ_E bounded) and a
    ``PointSolved`` source. 5 free parameters."""
    mass = af.Model(al.mp.Isothermal)
    mass.einstein_radius = af.UniformPrior(
        lower_limit=EINSTEIN_RADIUS_LOWER, upper_limit=EINSTEIN_RADIUS_UPPER
    )
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=af.Model(al.ps.PointSolved))
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _plot(png_path: Path, payload: dict) -> None:
    """Wall clock split into likelihood and sampler overhead, beside the JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    likelihood_s = payload.get("likelihood_s")
    overhead_s = payload.get("overhead_s")
    if likelihood_s is None or overhead_s is None:
        return
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.barh([0], [likelihood_s], color=plt.cm.viridis(0.25), height=0.5, label="likelihood")
    ax.barh(
        [0],
        [overhead_s],
        left=[likelihood_s],
        color=plt.cm.viridis(0.75),
        height=0.5,
        label="sampler overhead",
    )
    ax.set_yticks([])
    ax.set_xlabel("Sampler wall clock (s)")
    share = payload.get("likelihood_share")
    share_text = "—" if share is None else f"{share:.2%}"
    fig.suptitle(
        f"{payload.get('sampler')}: {payload.get('target')} — likelihood share {share_text}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"{payload.get('config_name')}  |  {payload.get('likelihood_evals'):,} evals  |  "
        f"{payload.get('per_call_s') * 1e6:.1f} µs/eval  |  AutoLens v{payload.get('version')}",
        fontsize=8,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def run_point_search(
    sampler: str = "nautilus",
    dataset_class: str = "point_source",
    default_instrument: str = "simple",
) -> int:
    """Fit the L5 point-source model with Nautilus for one ``(config_name, seed)`` leg.

    Returns the process exit code: 0 on success, 2 on a flag / environment
    refusal (a malformed or lying config name, ``--inversion sparse``, the
    ``numba_cpu`` backend, or ``--backend jax_gpu`` on a host with no GPU).
    """
    if sampler != "nautilus":  # pragma: no cover — one sampler today
        print(f"ERROR: sampler {sampler!r} is not implemented", file=sys.stderr)
        return 2

    cli = parse_inference_cli()
    instrument = cli.instrument or default_instrument
    cores = cli.cores

    error = validate_config_name(cli)
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if cli.inversion != "dense":
        print(
            "ERROR: a point source has no inversion; the config-name grammar carries the "
            "field, so this leaf accepts only --inversion dense (recorded as "
            "inversion_applies: false).",
            file=sys.stderr,
        )
        return 2
    if cli.backend == "numba_cpu":
        print(
            "ERROR: the numba_cpu leg is not implemented for the point-source search leaf: "
            "the admission bar times the jitted Fitness call Nautilus makes under use_jax=True.",
            file=sys.stderr,
        )
        return 2

    config_name = cli.config_name
    precision = config_name_precision(config_name)
    env = set_backend_env(cli.backend, cores)

    # --- imports: everything below here sees the backend environment above ---
    import autofit as af
    import autolens as al
    from autonerves import jax_wrapper  # noqa: F401 — sets the JAX env before autolens
    from autonerves.test_mode import is_test_mode, with_test_mode_segment

    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
        return 0

    import jax
    import numpy as np
    from autofit.non_linear.fitness import Fitness

    if cli.backend == "jax_gpu":
        try:
            backend_in_use = jax.default_backend()
        except Exception as exc:
            backend_in_use = f"unavailable ({type(exc).__name__}: {exc})"
        if backend_in_use != "gpu":
            print(
                f"ERROR: --backend jax_gpu but jax.default_backend() == {backend_in_use!r}; "
                "refusing to record a CPU run as a GPU leg.",
                file=sys.stderr,
            )
            return 2

    seed = cli.seed
    root = _ROOT
    variant = VARIANT

    output_root = cli.output_dir or Path(os.environ.get("PYAUTO_OUTPUT_DIR") or (root / "output"))
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    af.conf.instance.push(new_path=root / "config", output_path=output_root)

    print("=" * 78)
    print(
        f"{sampler} search — {dataset_class} / {instrument} / {variant} / {config_name} / seed {seed}"
    )
    print("=" * 78)
    print(f"  backend:    {cli.backend}  (cores={cores})")
    print(f"  precision:  {precision}")
    print(f"  env set:    {env}")
    print(f"  output:     {output_root}")
    print(f"  test mode:  {is_test_mode()}")

    from simulators.point_source import INSTRUMENTS

    if instrument not in INSTRUMENTS:
        print(
            f"ERROR: unknown --instrument {instrument!r}; choose from {sorted(INSTRUMENTS)}",
            file=sys.stderr,
        )
        return 2

    dataset_path = root / "dataset" / dataset_class / instrument
    auto_simulate_if_missing(
        dataset_path, dataset_type=dataset_class, instrument=instrument, workspace_root=root
    )
    dataset = al.from_json(file_path=dataset_path / "point_dataset_positions_only.json")
    truths = point_truth_dict(json.loads((dataset_path / "truth.json").read_text()))

    fit_positions_cls = getattr(al, FIT_POSITIONS_CLS)
    model = point_model(af, al)
    analysis = al.AnalysisPoint(
        dataset=dataset, solver=None, fit_positions_cls=fit_positions_cls, use_jax=True
    )

    path_prefix = output_path_prefix(dataset_class, sampler, instrument, variant, config_name, seed)
    search_name = "nautilus"
    search = af.Nautilus(
        path_prefix=path_prefix,
        name=search_name,
        unique_tag=None,
        n_live=N_LIVE,
        n_batch=N_BATCH,
        seed=seed,
        number_of_cores=1,
    )
    search_root = with_test_mode_segment(output_root) / path_prefix
    keys = posterior_keys(list(model.all_names))

    timing: dict = {}
    log_likelihood_at_truth = None

    def admission_bar_timing() -> None:
        """Time the Fitness call Nautilus makes, on a Fitness of our own."""
        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=None,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99,
            use_jax_vmap=True,
            batch_size=N_BATCH,
        )
        vector = np.asarray(model.physical_values_from_prior_medians, dtype=float)
        batch = np.tile(vector, (N_BATCH, 1))

        batched = fitness._vmap
        start = time.perf_counter()
        jax.block_until_ready(batched(batch))
        timing["compile_s"] = time.perf_counter() - start
        timing["batched"] = time_calls(batched, batch, block=jax.block_until_ready)

        single = jax.jit(fitness.call)
        start = time.perf_counter()
        jax.block_until_ready(single(vector))
        timing["compile_single_s"] = time.perf_counter() - start
        timing["single"] = time_calls(single, vector, block=jax.block_until_ready)
        timing["vector"] = vector.tolist()

        nonlocal log_likelihood_at_truth
        truth = truth_vector(keys, truths)
        if truth is not None:
            log_likelihood_at_truth = float(single(np.asarray(truth, dtype=float)))

    status = "complete"
    row: dict = {}
    total_wall_s = None
    resumed = checkpoint_exists(search_root, search_name)
    posterior: dict = {}
    max_log_likelihood = None
    try:
        if is_test_mode():
            timing["note"] = "admission-bar timing skipped under PYAUTO_TEST_MODE"
        else:
            admission_bar_timing()
            print(
                f"  per-call (batched, /{N_BATCH}): "
                f"{timing['batched']['median_s'] / N_BATCH * 1e6:.2f} µs; single: "
                f"{timing['single']['median_s'] * 1e6:.1f} µs; compile {timing['compile_s']:.2f} s"
            )

        print(f"\n--- {search_name} (n_live={N_LIVE}, n_batch={N_BATCH}) ---")
        start = time.perf_counter()
        result = search.fit(model=model, analysis=analysis)
        total_wall_s = time.perf_counter() - start

        try:
            posterior = posterior_from_result(af, result)
            max_log_likelihood = float(result.samples.max_log_likelihood_sample.log_likelihood)
        except Exception as exc:
            print(f"  WARNING: posterior capture failed — {exc}")
    except BaseException as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        output_path = stage_output_dir(search_root, search_name)
        row = _stage_row(
            search_name,
            output_path,
            n_live=N_LIVE,
            n_batch=N_BATCH,
            free_parameters=model.prior_count,
            total_wall_s=total_wall_s,
            compile_s=timing.get("compile_s"),
            resumed=resumed,
            inversion_applies=False,
            posterior=posterior,
            truth_delta_sigma=truth_delta_sigma_from(posterior, truths),
            max_log_likelihood=max_log_likelihood,
        )
        # Relative to the output root, so a committed row carries no host path.
        if output_path is not None:
            try:
                row["output_path"] = str(output_path.relative_to(output_root))
            except ValueError:
                pass
        batched = timing.get("batched") or {}
        single = timing.get("single") or {}
        per_batch_s = batched.get("median_s")
        per_call_s = per_batch_s / N_BATCH if per_batch_s is not None else None
        share = likelihood_share(per_call_s, row["likelihood_evals"], row["wall_s"])
        likelihood_s = (
            per_call_s * row["likelihood_evals"]
            if per_call_s is not None and row["likelihood_evals"] is not None
            else None
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "target": target_id(instrument, variant, seed),
            "sampler": sampler,
            "variant": variant,
            "model": "L5: Isothermal (5 free) + PointSolved source",
            "fit_positions_cls": FIT_POSITIONS_CLS,
            "priors": {
                "einstein_radius": f"U({EINSTEIN_RADIUS_LOWER}, {EINSTEIN_RADIUS_UPPER})",
                "others": "library-default Isothermal priors",
            },
            "config_name": config_name,
            "backend": cli.backend,
            "inversion": cli.inversion,
            "precision": precision,
            "instrument": instrument,
            "dataset_class": dataset_class,
            "seed": seed,
            "version": al.__version__,
            "library_revisions": library_revisions(),
            "device": _device_info(),
            "host": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cores": cores,
            "use_jax": True,
            "test_mode": bool(is_test_mode()),
            "status": status,
            # --- the search (same words as a SLaM stage row) -----------------
            **{k: v for k, v in row.items() if k != "name"},
            "log_evidence_err_note": LOG_EVIDENCE_ERR_NOTE,
            "log_likelihood_at_truth": log_likelihood_at_truth,
            "truths": truths,
            # --- the admission bar ------------------------------------------
            "per_call_s": per_call_s,
            "per_call_basis": "batched_vmap" if per_call_s is not None else None,
            "per_batch_s": per_batch_s,
            "per_call_single_s": single.get("median_s"),
            "compile_single_s": timing.get("compile_single_s"),
            "likelihood_s": likelihood_s,
            "overhead_s": (
                row["wall_s"] - likelihood_s
                if likelihood_s is not None and row["wall_s"] is not None
                else None
            ),
            "likelihood_share": share,
            # The same share on the single-point basis — NOT what the sampler pays
            # (it never calls one point at a time), but the ceiling a reader who
            # distrusts the batched basis can check the verdict against.
            "likelihood_share_single": likelihood_share(
                single.get("median_s"), row["likelihood_evals"], row["wall_s"]
            ),
            "timing": timing,
        }
        json_path, png_path = results_paths(
            root, dataset_class, sampler, instrument, variant, config_name, seed
        )
        json_path.write_text(json.dumps(payload, indent=2))
        try:
            _plot(png_path, payload)
        except Exception as exc:  # pragma: no cover — a plot must not kill a run
            print(f"  WARNING: plot failed — {exc}")
        print(
            f"\n  {status}: wall {row['wall_s']}s, evals {row['likelihood_evals']}, "
            f"log Z {row['log_evidence']}, share {share}"
        )
        print(f"  Result row: {json_path}")
    return 0
