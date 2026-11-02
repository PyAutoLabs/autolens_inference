"""The dataset-agnostic core of every validation driver: one leg, its stages, its row.

A *leg* is one ``(variant, config_name, seed)`` run of one likelihood's chain. Everything
about a leg that does not depend on the dataset lives here, once: the flag checks (the
config-name grammar, ``--stages-from``), the backend environment, the stage-row readers
(``samples_info.json``, ``.completed``, the resume marker), the posterior and truth
helpers, the PyAutoFit output prefix and the results path, the row provenance (library
revisions, release pin, SLURM job), the ``--stages-from`` shared-stage loading, and the
``stages.json`` writer. A *family* supplies only what is its own: how to load its dataset
and which stages it fits.

Two drivers sit on it:

- ``slam._runner.run_slam`` — the five-stage imaging SLaM chain (``rectangular``,
  ``delaunay``, ``delaunay_nn``), the stage builder every existing imaging leaf runs. It
  calls the pieces below directly, because its lens-light modes and prepared export are
  its own; its rows are byte-for-byte what they were before this split.
- :func:`run_family` — the generic driver every other family uses, SLaM chain or single
  search alike: a :class:`Family` names its likelihood, its stages and its loader, and the
  driver does the rest.

Adding a family (likelihood-lifecycle 4c)
-----------------------------------------

1. Write ``scripts/misc/slam/_<dataset_type>.py`` (e.g. ``_interferometer.py``): a
   ``load(leg)`` that reads the dataset (``leg.dataset_path`` is
   ``dataset/<dataset_class>/<instrument>/``; call ``auto_simulate_if_missing`` first) and
   returns a context object of the family's choosing, and one :class:`Stage` per search,
   whose ``build(leg, context, results)`` returns ``(model, analysis, search)`` — the
   search from :meth:`Leg.nautilus`, every ``Analysis`` given ``use_jax=leg.use_jax``.
   ``results`` maps each earlier stage key to its fitted result, for prior passing. A
   stage whose fit changes the dataset (adaptive over-sampling, say) re-derives it in its
   ``after`` hook, which runs once the stage is fitted. The pixelization helpers of the
   imaging builder (``_runner.mesh_models``, ``_adapt_images``, ``mesh_fields``) take any
   dataset with a mask and are meant to be shared.
2. Declare the :class:`Family`: its likelihood (``dataset_type`` / ``likelihood_variant``,
   exactly as the profiling catalogue names it, so the exporter pairs its stages), its
   stages, its loader, the inversions and transformers it accepts, and the payload fields
   that record its model (``model_fields``; a pixelized family records ``mesh``,
   ``mesh_pixels`` and ``mesh_shape``, which the exporter's size comparison reads).
3. Add the leaf, ``scripts/<dataset_type>/<variant>/validation/<name>.py``: the
   ``imaging/rectangular/validation/baseline_slam.py`` preamble, then
   ``sys.exit(run_family(FAMILY, default_instrument=...))``.

A single-search family (point source, cluster, weak lensing, the datacube) is a family
with one stage; it still writes ``stages.json`` with that one stage, through the same
writer, so the exporter reads it like any chain.

What a family row declares
--------------------------

Every family row carries its likelihood outright — ``likelihood: {dataset_type,
variant}`` and ``model_variant`` on the payload — and every stage carries
``likelihood_evaluated``: the same block when the stage's evaluations are that
likelihood, ``null`` when they are none of the profiled likelihoods (a parametric
``source_lp[1]`` ahead of a pixelized chain: :attr:`Stage.evaluates_likelihood` False).
The exporter files and pairs each stage by that declaration, so an MGE, point, cluster or
weak stage — which solves no inversion — pairs with its own likelihood's profiling
record. The imaging SLaM rows declare neither field and keep their old resolution (the
mesh or variant name, and ``inversion_applies``). A family with lens light
(:attr:`Family.light_fixed`, 4c-1) also writes ``lens_light_mode``, and its ``fixed`` row is
the ``light_fixed`` model variant. The fields are additive, so the schema stays at version 3.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _inference_cli import (  # noqa: E402
    BASE_VARIANT,
    CONFIG_NAME_GRAMMAR,
    LENS_LIGHT_FIXED_SUFFIX,
    InferenceCLI,
    auto_simulate_if_missing,
    device_info_dict,
    lens_light_modes,
    parse_config_name,
    parse_inference_cli,
    row_stem,
    split_config_name,
)

#: Version of the ``stages_seed<n>.json`` schema written by :func:`_write_results`.
#:
#: v2 added the run variant and the mesh it is named after: ``variant``,
#: ``mesh``, ``mesh_pixels`` and ``regularization`` (the resolved scheme name),
#: with ``mesh_shape`` now ``null`` on a row whose mesh has no shape (Delaunay
#: takes a vertex count). v1 rows are still read — every reader falls back to
#: the base variant when the field is absent, which is what those rows are.
#:
#: v3 added the lens-light mode and the row's provenance: ``lens_light_mode``
#: (``linear`` / ``fixed``; the ``fixed`` mode is also the ``_light_fixed``
#: variant suffix), ``library_revisions`` (the five import-selected PyAuto*
#: checkouts' HEAD commits, or ``null`` with ``library_revisions_note`` when
#: they cannot be pinned), ``release_pin`` (``PYAUTO_RELEASE_PIN``, exported by
#: a submit that runs against a pinned release checkout, else ``null``),
#: ``slurm_job_id`` (``<array job>_<task>`` inside an array, ``SLURM_JOB_ID``
#: otherwise, else ``null``) and ``results_tag`` (``--results-tag``, else
#: ``null``). v1/v2 rows still read: an absent ``lens_light_mode`` is
#: ``linear`` (the only mode that existed), absent provenance is ``null``.
SCHEMA_VERSION = 3

#: Exit code of a ``--stages-from`` run whose shared stage is not complete in
#: the other config's tree (nothing is fitted). 2 stays the flag/environment
#: refusal; 4 is the submit's in-job wait timing out.
SHARED_STAGE_INCOMPLETE_EXIT = 3

SHARED_STAGE_NOTE = (
    "loaded from another config's completed fit (--stages-from), not fitted by "
    "this run: wall_s and likelihood_evals are that fit's, kept for the record "
    "and excluded from chain_wall_s"
)
CHAIN_WALL_NOTE = (
    "chain_wall_s is the sum of the sampler walls (wall_s) of the stages this "
    "row's process fitted; a stage loaded from another config (shared_from_config) "
    "is excluded and listed in chain_wall_excludes"
)

LOG_EVIDENCE_ERR_NOTE = (
    "nautilus 1.0.5 exposes no log_z_err; samples_info carries only "
    "log_evidence, total_samples, time and number_live_points."
)
COMPILE_S_TEST_MODE_NOTE = (
    "compile probe skipped under PYAUTO_TEST_MODE (the fit itself makes the "
    "single likelihood call, so a probe would only double the run)."
)


# ---------------------------------------------------------------------------
# Flag validation
# ---------------------------------------------------------------------------


def validate_config_name(cli: InferenceCLI) -> str | None:
    """Return an error message if ``--config-name`` is absent, malformed, or disagrees
    with the run flags; ``None`` when everything agrees.

    The config name is the only thing that identifies a leg inside a parity row,
    so a name that does not describe the run it labels is a silent data
    corruption: two legs would sit side by side under one target claiming
    backends they did not use. Every field is checked against its flag: the
    backend, the inversion (``na`` included), the precision and the optional
    transformer (``--transformer``; a name without one must come from a run
    without one). Which values a leaf accepts at all is :func:`refuse_setup`'s.
    """
    if cli.config_name is None:
        return f"--config-name is required. Grammar: {CONFIG_NAME_GRAMMAR}"

    fields = parse_config_name(cli.config_name)
    if fields is None:
        return f"--config-name {cli.config_name!r} does not match the grammar {CONFIG_NAME_GRAMMAR}"

    backend, inversion, precision = fields.backend, fields.inversion, fields.precision
    expected_precision = "mp" if cli.use_mixed_precision else "fp64"

    mismatches = []
    if backend != cli.backend:
        mismatches.append(f"backend: name says {backend!r}, --backend says {cli.backend!r}")
    if inversion != cli.inversion:
        mismatches.append(f"inversion: name says {inversion!r}, --inversion says {cli.inversion!r}")
    if precision != expected_precision:
        mismatches.append(
            f"precision: name says {precision!r}, "
            f"--use-mixed-precision implies {expected_precision!r}"
        )
    transformer = getattr(cli, "transformer", None)
    if fields.transformer != transformer:
        mismatches.append(
            f"transformer: name says {fields.transformer!r}, --transformer says {transformer!r}"
        )

    if mismatches:
        return (
            f"--config-name {cli.config_name!r} disagrees with the run flags "
            "(a config name that lies about its run makes the parity row a lie): "
            + "; ".join(mismatches)
        )
    return None


def config_name_precision(config_name: str) -> str:
    """The ``fp64`` / ``mp`` field of a validated config name."""
    return parse_config_name(config_name).precision


def refuse_setup(
    cli: InferenceCLI,
    inversions: tuple[str, ...],
    transformers: tuple[str | None, ...] = (None,),
) -> str | None:
    """An error when this leaf's likelihood has no such setup, else ``None``.

    The grammar admits every inversion and transformer any likelihood has; a leaf
    admits only its own. The imaging chain solves a dense or sparse inversion and has
    no transformer, so ``--inversion na`` or ``--transformer nufft`` there would label
    a run as something it is not.
    """
    if cli.inversion not in inversions:
        return f"--inversion {cli.inversion!r}: this likelihood runs only {inversions}"
    transformer = getattr(cli, "transformer", None)
    if transformer not in transformers:
        allowed = tuple(t for t in transformers if t) or "none"
        return f"--transformer {transformer!r}: this likelihood takes {allowed}"
    return None


# ---------------------------------------------------------------------------
# Backend environment — set BEFORE autolens is imported
# ---------------------------------------------------------------------------


#: The thread-pool variables a ``numba_cpu`` leg pins to ``"1"`` — see
#: :func:`set_backend_env`. The same five autolens_profiling's
#: ``_production_config.pin_thread_env`` sets.
NUMBA_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def set_backend_env(backend: str, cores: int) -> dict:
    """Export the environment the backend needs, returning what was set.

    Must run before ``import autolens``: ``JAX_ENABLE_X64`` and ``JAX_PLATFORMS``
    are read by JAX at import, and ``PYAUTO_DISABLE_JAX`` by the library's
    ``jax_wrapper``. ``NPROC`` sizes XLA's CPU thread pool (an unset one lets a
    job take the whole node; ``NPROC=1`` costs ~3x on JAX-CPU).

    ``numba_cpu`` also pins the five BLAS / OpenMP pools
    (:data:`NUMBA_THREAD_ENV`) to one thread. A numba leg parallelises by
    forking ``cores`` Nautilus workers, and each worker that inherits an
    unpinned OpenBLAS / MKL pool spawns one thread per core of the node: the
    2026-10-17 RAL sparse rows ran 2.6 s wall per evaluation against the
    ~0.06-0.08 s the autolens_profiling single-thread rates predict for eight
    workers, and PyAutoFit warned about it every stage (autolens_inference#38).
    The profiling cells pin the same five (``_production_config.pin_thread_env``).
    """
    env = {"JAX_ENABLE_X64": "True"}

    if backend == "numba_cpu":
        # Belt (process-wide) to `use_jax=False`'s braces (per Analysis).
        env["PYAUTO_DISABLE_JAX"] = "1"
        # One BLAS/OpenMP thread per forked worker. This only reaches libraries
        # imported AFTER this call (numpy/scipy read these at import, and the
        # runner may already have imported numpy), which is why the numba
        # submits export the same five before `python3` as well.
        env.update(NUMBA_THREAD_ENV)
    elif backend == "jax_cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["NPROC"] = str(cores)
    elif backend == "jax_gpu":
        env["JAX_PLATFORMS"] = "cuda,cpu"
        env["NPROC"] = str(cores)
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:  # pragma: no cover — argparse constrains the vocabulary
        raise ValueError(f"Unknown backend {backend!r}")

    os.environ.update(env)
    return env


# ---------------------------------------------------------------------------
# Per-stage row capture
# ---------------------------------------------------------------------------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _as_float(value):
    """Coerce a numeric-looking value to ``float``, else ``None``.

    ``samples_info["time"]`` is a **string** of seconds, not a number:
    ``autofit.non_linear.timer.Timer.update`` writes
    ``str(time.time() - start)``. Stored as it comes, ``wall_s`` is a JSON
    string, which every numeric consumer downstream (``build_readme``'s
    ``_format_time``, the stage plot, a rate calculation) silently renders as an
    em dash or raises on. Coerced here, once, where the value enters the row.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stage_output_dirs(search_root: Path, search_name: str) -> list[Path]:
    """Every identifier directory written for ``search_name`` under ``search_root``.

    ``search_root`` is ``<output root>[/test_mode]/<path_prefix>``. The one
    directory level this function cannot predict is the autofit identifier hash,
    so it is globbed rather than derived — deriving it would mean recomputing
    ``paths.identifier``, which needs the post-``modify_model`` model and creates
    the output tree as a side effect.
    """
    base = search_root / search_name
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)


def stage_output_dir(search_root: Path, search_name: str) -> Path | None:
    """The most recently touched identifier directory for ``search_name``."""
    dirs = stage_output_dirs(search_root, search_name)
    return dirs[-1] if dirs else None


def checkpoint_exists(search_root: Path, search_name: str) -> bool:
    """Is a Nautilus resume marker present for this stage *right now*?

    ``files/search_internal/checkpoint.hdf5`` is deleted by
    ``NautilusSearch.write_search_internal`` the moment a fit completes, so this
    must be sampled **before** ``search.fit``, never after.
    """
    return any(
        (d / "files" / "search_internal" / "checkpoint.hdf5").exists()
        for d in stage_output_dirs(search_root, search_name)
    )


def stage_completed(search_root: Path, search_name: str) -> bool:
    """Is ``search_name`` complete under ``search_root``: any identifier directory
    holding PyAutoFit's ``.completed`` marker.

    The same marker :func:`_stage_row` reads for ``completed``, and the test the
    numba clean-slate submits poll (``python3 scripts/misc/slam/_runner.py
    --stage-complete <seed_dir>``) before a ``--stages-from`` run starts.
    ``.completed`` is written only after the fit's archive has landed, so a stage
    carrying it is loadable.
    """
    return any((d / ".completed").exists() for d in stage_output_dirs(search_root, search_name))


class SharedStageIncomplete(RuntimeError):
    """A ``--stages-from`` stage is not complete in the other config's tree."""


def validate_stages_from(cli: InferenceCLI) -> str | None:
    """Return an error for a ``--stages-from`` that disagrees with this run.

    The grammar and the shareable-stage rule are checked at parse time
    (``parse_stages_from``); what needs this run's flags is checked here: a run
    may not share from itself, and both chains must run at the same precision
    (a fp64 stage is not the mp chain's stage 1).
    """
    if cli.stages_from is None:
        return None
    shared_config, _stages = cli.stages_from
    if shared_config == cli.config_name:
        return f"--stages-from names this run's own config {shared_config!r}"
    own = parse_config_name(cli.config_name or "")
    other = parse_config_name(shared_config)
    if own is not None and other is not None and own.precision != other.precision:
        return (
            f"--stages-from config {shared_config!r} runs at precision "
            f"{other.precision!r}, this run at {own.precision!r}"
        )
    return None


def shared_stage_names(cli: InferenceCLI) -> dict[str, str]:
    """``{stage_key: shared_from_config}`` for the stages this run loads."""
    if cli.stages_from is None:
        return {}
    shared_config, stages = cli.stages_from
    return {stage: shared_config for stage in stages}


def shared_stage_prefixes(
    cli: InferenceCLI,
    dataset_class: str,
    instrument: str,
    variant: str,
    seed: int,
    shared_variant: str = BASE_VARIANT,
    pipeline: str = "slam",
) -> dict[str, Path]:
    """``{stage_key: path_prefix}`` for the shared stages: the OTHER config's prefix.

    The same output root and seed as this run, the config level naming the chain
    the stage is loaded from — and the variant level always
    :data:`BASE_VARIANT`, whatever this run's ``variant``. Every shareable stage
    (:data:`SHAREABLE_STAGES`, today only ``source_lp[1]``) is parametric: it
    has no mesh and no light mode, so its model, search and identifier are the
    same for ``slam_base``, ``rect_40`` and ``slam_delaunay``, and the one
    completed copy lives in the base run's tree. Keying it by this run's
    variant sent a ``slam_delaunay`` leg looking under
    ``slam_delaunay/<shared_config>/`` for a stage only ever fitted under
    ``slam_base/<shared_config>/`` (autolens_inference#38). ``variant`` is kept
    in the signature so the call site does not change if a mesh-dependent
    stage ever becomes shareable.

    ``shared_variant`` names the variant whose tree holds the shared copy; it is
    :data:`BASE_VARIANT` for the imaging chain, and a family names its own
    (``Family.shared_variant``).
    """
    del variant  # every shareable stage is mesh-free: see the docstring
    return {
        stage: output_path_prefix(
            dataset_class, instrument, shared_variant, shared_config, seed, pipeline
        )
        for stage, shared_config in shared_stage_names(cli).items()
    }


def chain_wall_from(stages: list[dict]) -> dict:
    """The row-level chain wall: fitted stages only.

    ``chain_wall_s`` sums ``wall_s`` over the stages this process fitted and is
    null when any of them lacks one (a partial sum would read as a chain wall).
    Stages loaded from another config are never in the sum: their wall is that
    config's, and counting it would make a shared chain look as slow as the
    chain it borrowed from, or a parity row compare one stage twice.
    """
    fitted = [stage for stage in stages if not stage.get("shared_from_config")]
    walls = [stage.get("wall_s") for stage in fitted]
    total = None
    if fitted and all(isinstance(w, (int, float)) for w in walls):
        total = float(sum(walls))
    return {
        "chain_wall_s": total,
        "chain_wall_excludes": [
            stage.get("name") for stage in stages if stage.get("shared_from_config")
        ],
        "chain_wall_note": CHAIN_WALL_NOTE,
    }


def mark_shared_stage(row: dict, shared_from_config: str, load_s: float | None) -> dict:
    """Turn a stage row built from a loaded fit into a declared shared stage."""
    fields = parse_config_name(shared_from_config)
    fitted_backend = fields.backend if fields else None
    row.update(
        shared_from_config=shared_from_config,
        shared_stage_note=SHARED_STAGE_NOTE,
        # The process that FITTED this stage, read from the config name: a
        # numba chain sharing a jax_cpu stage must not claim use_jax=False here.
        fitted_backend=fitted_backend,
        use_jax=None if fitted_backend is None else fitted_backend != "numba_cpu",
        # This process only loaded it; its search.fit call is not a stage wall.
        total_wall_s=None,
        shared_load_s=_as_float(load_s),
    )
    return row


def _stage_row(
    name: str,
    output_path: Path | None,
    *,
    n_live: int,
    n_batch: int,
    free_parameters: int | None,
    total_wall_s: float | None,
    compile_s: float | None,
    resumed: bool,
    inversion_applies: bool,
    posterior: dict | None = None,
    truth_delta_sigma: dict | None = None,
    max_log_likelihood: float | None = None,
    compile_s_note: str | None = None,
) -> dict:
    """Build one stage row from the search's on-disk artefacts plus the caller's timings.

    Everything read from disk is read here, so the row can be rebuilt (and
    unit-tested) from a search directory alone:

    - ``files/samples_info.json`` -> ``wall_s`` (the sampler's own clock),
      ``likelihood_evals`` (``total_samples``, which is nautilus's reject-inclusive
      ``n_like``) and ``log_evidence``.
    - ``positions.info`` -> ``positions_info_present``. A pixelized stage without
      it is not citable.
    - ``.completed`` -> ``completed``.

    ``total_accepted_samples`` is deliberately not read: nautilus 1.0.5 sets it
    to ``n_like`` as well, so it is a duplicate, not an acceptance count.
    """
    info = {}
    if output_path is not None:
        info = _read_json(output_path / "files" / "samples_info.json") or {}

    likelihood_evals = info.get("total_samples")
    row = {
        "name": name,
        "free_parameters": free_parameters,
        "n_live": n_live,
        "n_batch": n_batch,
        "wall_s": _as_float(info.get("time")),
        "total_wall_s": _as_float(total_wall_s),
        "compile_s": _as_float(compile_s),
        "likelihood_evals": int(likelihood_evals) if likelihood_evals is not None else None,
        "log_evidence": _as_float(info.get("log_evidence")),
        "log_evidence_err": None,
        "log_evidence_err_note": LOG_EVIDENCE_ERR_NOTE,
        "max_log_likelihood": max_log_likelihood,
        "posterior": posterior or {},
        "truth_delta_sigma": truth_delta_sigma or {},
        "inversion_applies": inversion_applies,
        "positions_info_present": bool(
            output_path is not None and (output_path / "positions.info").exists()
        ),
        "completed": bool(output_path is not None and (output_path / ".completed").exists()),
        "resumed": bool(resumed),
        "output_path": str(output_path) if output_path is not None else None,
    }
    if compile_s_note:
        row["compile_s_note"] = compile_s_note
    return row


# ---------------------------------------------------------------------------
# Posterior + truth
# ---------------------------------------------------------------------------


def posterior_path(name) -> str:
    """The one dotted path for an entry of ``samples.names``.

    ``AbstractPriorModel.all_names`` returns, per prior, a **tuple of every name
    that prior answers to** — ``('galaxies.lens.mass.einstein_radius',)`` for a
    unique one, and more entries where a prior is reachable by several paths.
    Each entry is already a dotted string, so indexing the tuple gives an alias,
    not a leaf; the first alias is taken as canonical. A plain tuple of path
    segments (what a hand-written test or an older autofit hands over) is joined
    instead, so both shapes land on the same string.
    """
    if isinstance(name, str):
        return name
    parts = [str(part) for part in name]
    if not parts:
        return ""
    if len(parts) == 1 or any("." in part for part in parts):
        return parts[0]
    return ".".join(parts)


def posterior_keys(names) -> list[str]:
    """Map ``samples.names`` onto result-row keys.

    A leaf name that is unique across the model's free parameters becomes the
    key on its own (``einstein_radius``, ``slope``, ``gamma_1``); anything that
    repeats (``centre_0`` on twenty Gaussians) keeps its full dotted path. This
    is what lets the parity view name a row ``einstein_radius``, and the derived
    ``shear_magnitude`` find its ``gamma_1`` / ``gamma_2`` columns, without
    inventing an alias table — while never silently merging two parameters that
    happen to share a leaf.
    """
    paths = [posterior_path(name) for name in names]
    leaves = [path.rsplit(".", 1)[-1] for path in paths]
    counts: dict[str, int] = {}
    for leaf in leaves:
        counts[leaf] = counts.get(leaf, 0) + 1
    return [leaf if counts[leaf] == 1 else path for leaf, path in zip(leaves, paths)]


def posterior_from_result(af, result) -> dict:
    """``{key: {median, sigma, lower_1, upper_1}}`` for every free parameter.

    Marginalised with the library's own ``af.marginalize`` over the weighted
    samples, rather than read off ``samples_summary.errors_at_sigma_1`` — the
    same numbers, but computed through one code path that also serves the
    derived quantities below (``shear_magnitude`` is not a free parameter and so
    has no summary entry of its own).
    """
    samples = result.samples
    names = list(samples.names)
    keys = posterior_keys(names)
    parameter_lists = samples.parameter_lists
    weight_list = samples.weight_list

    posterior: dict[str, dict] = {}
    columns: dict[str, list[float]] = {}
    for index, key in enumerate(keys):
        column = [float(vector[index]) for vector in parameter_lists]
        columns[key] = column
        posterior[key] = _marginalized(af, column, weight_list)

    # Derived: shear magnitude. Computed from the sampled (gamma_1, gamma_2)
    # pairs and then marginalised, so it carries the pair's covariance rather
    # than a first-order propagation of two independent errors.
    if "gamma_1" in columns and "gamma_2" in columns:
        magnitude = [
            math.sqrt(g1 * g1 + g2 * g2) for g1, g2 in zip(columns["gamma_1"], columns["gamma_2"])
        ]
        posterior["shear_magnitude"] = _marginalized(af, magnitude, weight_list)

    return posterior


def _marginalized(af, column: list[float], weight_list) -> dict:
    median, lower, upper = af.marginalize(parameter_list=column, sigma=1.0, weight_list=weight_list)
    return {
        "median": float(median),
        "sigma": float(upper - lower) / 2.0,
        "lower_1": float(lower),
        "upper_1": float(upper),
    }


def truth_dict(tracer) -> dict:
    """The truth values a stage row can be compared against, from ``tracer.json``.

    Keyed to agree with :func:`posterior_keys`, so ``truth_delta_sigma`` is a
    plain key intersection and never a name-matching heuristic.
    """
    truths: dict[str, float] = {}
    for galaxy in getattr(tracer, "galaxies", []) or []:
        mass = getattr(galaxy, "mass", None)
        if mass is not None:
            for attribute in ("einstein_radius", "slope"):
                value = getattr(mass, attribute, None)
                if isinstance(value, (int, float)):
                    truths[attribute] = float(value)
        shear = getattr(galaxy, "shear", None)
        if shear is not None:
            gamma_1 = getattr(shear, "gamma_1", None)
            gamma_2 = getattr(shear, "gamma_2", None)
            if isinstance(gamma_1, (int, float)) and isinstance(gamma_2, (int, float)):
                truths["gamma_1"] = float(gamma_1)
                truths["gamma_2"] = float(gamma_2)
                truths["shear_magnitude"] = math.sqrt(gamma_1**2 + gamma_2**2)
    return truths


def truth_delta_sigma_from(posterior: dict, truths: dict) -> dict:
    """``(median - truth) / sigma`` for every key the truth and the posterior share.

    A key whose posterior has **zero width** maps to ``None`` rather than being
    dropped. A degenerate posterior is a real outcome — a resumed test-mode fit
    collapses every sample onto one point, and so does a real fit that never
    moved — and "the parameter was compared and its σ was zero" is a different
    statement from "the parameter was not in the truth", which is what a missing
    key would say.
    """
    deltas = {}
    for key, truth in truths.items():
        entry = posterior.get(key)
        if not entry:
            continue
        sigma = entry.get("sigma")
        deltas[key] = None if not sigma else (entry["median"] - truth) / sigma
    return deltas


# ---------------------------------------------------------------------------
# Output tree and row provenance
# ---------------------------------------------------------------------------


def output_path_prefix(
    dataset_class: str,
    instrument: str,
    variant: str,
    config_name: str,
    seed: int,
    pipeline: str = "slam",
) -> Path:
    """The PyAutoFit ``path_prefix`` for one leg:
    ``autolens/<dataset_class>/<instrument>/<pipeline>/<variant>/<config_name>/seed_<n>``
    (``pipeline`` is ``slam`` for every chain; a single-search family names its own).

    Under the output root this reads dataset class, instrument, then pipeline,
    with the pipeline's variant and config as the innermost levels (#35 dropped
    the dated ``cleanslate_2026_10_17/slam/<dataset_class>/<instrument>`` root).

    ``config_name`` and ``seed_<n>`` live in the PATH, not only in the payload:
    ``seed`` is an autofit identifier field but ``config_name`` is not, so two
    backends run at one seed would otherwise hash to the same identifier and
    resume each other's completed fit in seconds, re-stamping it as their own.
    The same argument makes ``variant`` a path level — the mesh is not an
    identifier field either, so a Delaunay leg and a rectangular leg of the same
    config and seed would collide in exactly the same way.
    """
    return (
        Path("autolens")
        / dataset_class
        / instrument
        / pipeline
        / variant
        / config_name
        / f"seed_{seed}"
    )


def slurm_job_id_from(environ) -> str | None:
    """The SLURM job a row was measured in, from the job's environment.

    Inside an array task it is ``<SLURM_ARRAY_JOB_ID>_<SLURM_ARRAY_TASK_ID>`` —
    the id ``sacct`` and the ``.out`` file name (``%A_%a``) use, and the one a
    reader can look up. Outside an array it is ``SLURM_JOB_ID``; outside SLURM
    (a laptop, a CI witness) it is ``None``, never an invented id.
    """
    array_job = environ.get("SLURM_ARRAY_JOB_ID")
    array_task = environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{array_job}_{array_task}"
    return environ.get("SLURM_JOB_ID") or None


def release_pin_from(environ) -> str | None:
    """The PyAuto release a submit pinned its libraries to (``PYAUTO_RELEASE_PIN``).

    A pinned submit exports it next to ``PYAUTO_LIB_BASE`` (see
    ``hpc/README.md``, "Running against a pinned release"). It is a *claim*;
    ``library_revisions`` is the evidence, and a reader checks one against the
    other.
    """
    return environ.get("PYAUTO_RELEASE_PIN") or None


def library_revisions_or_note(root: Path) -> tuple[dict | None, str | None]:
    """``(library_revisions, note)`` for the row, never fatal.

    ``experiments.prepared.library_revisions`` pins the five import-selected
    PyAuto* checkouts' HEAD commits and refuses (``ValueError``) a library that
    is not a clean tracked checkout. A prepared export must refuse — a frozen
    problem bound to an unknown revision is unverifiable — but a SLaM leg must
    not die for it: the row records ``null`` and says why, which is the honest
    record of a run made against an unpinnable tree.
    """
    from experiments.prepared import library_revisions

    try:
        return library_revisions(root), None
    except (ValueError, OSError) as exc:
        return None, f"library revisions could not be pinned: {exc}"


def _write_results(json_path: Path, payload: dict) -> None:
    json_path.write_text(json.dumps(payload, indent=2))


def _plot_stages(png_path: Path, payload: dict) -> None:
    """Wall-clock per stage, as a bar chart beside the JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = payload.get("stages") or []
    labels = [stage["name"] for stage in stages]
    walls = [stage.get("wall_s") or 0.0 for stage in stages]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(9, max(3.0, len(labels) * 0.7)))
    positions = range(len(labels))
    bars = ax.barh(list(positions), walls, color=plt.cm.viridis(0.55), height=0.6)
    for bar, wall in zip(bars, walls):
        ax.text(
            bar.get_width() + (max(walls) or 1.0) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{wall:.1f} s",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Sampler wall clock (s)", fontsize=11)
    fig.suptitle(f"SLaM stages: {payload.get('target')}", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{payload.get('config_name')}  |  AutoLens v{payload.get('version')}",
        fontsize=9,
    )
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _device_info() -> dict:
    """``device_info_dict()``, never fatal.

    A numba leg runs with ``PYAUTO_DISABLE_JAX=1`` but JAX is still importable,
    so the dict is still the honest record of what the process was pointed at.
    If JAX cannot be imported at all the row says so rather than losing the run.
    """
    try:
        return device_info_dict()
    except Exception as exc:
        return {"backend": None, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Results path
# ---------------------------------------------------------------------------


def validation_row_paths(
    root: Path,
    dataset_type: str,
    likelihood_variant: str,
    instrument: str,
    model_variant: str,
    config_name: str,
    seed: int,
    results_tag: str | None = None,
) -> tuple[Path, Path]:
    """``(json_path, png_path)`` of one validation row, created on demand:
    ``results/<dataset_type>/<likelihood_variant>/<instrument>/validation/<model_variant>/
    <setup_id>[__<tag>]/seed_<n>/stages[_<host_class>].json``.

    The mirrored layout of autolens_inference#41, for a likelihood named outright. The
    config name splits into the ``setup_id`` folder and a ``host_class`` that is
    provenance only (a ``local_*`` row's file is ``stages_local_laptop.json``); a
    ``results_tag`` suffixes the setup folder so a re-measurement never overwrites the
    earlier row. The imaging chain reaches the same layout from its run variant
    (``_runner.results_paths``).
    """
    setup_id, host_class = split_config_name(config_name)
    setup_dir = f"{setup_id}__{results_tag}" if results_tag else setup_id
    directory = (
        root
        / "results"
        / dataset_type
        / likelihood_variant
        / instrument
        / "validation"
        / model_variant
        / setup_dir
        / f"seed_{seed}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = row_stem("stages", host_class)
    return directory / f"{stem}.json", directory / f"{stem}.png"


# ---------------------------------------------------------------------------
# One leg
# ---------------------------------------------------------------------------


class Leg:
    """One ``(variant, config_name, seed)`` run: the flags, the modelling stack, the
    output tree, the row provenance and the stage runner.

    Built by :func:`open_leg` once the backend environment is exported and the
    modelling stack imported. ``af`` / ``al`` are the imported namespaces, so a
    stage builder never re-enters the import.
    """

    def __init__(
        self,
        cli: InferenceCLI,
        *,
        af,
        al,
        dataset_class: str,
        instrument: str,
        variant: str,
        output_root: Path,
        root: Path,
        env: dict,
        pipeline: str = "slam",
    ):
        from autonerves.test_mode import is_test_mode, with_test_mode_segment

        self.cli = cli
        self.af = af
        self.al = al
        self.dataset_class = dataset_class
        self.instrument = instrument
        self.variant = variant
        self.output_root = output_root
        self.root = root
        self.env = env
        self.pipeline = pipeline
        self.config_name = cli.config_name
        self.precision = config_name_precision(cli.config_name)
        self.use_jax = cli.backend != "numba_cpu"
        self.seed = cli.seed
        self.cores = cli.cores
        self.test_mode = bool(is_test_mode())
        self._with_test_mode_segment = with_test_mode_segment
        self.dataset_path = root / "dataset" / dataset_class / instrument
        self.shared_from = shared_stage_names(cli)
        self.stage_settings_search: dict = {}
        self.stage_search_roots: dict[str, Path] = {}
        self.library_revisions = None
        self.library_revisions_note = None
        self.release_pin = None
        self.slurm_job_id = None
        #: The lens-light mode the stage being built is fitted under (``linear`` unless a
        #: family with ``light_fixed`` is fitting its last stage under ``fixed``).
        self.lens_light_mode = "linear"
        self.settings_search, self.search_root = self.settings_search_for(variant)

    # --- the output tree ------------------------------------------------------

    def settings_search_for(self, variant: str, config_name: str | None = None):
        """``(SettingsSearch, search_root)`` for ``variant`` (and another config's tree,
        for a shared stage).

        ``config_name``, ``variant`` and ``seed_<n>`` live in the PATH because ``seed``
        is an autofit identifier field and the other two are not, so legs that differ
        only in those would hash to one identifier and resume each other's completed fit
        (:func:`output_path_prefix`).
        """
        prefix = output_path_prefix(
            self.dataset_class,
            self.instrument,
            variant,
            config_name or self.config_name,
            self.seed,
            self.pipeline,
        )
        settings_search = self.af.SettingsSearch(
            path_prefix=prefix,
            unique_tag=None,
            info=None,
            session=None,
            # Nautilus routes on `analysis._use_jax`: under JAX it vmaps with
            # `n_batch` and ignores the pool, under numba it forks
            # `number_of_cores` workers. Never hand it a Pool object.
            number_of_cores=1 if self.use_jax else self.cores,
        )
        return settings_search, self._with_test_mode_segment(self.output_root) / prefix

    def share_stages(self, search_names: dict[str, str], shared_variant: str = BASE_VARIANT):
        """Resolve every ``--stages-from`` stage under the other config's tree.

        Returns :data:`SHARED_STAGE_INCOMPLETE_EXIT` (after saying why) when a shared
        stage is not complete there, else ``None``. Called before the dataset loads, so
        a refusal costs nothing and fits nothing.
        """
        prefixes = shared_stage_prefixes(
            self.cli,
            self.dataset_class,
            self.instrument,
            self.variant,
            self.seed,
            shared_variant,
            self.pipeline,
        )
        for stage_key, shared_config in self.shared_from.items():
            prefix = prefixes[stage_key]
            self.stage_settings_search[stage_key] = self.af.SettingsSearch(
                path_prefix=prefix,
                unique_tag=None,
                info=None,
                session=None,
                number_of_cores=1 if self.use_jax else self.cores,
            )
            stage_root = self._with_test_mode_segment(self.output_root) / prefix
            self.stage_search_roots[stage_key] = stage_root
            search_name = search_names[stage_key]
            complete = stage_completed(stage_root, search_name)
            print(
                f"  shared:     {search_name} from {shared_config} ({stage_root}) complete={complete}"
            )
            if not complete:
                print(
                    f"ERROR: --stages-from {shared_config}:{stage_key} but "
                    f"{search_name} is not complete under "
                    f"{stage_root} (no identifier directory holds "
                    f"`.completed`). Refusing to run: fitting it here would make this "
                    f"chain a different chain from the one it claims to share. Wait for "
                    f"that stage (the clean-slate submits poll `_runner.py "
                    f"--stage-complete`) or drop --stages-from. Nothing was fitted.",
                    file=sys.stderr,
                )
                return SHARED_STAGE_INCOMPLETE_EXIT
        return None

    def search_settings(self, stage_key: str):
        """The ``SettingsSearch`` a stage's search is built with: the other config's for
        a shared stage, this leg's otherwise."""
        return self.stage_settings_search.get(stage_key, self.settings_search)

    def nautilus(self, stage: Stage, settings_search=None):
        """The stage's Nautilus search, at its ``n_live`` / ``n_batch`` and this leg's seed."""
        settings_search = settings_search or self.search_settings(stage.key)
        return self.af.Nautilus(
            name=stage.search_name,
            **settings_search.search_dict,
            n_live=stage.n_live,
            n_batch=stage.n_batch,
            seed=self.seed,
        )

    # --- provenance -----------------------------------------------------------

    def sample_provenance(self) -> None:
        """Sample the row provenance once, before any fit: the import-selected library
        checkouts, the release the submit says it pinned, and the SLURM job this is."""
        self.library_revisions, self.library_revisions_note = library_revisions_or_note(self.root)
        self.release_pin = release_pin_from(os.environ)
        self.slurm_job_id = slurm_job_id_from(os.environ)
        print(f"  libraries:  {self.library_revisions or self.library_revisions_note}")
        print(f"  release pin: {self.release_pin}   slurm job: {self.slurm_job_id}")

    # --- one stage --------------------------------------------------------------

    def run_stage(
        self,
        stage_key: str,
        search_name: str,
        built,
        *,
        n_live: int,
        n_batch: int,
        inversion_applies: bool,
        truths: dict,
        stage_search_root: Path | None = None,
        before_fit: Callable | None = None,
        after_row: Callable | None = None,
        likelihood_evaluated=None,
        declares_likelihood: bool = False,
    ) -> tuple[object, dict]:
        """Fit one stage (or load it, for a shared stage) and build its row.

        ``before_fit(model, analysis)`` returns ``(model, context)`` and runs after the
        resume marker is sampled and before the compile probe (the imaging chain's
        prepared export); ``after_row(row, result, posterior, resumed, context)`` runs
        once the row is built. ``declares_likelihood`` writes ``likelihood_evaluated``
        on the row (a family-driver row; the imaging rows predate the field).
        """
        af = self.af
        model, analysis, search = built
        if stage_search_root is None:
            stage_search_root = self.stage_search_roots.get(stage_key)
        stage_root = stage_search_root if stage_search_root is not None else self.search_root
        shared_config = self.shared_from.get(stage_key)
        if shared_config is not None:
            # The directory-level test passed before the chain started; this is
            # the exact one. PyAutoFit loads a fit only when THIS model+search
            # hashes to the completed identifier — anything else would be a
            # fresh fit written into the other config's tree.
            search.paths.model = analysis.modify_model(model)
            search.paths.unique_tag = search.unique_tag
            if not search.paths.is_complete:
                raise SharedStageIncomplete(
                    f"--stages-from {shared_config}:{stage_key}: this run's "
                    f"{search_name} identifier {search.paths.identifier} is not a "
                    f"completed fit under {stage_root} (the model or search settings "
                    f"differ from the run that fitted it). Nothing was fitted."
                )

        # Sampled BEFORE the fit: nautilus deletes checkpoint.hdf5 on completion.
        resumed = checkpoint_exists(stage_root, search_name)
        context = None
        if before_fit is not None:
            model, context = before_fit(model, analysis)

        compile_s = None
        compile_note = None
        if shared_config is not None:
            compile_note = "no compile probe: stage loaded from another config, not fitted"
        elif self.use_jax and self.test_mode:
            compile_note = COMPILE_S_TEST_MODE_NOTE
        elif self.use_jax:
            try:
                instance = model.instance_from_prior_medians()
                start = time.perf_counter()
                analysis.log_likelihood_function(instance)
                compile_s = time.perf_counter() - start
            except Exception as exc:  # pragma: no cover — probe is best-effort
                compile_note = f"compile probe failed: {type(exc).__name__}: {exc}"

        print(f"\n--- {search_name} (n_live={n_live}, n_batch={n_batch}) ---")
        start = time.perf_counter()
        result = search.fit(model=model, analysis=analysis, **self.settings_search.fit_dict)
        total_wall_s = time.perf_counter() - start

        output_path = stage_output_dir(stage_root, search_name)

        posterior: dict = {}
        max_log_likelihood = None
        free_parameters = None
        try:
            posterior = posterior_from_result(af, result)
            max_log_likelihood = float(result.samples.max_log_likelihood_sample.log_likelihood)
            free_parameters = len(list(result.samples.names))
        except Exception as exc:
            print(f"  WARNING [{search_name}]: posterior capture failed — {exc}")
        if free_parameters is None:
            free_parameters = getattr(model, "prior_count", None)

        row = _stage_row(
            search_name,
            output_path,
            n_live=n_live,
            n_batch=n_batch,
            free_parameters=free_parameters,
            total_wall_s=total_wall_s,
            compile_s=compile_s,
            resumed=resumed,
            inversion_applies=inversion_applies,
            posterior=posterior,
            truth_delta_sigma=truth_delta_sigma_from(posterior, truths),
            max_log_likelihood=max_log_likelihood,
            compile_s_note=compile_note,
        )
        if shared_config is not None:
            mark_shared_stage(row, shared_config, total_wall_s)
        else:
            row["use_jax"] = self.use_jax
        if declares_likelihood:
            row["likelihood_evaluated"] = likelihood_evaluated
        if after_row is not None:
            after_row(row, result, posterior, resumed, context)
        print(
            f"  {search_name}: wall {row['wall_s']}s, evals {row['likelihood_evals']}, "
            f"log Z {row['log_evidence']}, positions.info "
            f"{row['positions_info_present']}, completed {row['completed']}"
        )
        return result, row

    # --- the row ----------------------------------------------------------------

    def payload(
        self,
        *,
        target: str,
        variant: str,
        stages: list[dict],
        status: str,
        identity: dict | None = None,
        model: dict | None = None,
    ) -> dict:
        """The ``stages.json`` payload. ``identity`` sits after ``variant`` (the imaging
        chain's ``lens_light_mode``; a family's ``likelihood`` / ``model_variant`` /
        ``transformer``), ``model`` after ``use_jax`` (the fields that record the fitted
        model: mesh, regularization). The key order is the imaging row's, unchanged."""
        cli = self.cli
        payload = {
            "schema_version": SCHEMA_VERSION,
            "target": target,
            "variant": variant,
            **(identity or {}),
            "results_tag": cli.results_tag,
            "library_revisions": self.library_revisions,
            "release_pin": self.release_pin,
            "slurm_job_id": self.slurm_job_id,
            "config_name": self.config_name,
            "backend": cli.backend,
            "inversion": cli.inversion,
            "precision": self.precision,
            "instrument": self.instrument,
            "dataset_class": self.dataset_class,
            "seed": self.seed,
            "sampler": "nautilus",
            "version": self.al.__version__,
            "device": _device_info(),
            "cores": self.cores,
            "use_jax": self.use_jax,
            **(model or {}),
            "stages_requested": cli.stages,
            "stages_from": (
                {"config_name": cli.stages_from[0], "stages": list(cli.stages_from[1])}
                if cli.stages_from is not None
                else None
            ),
            **chain_wall_from(stages),
            "test_mode": self.test_mode,
            "status": status,
            "stages": stages,
        }
        if self.library_revisions_note:
            payload["library_revisions_note"] = self.library_revisions_note
        return payload

    def write_row(self, json_path: Path, png_path: Path, payload: dict) -> None:
        """Write the row and its stage plot (a plot never kills a run)."""
        _write_results(json_path, payload)
        try:
            _plot_stages(png_path, payload)
        except Exception as exc:  # pragma: no cover — a plot must not kill a run
            print(f"  WARNING: stage plot failed — {exc}")
        print(f"\n  Result row: {json_path}")


def open_leg(
    cli: InferenceCLI,
    *,
    dataset_class: str,
    instrument: str,
    variant: str,
    env: dict,
    root: Path = _ROOT,
    pipeline: str = "slam",
    title: str = "SLaM run",
    details: tuple[str, ...] = (),
) -> Leg | int:
    """Import the modelling stack and open the leg, or return an exit code.

    Call it after :func:`set_backend_env` (whose exported environment ``env`` is): the
    imports here are the first that see it. Returns 0 under
    ``AUTOLENS_INFERENCE_SMOKE=1`` (the lint smoke: imports and flags only), 2 when
    ``--backend jax_gpu`` finds no GPU, else the :class:`Leg`. ``details`` are extra
    banner lines (the family's model, e.g. its mesh).
    """
    import autofit as af
    import autolens as al
    from autonerves import jax_wrapper  # noqa: F401 — sets the JAX env before autolens

    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {sys.argv[0]}: imports + module setup OK; exiting.")
        return 0

    if cli.backend == "jax_gpu":
        import jax

        # Two different non-GPU outcomes, one refusal. On a host with a CUDA
        # plugin that failed to find a device, `default_backend()` returns
        # 'cpu'; on a host with no plugin at all it *raises* ("Backend 'cuda' is
        # not in the list of known backends"). Both mean the same thing — this
        # is not a GPU run — so both take the same exit rather than one of them
        # surfacing as an unhandled traceback.
        try:
            backend_in_use = jax.default_backend()
        except Exception as exc:
            backend_in_use = f"unavailable ({type(exc).__name__}: {exc})"
        if backend_in_use != "gpu":
            print(
                f"ERROR: --backend jax_gpu but jax.default_backend() == "
                f"{backend_in_use!r}. Refusing to run: a 'GPU' leg that silently "
                f"fell back to CPU reports a wall clock that is nothing of the "
                f"kind, and would sit in a parity row as if it were an A100 "
                f"measurement. Run this leg on a CUDA host (RAL --partition=gpu "
                f"--gres=gpu:1) with JAX_PLATFORMS=cuda,cpu exported in the "
                f"submit script.",
                file=sys.stderr,
            )
            return 2

    output_root = cli.output_dir or Path(os.environ.get("PYAUTO_OUTPUT_DIR") or (root / "output"))
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    af.conf.instance.push(new_path=root / "config", output_path=output_root)

    leg = Leg(
        cli,
        af=af,
        al=al,
        dataset_class=dataset_class,
        instrument=instrument,
        variant=variant,
        output_root=output_root,
        root=root,
        env=env,
        pipeline=pipeline,
    )
    print("=" * 78)
    print(
        f"{title} — {dataset_class} / {instrument} / {variant} / {leg.config_name} / seed {leg.seed}"
    )
    print("=" * 78)
    print(f"  variant:    {variant}")
    for line in details:
        print(f"  {line}")
    print(f"  results tag: {cli.results_tag}")
    print(f"  stages from: {cli.stages_from}")
    print(f"  backend:    {cli.backend}  (use_jax={leg.use_jax}, cores={leg.cores})")
    print(f"  inversion:  {cli.inversion}  transformer: {cli.transformer}")
    print(f"  precision:  {leg.precision}")
    print(f"  env set:    {env}")
    print(f"  output:     {output_root}")
    print(f"  test mode:  {leg.test_mode}")
    return leg


# ---------------------------------------------------------------------------
# The generic family driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One search of a family's chain.

    ``key`` is the ``--stages`` / ``--stages-from`` vocabulary entry and ``search_name``
    the search's name — the output-tree directory and the row's stage ``name``, so the
    citable identity of the stage. ``build(leg, context, results)`` returns
    ``(model, analysis, search)``; ``after(leg, context, results)``, when given, runs
    once the stage is fitted (re-derive the dataset there). ``inversion_applies`` says
    whether ``--inversion`` touches the stage; ``evaluates_likelihood`` False marks a
    stage whose evaluations are none of the profiled likelihoods (the parametric stage
    ahead of a pixelized chain), which the row records as ``likelihood_evaluated: null``.
    """

    key: str
    search_name: str
    n_live: int
    n_batch: int
    build: Callable
    inversion_applies: bool = False
    evaluates_likelihood: bool = True
    after: Callable | None = None


@dataclass(frozen=True)
class Family:
    """One likelihood's validation chain: what :func:`run_family` needs to run it.

    ``dataset_type`` / ``likelihood_variant`` are the likelihood exactly as the profiling
    catalogue names it (``interferometer`` / ``mge``, ``point_source_image`` /
    ``image_plane``): the row declares them and is filed under them. ``dataset_class``
    is the ``dataset/<class>/`` directory and the PyAutoFit output level (defaults to
    ``dataset_type``). ``load(leg)`` returns the stages' context (or an int exit code).
    ``variant(cli)`` names the run variant from the flags (default
    ``<pipeline>_<likelihood_variant>``); ``model_variant`` is the results folder below
    ``validation/``, ``default`` for the production model the profiling records time.
    ``model_fields(leg, context)`` returns the payload fields that record the model.
    ``inversions`` / ``transformers`` are the setups the likelihood has
    (:func:`refuse_setup`); ``shareable`` the stage keys ``--stages-from`` may load,
    from the ``shared_variant`` tree (default: this leg's own variant).

    ``light_fixed`` (likelihood-lifecycle 4c-1) marks a family whose dataset has lens light:
    ``--lens-light fixed|both`` then fits its LAST stage once per mode, as the imaging chain's
    ``mass_total[1]`` is fitted, the earlier stages once. The stage builder reads the mode off
    ``leg.lens_light_mode``; a ``fixed`` stage carries the ``_light_fixed`` variant in its
    output prefix and its row is the ``light_fixed`` model variant, so ``both`` writes two rows.
    """

    dataset_type: str
    likelihood_variant: str
    stages: tuple[Stage, ...]
    load: Callable
    dataset_class: str | None = None
    pipeline: str = "slam"
    inversions: tuple[str, ...] = ("dense", "sparse")
    transformers: tuple[str | None, ...] = (None,)
    shareable: tuple[str, ...] = ()
    shared_variant: str | None = None
    variant: Callable | None = None
    model_variant: str = "default"
    model_fields: Callable | None = None
    title: str = field(default="Validation run")
    light_fixed: bool = False

    @property
    def stage_keys(self) -> tuple[str, ...]:
        return tuple(stage.key for stage in self.stages)

    @property
    def likelihood(self) -> dict:
        return {"dataset_type": self.dataset_type, "variant": self.likelihood_variant}

    @property
    def data_class(self) -> str:
        return self.dataset_class or self.dataset_type

    def variant_of(self, cli: InferenceCLI) -> str:
        if self.variant is not None:
            return self.variant(cli)
        return f"{self.pipeline}_{self.likelihood_variant}"

    def target(self, instrument: str, variant: str, seed: int) -> str:
        """The parity-row ``target``: every leg of one likelihood, instrument, variant and
        seed is one parity row, whatever its setup."""
        return f"{self.dataset_type}/{self.likelihood_variant}/{instrument}/{variant}/seed{seed}"


def run_family(
    family: Family,
    default_instrument: str,
    default_mesh: str | None = None,
    default_mesh_pixels: int | None = None,
) -> int:
    """Run one leg of ``family``'s chain and write its ``stages.json``.

    Returns the exit code: 0 on success, 2 on a flag / environment refusal,
    :data:`SHARED_STAGE_INCOMPLETE_EXIT` for an incomplete shared stage. A leg that dies
    part way through still writes the stages it finished, under ``status: "failed:
    ..."``, and re-raises.
    """
    cli = parse_inference_cli(
        default_mesh=default_mesh,
        default_mesh_pixels=default_mesh_pixels,
        stages_vocabulary=family.stage_keys,
        shareable_stages=family.shareable,
    )
    instrument = cli.instrument or default_instrument
    variant = family.variant_of(cli)

    error = (
        validate_config_name(cli)
        or refuse_setup(cli, family.inversions, family.transformers)
        or validate_stages_from(cli)
    )
    if error is None and cli.lens_light != "linear" and not family.light_fixed:
        error = "--lens-light is the imaging SLaM chain's; this family has one light model"
    if error is None and cli.export_prepared is not None:
        error = "--export-prepared is the imaging SLaM chain's mass_total[1] export"
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    env = set_backend_env(cli.backend, cli.cores)
    leg = open_leg(
        cli,
        dataset_class=family.data_class,
        instrument=instrument,
        variant=variant,
        env=env,
        pipeline=family.pipeline,
        title=family.title,
        details=(f"likelihood: {family.dataset_type}/{family.likelihood_variant}",),
    )
    if isinstance(leg, int):
        return leg

    code = leg.share_stages(
        {stage.key: stage.search_name for stage in family.stages},
        family.shared_variant or variant,
    )
    if code is not None:
        return code

    context = family.load(leg)
    if isinstance(context, int):
        return context
    leg.sample_provenance()
    truths = getattr(context, "truths", None) or {}
    model_fields = family.model_fields(leg, context) if family.model_fields else {}
    # One row per lens-light mode: ``linear`` only, unless the family has lens light.
    modes = lens_light_modes(cli) if family.light_fixed else ("linear",)
    variants = {
        mode: variant + LENS_LIGHT_FIXED_SUFFIX if mode == "fixed" else variant for mode in modes
    }
    model_variants = {
        mode: LENS_LIGHT_FIXED_SUFFIX.lstrip("_") if mode == "fixed" else family.model_variant
        for mode in modes
    }
    rows: list[dict] = []
    mode_rows: dict[str, dict] = {}
    written: set[str] = set()

    def write(status: str, mode: str) -> None:
        json_path, png_path = validation_row_paths(
            leg.root,
            family.dataset_type,
            family.likelihood_variant,
            instrument,
            model_variants[mode],
            leg.config_name,
            leg.seed,
            cli.results_tag,
        )
        identity = {
            "likelihood": family.likelihood,
            "model_variant": model_variants[mode],
            "transformer": cli.transformer,
        }
        if family.light_fixed:
            identity["lens_light_mode"] = mode
        payload = leg.payload(
            target=family.target(instrument, variants[mode], leg.seed),
            variant=variants[mode],
            stages=[*rows, *([mode_rows[mode]] if mode in mode_rows else [])],
            status=status,
            identity=identity,
            model={"memo": cli.memo, **model_fields},
        )
        leg.write_row(json_path, png_path, payload)
        written.add(mode)

    def write_all(status: str) -> None:
        for mode in modes:
            if mode not in written:
                write(status, mode)

    results: dict[str, object] = {}
    try:
        for index, stage in enumerate(family.stages):
            per_mode = family.light_fixed and index == len(family.stages) - 1
            for mode in modes if per_mode else (None,):
                stage_search_root = None
                if mode is not None:
                    # The mode's own prefix, as the imaging chain's mass_total[1]: a fixed fit
                    # never resumes the linear one, and a fixed-only run resumes the linear
                    # chain's earlier stages.
                    leg.lens_light_mode = mode
                    settings_search, stage_search_root = leg.settings_search_for(variants[mode])
                    leg.stage_settings_search[stage.key] = settings_search
                result, row = leg.run_stage(
                    stage.key,
                    stage.search_name,
                    stage.build(leg, context, results),
                    n_live=stage.n_live,
                    n_batch=stage.n_batch,
                    inversion_applies=stage.inversion_applies,
                    truths=truths,
                    stage_search_root=stage_search_root,
                    likelihood_evaluated=family.likelihood if stage.evaluates_likelihood else None,
                    declares_likelihood=True,
                )
                results[stage.key] = result
                if mode is None:
                    rows.append(row)
                else:
                    mode_rows[mode] = row
                    write("complete", mode)
            if cli.stages == stage.key:
                write_all("stopped_early")
                return 0
            if stage.after is not None:
                stage.after(leg, context, results)
        write_all("complete")
        return 0
    except SharedStageIncomplete as exc:
        # A refusal, not a failure: nothing was fitted, so no row is written.
        print(f"ERROR: {exc}", file=sys.stderr)
        return SHARED_STAGE_INCOMPLETE_EXIT
    except BaseException as exc:
        write_all(f"failed: {type(exc).__name__}: {exc}")
        raise
