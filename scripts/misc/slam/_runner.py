"""The backend-parameterised SLaM base-run driver.

One function, :func:`run_slam`, runs the standard 5-stage imaging SLaM chain
end to end under any ``--backend`` × ``--inversion`` combination, records every
stage as a comparable row, and writes one result JSON (+ PNG) per
``(config_name, seed)``. Every ``scripts/<dataset_class>/slam/<target>.py`` leaf
is a ~25-line shim over it.

The chain mirrors ``autolens_workspace/scripts/guides/modeling/slam_start_here.py``
(the five module-level stage functions there, at the time of writing lines
148 / 236 / 317 / 397 / 481): ``source_lp[1]`` (Nautilus n_live 200 / n_batch
50) -> ``source_pix[1]`` (150/20) -> ``source_pix[2]`` (75/20) -> ``light[1]``
(150/20) -> ``mass_total[1]`` (150/20), with a 28x28
``RectangularBilinearAdaptDensity`` then ``RectangularBilinearAdaptImage`` mesh
under ``reg.Adapt`` (the ``slam_base`` **run variant**; ``--mesh delaunay``
swaps both stages for ``al.mesh.Delaunay`` under ``reg.AdaptSplit`` and names
its own variant folder — see :func:`mesh_models` and :func:`target_id`),
MGE 20x2 lens light / 20x1 source, the S/N 3.0 adapt-image
cap, and ``Isothermal + ExternalShear`` chained into a ``PowerLaw`` via
``al.util.chaining.mass_and_fields_from(..., unfix_mass_centre=True)``.

Deliberate departures from ``slam_start_here.py``
-------------------------------------------------

1. **A positions likelihood is attached to all FOUR pixelized stages**
   (``source_pix[1]``, ``source_pix[2]``, ``light[1]``, ``mass_total[1]``),
   where the workspace script attaches one only to ``source_pix[1]`` and
   ``mass_total[1]``. A mesh stage with no ``positions.info`` beside it is not
   citable in this repo (``AGENTS.md``, and the phase-4 Cortex witness reads
   ``positions_info_present`` on every pixelized stage), so the two stages the
   workspace leaves bare would produce rows that cannot be ruled on.

2. **``log_evidence_err`` is always ``null``.** nautilus 1.0.5 exposes no
   ``log_z_err``: ``NautilusSearch.samples_info_from`` returns only
   ``log_evidence``, ``total_samples``, ``total_accepted_samples``, ``time`` and
   ``number_live_points``. A derived or invented error would be worse than an
   absent one, so the field is written as ``null`` alongside
   ``log_evidence_err_note`` saying why.

3. **``--output-dir`` overrides the PyAutoFit *run output* root**, not the
   results directory (which is what ``_inference_cli.resolve_output_paths``
   means by it for the single-JSON leaves). A SLaM leg's bulk state is the run
   tree, and that is the thing a RAL job needs to redirect; the small result row
   always lands under
   ``results/slam/<dataset_class>/<instrument>/<variant>/<config_name>/``.

4. **The JAX compile probe is skipped under ``PYAUTO_TEST_MODE``.** In
   production ``compile_s`` is one timed ``analysis.log_likelihood_function``
   call made before ``search.fit``, which is the first-call JIT cost. Under test
   mode the fit itself makes exactly one likelihood call, so the probe would
   double the witness's cost to measure a number no witness reads; the field is
   written as ``null`` with ``compile_s_note``.

5. **Under ``PYAUTO_TEST_MODE`` the positions likelihood is built from the
   simulator's ``positions.json`` rather than from the previous result's point
   solver** (see :func:`positions_likelihood_from` for the measurement that
   forced it). Every pixelized stage still carries a positions likelihood and
   still writes ``positions.info``; only the *threshold's* provenance changes,
   and only in test mode, where the model it would have been solved from is a
   near-random prior draw.

Hazards this module is built around (do not rediscover them)
------------------------------------------------------------

- ``Imaging.apply_over_sampling`` rebuilds the ``Imaging`` object from scratch
  (``PyAutoArray/autoarray/dataset/imaging/dataset.py``) and therefore **drops
  the sparse operator**. The over-sample-map step before ``source_pix[2]`` is
  followed immediately by a second :func:`_apply_inversion` call.
- ``Imaging.apply_sparse_operator`` raises if
  ``psf.convolve_over_sample_size > 1``; the dataset is built without PSF
  over-sampling for that reason.
- ``config_name`` is **not** an autofit identifier field but ``seed`` is, so two
  backends run at one seed would hash to the same identifier and silently resume
  each other's fit. Both live in ``path_prefix`` instead.
- The Nautilus resume marker is ``files/search_internal/checkpoint.hdf5`` and it
  is **deleted on completion**, so ``resumed`` is sampled *before* each
  ``search.fit``, never after.
- Nautilus routes on ``analysis._use_jax``: ``use_jax=False`` takes the
  ``number_of_cores`` multiprocessing path, ``use_jax=True`` vmaps with
  ``n_batch``. A ``Pool`` object is never handed to it. ``use_jax`` is therefore
  passed **explicitly on all five** ``al.AnalysisImaging`` calls — the workspace
  script omits it on three, which would silently make those stages JAX runs on a
  numba leg.
- Backend environment variables (``PYAUTO_DISABLE_JAX``, ``JAX_PLATFORMS``,
  ``JAX_ENABLE_X64``, ``NPROC``) must be exported **before** ``autolens`` is
  imported, so argv is parsed first (``parse_inference_cli`` imports no autolens)
  and the modelling stack is imported inside :func:`run_slam`.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _inference_cli import (  # noqa: E402
    BASE_VARIANT,
    CONFIG_NAME_RE,
    STAGES,
    InferenceCLI,
    auto_simulate_if_missing,
    device_info_dict,
    parse_inference_cli,
    rect_mesh_classes,
    variant_name,
)

#: Version of the ``stages_seed<n>.json`` schema written by :func:`_write_results`.
#:
#: v2 added the run variant and the mesh it is named after: ``variant``,
#: ``mesh``, ``mesh_pixels`` and ``regularization`` (the resolved scheme name),
#: with ``mesh_shape`` now ``null`` on a row whose mesh has no shape (Delaunay
#: takes a vertex count). v1 rows are still read — every reader falls back to
#: the base variant when the field is absent, which is what those rows are.
SCHEMA_VERSION = 2

#: Search name per stage key, in chain order. These are the names that appear in
#: the output tree and in every result row, so they are the citable identity of a
#: stage — ``--stages light`` stops after ``light[1]``.
STAGE_SEARCH_NAMES = {
    "source_lp": "source_lp[1]",
    "source_pix_1": "source_pix[1]",
    "source_pix_2": "source_pix[2]",
    "light": "light[1]",
    "mass_total": "mass_total[1]",
}

#: ``(n_live, n_batch)`` per stage, copied from ``slam_start_here.py``.
STAGE_SAMPLING = {
    "source_lp": (200, 50),
    "source_pix_1": (150, 20),
    "source_pix_2": (75, 20),
    "light": (150, 20),
    "mass_total": (150, 20),
}

#: Stages whose model contains a pixelization, and which therefore (a) care
#: about ``--inversion`` and (b) must carry a ``positions.info``.
PIXELIZED_STAGES = ("source_pix_1", "source_pix_2", "light", "mass_total")

#: The source adapt image is capped at this S/N before it steers the adaptive
#: image-mesh and the adaptive regularization (see ``slam_start_here.py``,
#: "__Adapt Image S/N Cap__").
ADAPT_IMAGE_SNR_CAP = 3.0

#: Pixels of the source S/N map above this threshold get pixelization
#: over-sampling 4, the rest get 2 (``slam_start_here.py``, "__Adaptive
#: Pixelization Over-Sampling__").
OVER_SAMPLE_SNR_THRESHOLD = 3.0

#: ``al.mesh.Delaunay``'s ``areas_factor`` — the multiplier on the sqrt of each
#: vertex's dual-cell area that sets the interpolation's split step. 0.5 is the
#: library default, and it is passed EXPLICITLY rather than inherited: it is a
#: science-relevant knob of the mesh under test, so the run states it and the
#: result row records it (``mesh_areas_factor``) instead of leaving a future
#: reader to look up what the library's default was on the day.
MESH_AREAS_FACTOR = 0.5

#: The circle of edge points appended to the Delaunay image-plane mesh grid
#: (``al.image_mesh.append_with_circle_edge_points``), and — the same number
#: written twice — the mesh's ``zeroed_pixels``. These two MUST move together:
#: ``Delaunay.total_pixels`` is ``pixels + zeroed_pixels`` and the mesh grid the
#: mapper receives has to be exactly that long, so 1250 Hilbert-drawn interior
#: vertices plus this 30-point ring is 1280 = 1250 + 30. Changing the ring size
#: without changing ``zeroed_pixels`` breaks the accounting silently — the ring
#: is not a free knob, it IS ``zeroed_pixels``.
MESH_EDGE_POINTS = 30

#: The Hilbert image-mesh weights that draw the Delaunay vertices, as production
#: runs them (``autolens_workspace/scripts/group/slam.py``). They are NOT the
#: library defaults, which are 0.0 / 0.0 and would draw a mesh that does not
#: adapt to the source at all — stating them here is the difference between the
#: production recipe and a uniform mesh wearing its name.
MESH_HILBERT_WEIGHT_POWER = 3.5
MESH_HILBERT_WEIGHT_FLOOR = 0.01

#: ``result.positions_likelihood_from`` arguments used on every pixelized stage.
POSITIONS_FACTOR = 3.0
POSITIONS_MINIMUM_THRESHOLD = 0.2

#: Threshold of the ``PositionsLH`` built from the simulator's ``positions.json``
#: when the point solver cannot solve the (essentially random) test-mode tracer.
TEST_MODE_POSITIONS_THRESHOLD = 0.3

REDSHIFT_LENS = 0.5
REDSHIFT_SOURCE = 1.0

#: Lens-light / source-light over-sampling of the fitted dataset, the workspace
#: default for ``slam_start_here.py``.
OVER_SAMPLE_SUB_SIZE_LIST = [4, 2, 2]
OVER_SAMPLE_RADIAL_LIST = [0.3, 0.6]

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
    backends they did not use.
    """
    if cli.config_name is None:
        return (
            "--config-name is required. Grammar: "
            "{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}"
        )

    match = CONFIG_NAME_RE.match(cli.config_name)
    if match is None:
        return (
            f"--config-name {cli.config_name!r} does not match the grammar "
            "{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}"
        )

    _where, backend, inversion, precision = match.groups()
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

    if mismatches:
        return (
            f"--config-name {cli.config_name!r} disagrees with the run flags "
            "(a config name that lies about its run makes the parity row a lie): "
            + "; ".join(mismatches)
        )
    return None


def config_name_precision(config_name: str) -> str:
    """The ``fp64`` / ``mp`` tail of a validated config name."""
    return CONFIG_NAME_RE.match(config_name).group(4)


# ---------------------------------------------------------------------------
# Backend environment — set BEFORE autolens is imported
# ---------------------------------------------------------------------------


def set_backend_env(backend: str, cores: int) -> dict:
    """Export the environment the backend needs, returning what was set.

    Must run before ``import autolens``: ``JAX_ENABLE_X64`` and ``JAX_PLATFORMS``
    are read by JAX at import, and ``PYAUTO_DISABLE_JAX`` by the library's
    ``jax_wrapper``. ``NPROC`` sizes XLA's CPU thread pool (an unset one lets a
    job take the whole node; ``NPROC=1`` costs ~3x on JAX-CPU).
    """
    env = {"JAX_ENABLE_X64": "True"}

    if backend == "numba_cpu":
        # Belt (process-wide) to `use_jax=False`'s braces (per Analysis).
        env["PYAUTO_DISABLE_JAX"] = "1"
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
# Inversion path
# ---------------------------------------------------------------------------


def _apply_inversion(dataset, backend: str, inversion: str):
    """Return ``dataset`` with this run's inversion path applied.

    ``dense`` leaves the mapping-matrix path in place. ``sparse`` is **two
    different calls**: ``apply_sparse_operator()`` is the JAX operator (on any
    device) and ``apply_sparse_operator_cpu()`` is the numba one. Calling the
    JAX one on a numba leg does not error — it produces a leg that is neither,
    which is why the dispatch lives in one function with one test.

    Call this again after **any** dataset re-derivation:
    ``Imaging.apply_over_sampling`` rebuilds the dataset and silently drops the
    operator.
    """
    if inversion == "dense":
        return dataset
    if inversion != "sparse":  # pragma: no cover — argparse constrains this
        raise ValueError(f"Unknown inversion {inversion!r}")
    if backend.startswith("jax"):
        return dataset.apply_sparse_operator()
    return dataset.apply_sparse_operator_cpu()


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
# The five SLaM stages (copied from slam_start_here.py)
# ---------------------------------------------------------------------------


def _adapt_images(al, result, *, meshes, dataset, mask_radius: float):
    """The S/N-capped adapt images built from ``result`` — and, for the Delaunay
    family, the image-plane mesh grid that places its vertices.

    The cap is applied to an explicit copy so the raw S/N image is untouched:
    without it the brightest peak dominates the mesh-density and regularization
    weights (which scale as a power of the adapt image) and fainter multiply
    imaged features get too few source pixels.

    **The Delaunay mesh does not place its own vertices.** Passing
    ``al.mesh.Delaunay`` alone raises ``MeshException: the mesh Delaunay was not
    given an image-plane mesh grid``: the points are drawn by an *image mesh*
    from the capped source adapt image, and reach the fit only through
    ``adapt_images``. ``Delaunay.pixels`` is documented as a description of that
    grid rather than a control over it. So this helper is mesh-aware, and builds
    the production recipe (``autolens_workspace/scripts/group/slam.py``):
    ``al.image_mesh.Hilbert`` at :data:`MESH_HILBERT_WEIGHT_POWER` /
    :data:`MESH_HILBERT_WEIGHT_FLOOR`, then a :data:`MESH_EDGE_POINTS`-point
    circle edge ring appended at the mask's radius.

    It is called at **every** pixelized stage, and the grid is rebuilt each time
    from *that* stage's own adapt image, as production does — ``light[1]`` and
    ``mass_total[1]`` inherit the source pixelization and so need the grid too.

    ``dataset`` and ``mask_radius`` are threaded in explicitly rather than read
    from anything ambient: the ring's radius is the mask's, and a helper that
    guessed it would silently drift from the mask the fit actually uses.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result)

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > ADAPT_IMAGE_SNR_CAP] = ADAPT_IMAGE_SNR_CAP
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    if meshes.image_mesh is None:
        return al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    mask = dataset.mask
    image_mesh = al.image_mesh.Hilbert(
        pixels=meshes.pixels,
        weight_power=meshes.weight_power,
        weight_floor=meshes.weight_floor,
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask,
        adapt_data=source_adapt_image,
    )
    # The ring is appended just outside the mask edge, and is exactly the mesh's
    # `zeroed_pixels`: pixels + MESH_EDGE_POINTS is the length the mapper expects
    # (`Delaunay.total_pixels`), and those last points are fixed to zero.
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=meshes.edge_points,
    )

    return al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )


def positions_likelihood_from(al, result, *, fallback_positions, stage: str):
    """``result.positions_likelihood_from``, with a stated test-mode substitute.

    In production this is the SLaM automation the workspace script relies on:
    the previous result's maximum-likelihood mass model is ray-traced by the
    point solver to find the multiple images and the source-plane threshold.

    Under ``PYAUTO_TEST_MODE`` the sampler stops after a single likelihood call,
    so that "maximum likelihood" model is a near-random draw from the priors —
    and the solver does not fail fast on one. Measured on this cell
    (2026-09-11): ``source_lp[1]``'s test-mode result sent
    ``positions_likelihood_from`` into ``Result``'s recovery loop
    ("Could not find multiple images ... incrementally moving source centre
    inwards"), which had not returned after nine minutes for a single stage —
    four such calls per leg, six legs, all to solve a model no one will read.

    So under test mode the solver is not called at all and the positions
    likelihood is built from the simulator's own ``positions.json`` instead.
    The stage still carries a positions likelihood and still writes the
    ``positions.info`` the witness reads — the substitution is announced on
    stdout and recorded in the result row's ``test_mode`` flag, never silently
    dropped. Outside test mode the solver's exception is re-raised: a real chain
    whose point solver fails has a real problem.
    """
    from autonerves.test_mode import is_test_mode

    if is_test_mode():
        print(
            f"  NOTE [{stage}]: PYAUTO_TEST_MODE — skipping "
            f"positions_likelihood_from and building the positions likelihood "
            f"from the simulator's positions.json at threshold "
            f"{TEST_MODE_POSITIONS_THRESHOLD}. The test-mode maximum-likelihood "
            f"tracer is a near-random draw, and solving it costs minutes per "
            f"stage for a threshold nothing reads. This substitution is NEVER "
            f"made outside test mode."
        )
        return al.PositionsLH(positions=fallback_positions, threshold=TEST_MODE_POSITIONS_THRESHOLD)

    return result.positions_likelihood_from(
        factor=POSITIONS_FACTOR, minimum_threshold=POSITIONS_MINIMUM_THRESHOLD
    )


def source_lp(
    af,
    al,
    settings_search,
    dataset,
    mask_radius: float,
    redshift_lens: float,
    redshift_source: float,
    *,
    use_jax: bool,
    seed: int,
    settings=None,
    n_live: int = 200,
    n_batch: int = 50,
):
    """Initialise the lens mass and a parametric source: MGE 20x2 lens light, 20x1
    source, ``Isothermal + ExternalShear``. No pixelization, so ``--inversion``
    does not touch this stage."""
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=use_jax, settings=settings)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )
    field = af.Model(
        al.MassField,
        redshift=redshift_lens,
        shear=af.Model(al.mp.ExternalShear),
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=af.Model(al.mp.Isothermal),
            ),
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
        fields=field,
    )

    search = af.Nautilus(
        name=STAGE_SEARCH_NAMES["source_lp"],
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        seed=seed,
    )

    return model, analysis, search


class MeshModels(NamedTuple):
    """What the two pixelized source stages are built from, for one ``--mesh``.

    ``mesh_init`` goes to ``source_pix[1]`` and ``mesh`` to ``source_pix[2]``
    (two distinct objects — the rectangular family uses a different class for
    each, and sharing one between two searches would share its priors). The
    rectangular family hands over ``af.Model``s (its shape is a free parameter
    with a prior configured for it); the Delaunay family hands over plain
    **instances**, because its mesh has no free parameters — see
    :func:`mesh_models`. ``regularization`` is always the **class**, which is
    what the stage helpers hand ``af.Model(al.Pixelization, ...)``; ``scheme``
    is the name the result row records; ``shape`` is the mesh shape for a family
    that has one and ``None`` for a family that does not.

    The remaining fields are the Delaunay recipe, ``None`` on the rectangular
    family, and they are carried rather than recomputed because they are the
    *identity of the experiment*: the result row records every one of them. A
    row that said only "delaunay, 1250" could not be reproduced —
    ``image_mesh`` is what actually places the vertices, and its weights decide
    where they go.
    """

    mesh_init: object
    mesh: object
    regularization: object
    scheme: str
    shape: tuple[int, int] | None
    areas_factor: float | None = None
    #: The image mesh that draws the vertices, and its arguments. ``pixels`` is
    #: the interior vertex count; ``edge_points`` the appended ring, which is
    #: also ``zeroed_pixels`` (see :data:`MESH_EDGE_POINTS`).
    image_mesh: str | None = None
    pixels: int | None = None
    weight_power: float | None = None
    weight_floor: float | None = None
    edge_points: int | None = None
    zeroed_pixels: int | None = None


def mesh_models(af, al, cli: InferenceCLI) -> MeshModels:
    """Resolve ``--mesh`` / ``--mesh-pixels`` into the two stages' pixelization models.

    ``rect`` is the workspace default: the adaptive rectangular density mesh
    then the adaptive rectangular image mesh (``--rect-mesh`` picks the family),
    at ``mesh_pixels x mesh_pixels``, under ``al.reg.Adapt``.

    ``delaunay`` is
    ``al.mesh.Delaunay(pixels=mesh_pixels, zeroed_pixels=30, areas_factor=0.5)``
    on both stages, under ``al.reg.AdaptSplit``, with its vertices drawn by the
    ``al.image_mesh.Hilbert`` recipe :func:`_adapt_images` builds. Three things
    about that pairing are not free choices:

    - **It cannot be ``al.reg.Adapt``.** That scheme takes its pixel
      neighbours from a ``scipy.spatial.Delaunay`` call on the *traced* source
      grid, which raises ``TracerArrayConversionError`` under ``jax.jit``. The
      mesh family, not the regularization alone, decides what can be traced:
      the same ``Adapt`` is exactly what the rectangular legs run under. The
      Split schemes are the traceable pairing for this family.
    - **The mesh is an instance, not an ``af.Model``.** This mesh has no free
      parameters — no more than the rectangular family's shape is meant to be
      one here — and wrapping it in ``af.Model`` asks PyAutoFit to give every
      constructor argument the caller did not pin a prior, which for
      ``areas_factor`` no ``config/priors`` tree in this stack defines:
      ``ConfigException: No prior config found for class: Delaunay ... for
      parameter name and path: areas_factor``, raised at the first
      ``search.fit`` of ``source_pix[1]``. Production passes the instance
      (``autolens_workspace/scripts/group/slam.py``, every ``autolens_profiling``
      Delaunay cell), and so does this. The rectangular branch keeps its
      ``af.Model`` because its ``shape`` *is* configured and pinned.
    - **The regularization is the class, not a fixed instance.** The stage helpers build
      ``af.Model(al.Pixelization, mesh=..., regularization=...)``, so a class
      leaves the regularization's coefficients as free parameters, exactly as
      ``al.reg.Adapt`` does on the rectangular legs — which is what makes the
      two runs comparable. ``_inference_cli.delaunay_regularization()`` builds a
      *fixed-coefficient* ``AdaptSplit`` for the profiling-style cells, and
      using it here would silently drop free parameters from the model.

    Delaunay needs an adapt image, which :func:`_adapt_images` already builds at
    every pixelized stage, so nothing else has to be plumbed through.
    """
    if cli.mesh == "delaunay":
        # zeroed_pixels IS the edge ring: the mesh grid handed to the mapper is
        # `pixels` Hilbert-drawn interior vertices plus MESH_EDGE_POINTS appended
        # circle points, and `Delaunay.total_pixels` is pixels + zeroed_pixels.
        # The ring is fixed to zero rather than solved for, which is what keeps
        # poorly constrained boundary vertices from absorbing flux.
        kwargs = dict(
            pixels=cli.mesh_pixels,
            zeroed_pixels=MESH_EDGE_POINTS,
            areas_factor=MESH_AREAS_FACTOR,
        )
        return MeshModels(
            mesh_init=al.mesh.Delaunay(**kwargs),
            mesh=al.mesh.Delaunay(**kwargs),
            regularization=al.reg.AdaptSplit,
            scheme="adapt_split",
            shape=None,
            areas_factor=MESH_AREAS_FACTOR,
            image_mesh="hilbert",
            pixels=cli.mesh_pixels,
            weight_power=MESH_HILBERT_WEIGHT_POWER,
            weight_floor=MESH_HILBERT_WEIGHT_FLOOR,
            edge_points=MESH_EDGE_POINTS,
            zeroed_pixels=MESH_EDGE_POINTS,
        )

    density_cls, image_cls = rect_mesh_classes(cli, al=al)
    shape = (cli.mesh_pixels, cli.mesh_pixels)
    return MeshModels(
        mesh_init=af.Model(density_cls, shape=shape),
        mesh=af.Model(image_cls, shape=shape),
        regularization=al.reg.Adapt,
        scheme="adapt",
        shape=shape,
        areas_factor=None,
    )


def source_pix_1(
    af,
    al,
    settings_search,
    dataset,
    source_lp_result,
    mesh_init,
    regularization_init,
    *,
    meshes,
    mask_radius: float,
    use_jax: bool,
    seed: int,
    positions_likelihood,
    settings=None,
    n_live: int = 150,
    n_batch: int = 20,
):
    """Build the high-quality adapt image: a pixelized source on the SOURCE LP mass."""
    adapt_images = _adapt_images(
        al, source_lp_result, meshes=meshes, dataset=dataset, mask_radius=mask_radius
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=use_jax,
        settings=settings,
    )

    mass, field = al.util.chaining.mass_and_fields_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        fields_result=source_lp_result.model.fields,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization, mesh=mesh_init, regularization=regularization_init
                ),
            ),
        ),
        fields=field,
    )

    search = af.Nautilus(
        name=STAGE_SEARCH_NAMES["source_pix_1"],
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        seed=seed,
    )

    return model, analysis, search


def source_pix_2(
    af,
    al,
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh,
    regularization,
    *,
    meshes,
    mask_radius: float,
    use_jax: bool,
    seed: int,
    positions_likelihood,
    settings=None,
    n_live: int = 75,
    n_batch: int = 20,
):
    """The final pixelized source, on the adapt images from ``source_pix[1]``.

    Departure from ``slam_start_here.py``: a positions likelihood is attached
    here too (see the module docstring, departure 1)."""
    adapt_images = _adapt_images(
        al, source_pix_result_1, meshes=meshes, dataset=dataset, mask_radius=mask_radius
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=use_jax,
        settings=settings,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(al.Pixelization, mesh=mesh, regularization=regularization),
            ),
        ),
        fields=source_pix_result_1.instance.fields,
    )

    search = af.Nautilus(
        name=STAGE_SEARCH_NAMES["source_pix_2"],
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        seed=seed,
    )

    return model, analysis, search


def light_lp(
    af,
    al,
    settings_search,
    dataset,
    mask_radius: float,
    source_result_for_lens,
    source_result_for_source,
    *,
    meshes,
    use_jax: bool,
    seed: int,
    positions_likelihood,
    settings=None,
    n_live: int = 150,
    n_batch: int = 20,
):
    """A complex lens light model with mass and source fixed.

    Departure from ``slam_start_here.py``: a positions likelihood is attached
    here too (see the module docstring, departure 1)."""
    adapt_images = _adapt_images(
        al, source_result_for_lens, meshes=meshes, dataset=dataset, mask_radius=mask_radius
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=use_jax,
        settings=settings,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=None,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
            ),
            source=source,
        ),
        fields=source_result_for_lens.instance.fields,
    )

    search = af.Nautilus(
        name=STAGE_SEARCH_NAMES["light"],
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        seed=seed,
    )

    return model, analysis, search


def mass_total(
    af,
    al,
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    *,
    meshes,
    mask_radius: float,
    use_jax: bool,
    seed: int,
    positions_likelihood,
    settings=None,
    n_live: int = 150,
    n_batch: int = 20,
):
    """The headline stage: a ``PowerLaw`` mass, priors chained from the SOURCE PIX
    ``Isothermal`` with the mass centre unfixed. This is the stage the parity
    view reads."""
    adapt_images = _adapt_images(
        al, source_result_for_lens, meshes=meshes, dataset=dataset, mask_radius=mask_radius
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=use_jax,
        settings=settings,
    )

    mass, field = al.util.chaining.mass_and_fields_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        fields_result=source_result_for_lens.model.fields,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=light_result.instance.galaxies.lens.bulge,
                disk=light_result.instance.galaxies.lens.disk,
                mass=mass,
            ),
            source=source,
        ),
        fields=field,
    )

    search = af.Nautilus(
        name=STAGE_SEARCH_NAMES["mass_total"],
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        seed=seed,
    )

    return model, analysis, search


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------


def results_paths(
    root: Path,
    dataset_class: str,
    instrument: str,
    variant: str,
    config_name: str,
    seed: int,
):
    """``(json_path, png_path)`` for one ``(variant, config_name, seed)``.

    ``results/slam/<dataset_class>/<instrument>/<variant>/<config_name>/stages_seed<n>.json``.

    One file per ``(config_name, seed)`` — the seed is in the *filename*, not
    only in the payload, so a reliability sweep over seeds does not overwrite
    itself. The ``<variant>`` level above it separates *experiments*: the legs
    of one variant are a parity row, and two variants sharing a directory would
    read as backends of one run rather than as two different models.
    """
    directory = root / "results" / "slam" / dataset_class / instrument / variant / config_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"stages_seed{seed}.json", directory / f"stages_seed{seed}.png"


def output_path_prefix(
    dataset_class: str, instrument: str, variant: str, config_name: str, seed: int
) -> Path:
    """The PyAutoFit ``path_prefix`` for one leg.

    ``config_name`` and ``seed_<n>`` live in the PATH, not only in the payload:
    ``seed`` is an autofit identifier field but ``config_name`` is not, so two
    backends run at one seed would otherwise hash to the same identifier and
    resume each other's completed fit in seconds, re-stamping it as their own.
    The same argument makes ``variant`` a path level — the mesh is not an
    identifier field either, so a Delaunay leg and a rectangular leg of the same
    config and seed would collide in exactly the same way.
    """
    return Path("slam") / dataset_class / instrument / variant / config_name / f"seed_{seed}"


def target_id(instrument: str, variant: str, seed: int) -> str:
    """The ``target`` every result row of this leg carries.

    The dashboard groups rows into a **parity row** by ``target``, so the target
    is what decides which legs are claimed to be the same run under different
    backends. A different experiment must never land in the base run's parity
    group: only :data:`BASE_VARIANT` keeps the bare ``<instrument>/slam5/seed<n>``
    id (the rows already on disk were written with it), and every other variant
    says so in the id — ``hst/slam5_delaunay_1250/seed0``.
    """
    if variant == BASE_VARIANT:
        return f"{instrument}/slam5/seed{seed}"
    return f"{instrument}/slam5_{variant}/seed{seed}"


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


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def run_slam(
    dataset_class: str = "imaging",
    default_instrument: str = "hst",
    default_mesh: str | None = None,
    default_mesh_pixels: int | None = None,
) -> int:
    """Run the 5-stage SLaM chain for one ``(variant, backend, inversion, seed)`` leg.

    ``default_mesh`` / ``default_mesh_pixels`` are the leaf's mesh defaults, in
    the style of ``default_instrument``: a leaf that exists to run one mesh
    names it once, and ``--mesh`` / ``--mesh-pixels`` still override.

    Returns the process exit code: 0 on success, 2 on a flag / environment
    refusal (a malformed config name, a config name that disagrees with the
    flags, or ``--backend jax_gpu`` on a host with no GPU).
    """
    cli = parse_inference_cli(default_mesh=default_mesh, default_mesh_pixels=default_mesh_pixels)
    instrument = cli.instrument or default_instrument
    variant = variant_name(cli)
    cores = cli.cores

    error = validate_config_name(cli)
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    config_name = cli.config_name
    precision = config_name_precision(config_name)

    if cli.stages is not None and cli.stages not in STAGES:
        print(f"ERROR: --stages {cli.stages!r} is not one of {STAGES}", file=sys.stderr)
        return 2

    env = set_backend_env(cli.backend, cores)

    # --- imports: everything below here sees the backend environment above ---
    import autofit as af
    import autolens as al
    from autonerves import jax_wrapper  # noqa: F401 — sets the JAX env before autolens
    from autonerves.test_mode import is_test_mode, with_test_mode_segment

    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
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

    use_jax = cli.backend != "numba_cpu"
    seed = cli.seed
    root = _ROOT

    output_root = cli.output_dir or Path(os.environ.get("PYAUTO_OUTPUT_DIR") or (root / "output"))
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    af.conf.instance.push(new_path=root / "config", output_path=output_root)

    print("=" * 78)
    print(f"SLaM run — {dataset_class} / {instrument} / {variant} / {config_name} / seed {seed}")
    print("=" * 78)
    print(f"  variant:    {variant}  (mesh={cli.mesh}, mesh_pixels={cli.mesh_pixels})")
    print(f"  backend:    {cli.backend}  (use_jax={use_jax}, cores={cores})")
    print(f"  inversion:  {cli.inversion}")
    print(f"  precision:  {precision}")
    print(f"  env set:    {env}")
    print(f"  output:     {output_root}")
    print(f"  test mode:  {is_test_mode()}")

    # --- dataset ----------------------------------------------------------
    from instruments.imaging import INSTRUMENTS

    if instrument not in INSTRUMENTS:
        print(
            f"ERROR: unknown --instrument {instrument!r}; choose from {sorted(INSTRUMENTS)}",
            file=sys.stderr,
        )
        return 2

    preset = INSTRUMENTS[instrument]
    pixel_scale = preset["pixel_scale"]
    mask_radius = preset["mask_radius"]

    dataset_path = root / "dataset" / dataset_class / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type=dataset_class,
        instrument=instrument,
        workspace_root=root,
    )

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        psf_path=dataset_path / "psf.fits",
        pixel_scales=pixel_scale,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=OVER_SAMPLE_SUB_SIZE_LIST,
        radial_list=OVER_SAMPLE_RADIAL_LIST,
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    # The inversion path is applied AFTER mask + over-sampling, and re-applied
    # after every later dataset re-derivation (`apply_over_sampling` drops it).
    dataset = _apply_inversion(dataset, cli.backend, cli.inversion)

    fallback_positions = al.Grid2DIrregular(
        al.from_json(file_path=str(dataset_path / "positions.json"))
    )
    truth_tracer = al.from_json(file_path=str(dataset_path / "tracer.json"))
    truths = truth_dict(truth_tracer)

    settings = al.Settings(
        use_mixed_precision=cli.use_mixed_precision,
        nnls_warm_start_memo=cli.memo == "on",
    )

    # --- search settings --------------------------------------------------
    # See `output_path_prefix`: config_name, variant and seed_<n> live in the
    # PATH because `seed` is an autofit identifier field and the other two are
    # not, so legs that differ only in those would hash to one identifier and
    # resume each other's completed fit.
    path_prefix = output_path_prefix(dataset_class, instrument, variant, config_name, seed)
    settings_search = af.SettingsSearch(
        path_prefix=path_prefix,
        unique_tag=None,
        info=None,
        session=None,
        # Nautilus routes on `analysis._use_jax`: under JAX it vmaps with
        # `n_batch` and ignores the pool, under numba it forks
        # `number_of_cores` workers. Never hand it a Pool object.
        number_of_cores=1 if use_jax else cores,
    )
    search_root = with_test_mode_segment(output_root) / path_prefix

    meshes = mesh_models(af, al, cli)

    stages: list[dict] = []
    stop_after = cli.stages

    def run_stage(stage_key: str, built) -> tuple[object, dict]:
        """Fit one stage and capture its row."""
        model, analysis, search = built
        n_live, n_batch = STAGE_SAMPLING[stage_key]
        search_name = STAGE_SEARCH_NAMES[stage_key]

        # Sampled BEFORE the fit: nautilus deletes checkpoint.hdf5 on completion.
        resumed = checkpoint_exists(search_root, search_name)

        compile_s = None
        compile_note = None
        if use_jax and is_test_mode():
            compile_note = COMPILE_S_TEST_MODE_NOTE
        elif use_jax:
            try:
                instance = model.instance_from_prior_medians()
                start = time.perf_counter()
                analysis.log_likelihood_function(instance)
                compile_s = time.perf_counter() - start
            except Exception as exc:  # pragma: no cover — probe is best-effort
                compile_note = f"compile probe failed: {type(exc).__name__}: {exc}"

        print(f"\n--- {search_name} (n_live={n_live}, n_batch={n_batch}) ---")
        start = time.perf_counter()
        result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
        total_wall_s = time.perf_counter() - start

        output_path = stage_output_dir(search_root, search_name)

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
            inversion_applies=stage_key in PIXELIZED_STAGES,
            posterior=posterior,
            truth_delta_sigma=truth_delta_sigma_from(posterior, truths),
            max_log_likelihood=max_log_likelihood,
            compile_s_note=compile_note,
        )
        stages.append(row)
        print(
            f"  {search_name}: wall {row['wall_s']}s, evals {row['likelihood_evals']}, "
            f"log Z {row['log_evidence']}, positions.info "
            f"{row['positions_info_present']}, completed {row['completed']}"
        )
        return result, row

    def write(status: str) -> None:
        json_path, png_path = results_paths(
            root, dataset_class, instrument, variant, config_name, seed
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "target": target_id(instrument, variant, seed),
            "variant": variant,
            "config_name": config_name,
            "backend": cli.backend,
            "inversion": cli.inversion,
            "precision": precision,
            "instrument": instrument,
            "dataset_class": dataset_class,
            "seed": seed,
            "version": al.__version__,
            "device": _device_info(),
            "cores": cores,
            "use_jax": use_jax,
            "rect_mesh": cli.rect_mesh,
            "memo": cli.memo,
            "mesh": cli.mesh,
            "mesh_pixels": cli.mesh_pixels,
            # The Delaunay split-step knob, recorded rather than inherited; a
            # family without one says null rather than borrowing a number.
            "mesh_areas_factor": meshes.areas_factor,
            # The rest of the recipe: which image mesh placed the vertices and
            # how. Without these a row cannot be reproduced — "delaunay, 1250"
            # says nothing about where the 1250 went. Null on the rectangular
            # family, which draws no vertices.
            "mesh_zeroed_pixels": meshes.zeroed_pixels,
            "image_mesh": meshes.image_mesh,
            "image_mesh_weight_power": meshes.weight_power,
            "image_mesh_weight_floor": meshes.weight_floor,
            "image_mesh_edge_points": meshes.edge_points,
            "regularization": meshes.scheme,
            # A mesh family with no shape (Delaunay takes a vertex count) says
            # so rather than inventing one.
            "mesh_shape": list(meshes.shape) if meshes.shape is not None else None,
            "stages_requested": stop_after,
            "test_mode": bool(is_test_mode()),
            "status": status,
            "stages": stages,
        }
        _write_results(json_path, payload)
        try:
            _plot_stages(png_path, payload)
        except Exception as exc:  # pragma: no cover — a plot must not kill a run
            print(f"  WARNING: stage plot failed — {exc}")
        print(f"\n  Result row: {json_path}")

    # The whole chain runs inside one try/except so that a leg which dies part
    # way through still leaves the stages it *did* finish on disk, under
    # `status: "failed: ..."`. A run whose only trace is a missing file is a
    # failure discoverable solely by noticing an absence (the lesson recorded in
    # activate.sh's SLURM exit-code guard); the row names the stage it reached
    # and the exception that stopped it. The exception is re-raised so the
    # process still exits non-zero.
    try:
        # --- the chain --------------------------------------------------------
        source_lp_result, _ = run_stage(
            "source_lp",
            source_lp(
                af,
                al,
                settings_search,
                dataset,
                mask_radius=mask_radius,
                redshift_lens=REDSHIFT_LENS,
                redshift_source=REDSHIFT_SOURCE,
                use_jax=use_jax,
                seed=seed,
                settings=settings,
            ),
        )
        if stop_after == "source_lp":
            write("stopped_early")
            return 0

        source_pix_result_1, _ = run_stage(
            "source_pix_1",
            source_pix_1(
                af,
                al,
                settings_search,
                dataset,
                source_lp_result,
                mesh_init=meshes.mesh_init,
                regularization_init=meshes.regularization,
                meshes=meshes,
                mask_radius=mask_radius,
                use_jax=use_jax,
                seed=seed,
                settings=settings,
                positions_likelihood=positions_likelihood_from(
                    al,
                    source_lp_result,
                    fallback_positions=fallback_positions,
                    stage=STAGE_SEARCH_NAMES["source_pix_1"],
                ),
            ),
        )
        if stop_after == "source_pix_1":
            write("stopped_early")
            return 0

        # Adaptive pixelization over-sampling (workspace default, kept): the source
        # S/N map from source_pix[1] is thresholded at 3.0, bright source pixels get
        # sub-size 4 and the rest 2. `apply_over_sampling` rebuilds the Imaging and
        # DROPS the sparse operator, so the inversion path is re-applied right after.
        import numpy as np

        source_image_raw = al.galaxy_name_image_dict_via_result_from(result=source_pix_result_1)[
            "('galaxies', 'source')"
        ]
        over_sample_size_pixelization = al.Array2D(
            values=np.where(source_image_raw > OVER_SAMPLE_SNR_THRESHOLD, 4, 2),
            mask=dataset.mask,
        )
        dataset = dataset.apply_over_sampling(
            over_sample_size_pixelization=over_sample_size_pixelization
        )
        dataset = _apply_inversion(dataset, cli.backend, cli.inversion)

        source_pix_result_2, _ = run_stage(
            "source_pix_2",
            source_pix_2(
                af,
                al,
                settings_search,
                dataset,
                source_lp_result,
                source_pix_result_1,
                mesh=meshes.mesh,
                regularization=meshes.regularization,
                meshes=meshes,
                mask_radius=mask_radius,
                use_jax=use_jax,
                seed=seed,
                settings=settings,
                positions_likelihood=positions_likelihood_from(
                    al,
                    source_pix_result_1,
                    fallback_positions=fallback_positions,
                    stage=STAGE_SEARCH_NAMES["source_pix_2"],
                ),
            ),
        )
        if stop_after == "source_pix_2":
            write("stopped_early")
            return 0

        light_result, _ = run_stage(
            "light",
            light_lp(
                af,
                al,
                settings_search,
                dataset,
                mask_radius=mask_radius,
                source_result_for_lens=source_pix_result_1,
                source_result_for_source=source_pix_result_2,
                meshes=meshes,
                use_jax=use_jax,
                seed=seed,
                settings=settings,
                positions_likelihood=positions_likelihood_from(
                    al,
                    source_pix_result_2,
                    fallback_positions=fallback_positions,
                    stage=STAGE_SEARCH_NAMES["light"],
                ),
            ),
        )
        if stop_after == "light":
            write("stopped_early")
            return 0

        run_stage(
            "mass_total",
            mass_total(
                af,
                al,
                settings_search,
                dataset,
                source_result_for_lens=source_pix_result_1,
                source_result_for_source=source_pix_result_2,
                light_result=light_result,
                meshes=meshes,
                mask_radius=mask_radius,
                use_jax=use_jax,
                seed=seed,
                settings=settings,
                positions_likelihood=positions_likelihood_from(
                    al,
                    source_pix_result_2,
                    fallback_positions=fallback_positions,
                    stage=STAGE_SEARCH_NAMES["mass_total"],
                ),
            ),
        )

        write("complete")
        return 0
    except BaseException as exc:
        write(f"failed: {type(exc).__name__}: {exc}")
        raise


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
