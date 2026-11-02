"""The imaging SLaM stage builder and its driver, on the shared core.

One function, :func:`run_slam`, runs the standard 5-stage imaging SLaM chain
end to end under any ``--backend`` × ``--inversion`` combination, records every
stage as a comparable row, and writes one result JSON (+ PNG) per
``(config_name, seed)``. Every ``scripts/imaging/<variant>/validation/`` leaf
is a ~25-line shim over it.

Everything about a leg that is not imaging — the flag checks, the backend
environment, the stage-row readers, the posterior and truth helpers, the output
prefix, the row provenance, ``--stages-from`` and the ``stages.json`` writer —
lives in :mod:`slam._core`, and is re-exported here under the names it has
always had (``from slam._runner import validate_config_name`` keeps working).
This module is the *imaging family*: the dataset, the five stage builders, the
mesh models and the lens-light modes. Other families register their own chain
with :class:`slam._core.Family` and run through :func:`slam._core.run_family`;
see that module's docstring, and ``scripts/misc/slam/README.md``. The
pixelization helpers here (:func:`mesh_models`, :func:`_adapt_images`,
:func:`mesh_fields`) take any masked dataset and are meant to be shared.

The chain mirrors ``autolens_workspace/scripts/guides/modeling/slam_start_here.py``
(the five module-level stage functions there, at the time of writing lines
148 / 236 / 317 / 397 / 481): ``source_lp[1]`` (Nautilus n_live 200 / n_batch
50) -> ``source_pix[1]`` (150/20) -> ``source_pix[2]`` (75/20) -> ``light[1]``
(150/20) -> ``mass_total[1]`` (150/20), with a 28x28
``RectangularBilinearAdaptDensity`` then ``RectangularBilinearAdaptImage`` mesh
under ``reg.Adapt`` (the ``slam_base`` **run variant**; ``--mesh delaunay``
swaps both stages for ``al.mesh.Delaunay`` under ``reg.AdaptSplit`` and names
its own variant folder, and ``--mesh delaunay_nn`` for ``al.mesh.DelaunayNN`` —
see :func:`mesh_models` and :func:`target_id`),
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
   ``results/<dataset_class>/<variant>/<instrument>/validation/<model_variant>/<setup_id>/``.

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

6. **``--stages-from`` loads a stage from another config's chain instead of
   fitting it** (autolens_inference#31). The stage's PyAutoFit ``path_prefix``
   names the other config (same output root and seed; the variant level is
   always ``slam_base``, because the parametric stage has no mesh — a
   ``slam_delaunay`` leg loads the base run's copy), so its
   completed fit is loaded exactly as a PyAutoFit resume loads one, and the
   result is used only as a result object (prior passing). Only the parametric
   ``source_lp[1]`` may be shared (no inversion, so both CPU backends fit the
   same likelihood there). The run refuses with exit 3, fitting nothing, unless
   that stage is complete there *for this run's own model and search
   identifier*. The row names the departure: the stage entry carries
   ``shared_from_config`` and the payload's ``chain_wall_s`` excludes it
   (``chain_wall_excludes``).

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

import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _inference_cli import (  # noqa: E402
    BASE_VARIANT,
    STAGES,
    InferenceCLI,
    auto_simulate_if_missing,
    base_variant_name,
    lens_light_modes,
    parse_inference_cli,
    rect_mesh_classes,
    validation_layout,
    variant_name,
)

# The dataset-agnostic core, re-exported under the names this module has always
# exposed: the point-source runner, the tests and every reader import them from here.
from slam._core import (  # noqa: E402
    CHAIN_WALL_NOTE,
    COMPILE_S_TEST_MODE_NOTE,
    LOG_EVIDENCE_ERR_NOTE,
    NUMBA_THREAD_ENV,
    SCHEMA_VERSION,
    SHARED_STAGE_INCOMPLETE_EXIT,
    SHARED_STAGE_NOTE,
    SharedStageIncomplete,
    _as_float,
    _device_info,
    _marginalized,
    _plot_stages,
    _read_json,
    _stage_row,
    _write_results,
    chain_wall_from,
    checkpoint_exists,
    config_name_precision,
    library_revisions_or_note,
    mark_shared_stage,
    open_leg,
    output_path_prefix,
    posterior_from_result,
    posterior_keys,
    posterior_path,
    refuse_setup,
    release_pin_from,
    set_backend_env,
    shared_stage_names,
    shared_stage_prefixes,
    slurm_job_id_from,
    stage_completed,
    stage_output_dir,
    stage_output_dirs,
    truth_delta_sigma_from,
    truth_dict,
    validate_config_name,
    validate_stages_from,
    validation_row_paths,
)

#: The setups the imaging chain has: a dense or sparse inversion, no transformer.
IMAGING_INVERSIONS = ("dense", "sparse")


#: What a v1/v2 row's ``lens_light_mode`` is when the field is absent: the only
#: mode the chain had before v3.
DEFAULT_LENS_LIGHT_MODE = "linear"

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

    ``delaunay_nn`` is the same recipe on ``al.mesh.DelaunayNN`` — a ``Delaunay``
    subclass with the identical ``(pixels, zeroed_pixels, areas_factor)``
    constructor, interpolating by Sibson natural-neighbour weights instead of the
    containing triangle's barycentric ones — exactly as autolens_profiling's
    ``imaging/delaunay_nn`` cell profiles it (``_production_config.standard_jax_cell``:
    the HST Delaunay preset, 1250 Hilbert vertices at 3.5 / 0.01 plus the 30-point
    zeroed ring, ``AdaptSplit``, with only the mesh class swapped). The two meshes
    differ in nothing else, so a ``slam_delaunay_nn`` row against a
    ``slam_delaunay`` row isolates the interpolation.
    """
    if cli.mesh in ("delaunay", "delaunay_nn"):
        mesh_cls = al.mesh.DelaunayNN if cli.mesh == "delaunay_nn" else al.mesh.Delaunay
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
            mesh_init=mesh_cls(**kwargs),
            mesh=mesh_cls(**kwargs),
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


def linear_light_profiles_in(al, obj) -> list:
    """Every linear light profile reachable from ``obj`` (a galaxy, a profile, a basis).

    ``Galaxy.cls_list_from`` does not look inside a ``Basis`` (measured: it
    returns nothing for a galaxy whose bulge is a 20-Gaussian linear MGE), so
    the walk is explicit — galaxy attributes, ``Basis.profile_list`` and plain
    lists.
    """
    linear_cls = al.lp_linear.LightProfileLinear
    if obj is None:
        return []
    if isinstance(obj, linear_cls):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [p for item in obj for p in linear_light_profiles_in(al, item)]
    profile_list = getattr(obj, "profile_list", None)
    if profile_list is not None:
        return linear_light_profiles_in(al, list(profile_list))
    if isinstance(obj, al.Galaxy):
        return [
            p
            for value in vars(obj).values()
            if not isinstance(value, (int, float, str))
            for p in linear_light_profiles_in(al, value)
        ]
    return []


def fixed_lens_light_from(al, light_result, *, lens_redshift: float):
    """The lens galaxy of ``light[1]``'s maximum-likelihood fit, with every linear
    light profile replaced by an ordinary one carrying its solved intensity.

    ``FitImaging.model_obj_linear_light_profiles_to_light_profiles``
    (``autogalaxy/abstract_fit.py``) returns the fit's tracer with that
    replacement made. The lens is found by name when the tracer's galaxies are a
    model instance (they are, for a tracer built from a SLaM result), else as the
    single galaxy at the lens redshift; anything else is refused rather than
    guessed.
    """
    tracer = light_result.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
    galaxies = tracer.galaxies
    lens = getattr(galaxies, "lens", None)
    if lens is None:
        candidates = [g for g in galaxies if getattr(g, "redshift", None) == lens_redshift]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot identify the lens galaxy in light[1]'s fixed-light tracer: "
                f"{len(candidates)} galaxies at z={lens_redshift}"
            )
        lens = candidates[0]
    leftover = linear_light_profiles_in(al, lens)
    if leftover:
        raise ValueError(
            f"model_obj_linear_light_profiles_to_light_profiles left {len(leftover)} linear "
            f"light profile(s) on the lens galaxy"
        )
    return lens


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
    lens_light_mode: str = "linear",
):
    """The headline stage: a ``PowerLaw`` mass, priors chained from the SOURCE PIX
    ``Isothermal`` with the mass centre unfixed. This is the stage the parity
    view reads.

    ``lens_light_mode`` decides only where the lens light comes from:

    - ``linear`` (the chain as it has always run): ``light[1]``'s instance, whose
      MGE is linear light profiles, so every mass-stage inversion re-solves the
      lens-light intensities alongside the source.
    - ``fixed``: the lens galaxy of
      ``light_result.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles``
      — the same profiles as ordinary ones carrying the intensities ``light[1]``'s
      maximum-likelihood fit solved for — so the inversion carries no lens-light
      columns. The built model is checked to hold no linear light profile.

    Everything else (the mass chaining, the source, the positions likelihood,
    the adapt images and the Nautilus settings) is identical between the two."""
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

    lens_redshift = source_result_for_lens.instance.galaxies.lens.redshift
    if lens_light_mode == "fixed":
        lens_light = fixed_lens_light_from(al, light_result, lens_redshift=lens_redshift)
    elif lens_light_mode == "linear":
        lens_light = light_result.instance.galaxies.lens
    else:
        raise ValueError(f"Unknown lens_light_mode {lens_light_mode!r}")

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=lens_redshift,
                bulge=lens_light.bulge,
                disk=lens_light.disk,
                mass=mass,
            ),
            source=source,
        ),
        fields=field,
    )

    if lens_light_mode == "fixed":
        leftover = linear_light_profiles_in(al, model.instance_from_prior_medians().galaxies.lens)
        if leftover:
            raise ValueError(
                f"lens_light_mode='fixed' built a mass_total[1] model whose lens still holds "
                f"{len(leftover)} linear light profile(s) "
                f"({sorted({type(p).__name__ for p in leftover})}); the fixed-light variant "
                f"would silently be the linear one."
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
# Result paths and identity
# ---------------------------------------------------------------------------


def results_paths(
    root: Path,
    dataset_class: str,
    instrument: str,
    variant: str,
    config_name: str,
    seed: int,
    results_tag: str | None = None,
):
    """``(json_path, png_path)`` for one ``(variant, config_name, seed)``.

    ``results/<dataset_class>/<rectangular|delaunay|delaunay_nn>/<instrument>/validation/
    <model_variant>/<setup_id>/seed_<n>/stages.json`` — the mirrored layout of
    autolens_inference#41: the run variant splits into the likelihood variant and
    the model variant (``_inference_cli.validation_layout``), the config name into
    the ``setup_id`` folder and a ``host_class`` that is provenance only
    (``_inference_cli.split_config_name``; a ``local_*`` row's file is
    ``stages_local_laptop.json``). The path itself is the core's
    (``_core.validation_row_paths``), which a family that names its likelihood
    outright calls directly.

    A ``results_tag`` (``--results-tag``) suffixes the setup directory as
    ``<setup_id>__<tag>`` so a re-measurement of the same variant, config and
    seed can never overwrite the earlier row, locally or through
    ``hpc/sync pull``. The row's ``config_name`` field is unchanged.

    One directory per seed, so a reliability sweep over seeds does not
    overwrite itself, and one model variant per directory, so two models never
    read as backends of one run.
    """
    family, model_variant = validation_layout(variant)
    return validation_row_paths(
        root, dataset_class, family, instrument, model_variant, config_name, seed, results_tag
    )


def target_id(instrument: str, variant: str, seed: int) -> str:
    """The ``target`` every result row of this leg carries.

    The dashboard groups rows into a **parity row** by ``target``, so the target
    is what decides which legs are claimed to be the same run under different
    backends. A different experiment must never land in the base run's parity
    group: only :data:`BASE_VARIANT` keeps the bare ``<instrument>/slam5/seed<n>``
    id (the rows already on disk were written with it), and every other variant
    says so in the id — ``hst/slam5_slam_delaunay/seed0``.
    """
    if variant == BASE_VARIANT:
        return f"{instrument}/slam5/seed{seed}"
    return f"{instrument}/slam5_{variant}/seed{seed}"


def lens_light_mode_of(payload: dict) -> str:
    """The light mode of a result row of any schema version (absent = linear)."""
    return payload.get("lens_light_mode") or DEFAULT_LENS_LIGHT_MODE


def prepared_export_dir(export_prepared: Path, modes: tuple[str, ...], variant: str) -> Path:
    """Where one light mode's frozen ``mass_total[1]`` problem is exported.

    One directory identifies one problem (``export_problem`` refuses to put a
    second into it), so a single-mode run exports to the directory given — as
    before — and a ``both`` run exports each mode to ``<dir>/<variant>/``. The two
    problems' ids differ anyway: the model differs, and the controls carry the
    variant and ``lens_light_mode``.
    """
    if len(modes) == 1:
        return Path(export_prepared)
    return Path(export_prepared) / variant


def mesh_fields(cli: InferenceCLI, meshes: MeshModels) -> dict:
    """The payload fields that record the source pixelization, in the row's key order.

    Without the image-mesh fields a row cannot be reproduced — "delaunay, 1250" says
    nothing about where the 1250 went — and the exporter's size comparison reads
    ``mesh`` / ``mesh_pixels`` / ``mesh_shape``. A family without a field says null
    rather than borrowing a number.
    """
    return {
        "rect_mesh": cli.rect_mesh,
        "memo": cli.memo,
        "mesh": cli.mesh,
        "mesh_pixels": cli.mesh_pixels,
        # The Delaunay split-step knob, recorded rather than inherited; a
        # family without one says null rather than borrowing a number.
        "mesh_areas_factor": meshes.areas_factor,
        # The rest of the recipe: which image mesh placed the vertices and
        # how. Null on the rectangular family, which draws no vertices.
        "mesh_zeroed_pixels": meshes.zeroed_pixels,
        "image_mesh": meshes.image_mesh,
        "image_mesh_weight_power": meshes.weight_power,
        "image_mesh_weight_floor": meshes.weight_floor,
        "image_mesh_edge_points": meshes.edge_points,
        "regularization": meshes.scheme,
        # A mesh family with no shape (Delaunay takes a vertex count) says
        # so rather than inventing one.
        "mesh_shape": list(meshes.shape) if meshes.shape is not None else None,
    }


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
    flags, a setup the imaging chain does not have — ``--inversion na`` or a
    ``--transformer`` — or ``--backend jax_gpu`` on a host with no GPU).
    """
    cli = parse_inference_cli(default_mesh=default_mesh, default_mesh_pixels=default_mesh_pixels)
    instrument = cli.instrument or default_instrument
    # Stages 1 to 4 are the mesh-only variant whatever the light mode; each
    # light mode's mass_total[1] (and its result row) is that mode's variant.
    variant = base_variant_name(cli)
    modes = lens_light_modes(cli)
    variants = {mode: variant_name(cli, mode) for mode in modes}
    cores = cli.cores

    error = (
        validate_config_name(cli)
        or refuse_setup(cli, IMAGING_INVERSIONS)
        or validate_stages_from(cli)
    )
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if cli.stages is not None and cli.stages not in STAGES:
        print(f"ERROR: --stages {cli.stages!r} is not one of {STAGES}", file=sys.stderr)
        return 2

    env = set_backend_env(cli.backend, cores)

    # --- imports: everything below here sees the backend environment above ---
    leg = open_leg(
        cli,
        dataset_class=dataset_class,
        instrument=instrument,
        variant=variant,
        env=env,
        root=_ROOT,
        details=(
            f"mesh:       {cli.mesh}  (mesh_pixels={cli.mesh_pixels})",
            f"lens light: {cli.lens_light}  (mass_total[1] variants: {list(variants.values())})",
        ),
    )
    if isinstance(leg, int):
        return leg
    af, al = leg.af, leg.al
    use_jax = leg.use_jax
    seed = leg.seed
    root = leg.root
    config_name = leg.config_name
    precision = leg.precision

    # `--stages-from`: a shared stage's prefix names the OTHER config, so its
    # completed fit is loaded (PyAutoFit resume), never refitted here.
    # Checked before the dataset is even loaded, so a refusal costs nothing.
    code = leg.share_stages(STAGE_SEARCH_NAMES, BASE_VARIANT)
    if code is not None:
        return code

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

    dataset_path = leg.dataset_path
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

    # --- row provenance (schema v3) ----------------------------------------
    leg.sample_provenance()

    # --- search settings --------------------------------------------------
    settings_search = leg.settings_search

    # The mass stage alone carries the light-mode variant in its path_prefix:
    # stages 1 to 4 above keep the mesh-only variant, so a `fixed` run resumes a
    # completed `linear` chain's first four stages, while the two mass fits can
    # never resume each other (the lens light is not an identifier field any
    # more than the mesh is). For `linear` this is exactly the prefix above.
    mass_settings_search = {}
    mass_search_root = {}
    for mode in modes:
        mass_settings_search[mode], mass_search_root[mode] = leg.settings_search_for(variants[mode])

    meshes = mesh_models(af, al, cli)

    shared_stages: list[dict] = []
    mass_rows: dict[str, dict] = {}
    written: set[str] = set()
    stop_after = cli.stages
    preparation_started = time.perf_counter()
    # Wall time of mass fits already run in this process. Under `both` the
    # fixed export's preparation_s must not include the linear mass fit.
    mass_fit_elapsed = [0.0]

    def export_hooks(mode: str):
        """The prepared-export hooks of one light mode's ``mass_total[1]``."""
        if cli.export_prepared is None:
            return None, None
        search_name = STAGE_SEARCH_NAMES["mass_total"]

        def before_fit(model, analysis):
            import hashlib

            from experiments.prepared import export_problem

            model = analysis.modify_model(model)
            mode_variant = variants[mode]
            baseline_path = results_paths(
                root,
                dataset_class,
                instrument,
                mode_variant,
                config_name,
                seed,
                cli.results_tag,
            )[0]
            baseline_id = (
                baseline_path.relative_to(root).as_posix()
                + "#stage-"
                + hashlib.sha256(search_name.encode()).hexdigest()[:16]
            )
            export_dir = prepared_export_dir(cli.export_prepared, modes, mode_variant)
            manifest = export_problem(
                root,
                export_dir,
                model,
                analysis,
                setup_id=f"imaging/{validation_layout(variant)[0]}/{instrument}",
                baseline_record_id=baseline_id,
                controls=dict(
                    dataset_class=dataset_class,
                    instrument=instrument,
                    variant=mode_variant,
                    lens_light_mode=mode,
                    config_name=config_name,
                    backend=cli.backend,
                    precision=precision,
                    seed=seed,
                    use_jax=use_jax,
                ),
                preparation_s=(time.perf_counter() - preparation_started - mass_fit_elapsed[0]),
            )
            return model, manifest

        def after_row(row, result, posterior, resumed, prepared_manifest):
            from experiments.prepared import complete_baseline, verify_manifest

            complete_baseline(root, prepared_manifest, result, posterior)
            manifest = verify_manifest(root, prepared_manifest)
            row.update(
                setup_id=manifest["setup_id"],
                problem_id=manifest["id"],
                dataset_id=manifest["dataset_id"],
                model=manifest["model_id"],
                experiment_protocol="frozen-mass-total-v1",
                initialization=dict(
                    mode="unknown" if resumed else "cold",
                    recipe="Nautilus prepared priors; no baseline sampler information reused",
                    sources=[],
                    **({"reason": "Preexisting checkpoint not pinned"} if resumed else {}),
                ),
                preparation_s=manifest["preparation_s"],
                max_likelihood_parameters=manifest["baseline_evidence"][
                    "max_likelihood_parameters"
                ],
                library_revisions=manifest["dependency_revisions"],
            )

        return before_fit, after_row

    def run_stage(
        stage_key: str,
        built,
        *,
        stage_search_root: Path | None = None,
        mode: str | None = None,
    ) -> tuple[object, dict]:
        """Fit one stage and capture its row."""
        n_live, n_batch = STAGE_SAMPLING[stage_key]
        stage_started = time.perf_counter()
        before_fit, after_row = export_hooks(mode) if stage_key == "mass_total" else (None, None)
        result, row = leg.run_stage(
            stage_key,
            STAGE_SEARCH_NAMES[stage_key],
            built,
            n_live=n_live,
            n_batch=n_batch,
            inversion_applies=stage_key in PIXELIZED_STAGES,
            truths=truths,
            stage_search_root=stage_search_root,
            before_fit=before_fit,
            after_row=after_row,
        )
        if stage_key == "mass_total":
            mass_rows[mode] = row
            mass_fit_elapsed[0] += time.perf_counter() - stage_started
        else:
            shared_stages.append(row)
        return result, row

    def write(status: str, mode: str) -> None:
        """Write the result row of one light mode.

        Every row carries the shared stages 1 to 4 plus *that* mode's
        ``mass_total[1]`` (when it ran), so a ``both`` run's two rows read
        exactly like two single-mode runs that shared a prefix.
        """
        mode_variant = variants[mode]
        stages = list(shared_stages)
        if mode in mass_rows:
            stages.append(mass_rows[mode])
        json_path, png_path = results_paths(
            root, dataset_class, instrument, mode_variant, config_name, seed, cli.results_tag
        )
        payload = leg.payload(
            target=target_id(instrument, mode_variant, seed),
            variant=mode_variant,
            stages=stages,
            status=status,
            identity={"lens_light_mode": mode},
            model=mesh_fields(cli, meshes),
        )
        leg.write_row(json_path, png_path, payload)
        written.add(mode)

    def write_all(status: str) -> None:
        for mode in modes:
            if mode not in written:
                write(status, mode)

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
                leg.search_settings("source_lp"),
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
            write_all("stopped_early")
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
            write_all("stopped_early")
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
            write_all("stopped_early")
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
            write_all("stopped_early")
            return 0

        # One positions likelihood for every light mode: both mass fits must
        # see the same threshold, or they would differ by more than the light.
        mass_positions_likelihood = positions_likelihood_from(
            al,
            source_pix_result_2,
            fallback_positions=fallback_positions,
            stage=STAGE_SEARCH_NAMES["mass_total"],
        )
        for mode in modes:
            print(f"\n=== mass_total[1] — lens light {mode} (variant {variants[mode]}) ===")
            run_stage(
                "mass_total",
                mass_total(
                    af,
                    al,
                    mass_settings_search[mode],
                    dataset,
                    source_result_for_lens=source_pix_result_1,
                    source_result_for_source=source_pix_result_2,
                    light_result=light_result,
                    meshes=meshes,
                    mask_radius=mask_radius,
                    use_jax=use_jax,
                    seed=seed,
                    settings=settings,
                    positions_likelihood=mass_positions_likelihood,
                    lens_light_mode=mode,
                ),
                stage_search_root=mass_search_root[mode],
                mode=mode,
            )
            write("complete", mode)

        return 0
    except SharedStageIncomplete as exc:
        # A refusal, not a failure: nothing was fitted, so no row is written.
        print(f"ERROR: {exc}", file=sys.stderr)
        return SHARED_STAGE_INCOMPLETE_EXIT
    except BaseException as exc:
        write_all(f"failed: {type(exc).__name__}: {exc}")
        raise


def main_stage_complete(argv: list[str] | None = None) -> int:
    """``python3 scripts/misc/slam/_runner.py --stage-complete <seed_dir>``.

    Exit 0 when the stage is complete under ``<seed_dir>`` (the
    ``.../<variant>/<config_name>/seed_<n>`` directory of a chain), 1 when not.
    It runs :func:`stage_completed`, the test the runner itself applies to a
    ``--stages-from`` stage, and imports nothing beyond the stdlib, so a SLURM
    wait loop can poll it cheaply.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main_stage_complete.__doc__)
    parser.add_argument("--stage-complete", type=Path, required=True, metavar="SEED_DIR")
    parser.add_argument(
        "--stage",
        default="source_lp",
        choices=tuple(STAGE_SEARCH_NAMES),
        help="Stage key (default source_lp, i.e. source_lp[1]).",
    )
    args = parser.parse_args(argv)
    complete = stage_completed(args.stage_complete, STAGE_SEARCH_NAMES[args.stage])
    print(
        f"{STAGE_SEARCH_NAMES[args.stage]} under {args.stage_complete}: "
        f"{'complete' if complete else 'not complete'}"
    )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main_stage_complete())
