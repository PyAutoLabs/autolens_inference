"""The interferometer families: the workspace interferometer SLaM chains, on the shared core.

Likelihood-lifecycle 4c-1 (autolens_inference#54). Three likelihoods, as the profiling catalogue
names them, each one :class:`slam._core.Family` on the reference instrument ``sma``:

- ``interferometer/rectangular`` and ``interferometer/delaunay`` mirror
  ``autolens_workspace/scripts/interferometer/features/pixelization/slam.py``, four stages and
  no lens light (interferometer data carries none): ``source_lp[1]`` (Nautilus 200 / 50, an MGE
  source of 5 Gaussians) -> ``source_pix[1]`` (150 / 20) -> ``source_pix[2]`` (75 / 20) ->
  ``mass_total[1]`` (150 / 20, a ``PowerLaw`` chained from the ``Isothermal`` with the mass
  centre unfixed). The three pixelized stages carry the profiled standard-size cell's mesh,
  regularization and size (autolens_profiling ``_production_config.interferometer_standard_preset``,
  ruling R3): ``RectangularBilinearAdaptImage`` 35 x 35 under a free ``Adapt`` (``source_pix[1]``
  keeps the workspace's ``RectangularBilinearAdaptDensity`` initialiser, at the same size, as
  the imaging chain does), or Delaunay on 1250 Hilbert vertices (weight power 3.5, floor 0.01)
  plus the 30 zeroed edge points under a free ``AdaptSplit``. The workspace has no Delaunay
  interferometer SLaM, so that chain is the rectangular one with the profiled Delaunay preset
  swapped in: the nearest analogue (human ruling 2026-11-02). Those three stages evaluate the
  likelihood and pair with its profiling record; ``source_lp[1]`` is parametric and declares
  ``likelihood_evaluated: null``.
- ``interferometer/mge`` is the profiled MGE cell (a linear MGE source of 20 Gaussians, one
  basis, on the NUFFT). The workspace's ``features/multi_gaussian_expansion/slam.py`` is the
  pixelization chain above behind an MGE ``source_lp[1]``, so its only MGE stage is that one;
  the family keeps it, at the profiled 20 Gaussians, and closes the chain the way the imaging
  MGE SLaM does, with ``mass_total[1]`` on the same MGE source (no ``light[1]``: there is no
  lens light). Both stages evaluate ``interferometer/mge``.

The setups are the visibility-space routes the profiling records time (``--transformer``):
``wtilde`` (``--inversion sparse``) fits every pixelized stage on ``TransformerNUFFT`` plus the
NUFFT-built W~ operator (``apply_sparse_operator(method="nufft")``, its preload built by JAX on a
JAX leg and by finufft on a numba one), ``nufft`` (``--inversion dense``) on the plain
``TransformerNUFFT`` mapping matrix. ``source_lp[1]`` always runs on the plain NUFFT dataset, as
the workspace's ``dataset_nufft`` does.

Deliberate departures from the workspace scripts, each the imaging chain's own
(``slam._runner``): ``use_jax=leg.use_jax`` on every ``AnalysisInterferometer``; a positions
likelihood on every pixelized stage (``source_pix[2]`` too, so each writes the
``positions.info`` this repo cites a mesh stage by; under ``PYAUTO_TEST_MODE`` it is built from
the simulator's ``positions.json``); and ``al.Settings`` with the library's positive-only solver
(the workspace turns it off, which would time a different solve from the profiled cell's)
carrying ``--use-mixed-precision`` and ``--memo``. The adapt images are the workspace's: model
images, capped at the S/N 3.0 contour of their dirty image.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _inference_cli import MESH_PIXELS_STANDARD, auto_simulate_if_missing  # noqa: E402
from slam._core import Family, Stage, truth_dict  # noqa: E402
from slam._runner import (  # noqa: E402
    ADAPT_IMAGE_SNR_CAP,
    REDSHIFT_LENS,
    REDSHIFT_SOURCE,
    mesh_fields,
    mesh_models,
    positions_likelihood_from,
)

#: The workspace ``source_lp[1]`` MGE source ahead of a pixelized chain.
SOURCE_LP_GAUSSIANS = 5
#: The profiled ``interferometer/mge`` cell's source: 20 Gaussians in one basis.
MGE_TOTAL_GAUSSIANS = 20

#: The pixelized families' ``--mesh``. The transformer is paired with the inversion by the
#: config-name grammar (``nufft`` dense, ``wtilde`` sparse), so a leaf need not check it again.
MESH_OF = {"rectangular": "rect", "delaunay": "delaunay"}


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def transformer_class(al, chunk_size):
    """``TransformerNUFFT`` at the instrument's chunk size (``None`` = one shot)."""

    def build(uv_wavelengths, real_space_mask):
        return al.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask, chunk_size=chunk_size
        )

    return build


def apply_wtilde(dataset, leg, chunk_size, nufft_precision_operator=None):
    """``dataset`` with the NUFFT-built W~ operator the ``wtilde`` setups time.

    ``use_jax`` selects only the backend that BUILDS the preload (the fit applies it with its
    own ``xp``), so a numba leg builds it with finufft and never starts XLA before Nautilus
    forks its workers.
    """
    return dataset.apply_sparse_operator(
        nufft_precision_operator=nufft_precision_operator,
        method="nufft",
        nufft_chunk_size=chunk_size,
        use_jax=leg.use_jax,
    )


def load_interferometer(leg):
    """The masked ``sma`` interferometer dataset (plain NUFFT, and the fitted route), its
    truths, the test-mode positions and the analysis settings. An int is an exit code."""
    al = leg.al
    from instruments.interferometer import INSTRUMENTS

    if leg.instrument not in INSTRUMENTS:
        print(
            f"ERROR: unknown --instrument {leg.instrument!r}; choose from {sorted(INSTRUMENTS)}",
            file=sys.stderr,
        )
        return 2
    preset = INSTRUMENTS[leg.instrument]
    mask_radius = preset["mask_radius"]
    chunk_size = preset.get("transformer_chunk_size")
    auto_simulate_if_missing(
        leg.dataset_path,
        dataset_type="interferometer",
        instrument=leg.instrument,
        workspace_root=leg.root,
    )
    real_space_mask = al.Mask2D.circular(
        shape_native=preset["real_space_shape"],
        pixel_scales=preset["pixel_scale"],
        radius=mask_radius,
    )
    dataset = al.Interferometer.from_fits(
        data_path=leg.dataset_path / "data.fits",
        noise_map_path=leg.dataset_path / "noise_map.fits",
        uv_wavelengths_path=leg.dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=transformer_class(al, chunk_size),
    )
    fit_dataset = (
        apply_wtilde(dataset, leg, chunk_size) if leg.cli.transformer == "wtilde" else dataset
    )
    tracer = al.from_json(file_path=str(leg.dataset_path / "tracer.json"))
    return SimpleNamespace(
        dataset_nufft=dataset,
        dataset=fit_dataset,
        mask_radius=mask_radius,
        chunk_size=chunk_size,
        truths=truth_dict(tracer),
        fallback_positions=al.Grid2DIrregular(
            al.from_json(file_path=str(leg.dataset_path / "positions.json"))
        ),
        settings=al.Settings(
            use_mixed_precision=leg.cli.use_mixed_precision,
            nnls_warm_start_memo=leg.cli.memo == "on",
        ),
    )


def _refuse_mesh(leg, variant: str) -> int | None:
    """A pixelized family runs only its own mesh: the row is filed under that likelihood."""
    if leg.cli.mesh != MESH_OF[variant]:
        print(
            f"ERROR: --mesh {leg.cli.mesh!r}: this is the interferometer/{variant} chain "
            f"(--mesh {MESH_OF[variant]})",
            file=sys.stderr,
        )
        return 2
    return None


def pixelized_loader(variant: str):
    def load(leg):
        code = _refuse_mesh(leg, variant)
        if code is not None:
            return code
        context = load_interferometer(leg)
        if isinstance(context, int):
            return context
        context.meshes = mesh_models(leg.af, leg.al, leg.cli)
        return context

    return load


# ---------------------------------------------------------------------------
# Adapt images
# ---------------------------------------------------------------------------


def capped_source_adapt_image(dataset, source_image):
    """The workspace's interferometer S/N cap: a copy of the model image, clipped at the flux
    of its S/N 3.0 contour in the dirty image (the homogeneous image-plane noise is the square
    root of half the summed visibility variances)."""
    import numpy as np

    capped = source_image.copy()
    sigma_pix = np.sqrt(0.5 * np.sum(np.abs(dataset.noise_map.array) ** 2))
    dirty = dataset.transformer.image_from(
        visibilities=dataset.transformer.visibilities_from(image=capped)
    )
    above_cap = capped.array[dirty.array / sigma_pix > ADAPT_IMAGE_SNR_CAP]
    if above_cap.size:
        capped[capped > above_cap.min()] = above_cap.min()
    else:
        print("  Adapt image S/N cap not applied: no pixel reaches S/N 3.0 in the dirty image.")
    return capped


def hilbert_mesh_grid(al, meshes, mask, mask_radius: float, adapt_data):
    """The Delaunay vertices: the Hilbert draw on the real-space mask plus the zeroed ring
    (``slam._runner._adapt_images``, the profiled preset's recipe)."""
    image_mesh = al.image_mesh.Hilbert(
        pixels=meshes.pixels, weight_power=meshes.weight_power, weight_floor=meshes.weight_floor
    )
    grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data)
    return al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=meshes.edge_points,
    )


def adapt_images_from(al, result, context):
    """The capped adapt images of ``result`` and, for the Delaunay family, its vertices."""
    name = "('galaxies', 'source')"
    images = al.galaxy_name_image_dict_via_result_from(result=result, use_model_images=True)
    images[name] = capped_source_adapt_image(context.dataset, images[name])
    meshes = context.meshes
    if meshes.image_mesh is None:
        return al.AdaptImages(galaxy_name_image_dict=images)
    grid = hilbert_mesh_grid(
        al, meshes, context.dataset.real_space_mask, context.mask_radius, images[name]
    )
    return al.AdaptImages(
        galaxy_name_image_dict=images, galaxy_name_image_plane_mesh_grid_dict={name: grid}
    )


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def _positions(leg, context, result, search_name):
    return positions_likelihood_from(
        leg.al, result, fallback_positions=context.fallback_positions, stage=search_name
    )


def source_lp_stage(total_gaussians: int, dataset_key: str, evaluates: bool) -> Stage:
    """``source_lp[1]``: the lens mass and an MGE source of ``total_gaussians``; no lens light.

    ``dataset_key`` names the context's dataset it fits (the plain NUFFT one ahead of a
    pixelized chain, as the workspace's ``dataset_nufft``); ``evaluates`` whether its
    evaluations are the family's likelihood.
    """

    def build(leg, context, results):
        af, al = leg.af, leg.al
        source_bulge = al.model_util.mge_model_from(
            mask_radius=context.mask_radius,
            total_gaussians=total_gaussians,
            centre_prior_is_uniform=False,
        )
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=REDSHIFT_LENS,
                    bulge=None,
                    disk=None,
                    mass=af.Model(al.mp.Isothermal),
                ),
                source=af.Model(al.Galaxy, redshift=REDSHIFT_SOURCE, bulge=source_bulge, disk=None),
            ),
            fields=af.Model(
                al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)
            ),
        )
        analysis = al.AnalysisInterferometer(
            dataset=getattr(context, dataset_key), use_jax=leg.use_jax, settings=context.settings
        )
        return model, analysis, leg.nautilus(stage)

    stage = Stage("source_lp", "source_lp[1]", 200, 50, build, evaluates_likelihood=evaluates)
    return stage


def build_source_pix_1(leg, context, results):
    """A pixelized source on the ``source_lp[1]`` mass, for the high-quality adapt image."""
    af, al = leg.af, leg.al
    source_lp_result = results["source_lp"]
    analysis = al.AnalysisInterferometer(
        dataset=context.dataset,
        adapt_images=adapt_images_from(al, source_lp_result, context),
        positions_likelihood_list=[
            _positions(leg, context, source_lp_result, PIXELIZED_STAGES[1].search_name)
        ],
        use_jax=leg.use_jax,
        settings=context.settings,
    )
    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=context.meshes.mesh_init,
                    regularization=context.meshes.regularization,
                ),
            ),
        ),
        fields=source_lp_result.model.fields,
    )
    return model, analysis, leg.nautilus(PIXELIZED_STAGES[1])


def build_source_pix_2(leg, context, results):
    """The final pixelized source, on the adapt images of ``source_pix[1]``."""
    af, al = leg.af, leg.al
    source_lp_result, source_pix_result_1 = results["source_lp"], results["source_pix_1"]
    analysis = al.AnalysisInterferometer(
        dataset=context.dataset,
        adapt_images=adapt_images_from(al, source_pix_result_1, context),
        positions_likelihood_list=[
            _positions(leg, context, source_pix_result_1, PIXELIZED_STAGES[2].search_name)
        ],
        use_jax=leg.use_jax,
        settings=context.settings,
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=context.meshes.mesh,
                    regularization=context.meshes.regularization,
                ),
            ),
        ),
        fields=source_pix_result_1.instance.fields,
    )
    return model, analysis, leg.nautilus(PIXELIZED_STAGES[2])


def build_pixelized_mass_total(leg, context, results):
    """The headline stage: a ``PowerLaw`` chained from ``source_pix[1]``, the source of
    ``source_pix[2]``."""
    af, al = leg.af, leg.al
    source_pix_result_1, source_pix_result_2 = results["source_pix_1"], results["source_pix_2"]
    analysis = al.AnalysisInterferometer(
        dataset=context.dataset,
        adapt_images=adapt_images_from(al, source_pix_result_1, context),
        positions_likelihood_list=[
            _positions(leg, context, source_pix_result_1, PIXELIZED_STAGES[3].search_name)
        ],
        use_jax=leg.use_jax,
        settings=context.settings,
    )
    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_pix_result_1.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_pix_result_1.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
            ),
            source=al.util.chaining.source_from(result=source_pix_result_2),
        ),
        fields=source_pix_result_1.model.fields,
    )
    return model, analysis, leg.nautilus(PIXELIZED_STAGES[3])


def build_mge_mass_total(leg, context, results):
    """The MGE chain's mass stage: a ``PowerLaw`` chained from ``source_lp[1]``, on its MGE
    source (the imaging MGE SLaM's ``mass_total[1]``, less the lens light)."""
    af, al = leg.af, leg.al
    source_lp_result = results["source_lp"]
    analysis = al.AnalysisInterferometer(
        dataset=context.dataset, use_jax=leg.use_jax, settings=context.settings
    )
    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
            ),
            source=al.util.chaining.source_from(result=source_lp_result),
        ),
        fields=source_lp_result.model.fields,
    )
    return model, analysis, leg.nautilus(MGE_STAGES[1])


#: The pixelized chain. ``source_lp[1]`` runs on the plain NUFFT dataset and evaluates none
#: of the profiled likelihoods; the three pixelized stages evaluate the family's.
PIXELIZED_STAGES = (
    source_lp_stage(SOURCE_LP_GAUSSIANS, "dataset_nufft", evaluates=False),
    Stage("source_pix_1", "source_pix[1]", 150, 20, build_source_pix_1, inversion_applies=True),
    Stage("source_pix_2", "source_pix[2]", 75, 20, build_source_pix_2, inversion_applies=True),
    Stage(
        "mass_total", "mass_total[1]", 150, 20, build_pixelized_mass_total, inversion_applies=True
    ),
)

#: The MGE chain: both stages evaluate ``interferometer/mge``, on the fitted (NUFFT) dataset.
MGE_STAGES = (
    source_lp_stage(MGE_TOTAL_GAUSSIANS, "dataset", evaluates=True),
    Stage("mass_total", "mass_total[1]", 150, 20, build_mge_mass_total),
)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------


def pixelized_variant(likelihood_variant: str):
    """``slam_<variant>`` at the profiled standard size; any other size names itself."""

    def variant(cli) -> str:
        if cli.mesh_pixels == MESH_PIXELS_STANDARD[cli.mesh]:
            return f"slam_{likelihood_variant}"
        return f"{cli.mesh}_{cli.mesh_pixels}"

    return variant


def pixelized_fields(leg, context) -> dict:
    """The source pixelization, in the imaging row's fields (``mesh`` / ``mesh_pixels`` /
    ``mesh_shape`` are what the exporter's size comparison reads)."""
    return mesh_fields(leg.cli, context.meshes)


def mge_fields(leg, context) -> dict:
    return {
        "mesh": None,
        "mesh_pixels": None,
        "mesh_shape": None,
        "mge_total_gaussians": MGE_TOTAL_GAUSSIANS,
        "mge_gaussian_per_basis": 1,
    }


def pixelized_family(likelihood_variant: str) -> Family:
    return Family(
        dataset_type="interferometer",
        likelihood_variant=likelihood_variant,
        stages=PIXELIZED_STAGES,
        load=pixelized_loader(likelihood_variant),
        inversions=("dense", "sparse"),
        transformers=("nufft", "wtilde"),
        variant=pixelized_variant(likelihood_variant),
        model_fields=pixelized_fields,
        title=f"Interferometer {likelihood_variant} SLaM run",
    )


RECTANGULAR = pixelized_family("rectangular")
DELAUNAY = pixelized_family("delaunay")
MGE = Family(
    dataset_type="interferometer",
    likelihood_variant="mge",
    stages=MGE_STAGES,
    load=load_interferometer,
    inversions=("dense",),
    transformers=("nufft",),
    model_fields=mge_fields,
    title="Interferometer MGE SLaM run",
)
