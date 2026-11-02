"""The multi_dataset family: the HST simultaneous two-dataset SLaM chain of
``multi_dataset/delaunay``.

Likelihood-lifecycle 4c-2 (autolens_inference#54). The profiled cell
(``autolens_profiling/scripts/misc/likelihood_breakdown/multi_dataset_standard.py``,
decision D6) is the imaging Delaunay standard cell of hst fitted to two datasets at once: the
hst imaging dataset (``simulators/imaging.py --instrument hst``, the 3.5" mask) and a copy whose
data AND noise map are scaled by 0.5 — a second exposure with another flux calibration and the
same signal-to-noise — summed by an ``af.FactorGraphModel`` of one ``AnalysisImaging`` per
dataset, each solving its own inversion. :func:`load` builds both datasets in memory from
``dataset/imaging/hst/`` (auto-simulated first); nothing new is written to disk.

The chain mirrors ``multi_dataset/features/slam/simultaneous.py`` without its three subhalo stages
(human ruling 2026-11-02): ``source_lp[1]`` -> ``source_pix[1]`` -> ``source_pix[2]`` ->
``light[1]`` -> ``mass_total[1]``, one lens model shared by both datasets, one ``DatasetModel``
per dataset (the first the astrometric reference at a fixed zero offset, the second's grid offset
free in +-1"), and a free regularization per dataset in the two SOURCE PIX stages. The profiled
preset is swapped into the mesh stages: ``al.mesh.Delaunay`` at 1250 Hilbert vertices (+30
zeroed edge points) under a free ``AdaptSplit``, the image mesh drawn per dataset from its own
S/N-capped adapt image (``_runner._adapt_images``), where the workspace runs a 28 x 28
rectangular mesh under ``Adapt``. The four mesh stages evaluate the profiled likelihood;
``source_lp[1]`` (an MGE source) declares ``null``.

The hst dataset has lens light, so the family is ``Family.light_fixed`` as the imaging chain
is: ``mass_total[1]`` is fitted once per lens-light mode, the ``fixed`` one carrying each
dataset's ``light[1]`` maximum-likelihood lens light as ordinary profiles
(``_group.lens_light_galaxies``).

Departures from ``simultaneous.py``: a positions likelihood on all four mesh stages (the
workspace has ``source_pix[1]`` and ``mass_total[1]``), solved from dataset 0's result at factor
3.0 (under ``PYAUTO_TEST_MODE``, the fixed test-mode threshold on the simulator's
``positions.json``); the profiled 3.5" mask, not 3.0"; ``use_jax`` on every ``Analysis``. The
pixelization over-sampling stays the dataset's uniform default on every stage, as
``simultaneous.py`` keeps it (the profiled cell's adaptive map is not swapped in: it is data
preparation, not the mesh, and the workspace documents why a simultaneous fit keeps it uniform).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _inference_cli import MESH_PIXELS_STANDARD, auto_simulate_if_missing  # noqa: E402
from slam._core import Family, Stage, truth_dict  # noqa: E402
from slam._group import lens_light_galaxies  # noqa: E402
from slam._runner import (  # noqa: E402
    OVER_SAMPLE_RADIAL_LIST,
    OVER_SAMPLE_SUB_SIZE_LIST,
    POSITIONS_MINIMUM_THRESHOLD,
    TEST_MODE_POSITIONS_THRESHOLD,
    _adapt_images,
    _apply_inversion,
    mesh_fields,
    mesh_models,
)

INSTRUMENT = "hst"
#: The profiled cell's mask (``multi_dataset_standard.MASK_RADIUS``).
MASK_RADIUS = 3.5
#: Each dataset's flux-calibration scale: data and noise map times ``s`` (decision D6).
DATASET_SCALES = (1.0, 0.5)
#: The second dataset's grid-offset prior half-width (``simultaneous.py``).
GRID_OFFSET_LIMIT = 1.0
REDSHIFT_LENS = 0.5
REDSHIFT_SOURCE = 1.0
MESH = "delaunay"


@dataclass
class MultiDatasetContext:
    """The two masked datasets, the shared fallback positions, the truth and the mesh."""

    datasets: list
    positions: object
    truths: dict
    settings: object
    meshes: object
    cli: object
    stages: dict = field(default_factory=dict)


def scaled_dataset(al, base, scale: float):
    """``base`` with its data and noise map times ``scale``, everything else shared
    (``multi_dataset_standard.scaled_datasets``)."""
    import numpy as np

    if scale == 1.0:
        return base
    return al.Imaging(
        data=al.Array2D(values=np.asarray(base.data) * scale, mask=base.mask),
        noise_map=al.Array2D(values=np.asarray(base.noise_map) * scale, mask=base.mask),
        psf=base.psf,
        noise_covariance_matrix=base.noise_covariance_matrix,
        over_sample_size_lp=base.over_sample_size_lp,
        over_sample_size_pixelization=base.over_sample_size_pixelization,
        convolve_over_sample_size_lp=base.convolve_over_sample_size_lp,
        convolve_over_sample_size_pixelization=base.convolve_over_sample_size_pixelization,
        check_noise_map=False,
    )


def load(leg, stages: tuple = ()):
    """Both datasets: hst imaging masked at 3.5", light-profile over-sampled about the lens
    (the workspace's 4 / 2 / 2 at 0.3" / 0.6"), scaled by :data:`DATASET_SCALES`, each with this
    run's inversion path (the sparse operator depends on the noise map, so each builds its own)."""
    al, cli = leg.al, leg.cli
    from instruments.imaging import INSTRUMENTS

    if leg.instrument != INSTRUMENT:
        print(
            f"ERROR: multi_dataset is profiled at {INSTRUMENT} only; got --instrument "
            f"{leg.instrument}",
            file=sys.stderr,
        )
        return 2
    if cli.mesh != MESH or cli.mesh_pixels != MESH_PIXELS_STANDARD[MESH]:
        print(
            f"ERROR: this leaf runs the profiled cell's mesh, --mesh {MESH} at --mesh-pixels "
            f"{MESH_PIXELS_STANDARD[MESH]}; got --mesh {cli.mesh} --mesh-pixels {cli.mesh_pixels}",
            file=sys.stderr,
        )
        return 2

    path = leg.root / "dataset" / "imaging" / INSTRUMENT
    auto_simulate_if_missing(
        path, dataset_type="imaging", instrument=INSTRUMENT, workspace_root=leg.root
    )
    base = al.Imaging.from_fits(
        data_path=path / "data.fits",
        noise_map_path=path / "noise_map.fits",
        psf_path=path / "psf.fits",
        pixel_scales=INSTRUMENTS[INSTRUMENT]["pixel_scale"],
    )
    mask = al.Mask2D.circular(
        shape_native=base.shape_native, pixel_scales=base.pixel_scales, radius=MASK_RADIUS
    )
    base = base.apply_mask(mask=mask)
    base = base.apply_over_sampling(
        over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=base.grid,
            sub_size_list=OVER_SAMPLE_SUB_SIZE_LIST,
            radial_list=OVER_SAMPLE_RADIAL_LIST,
            centre_list=[(0.0, 0.0)],
        )
    )
    datasets = [
        _apply_inversion(scaled_dataset(al, base, scale), cli.backend, cli.inversion)
        for scale in DATASET_SCALES
    ]
    return MultiDatasetContext(
        datasets=datasets,
        positions=al.Grid2DIrregular(al.from_json(file_path=str(path / "positions.json"))),
        truths=truth_dict(al.from_json(file_path=str(path / "tracer.json"))),
        settings=al.Settings(
            use_mixed_precision=cli.use_mixed_precision,
            nnls_warm_start_memo=cli.memo == "on",
        ),
        meshes=mesh_models(leg.af, al, cli),
        cli=cli,
        stages={stage.key: stage for stage in stages},
    )


def model_fields(leg, context) -> dict:
    return {**mesh_fields(context.cli, context.meshes), "dataset_scales": list(DATASET_SCALES)}


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def dataset_models(af, al) -> list:
    """One ``DatasetModel`` per dataset: dataset 0 the reference (offset fixed at zero), every
    other one's grid offset free in +-:data:`GRID_OFFSET_LIMIT`."""
    models = []
    for i in range(len(DATASET_SCALES)):
        model = af.Model(al.DatasetModel)
        if i > 0:
            model.grid_offset.grid_offset_0 = af.UniformPrior(
                lower_limit=-GRID_OFFSET_LIMIT, upper_limit=GRID_OFFSET_LIMIT
            )
            model.grid_offset.grid_offset_1 = af.UniformPrior(
                lower_limit=-GRID_OFFSET_LIMIT, upper_limit=GRID_OFFSET_LIMIT
            )
        models.append(model)
    return models


def factor_graph(leg, context, models, *, results=None, adapt_from=None, positions_from=None):
    """``(global model, FactorGraphModel)``: one ``AnalysisImaging`` per dataset, with the adapt
    images of ``adapt_from``'s matching result and a positions likelihood solved from
    ``positions_from``'s dataset-0 result."""
    af, al = leg.af, leg.al
    positions = None
    if positions_from is not None:
        if leg.test_mode:
            print(
                "  NOTE: PYAUTO_TEST_MODE — positions likelihood on the simulator's positions.json "
                f"at threshold {TEST_MODE_POSITIONS_THRESHOLD}, not solved from a near-random result."
            )
            positions = al.PositionsLH(
                positions=context.positions, threshold=TEST_MODE_POSITIONS_THRESHOLD
            )
        else:
            positions = results[positions_from][0].positions_likelihood_from(
                factor=3.0, minimum_threshold=POSITIONS_MINIMUM_THRESHOLD
            )
    factors = []
    for i, (dataset, model) in enumerate(zip(context.datasets, models)):
        kwargs = {}
        if adapt_from is not None:
            kwargs["adapt_images"] = _adapt_images(
                al,
                results[adapt_from][i],
                meshes=context.meshes,
                dataset=dataset,
                mask_radius=MASK_RADIUS,
            )
        if positions is not None:
            kwargs["positions_likelihood_list"] = [positions]
        analysis = al.AnalysisImaging(
            dataset=dataset, use_jax=leg.use_jax, settings=context.settings, **kwargs
        )
        factors.append(af.AnalysisFactor(prior_model=model, analysis=analysis))
    graph = af.FactorGraphModel(*factors, use_jax=leg.use_jax)
    return graph.global_prior_model, graph


def built(leg, context, key, graph):
    model, analysis = graph
    return model, analysis, leg.nautilus(context.stages[key])


# ---------------------------------------------------------------------------
# The chain (simultaneous.py, without the subhalo stages)
# ---------------------------------------------------------------------------


def build_source_lp(leg, context, results):
    """One parametric lens and source for every dataset: MGE 20 x 2 lens light, ``Isothermal``
    at the fixed centre (0, 0), shear, MGE source; only the ``DatasetModel`` differs."""
    af, al = leg.af, leg.al
    pixel_scale = context.datasets[0].pixel_scales[0]
    mass = af.Model(al.mp.Isothermal)
    mass.centre = (0.0, 0.0)
    lens = af.Model(
        al.Galaxy,
        redshift=REDSHIFT_LENS,
        bulge=al.model_util.mge_model_from(
            mask_radius=MASK_RADIUS,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            sigma_min=pixel_scale / 10.0,
        ),
        disk=None,
        mass=mass,
    )
    source = af.Model(
        al.Galaxy,
        redshift=REDSHIFT_SOURCE,
        bulge=al.model_util.mge_model_from(
            mask_radius=MASK_RADIUS, total_gaussians=20, centre_prior_is_uniform=False
        ),
    )
    field_model = af.Model(
        al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)
    )
    models = [
        af.Collection(
            galaxies=af.Collection(lens=lens, source=source),
            fields=field_model,
            dataset_model=dataset_model,
        )
        for dataset_model in dataset_models(af, al)
    ]
    return built(leg, context, "source_lp", factor_graph(leg, context, models))


def build_source_pix_1(leg, context, results):
    """The pixelized source per dataset on the ``source_lp[1]`` light; the mass chained with its
    centre freed and shared by every dataset; a free regularization per dataset."""
    af, al = leg.af, leg.al
    lp = results["source_lp"]
    meshes = context.meshes
    models = []
    for i, dataset_model in enumerate(dataset_models(af, al)):
        mass = al.util.chaining.mass_from(
            mass=lp[i].model.galaxies.lens.mass,
            mass_result=lp[i].model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )
        if i > 0:
            mass.centre = models[0].galaxies.lens.mass.centre
        lens = lp[i].instance.galaxies.lens
        models.append(
            af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=lens.redshift,
                        bulge=lens.bulge,
                        disk=lens.disk,
                        mass=mass,
                    ),
                    source=af.Model(
                        al.Galaxy,
                        redshift=lp[i].instance.galaxies.source.redshift,
                        pixelization=af.Model(
                            al.Pixelization,
                            mesh=meshes.mesh_init,
                            regularization=meshes.regularization,
                        ),
                    ),
                ),
                dataset_model=dataset_model,
                fields=lp[i].model.fields,
            )
        )
    graph = factor_graph(
        leg, context, models, results=results, adapt_from="source_lp", positions_from="source_lp"
    )
    return built(leg, context, "source_pix_1", graph)


def build_source_pix_2(leg, context, results):
    """The final pixelized source per dataset on ``source_pix[1]``'s adapt images, the lens
    fixed."""
    af, al = leg.af, leg.al
    lp, pix1 = results["source_lp"], results["source_pix_1"]
    meshes = context.meshes
    models = []
    for i, dataset_model in enumerate(dataset_models(af, al)):
        lens = lp[i].instance.galaxies.lens
        models.append(
            af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=lens.redshift,
                        bulge=lens.bulge,
                        disk=lens.disk,
                        mass=pix1[i].instance.galaxies.lens.mass,
                    ),
                    source=af.Model(
                        al.Galaxy,
                        redshift=lp[i].instance.galaxies.source.redshift,
                        pixelization=af.Model(
                            al.Pixelization, mesh=meshes.mesh, regularization=meshes.regularization
                        ),
                    ),
                ),
                dataset_model=dataset_model,
                fields=pix1[i].instance.fields,
            )
        )
    graph = factor_graph(
        leg,
        context,
        models,
        results=results,
        adapt_from="source_pix_1",
        positions_from="source_pix_1",
    )
    return built(leg, context, "source_pix_2", graph)


def build_light(leg, context, results):
    """One fresh MGE lens light shared by every dataset; mass and source fixed."""
    af, al = leg.af, leg.al
    pix1, pix2 = results["source_pix_1"], results["source_pix_2"]
    pixel_scale = context.datasets[0].pixel_scales[0]
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=MASK_RADIUS,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        sigma_min=pixel_scale / 10.0,
    )
    models = [
        af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=pix1[i].instance.galaxies.lens.redshift,
                    bulge=lens_bulge,
                    disk=None,
                    mass=pix1[i].instance.galaxies.lens.mass,
                ),
                source=al.util.chaining.source_custom_model_from(
                    result=pix2[i], source_is_model=False
                ),
            ),
            dataset_model=dataset_model,
            fields=pix1[i].instance.fields,
        )
        for i, dataset_model in enumerate(dataset_models(af, al))
    ]
    graph = factor_graph(
        leg,
        context,
        models,
        results=results,
        adapt_from="source_pix_1",
        positions_from="source_pix_2",
    )
    return built(leg, context, "light", graph)


def build_mass_total(leg, context, results):
    """A ``PowerLaw`` chained per dataset from ``source_pix[1]``, the light fixed from
    ``light[1]``, the source from ``source_pix[2]``."""
    af, al = leg.af, leg.al
    pix1, pix2, light = results["source_pix_1"], results["source_pix_2"], results["light"]
    mass = af.Model(al.mp.PowerLaw)
    models = []
    for i, dataset_model in enumerate(dataset_models(af, al)):
        ((lens,), _) = lens_light_galaxies(leg, light[i], ["lens"])
        models.append(
            af.Collection(
                galaxies=af.Collection(
                    lens=af.Model(
                        al.Galaxy,
                        redshift=pix1[i].instance.galaxies.lens.redshift,
                        bulge=lens.bulge,
                        disk=lens.disk,
                        mass=al.util.chaining.mass_from(
                            mass=mass,
                            mass_result=pix1[i].model.galaxies.lens.mass,
                            unfix_mass_centre=True,
                        ),
                    ),
                    source=al.util.chaining.source_from(result=pix2[i]),
                ),
                dataset_model=dataset_model,
                fields=pix1[i].model.fields,
            )
        )
    graph = factor_graph(
        leg,
        context,
        models,
        results=results,
        adapt_from="source_pix_1",
        positions_from="source_pix_2",
    )
    return built(leg, context, "mass_total", graph)


#: ``(n_live, n_batch)`` per stage, as ``simultaneous.py`` sets them.
STAGES = (
    Stage("source_lp", "source_lp[1]", 200, 50, build_source_lp, evaluates_likelihood=False),
    Stage("source_pix_1", "source_pix[1]", 150, 20, build_source_pix_1, inversion_applies=True),
    Stage("source_pix_2", "source_pix[2]", 75, 20, build_source_pix_2, inversion_applies=True),
    Stage("light", "light[1]", 250, 20, build_light, inversion_applies=True),
    Stage("mass_total", "mass_total[1]", 150, 20, build_mass_total, inversion_applies=True),
)

FAMILY_DELAUNAY = Family(
    dataset_type="multi_dataset",
    likelihood_variant="delaunay",
    stages=STAGES,
    load=lambda leg: load(leg, STAGES),
    model_fields=model_fields,
    title="Multi-dataset simultaneous SLaM run",
    light_fixed=True,
)
