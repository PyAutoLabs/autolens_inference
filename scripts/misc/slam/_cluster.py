"""The cluster families: one Nautilus search over a factor graph, on the shared core.

The pipeline is ``autolens_workspace/scripts/cluster/modeling.py``: ONE Nautilus search
(``n_live=100``, ``n_batch=50``) of one shared lens model, fitted to every multiply-imaged
source at once — one ``AnalysisPoint`` per source, each wrapped in an ``af.AnalysisFactor``
of the same model and summed by an ``af.FactorGraphModel``. Each source is a parameter-free
``PointSolved`` at its own redshift, so the lens is multi-plane and the source centres are
solved analytically.

The model, the fit classes and the ``PointSolver`` are the profiled cell's
(``autolens_profiling/scripts/misc/likelihood_breakdown/cluster_standard.py``, which both
``scripts/cluster/<variant>/likelihood_runtime.py`` cells run), on the same
``dataset/cluster/simple`` (``scripts/misc/simulators/cluster.py``, ported byte-for-byte):

- 2 main ``dPIEMassB0Sph`` lenses with free centre, ``ra``, ``rs`` and ``b0`` (10 parameters);
- 10 scaling-relation ``dPIEMassB0Sph`` members at their catalogue centres, ``rs`` and ``b0``
  scaled from one shared free normalisation ``b0_ref`` by ``(L / L_ref)^0.5`` (1 parameter);
- an ``NFWMCRLudlowSph`` host with a free centre and ``mass_at_200`` fixed (2 parameters);
- 2 ``PointSolved`` sources at z = 1 and 2 (0 parameters). 13 free parameters in all.

The two likelihoods differ only in the fit class (decision D3 of the profiling cluster
campaign, human 2026-10-29: "the cluster default setup should use solved"):

- :data:`SOURCE_PLANE` (``cluster/source_plane``) — ``al.FitPositionsSourceSolved``, the
  workspace's search-stage fit (``modeling.py`` ~:580), which ray-traces the observed images
  back and never forward-solves;
- :data:`IMAGE_PLANE` (``cluster/image_plane``) — ``al.FitPositionsImagePairRepeatSolved``,
  the image-plane chi-squared the profiled cell times, forward-solving every source's images
  with the standard cluster ``PointSolver`` (decision D1: 100 x 100 @ 1.0",
  ``pixel_scale_precision`` 0.001, ``magnification_threshold`` 0.1, ``residual_threshold``
  10, every setting passed explicitly). The workspace has no image-plane cluster search (it
  validates the max-likelihood model with the image-plane fit instead), so this family is the
  nearest analogue: ``modeling.py``'s search with the profiled image-plane fit class swapped
  in.

Departures from ``modeling.py``, each forced by the ruling "workspace pipeline + profiled
preset" (human, 2026-11-02):

1. **The parameterisation is the profiled cell's**, not the workspace's ``dPIEMassSph``
   ``sigma`` / ``r_cut`` with centres fixed and a free halo mass: the per-evaluation cost
   being validated is the profiled model's, and the ported simulator writes no ``mass.csv``
   for the workspace's CSV model loaders.
2. **The priors are a search's, not the profiled cell's.** Profiling times one call at the
   prior medians, so it puts 1%-wide Gaussians on the truth; a search that started there
   would measure a fit nobody runs. Each main-lens centre takes a Gaussian of sigma
   :data:`MAIN_CENTRE_SIGMA` on its light centre (``main_lens_centres.json``: the workspace
   pins mass centres to light centres, this frees them a little, as the profiled cell does);
   the host centre a Gaussian of sigma :data:`HOST_CENTRE_SIGMA` on ``host_halo_centre.json``;
   every main-lens ``ra`` / ``rs`` / ``b0`` and the scaling ``b0_ref`` a factor-of-two box
   ``U(truth / 2, 2 truth)`` around the simulator truth (:data:`PRIOR_BOX`).
3. **Every analysis is given** ``use_jax=leg.use_jax`` (the workspace hard-codes ``True``),
   so the row says what it ran.

The likelihood has no linear inversion, so the leaves accept only ``--inversion na`` and the
Optimal setups profiling measured (``jax_cpu_na_fp64`` on a RAL CPU node, ``jax_gpu_na_fp64``
on an A100).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _inference_cli import auto_simulate_if_missing  # noqa: E402
from slam._core import Family, Stage, posterior_keys  # noqa: E402
from slam._point import (  # noqa: E402
    PROFILED_SETUPS,
    SEARCH_NAME,
    STAGE_KEY,
    refuse_unprofiled_setup,
)

#: The standard cluster's constants (``scripts/misc/simulators/cluster.py``; the profiled
#: cell's ``cluster_standard.MAIN_LENS_PARAMS`` etc.). ``(centre, ra, rs, b0)`` per main lens.
REDSHIFT_LENS = 0.5
MAIN_LENS_PARAMS = (
    ((0.0, 0.0), 8.0, 20.0, 3.0),
    ((10.0, 8.0), 5.0, 12.0, 1.2),
)
SCALING_B0_REF, SCALING_RS_REF, SCALING_RA, SCALING_EXPONENT = 0.12, 10.0, 0.1, 0.5
MASS_AT_200 = 10**15.3

#: The search's priors (departure 2 in the module docstring).
MAIN_CENTRE_SIGMA = 0.1
HOST_CENTRE_SIGMA = 1.0
PRIOR_BOX = 2.0

#: The standard cell's PointSolver (decision D1), every setting explicit. Used by the
#: image-plane fit at every evaluation; the source-plane fit carries it unused, as the
#: workspace's does.
STANDARD_SOLVER = {
    "shape_native": (100, 100),
    "pixel_scales": 1.0,
    "pixel_scale_precision": 0.001,
    "magnification_threshold": 0.1,
    "residual_threshold": 10.0,
}


def _centres(path: Path) -> list[tuple[float, float]]:
    """The (y, x) centres of a simulator ``Grid2DIrregular`` JSON."""
    values = json.loads(path.read_text())["arguments"]["values"]["array"]
    return [(float(y), float(x)) for y, x in values]


def _load(fit_positions_cls: str):
    def load(leg):
        code = refuse_unprofiled_setup(leg, PROFILED_SETUPS)
        if code is not None:
            return code
        al = leg.al
        auto_simulate_if_missing(
            leg.dataset_path,
            dataset_type=leg.dataset_class,
            instrument=leg.instrument,
            workspace_root=leg.root,
        )
        dataset_list = al.list_from_csv(file_path=leg.dataset_path / "point_datasets.csv")
        scaling = al.galaxy_table_from_csv(file_path=leg.dataset_path / "scaling_galaxies.csv")
        solver = al.PointSolver.for_grid(
            grid=al.Grid2D.uniform(
                shape_native=STANDARD_SOLVER["shape_native"],
                pixel_scales=STANDARD_SOLVER["pixel_scales"],
            ),
            pixel_scale_precision=STANDARD_SOLVER["pixel_scale_precision"],
            magnification_threshold=STANDARD_SOLVER["magnification_threshold"],
            residual_threshold=STANDARD_SOLVER["residual_threshold"],
            use_jax=leg.use_jax,
        )
        context = SimpleNamespace(
            dataset_list=dataset_list,
            scaling_centres=[tuple(c) for c in scaling.centres],
            scaling_luminosities=list(scaling.luminosities),
            main_centres=_centres(leg.dataset_path / "main_lens_centres.json"),
            host_centre=_centres(leg.dataset_path / "host_halo_centre.json")[0],
            solver=solver,
            fit_positions_cls=getattr(al, fit_positions_cls),
        )
        _, context.truths = cluster_model(leg.af, al, context)
        return context

    return load


def _box(af, truth: float, truth_by_id: dict):
    prior = af.UniformPrior(lower_limit=truth / PRIOR_BOX, upper_limit=truth * PRIOR_BOX)
    truth_by_id[prior.id] = truth
    return prior


def _gaussian(af, mean: float, sigma: float, truth: float, truth_by_id: dict):
    prior = af.GaussianPrior(mean=mean, sigma=sigma)
    truth_by_id[prior.id] = truth
    return prior


def cluster_model(af, al, context):
    """``(model, truths)``: the profiled cluster model under the search's priors, and the
    simulator truth of every free parameter keyed as the row's posterior is
    (:func:`slam._core.posterior_keys`), so ``truth_delta_sigma`` is a key intersection."""
    truth_by_id: dict[int, float] = {}
    galaxies = {}
    for i, ((centre, ra, rs, b0), light_centre) in enumerate(
        zip(MAIN_LENS_PARAMS, context.main_centres)
    ):
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre.centre_0 = _gaussian(
            af, light_centre[0], MAIN_CENTRE_SIGMA, centre[0], truth_by_id
        )
        mass.centre.centre_1 = _gaussian(
            af, light_centre[1], MAIN_CENTRE_SIGMA, centre[1], truth_by_id
        )
        mass.ra = _box(af, ra, truth_by_id)
        mass.rs = _box(af, rs, truth_by_id)
        mass.b0 = _box(af, b0, truth_by_id)
        galaxies[f"main_{i}"] = af.Model(al.Galaxy, redshift=REDSHIFT_LENS, mass=mass)

    # One shared normalisation; each member's b0 / rs follow it by its luminosity ratio.
    lum_ref = max(context.scaling_luminosities)
    b0_ref = _box(af, SCALING_B0_REF, truth_by_id)
    for i, (centre, lum) in enumerate(zip(context.scaling_centres, context.scaling_luminosities)):
        ratio = (lum / lum_ref) ** SCALING_EXPONENT
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre = tuple(centre)
        mass.ra = SCALING_RA
        mass.rs = SCALING_RS_REF * ratio
        mass.b0 = b0_ref * ratio
        galaxies[f"scaling_{i}"] = af.Model(al.Galaxy, redshift=REDSHIFT_LENS, mass=mass)

    dark = af.Model(
        al.mp.NFWMCRLudlowSph,
        mass_at_200=MASS_AT_200,
        redshift_object=REDSHIFT_LENS,
        redshift_source=max(float(d.redshift) for d in context.dataset_list),
    )
    host_0, host_1 = context.host_centre
    dark.centre.centre_0 = _gaussian(af, host_0, HOST_CENTRE_SIGMA, host_0, truth_by_id)
    dark.centre.centre_1 = _gaussian(af, host_1, HOST_CENTRE_SIGMA, host_1, truth_by_id)
    galaxies["host_halo"] = af.Model(al.Galaxy, redshift=REDSHIFT_LENS, dark=dark)

    # The source galaxy carries its point under the dataset's name, which pairs it with
    # that dataset's analysis.
    for i, dataset in enumerate(context.dataset_list):
        galaxies[f"source_{i}"] = af.Model(
            al.Galaxy,
            redshift=float(dataset.redshift),
            **{dataset.name: af.Model(al.ps.PointSolved)},
        )

    model = af.Collection(galaxies=af.Collection(**galaxies))
    keys = posterior_keys(list(model.all_names))
    truths = {key: truth_by_id[prior.id] for key, prior in zip(keys, model.priors_ordered_by_id)}
    return model, truths


def build_modeling(leg, context, results):
    """``modeling.py``'s one search: one ``AnalysisPoint`` per source, each an
    ``AnalysisFactor`` of the shared model, summed by a ``FactorGraphModel``."""
    af, al = leg.af, leg.al
    model, _ = cluster_model(af, al, context)
    factors = [
        af.AnalysisFactor(
            prior_model=model,
            analysis=al.AnalysisPoint(
                dataset=dataset,
                solver=context.solver,
                fit_positions_cls=context.fit_positions_cls,
                use_jax=leg.use_jax,
            ),
        )
        for dataset in context.dataset_list
    ]
    factor_graph = af.FactorGraphModel(*factors, use_jax=leg.use_jax)
    return factor_graph.global_prior_model, factor_graph, leg.nautilus(MODELING)


#: Nautilus at ``autolens_workspace/scripts/cluster/modeling.py``'s settings.
MODELING = Stage(STAGE_KEY, SEARCH_NAME, 100, 50, build_modeling)


def _model_fields(fit_positions_cls: str):
    def model_fields(leg, context) -> dict:
        return {
            "fit_positions_cls": fit_positions_cls,
            "point_solver": {
                "grid_shape_native": list(STANDARD_SOLVER["shape_native"]),
                "grid_pixel_scales": STANDARD_SOLVER["pixel_scales"],
                "pixel_scale_precision": STANDARD_SOLVER["pixel_scale_precision"],
                "magnification_threshold": STANDARD_SOLVER["magnification_threshold"],
                "residual_threshold": STANDARD_SOLVER["residual_threshold"],
            },
            "point_datasets": len(context.dataset_list),
            "scaling_members": len(context.scaling_centres),
        }

    return model_fields


def _family(variant: str, fit_positions_cls: str) -> Family:
    return Family(
        dataset_type="cluster",
        likelihood_variant=variant,
        stages=(MODELING,),
        load=_load(fit_positions_cls),
        pipeline="modeling",
        inversions=("na",),
        model_fields=_model_fields(fit_positions_cls),
        title="Cluster modeling run",
    )


SOURCE_PLANE = _family("source_plane", "FitPositionsSourceSolved")
IMAGE_PLANE = _family("image_plane", "FitPositionsImagePairRepeatSolved")
