"""The group family: the Euclid group-scale SLaM chains of ``group/{rectangular,delaunay,mge}``.

Likelihood-lifecycle 4c-2 (autolens_inference#54). Three likelihoods, one dataset: the profiled
group cell (``simulators/_galaxy_group.SPECS["group"]``, byte-for-byte the data
autolens_profiling's ``cell_builders/galaxy_group.py`` fits) — one main lens, two extra galaxies
with ``dPIEMassSph`` masses, a SersicCore source, at euclid (0.1"/px, 7.5" mask, light-profile
over-sampling 4 / 2 / 2 at 0.3" / 0.6" on every galaxy centre).

The chains mirror the autolens_workspace group pipelines (human ruling 2026-11-02, "Workspace
pipeline + profiled preset"), with the profiled cell's mesh, regularization and size swapped into
the stages whose likelihood is the one timed:

- ``group/delaunay`` — ``group/slam.py``, six stages: ``source_lp[0]`` (light only) ->
  ``source_lp[1]`` (mass, MGE source) -> ``source_pix[1]`` -> ``source_pix[2]`` -> ``light[1]`` ->
  ``mass_total[1]``. The Hilbert image mesh is drawn at the profiled standard 1250 vertices (+30
  zeroed edge points), not ``hilbert_pixels_from_pixel_scale``; ``AdaptSplit`` free, as there.
- ``group/rectangular`` — ``group/features/pixelization/slam.py``, the same six stages on
  ``RectangularBilinearAdaptDensity`` then ``RectangularBilinearAdaptImage`` under ``Adapt``, at
  the profiled 35 x 35 instead of the workspace 28 x 28, and that script's ``n_live`` (150 on
  ``light[1]`` and ``mass_total[1]``).
- ``group/mge`` — ``group/features/multi_gaussian_expansion/slam.py``, three stages:
  ``source_lp[1]`` -> ``light[1]`` -> ``mass_total[1]``, every one an MGE linear solve.

The mesh models, the S/N-capped adapt images (and the Hilbert mesh grid), the inversion path and
the row's mesh fields are the imaging builder's (``_runner.mesh_models`` / ``_adapt_images`` /
``_apply_inversion`` / ``mesh_fields``), which take any masked dataset.

Which stages evaluate the profiled likelihood
---------------------------------------------

The profiled cell is the full model: MGE lens light, Isothermal + shear, the two extra galaxies'
MGE + dPIE, and the source. A pixelized chain's four mesh stages evaluate it (``source_pix[1]``
on the uniform pixelization sub-size, the rest on the adaptive 4 / 2 map, as the workspace); its
two parametric stages do not (``source_lp[0]`` has no mass and no source, ``source_lp[1]`` an MGE
source), so they declare ``likelihood_evaluated: null``. Every stage of the MGE chain is the MGE
likelihood.

The lens-light modes
--------------------

The dataset has lens light, so every family here is ``Family.light_fixed``: ``--lens-light
fixed|both`` fits ``mass_total[1]`` once per mode, as the imaging chain does. ``linear`` carries
``light[1]``'s linear MGEs (the workspace); ``fixed`` carries the same profiles at the
intensities ``light[1]``'s maximum-likelihood fit solved, main lenses and extra galaxies alike
(:func:`lens_light_galaxies`), and its row is the ``light_fixed`` model variant.

Departures from the workspace pipelines
---------------------------------------

1. **A positions likelihood on all four pixelized stages** (the workspace attaches one to
   ``source_pix[1]`` and ``mass_total[1]`` only), as the imaging chain does, so every mesh stage
   writes a ``positions.info``. The positions are the truth tracer's multiple images, solved at
   load (the workspace simulator writes them to ``positions.json``; the ported simulator does
   not), with the workspace factors (2.0 on ``source_pix[1]``, 3.0 elsewhere). Under
   ``PYAUTO_TEST_MODE`` the threshold is the fixed test-mode one instead of a solved one, as the
   imaging chain's.
2. **The rectangular chain's model is ``group/slam.py``'s**, mesh aside: the luminosity-bounded
   dPIE ``sigma`` prior on ``mass_total[1]`` (the pixelization script frees it to 300) and the
   1.0" source MGE. A luminosity of zero or less (a test-mode draw) bounds the Einstein radius at
   5.0", as the pixelization script guards it.
3. **No scaling galaxies.** The profiled dataset has none, so the tier is omitted rather than
   carried empty.
4. **The mask and over-sampling are the profiled cell's** (7.5", 0.3" / 0.6" bins), not
   ``group/slam.py``'s 6.0" / 0.1" / 0.3"; the lens light is the workspace's (an MGE of 30 x 2 in
   the pixelized chain, 20 in the MGE chain), not the cell's 20.
5. ``use_jax`` is passed on every ``Analysis``, and ``n_batch`` is the workspace's (50 on the
   parametric stages, 20 after).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from simulators._galaxy_group import SPECS  # noqa: E402

from _inference_cli import MESH_PIXELS_STANDARD, auto_simulate_if_missing  # noqa: E402
from slam._core import Family, Stage, truth_dict  # noqa: E402
from slam._runner import (  # noqa: E402
    OVER_SAMPLE_SNR_THRESHOLD,
    POSITIONS_MINIMUM_THRESHOLD,
    TEST_MODE_POSITIONS_THRESHOLD,
    _adapt_images,
    _apply_inversion,
    linear_light_profiles_in,
    mesh_fields,
    mesh_models,
)

REDSHIFT_LENS = 0.5
REDSHIFT_SOURCE = 1.0

#: The ``--mesh`` each pixelized likelihood runs, at the profiled standard size.
MESH_OF = {"rectangular": "rect", "delaunay": "delaunay"}

#: The fixed positions threshold of ``group/slam.py`` ``source_lp[1]``.
SOURCE_LP_POSITIONS_THRESHOLD = 0.3

#: The source MGE radius of ``group/slam.py`` ``source_lp[1]``.
SOURCE_MGE_RADIUS = 1.0

#: The point solver the workspace group simulator writes ``positions.json`` with.
POSITIONS_SOLVER_PRECISION = 0.001
POSITIONS_MAGNIFICATION_THRESHOLD = 0.01

#: The dPIE's non-parameters, fixed as the workspace fixes them.
DPIE_CONSTANTS = dict(r_core=0.0, r_cut=10.0, H0=67.66, Om0=0.30966)


# ---------------------------------------------------------------------------
# The dataset (shared with the multi_galaxy family)
# ---------------------------------------------------------------------------


@dataclass
class GalaxyGroupContext:
    """What every stage of a group / multi_galaxy chain reads: the masked dataset (re-derived
    after ``source_pix[1]``), its galaxy centres and truth positions, and the mesh."""

    spec: object
    dataset: object
    over_sample_size_lp: object
    positions: object
    truths: dict
    settings: object
    meshes: object | None
    cli: object
    #: The family's stages by key: each builder takes its search's ``n_live`` from here.
    stages: dict = field(default_factory=dict)

    @property
    def mask_radius(self) -> float:
        return self.spec.mask_radius


def refuse_mesh(leg, likelihood_variant: str) -> int | None:
    """Exit 2 unless the run's mesh is the profiled cell's: the row is filed under the
    likelihood, so a leaf asked for another mesh would file a different model under it."""
    cli = leg.cli
    if likelihood_variant == "mge":
        return None
    mesh = MESH_OF[likelihood_variant]
    if cli.mesh != mesh or cli.mesh_pixels != MESH_PIXELS_STANDARD[mesh]:
        print(
            f"ERROR: this leaf runs the profiled {likelihood_variant} cell's mesh, --mesh {mesh} "
            f"at --mesh-pixels {MESH_PIXELS_STANDARD[mesh]}; got --mesh {cli.mesh} "
            f"--mesh-pixels {cli.mesh_pixels}",
            file=sys.stderr,
        )
        return 2
    return None


def truth_positions(al, tracer, spec):
    """The truth tracer's multiple images of the source centre, solved on the data grid."""
    source = next(g for g in tracer.galaxies if g.redshift == REDSHIFT_SOURCE)
    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=spec.shape_native, pixel_scales=spec.pixel_scale),
        pixel_scale_precision=POSITIONS_SOLVER_PRECISION,
        magnification_threshold=POSITIONS_MAGNIFICATION_THRESHOLD,
    )
    return solver.solve(tracer=tracer, source_plane_coordinate=source.bulge.centre)


def load_galaxy_group(leg, dataset_type: str, likelihood_variant: str, stages: tuple = ()):
    """The profiled cell's masked dataset (``galaxy_group.load_masked``): noise-scaled where the
    spec says (multi_galaxy), masked, light-profile over-sampled on every galaxy centre, then
    this run's inversion path."""
    al = leg.al
    spec = SPECS[dataset_type]
    if leg.instrument != spec.instrument:
        print(
            f"ERROR: {dataset_type} is profiled at {spec.instrument} only; got "
            f"--instrument {leg.instrument}",
            file=sys.stderr,
        )
        return 2
    code = refuse_mesh(leg, likelihood_variant)
    if code is not None:
        return code

    path = leg.dataset_path
    auto_simulate_if_missing(
        path, dataset_type=dataset_type, instrument=spec.instrument, workspace_root=leg.root
    )
    dataset = al.Imaging.from_fits(
        data_path=path / "data.fits",
        psf_path=path / "psf.fits",
        noise_map_path=path / "noise_map.fits",
        pixel_scales=spec.pixel_scale,
    )
    if spec.noise_scaled is not None:
        centre, radius = spec.noise_scaled
        # apply_noise_scaling scales the pixels where the mask is False: inside this circle
        # (the profiled cell's reading, galaxy_group.load_masked).
        dataset = dataset.apply_noise_scaling(
            mask=al.Mask2D.circular(
                shape_native=dataset.shape_native,
                pixel_scales=dataset.pixel_scales,
                centre=centre,
                radius=radius,
            )
        )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=spec.mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size_lp = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=list(spec.lp_sub_size_list),
        radial_list=list(spec.lp_radial_list),
        centre_list=spec.centres,
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size_lp)
    dataset = _apply_inversion(dataset, leg.cli.backend, leg.cli.inversion)

    tracer = al.from_json(file_path=str(path / "tracer.json"))
    meshes = None if likelihood_variant == "mge" else mesh_models(leg.af, al, leg.cli)
    return GalaxyGroupContext(
        spec=spec,
        dataset=dataset,
        over_sample_size_lp=over_sample_size_lp,
        positions=truth_positions(al, tracer, spec),
        truths=truth_dict(tracer),
        settings=al.Settings(
            use_mixed_precision=leg.cli.use_mixed_precision,
            nnls_warm_start_memo=leg.cli.memo == "on",
        ),
        meshes=meshes,
        cli=leg.cli,
        stages={stage.key: stage for stage in stages},
    )


def model_fields(leg, context) -> dict:
    """The row's model fields: the mesh recipe of a pixelized chain (what the exporter's size
    comparison reads), nulls for the MGE chain, which has no mesh."""
    if context.meshes is None:
        return {"mesh": None, "mesh_pixels": None, "mesh_shape": None, "regularization": None}
    return mesh_fields(context.cli, context.meshes)


# ---------------------------------------------------------------------------
# Stage helpers (shared with the multi_galaxy family)
# ---------------------------------------------------------------------------


def nautilus(leg, stage, **extra):
    """``Leg.nautilus`` plus the workspace's extra Nautilus settings (``n_like_max``)."""
    if not extra:
        return leg.nautilus(stage)
    settings_search = leg.search_settings(stage.key)
    return leg.af.Nautilus(
        name=stage.search_name,
        **settings_search.search_dict,
        n_live=stage.n_live,
        n_batch=stage.n_batch,
        seed=leg.seed,
        **extra,
    )


def analysis(leg, context, **kwargs):
    """``AnalysisImaging`` on the context's current dataset, ``use_jax`` always explicit."""
    return leg.al.AnalysisImaging(
        dataset=context.dataset, use_jax=leg.use_jax, settings=context.settings, **kwargs
    )


def adapt_images(leg, context, result):
    """The S/N-capped adapt images of ``result`` (and the Hilbert mesh grid, Delaunay)."""
    return _adapt_images(
        leg.al,
        result,
        meshes=context.meshes,
        dataset=context.dataset,
        mask_radius=context.mask_radius,
    )


def positions_likelihood(leg, context, result, *, factor: float, stage: str, positions=True):
    """``result.positions_likelihood_from`` at the workspace ``factor``, or the fixed test-mode
    threshold on the truth positions (``_runner.positions_likelihood_from``'s reason).
    ``positions`` passes the truth positions to the solver, as ``group/slam.py`` does."""
    al = leg.al
    if leg.test_mode:
        print(
            f"  NOTE [{stage}]: PYAUTO_TEST_MODE — positions likelihood on the truth positions at "
            f"threshold {TEST_MODE_POSITIONS_THRESHOLD}, not solved from a near-random result."
        )
        return al.PositionsLH(positions=context.positions, threshold=TEST_MODE_POSITIONS_THRESHOLD)
    return result.positions_likelihood_from(
        factor=factor,
        minimum_threshold=POSITIONS_MINIMUM_THRESHOLD,
        positions=context.positions if positions else None,
    )


def pixelization(leg, mesh, regularization):
    return leg.af.Model(leg.al.Pixelization, mesh=mesh, regularization=regularization)


def adaptive_pixelization_over_sampling(leg, context, results, key: str = "source_pix_1"):
    """The workspace's adaptive pixelization over-sampling from ``source_pix[2]`` on: sub-size 4
    where ``source_pix[1]``'s source S/N exceeds 3.0, else 2. ``apply_over_sampling`` rebuilds
    the dataset and drops the sparse operator, so the inversion path is re-applied."""
    import numpy as np

    al = leg.al
    source_image = al.galaxy_name_image_dict_via_result_from(result=results[key])[
        "('galaxies', 'source')"
    ]
    over_sample_size_pixelization = al.Array2D(
        values=np.where(source_image > OVER_SAMPLE_SNR_THRESHOLD, 4, 2),
        mask=context.dataset.mask,
    )
    context.dataset = context.dataset.apply_over_sampling(
        over_sample_size_lp=context.over_sample_size_lp,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )
    context.dataset = _apply_inversion(context.dataset, leg.cli.backend, leg.cli.inversion)


def n_main(instance) -> int:
    return sum(1 for key in vars(instance.galaxies) if key.startswith("lens_"))


def n_extra(instance) -> int:
    extras = getattr(instance, "extra_galaxies", None)
    return len(list(extras)) if extras is not None else 0


def luminosity(profile_list, pixel_scale: float) -> float:
    """The workspace luminosity of an MGE: the summed Gaussian fluxes over the pixel area."""
    import numpy as np

    return float(
        np.sum([2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity for g in profile_list])
        / pixel_scale**2
    )


def dpie(leg, centre, sigma_prior):
    """A truncated ``dPIEMassSph`` at ``centre`` with the workspace's fixed constants."""
    mass = leg.af.Model(leg.al.mp.dPIEMassSph)
    mass.centre = tuple(centre)
    mass.sigma = sigma_prior
    for name, value in DPIE_CONSTANTS.items():
        setattr(mass, name, value)
    mass.redshift_object = REDSHIFT_LENS
    mass.redshift_source = REDSHIFT_SOURCE
    return mass


def luminosity_bounded_sigma(af, total_luminosity: float):
    """``group/slam.py``'s bound: the SIS-equivalent dispersion of
    ``min(5 * 0.5 * L**0.6, 5.0)``, 5.0" when the luminosity is not positive."""
    cap = 5 * 0.5 * total_luminosity**0.6 if total_luminosity > 0 else 5.0
    return af.UniformPrior(lower_limit=0.0, upper_limit=287.0 * min(cap, 5.0) ** 0.5)


def mge(al, mask_radius, *, total_gaussians, sigma_min, **kwargs):
    return al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=total_gaussians, sigma_min=sigma_min, **kwargs
    )


def lens_light_galaxies(leg, light_result, lens_names, n_extra_galaxies: int = 0):
    """``(lenses, extras)``: the lens-plane galaxies whose light the last stage carries, per
    ``leg.lens_light_mode`` (``Family.light_fixed``).

    ``linear`` (the chain as the workspace runs it): ``light[1]``'s instance, linear MGEs, so
    every evaluation re-solves the lens-light intensities. ``fixed``: the galaxies of
    ``light[1]``'s maximum-likelihood fit with every linear light profile replaced by an
    ordinary one at its solved intensity
    (``FitImaging.model_obj_linear_light_profiles_to_light_profiles``), so the inversion carries
    no lens-light columns. The fit's galaxies are ``instance.galaxies`` (the lenses, then the
    source) followed by the extra galaxies (``Analysis.galaxies_via_instance_from``), so they are
    read by position, the order and redshifts checked, and no linear profile may remain.
    """
    instance = light_result.instance
    if leg.lens_light_mode != "fixed":
        lenses = [getattr(instance.galaxies, name) for name in lens_names]
        extras = list(instance.extra_galaxies) if n_extra_galaxies else []
        return lenses, extras
    tracer = light_result.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
    galaxies = list(tracer.galaxies)
    n = len(lens_names)
    if len(galaxies) != n + 1 + n_extra_galaxies or galaxies[n].redshift != REDSHIFT_SOURCE:
        raise ValueError(
            f"light[1]'s fixed-light fit has {len(galaxies)} galaxies; expected {n} lenses, the "
            f"source at z={REDSHIFT_SOURCE}, then {n_extra_galaxies} extra galaxies"
        )
    lenses, extras = galaxies[:n], galaxies[n + 1 :]
    if any(g.redshift != REDSHIFT_LENS for g in lenses + extras):
        raise ValueError(
            "light[1]'s fixed-light lens-plane galaxies are not all at the lens redshift"
        )
    leftover = linear_light_profiles_in(leg.al, lenses + extras)
    if leftover:
        raise ValueError(
            f"lens_light_mode='fixed' left {len(leftover)} linear light profile(s) on the lens plane"
        )
    return lenses, extras


# ---------------------------------------------------------------------------
# The pixelized chain (group/slam.py, group/features/pixelization/slam.py)
# ---------------------------------------------------------------------------


def build_source_lp_0(leg, context, results):
    """Light only — every galaxy's MGE, no mass, no source — for clean fixed light models."""
    af, al = leg.af, leg.al
    spec = context.spec
    sigma_min = spec.pixel_scale / 10.0
    lenses = {
        f"lens_{i}": af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=30,
                gaussian_per_basis=2,
                centre_prior_is_uniform=False,
                centre=tuple(centre),
                centre_sigma=0.1,
                sigma_min=sigma_min,
            ),
            disk=None,
            point=None,
        )
        for i, centre in enumerate(spec.main_lens_centres)
    }
    extras = [
        af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=10,
                centre_prior_is_uniform=True,
                centre=tuple(centre),
                ell_comps_prior_is_uniform=True,
                sigma_min=sigma_min,
            ),
        )
        for centre in spec.extra_galaxies_centres
    ]
    model = af.Collection(galaxies=af.Collection(**lenses), extra_galaxies=af.Collection(extras))
    return (
        model,
        analysis(leg, context),
        nautilus(leg, context.stages["source_lp_0"], n_like_max=1000000),
    )


def build_source_lp_1(leg, context, results):
    """Mass and an MGE source on the fixed ``source_lp[0]`` light: Isothermal per main lens, one
    shear, luminosity-bounded truncated dPIEs on the extra galaxies."""
    af, al = leg.af, leg.al
    spec = context.spec
    lp0 = results["source_lp_0"]
    tracer = lp0.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    mains = n_main(lp0.instance)
    lenses = {}
    for i in range(mains):
        lens = getattr(lp0.instance.galaxies, f"lens_{i}")
        mass = af.Model(al.mp.Isothermal)
        mass.centre = lens.bulge.centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=lens.bulge,
            disk=lens.disk,
            point=lens.point,
            mass=mass,
        )
    extras = []
    for i in range(n_extra(lp0.instance)):
        extra = lp0.instance.extra_galaxies[i]
        total = luminosity(tracer.galaxies[mains + i].bulge.profile_list, spec.pixel_scale)
        mass = dpie(leg, extra.bulge.centre, luminosity_bounded_sigma(af, total))
        extras.append(af.Model(al.Galaxy, redshift=REDSHIFT_LENS, bulge=extra.bulge, mass=mass))
    source_bulge = al.model_util.mge_model_from(
        mask_radius=SOURCE_MGE_RADIUS,
        total_gaussians=30,
        centre=lp0.instance.galaxies.lens_0.bulge.centre,
        centre_prior_is_uniform=False,
        centre_sigma=0.6,
    )
    model = af.Collection(
        galaxies=af.Collection(
            **lenses, source=af.Model(al.Galaxy, redshift=REDSHIFT_SOURCE, bulge=source_bulge)
        ),
        fields=af.Model(al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)),
        extra_galaxies=af.Collection(extras),
    )
    built = analysis(
        leg,
        context,
        positions_likelihood_list=[
            al.PositionsLH(positions=context.positions, threshold=SOURCE_LP_POSITIONS_THRESHOLD)
        ],
    )
    return model, built, nautilus(leg, context.stages["source_lp_1"], n_like_max=200000)


def build_source_pix_1(leg, context, results):
    """The pixelized source on the ``source_lp[1]`` mass (centres unfixed); the extra galaxies
    carried forward as free models."""
    af, al = leg.af, leg.al
    lp1 = results["source_lp_1"]
    lenses = {}
    for i in range(n_main(lp1.instance)):
        instance = getattr(lp1.instance.galaxies, f"lens_{i}")
        lens_model = getattr(lp1.model.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=instance.bulge,
            disk=instance.disk,
            point=instance.point,
            mass=al.util.chaining.mass_from(
                mass=lens_model.mass, mass_result=lens_model.mass, unfix_mass_centre=True
            ),
        )
    meshes = context.meshes
    source = af.Model(
        al.Galaxy,
        redshift=lp1.instance.galaxies.source.redshift,
        pixelization=pixelization(leg, meshes.mesh_init, meshes.regularization),
    )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source),
        fields=lp1.model.fields,
        extra_galaxies=lp1.model.extra_galaxies,
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, lp1),
        positions_likelihood_list=[
            positions_likelihood(leg, context, lp1, factor=2.0, stage="source_pix[1]")
        ],
    )
    return model, built, nautilus(leg, context.stages["source_pix_1"])


def build_source_pix_2(leg, context, results):
    """The final pixelized source on ``source_pix[1]``'s adapt images, everything else fixed."""
    af, al = leg.af, leg.al
    lp1, pix1 = results["source_lp_1"], results["source_pix_1"]
    lenses = {}
    for i in range(n_main(pix1.instance)):
        instance = getattr(lp1.instance.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=instance.bulge,
            disk=instance.disk,
            point=instance.point,
            mass=getattr(pix1.instance.galaxies, f"lens_{i}").mass,
        )
    meshes = context.meshes
    source = af.Model(
        al.Galaxy,
        redshift=lp1.instance.galaxies.source.redshift,
        pixelization=pixelization(leg, meshes.mesh, meshes.regularization),
    )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source),
        fields=pix1.instance.fields,
        extra_galaxies=pix1.instance.extra_galaxies,
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, pix1),
        positions_likelihood_list=[
            positions_likelihood(leg, context, pix1, factor=3.0, stage="source_pix[2]")
        ],
    )
    return model, built, nautilus(leg, context.stages["source_pix_2"])


def build_light(leg, context, results):
    """Fresh lens and extra-galaxy MGEs with mass and source fixed from the SOURCE PIX stages."""
    af, al = leg.af, leg.al
    spec = context.spec
    sigma_min = spec.pixel_scale / 10.0
    lp0, pix1, pix2 = results["source_lp_0"], results["source_pix_1"], results["source_pix_2"]
    lenses = {}
    for i in range(n_main(pix1.instance)):
        instance = getattr(pix1.instance.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=30,
                gaussian_per_basis=2,
                centre_prior_is_uniform=True,
                centre=getattr(lp0.instance.galaxies, f"lens_{i}").bulge.centre,
                sigma_min=sigma_min,
            ),
            disk=None,
            point=None,
            mass=instance.mass,
        )
    extras = []
    for i in range(n_extra(pix1.instance)):
        extra = pix1.instance.extra_galaxies[i]
        bulge = mge(
            al,
            spec.mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=True,
            centre=extra.mass.centre,
            sigma_min=sigma_min,
        )
        extras.append(af.Model(al.Galaxy, redshift=REDSHIFT_LENS, bulge=bulge, mass=extra.mass))
    model = af.Collection(
        galaxies=af.Collection(
            **lenses,
            source=al.util.chaining.source_custom_model_from(result=pix2, source_is_model=False),
        ),
        fields=pix1.instance.fields,
        extra_galaxies=af.Collection(extras),
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, pix1),
        positions_likelihood_list=[
            positions_likelihood(leg, context, pix2, factor=3.0, stage="light[1]")
        ],
    )
    return model, built, nautilus(leg, context.stages["light"])


def build_mass_total(leg, context, results):
    """``PowerLaw`` main-lens masses chained from ``source_pix[1]``, fresh luminosity-bounded
    dPIEs on the extra galaxies, light fixed from ``light[1]``, source from ``source_pix[2]``."""
    af, al = leg.af, leg.al
    spec = context.spec
    pix1, pix2, light = results["source_pix_1"], results["source_pix_2"], results["light"]
    names = [f"lens_{i}" for i in range(n_main(light.instance))]
    light_lenses, light_extras = lens_light_galaxies(leg, light, names, n_extra(light.instance))
    lenses = {}
    for i, light_lens in enumerate(light_lenses):
        lens_model = getattr(pix1.model.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            bulge=light_lens.bulge,
            disk=light_lens.disk,
            point=light_lens.point,
            mass=al.util.chaining.mass_from(
                mass=af.Model(al.mp.PowerLaw),
                mass_result=lens_model.mass,
                unfix_mass_centre=True,
            ),
        )
    extras = []
    for i, light_extra in enumerate(light_extras):
        # The prior bound reads the linear instance in both modes, so the two light modes'
        # models differ only in the light.
        extra = light.instance.extra_galaxies[i]
        total = luminosity(extra.bulge.profile_list, spec.pixel_scale)
        mass = dpie(leg, extra.bulge.centre, luminosity_bounded_sigma(af, total))
        extras.append(
            af.Model(al.Galaxy, redshift=REDSHIFT_LENS, bulge=light_extra.bulge, mass=mass)
        )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=al.util.chaining.source_from(result=pix2)),
        fields=pix1.model.fields,
        extra_galaxies=af.Collection(extras),
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, pix1),
        positions_likelihood_list=[
            positions_likelihood(leg, context, light, factor=3.0, stage="mass_total[1]")
        ],
    )
    return model, built, nautilus(leg, context.stages["mass_total"])


def after_source_pix_1(leg, context, results):
    adaptive_pixelization_over_sampling(leg, context, results)


#: ``n_live`` of the pixelized chain: ``group/slam.py``'s formulas at one main lens and two
#: extra galaxies (``source_lp[0]`` 100 + 30 x 3, ``source_lp[1]`` 150 + 30 x 3), and the
#: two stages where the pixelization script differs (150 on ``light[1]`` / ``mass_total[1]``).
N_LIVE = {
    "delaunay": {"light": 300, "mass_total": 200},
    "rectangular": {"light": 150, "mass_total": 150},
}


def pixelized_stages(likelihood_variant: str) -> tuple[Stage, ...]:
    n_live = N_LIVE[likelihood_variant]
    return (
        Stage(
            "source_lp_0", "source_lp[0]", 190, 50, build_source_lp_0, evaluates_likelihood=False
        ),
        Stage(
            "source_lp_1", "source_lp[1]", 240, 50, build_source_lp_1, evaluates_likelihood=False
        ),
        Stage(
            "source_pix_1",
            "source_pix[1]",
            150,
            20,
            build_source_pix_1,
            inversion_applies=True,
            after=after_source_pix_1,
        ),
        Stage("source_pix_2", "source_pix[2]", 75, 20, build_source_pix_2, inversion_applies=True),
        Stage("light", "light[1]", n_live["light"], 20, build_light, inversion_applies=True),
        Stage(
            "mass_total",
            "mass_total[1]",
            n_live["mass_total"],
            20,
            build_mass_total,
            inversion_applies=True,
        ),
    )


# ---------------------------------------------------------------------------
# The MGE chain (group/features/multi_gaussian_expansion/slam.py)
# ---------------------------------------------------------------------------


def build_mge_source_lp(leg, context, results):
    """Lens light, mass, shear, extra galaxies and an MGE source, all at once."""
    af, al = leg.af, leg.al
    spec = context.spec
    sigma_min = spec.pixel_scale / 10.0
    lenses = {
        f"lens_{i}": af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=20,
                gaussian_per_basis=1,
                centre_prior_is_uniform=True,
                sigma_min=sigma_min,
            ),
            mass=af.Model(al.mp.Isothermal),
        )
        for i, _ in enumerate(spec.main_lens_centres)
    }
    extras = [
        af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=10,
                centre_fixed=tuple(centre),
                sigma_min=sigma_min,
            ),
            mass=dpie(leg, centre, af.UniformPrior(lower_limit=0.0, upper_limit=300.0)),
        )
        for centre in spec.extra_galaxies_centres
    ]
    source_bulge = mge(
        al,
        spec.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
        sigma_min=0.0001,
    )
    model = af.Collection(
        galaxies=af.Collection(
            **lenses, source=af.Model(al.Galaxy, redshift=REDSHIFT_SOURCE, bulge=source_bulge)
        ),
        fields=af.Model(al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)),
        extra_galaxies=af.Collection(extras),
    )
    return model, analysis(leg, context), nautilus(leg, context.stages["source_lp"])


def build_mge_light(leg, context, results):
    """Fresh MGEs on every galaxy; mass and source fixed from ``source_lp[1]``."""
    af, al = leg.af, leg.al
    spec = context.spec
    sigma_min = spec.pixel_scale / 10.0
    lp = results["source_lp"]
    lenses = {}
    for i in range(n_main(lp.instance)):
        instance = getattr(lp.instance.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=20,
                gaussian_per_basis=1,
                centre_prior_is_uniform=True,
                sigma_min=sigma_min,
            ),
            mass=instance.mass,
        )
    extras = [
        af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=mge(
                al,
                spec.mask_radius,
                total_gaussians=10,
                centre_fixed=tuple(centre),
                sigma_min=sigma_min,
            ),
            mass=lp.instance.extra_galaxies[i].mass,
        )
        for i, centre in enumerate(spec.extra_galaxies_centres)
    ]
    source = af.Model(
        al.Galaxy,
        redshift=lp.instance.galaxies.source.redshift,
        bulge=lp.instance.galaxies.source.bulge,
    )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source),
        fields=lp.instance.fields,
        extra_galaxies=af.Collection(extras),
    )
    return model, analysis(leg, context), nautilus(leg, context.stages["light"])


def build_mge_mass_total(leg, context, results):
    """``PowerLaw`` main-lens masses chained from ``source_lp[1]``, free dPIEs, the light fixed
    from ``light[1]`` and the source MGE free again."""
    af, al = leg.af, leg.al
    spec = context.spec
    lp, light = results["source_lp"], results["light"]
    names = [f"lens_{i}" for i in range(n_main(light.instance))]
    light_lenses, light_extras = lens_light_galaxies(leg, light, names, n_extra(light.instance))
    lenses = {}
    for i, light_lens in enumerate(light_lenses):
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=light_lens.redshift,
            bulge=light_lens.bulge,
            mass=al.util.chaining.mass_from(
                mass=af.Model(al.mp.PowerLaw),
                mass_result=getattr(lp.model.galaxies, f"lens_{i}").mass,
                unfix_mass_centre=True,
            ),
        )
    extras = [
        af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=light_extras[i].bulge,
            mass=dpie(leg, centre, af.UniformPrior(lower_limit=0.0, upper_limit=300.0)),
        )
        for i, centre in enumerate(spec.extra_galaxies_centres)
    ]
    source = af.Model(
        al.Galaxy,
        redshift=lp.instance.galaxies.source.redshift,
        bulge=lp.model.galaxies.source.bulge,
    )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source),
        fields=lp.model.fields,
        extra_galaxies=af.Collection(extras),
    )
    return model, analysis(leg, context), nautilus(leg, context.stages["mass_total"])


MGE_STAGES = (
    Stage("source_lp", "source_lp[1]", 200, 50, build_mge_source_lp),
    Stage("light", "light[1]", 150, 20, build_mge_light),
    Stage("mass_total", "mass_total[1]", 150, 20, build_mge_mass_total),
)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------


def _family(likelihood_variant: str) -> Family:
    if likelihood_variant == "mge":
        stages, inversions = MGE_STAGES, ("dense",)
    else:
        stages, inversions = pixelized_stages(likelihood_variant), ("dense", "sparse")
    return Family(
        dataset_type="group",
        likelihood_variant=likelihood_variant,
        stages=stages,
        load=lambda leg: load_galaxy_group(leg, "group", likelihood_variant, stages),
        inversions=inversions,
        model_fields=model_fields,
        title="Group SLaM run",
        light_fixed=True,
    )


FAMILY_RECTANGULAR = _family("rectangular")
FAMILY_DELAUNAY = _family("delaunay")
FAMILY_MGE = _family("mge")
