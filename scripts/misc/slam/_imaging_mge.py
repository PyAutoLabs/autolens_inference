"""The imaging MGE family: the three-stage workspace MGE SLaM chain, on the shared core.

Likelihood-lifecycle 4c-1 (autolens_inference#54). The likelihood is ``imaging/mge`` as the
profiling catalogue names it: a linear multi-Gaussian expansion of the lens light (20
Gaussians, one basis) and of the source (20, one basis), solved by one dense linear
inversion per evaluation, with no pixelization. The chain mirrors
``autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/slam.py``:
``source_lp[1]`` (Nautilus 200 / 50) -> ``light[1]`` (150 / 20) -> ``mass_total[1]``
(150 / 20), ``Isothermal + ExternalShear`` chained into a ``PowerLaw`` with the mass centre
unfixed. Every stage solves the same lens + source MGE, so every stage evaluates
``imaging/mge`` and pairs with its profiling record.

The profiled cell (autolens_profiling ``scripts/misc/cell_builders/imaging_mge.py``) is the
same 20 + 20 MGE at the same over-sampling; it fixes the lens mass to the simulator truth
only so its gradient harness can trace the mass, which leaves the per-evaluation solve
unchanged. Under the human ruling of 2026-11-02 ("Workspace pipeline + profiled preset") the
chain keeps the workspace's free mass.

Deliberate departures from the workspace script, each the imaging chain's own
(``slam._runner``): ``use_jax=leg.use_jax`` on every ``AnalysisImaging`` (the workspace
hard-codes ``True``), the instrument preset's mask radius (hst 3.5", the profiled cell's,
where the workspace uses 3.0") and ``al.Settings`` carrying ``--use-mixed-precision`` and
``--memo``. ``--lens-light fixed|both`` refits ``mass_total[1]`` with the lens light
``light[1]`` solved for held fixed (``Family.light_fixed``), the imaging chain's
``light_fixed`` model variant.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _inference_cli import auto_simulate_if_missing  # noqa: E402
from slam._core import Family, Stage, truth_dict  # noqa: E402
from slam._runner import (  # noqa: E402
    OVER_SAMPLE_RADIAL_LIST,
    OVER_SAMPLE_SUB_SIZE_LIST,
    REDSHIFT_LENS,
    REDSHIFT_SOURCE,
    fixed_lens_light_from,
)

#: The MGE of the profiled cell and of the workspace chain: 20 Gaussians in one basis, for
#: the lens light and for the source alike.
MGE_TOTAL_GAUSSIANS = 20
MGE_GAUSSIAN_PER_BASIS = 1


def load(leg):
    """The masked, over-sampled HST imaging dataset, its truths and the analysis settings."""
    al = leg.al
    from instruments.imaging import INSTRUMENTS

    if leg.instrument not in INSTRUMENTS:
        print(
            f"ERROR: unknown --instrument {leg.instrument!r}; choose from {sorted(INSTRUMENTS)}",
            file=sys.stderr,
        )
        return 2
    preset = INSTRUMENTS[leg.instrument]
    mask_radius = preset["mask_radius"]
    auto_simulate_if_missing(
        leg.dataset_path,
        dataset_type=leg.dataset_class,
        instrument=leg.instrument,
        workspace_root=leg.root,
    )
    dataset = al.Imaging.from_fits(
        data_path=leg.dataset_path / "data.fits",
        noise_map_path=leg.dataset_path / "noise_map.fits",
        psf_path=leg.dataset_path / "psf.fits",
        pixel_scales=preset["pixel_scale"],
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=OVER_SAMPLE_SUB_SIZE_LIST,
        radial_list=OVER_SAMPLE_RADIAL_LIST,
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
    tracer = al.from_json(file_path=str(leg.dataset_path / "tracer.json"))
    return SimpleNamespace(
        dataset=dataset,
        mask_radius=mask_radius,
        truths=truth_dict(tracer),
        settings=al.Settings(
            use_mixed_precision=leg.cli.use_mixed_precision,
            nnls_warm_start_memo=leg.cli.memo == "on",
        ),
    )


def _analysis(leg, context):
    return leg.al.AnalysisImaging(
        dataset=context.dataset, use_jax=leg.use_jax, settings=context.settings
    )


def _lens_bulge(al, context):
    return al.model_util.mge_model_from(
        mask_radius=context.mask_radius,
        total_gaussians=MGE_TOTAL_GAUSSIANS,
        gaussian_per_basis=MGE_GAUSSIAN_PER_BASIS,
        centre_prior_is_uniform=True,
        sigma_min=context.dataset.pixel_scales[0] / 10.0,
    )


def build_source_lp(leg, context, results):
    """The lens mass, the MGE lens light and the MGE source, fitted together."""
    af, al = leg.af, leg.al
    source_bulge = al.model_util.mge_model_from(
        mask_radius=context.mask_radius,
        total_gaussians=MGE_TOTAL_GAUSSIANS,
        gaussian_per_basis=MGE_GAUSSIAN_PER_BASIS,
        centre_prior_is_uniform=False,
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=REDSHIFT_LENS,
                bulge=_lens_bulge(al, context),
                disk=None,
                mass=af.Model(al.mp.Isothermal),
            ),
            source=af.Model(al.Galaxy, redshift=REDSHIFT_SOURCE, bulge=source_bulge, disk=None),
        ),
        fields=af.Model(al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)),
    )
    return model, _analysis(leg, context), leg.nautilus(STAGES[0])


def build_light(leg, context, results):
    """A fresh MGE lens light, with the mass and the source of ``source_lp[1]`` fixed."""
    af, al = leg.af, leg.al
    source_lp_result = results["source_lp"]
    source = al.util.chaining.source_custom_model_from(
        result=source_lp_result, source_is_model=False
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=_lens_bulge(al, context),
                disk=None,
                mass=source_lp_result.instance.galaxies.lens.mass,
            ),
            source=source,
        ),
        fields=source_lp_result.instance.fields,
    )
    return model, _analysis(leg, context), leg.nautilus(STAGES[1])


def build_mass_total(leg, context, results):
    """The headline stage: a ``PowerLaw`` chained from ``source_lp[1]``'s ``Isothermal``.

    ``leg.lens_light_mode`` decides where the lens light comes from: ``light[1]``'s linear
    MGE (``linear``, the workspace chain), or the same profiles carrying the intensities its
    maximum-likelihood fit solved (``fixed``), as the imaging chain's ``mass_total[1]``.
    """
    af, al = leg.af, leg.al
    source_lp_result, light_result = results["source_lp"], results["light"]
    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    lens_redshift = source_lp_result.instance.galaxies.lens.redshift
    if leg.lens_light_mode == "fixed":
        lens_light = fixed_lens_light_from(al, light_result, lens_redshift=lens_redshift)
    else:
        lens_light = light_result.instance.galaxies.lens
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=lens_redshift,
                bulge=lens_light.bulge,
                disk=lens_light.disk,
                mass=mass,
            ),
            source=al.util.chaining.source_from(result=source_lp_result),
        ),
        fields=source_lp_result.model.fields,
    )
    return model, _analysis(leg, context), leg.nautilus(STAGES[2])


STAGES = (
    Stage("source_lp", "source_lp[1]", 200, 50, build_source_lp),
    Stage("light", "light[1]", 150, 20, build_light),
    Stage("mass_total", "mass_total[1]", 150, 20, build_mass_total),
)


def model_fields(leg, context) -> dict:
    """No mesh: the row says so (``null``), and records the MGE it fits instead."""
    return {
        "mesh": None,
        "mesh_pixels": None,
        "mesh_shape": None,
        "mge_total_gaussians": MGE_TOTAL_GAUSSIANS,
        "mge_gaussian_per_basis": MGE_GAUSSIAN_PER_BASIS,
    }


FAMILY = Family(
    dataset_type="imaging",
    likelihood_variant="mge",
    stages=STAGES,
    load=load,
    inversions=("dense",),
    model_fields=model_fields,
    title="Imaging MGE SLaM run",
    light_fixed=True,
)
