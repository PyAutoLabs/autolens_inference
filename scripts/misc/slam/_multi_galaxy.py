"""The multi_galaxy family: the HST two-deflector SLaM chains of ``multi_galaxy/{rectangular,
delaunay,mge}``.

Likelihood-lifecycle 4c-2 (autolens_inference#54). One dataset: the profiled multi_galaxy cell
(``simulators/_galaxy_group.SPECS["multi_galaxy"]``) — two co-dominant lenses, an external shear,
a light-only contaminant whose pixels are noise-scaled, a SersicCore source, at hst (0.05"/px,
3.0" mask, light-profile over-sampling on both lens centres). It is loaded exactly as the group
family loads its dataset (``_group.load_galaxy_group``), and the stage helpers are that module's.

The workspace has one multi-galaxy pipeline, ``multi_galaxy/slam.py`` (five stages, rectangular
only): ``source_lp[1]`` -> ``source_pix[1]`` -> ``source_pix[2]`` -> ``light[1]`` ->
``mass_total[1]``, every deflector an MGE of 20 x 2 and an ``Isothermal`` whose centre is fixed to
its light centre in ``source_lp[1]`` and freed from ``source_pix[1]`` on. The three likelihoods
(human ruling 2026-11-02, "Workspace pipeline + profiled preset"; the nearest analogue where no
workspace pipeline exists):

- ``multi_galaxy/rectangular`` — that pipeline, its ``RectangularBilinearAdaptDensity`` /
  ``AdaptImage`` mesh under ``Adapt`` at the profiled 35 x 35 instead of 28 x 28.
- ``multi_galaxy/delaunay`` — no workspace variant: the same five stages with the profiled
  Delaunay (1250 Hilbert vertices + 30 zeroed edge points, ``AdaptSplit``), drawn as the group
  Delaunay pipeline draws it.
- ``multi_galaxy/mge`` — no workspace variant: the nearest analogue is the group MGE pipeline's
  shape (``group/features/multi_gaussian_expansion/slam.py``) on this pipeline's models —
  ``source_lp[1]`` (as above) -> ``light[1]`` (fresh MGEs, mass and source fixed) ->
  ``mass_total[1]`` (``PowerLaw`` masses, the source MGE free again), at this pipeline's
  deflector-scaled ``n_live``.

Every family is ``Family.light_fixed``: ``mass_total[1]`` is fitted once per lens-light mode,
the ``fixed`` one on ``light[1]``'s solved deflector light (``_group.lens_light_galaxies``).

Stages evaluating the profiled likelihood: the four mesh stages of a pixelized chain
(``source_lp[1]`` declares ``null``), and all three stages of the MGE chain.

Departures from ``multi_galaxy/slam.py``, besides the mesh: a positions likelihood on all four
pixelized stages (as the imaging chain; the workspace has ``source_pix[1]`` and
``mass_total[1]``), solved from the previous result at factor 3.0 and, under
``PYAUTO_TEST_MODE``, the fixed test-mode threshold on the truth positions; the noise scaling of
the profiled cell (a 0.9" circle on the contaminant, not the workspace's ``mask_extra_galaxies``
file); ``use_jax`` on every ``Analysis``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import Family, Stage  # noqa: E402
from slam._group import (  # noqa: E402
    REDSHIFT_LENS,
    REDSHIFT_SOURCE,
    adapt_images,
    adaptive_pixelization_over_sampling,
    analysis,
    lens_light_galaxies,
    load_galaxy_group,
    mge,
    model_fields,
    n_main,
    nautilus,
    pixelization,
    positions_likelihood,
)

#: ``source_lp[1]``'s Einstein-radius ceiling (``multi_galaxy/slam.py``).
UPPER_EINSTEIN_RADIUS = 3.0

#: The two deflectors of the profiled dataset, which every ``n_live`` below scales with.
N_DEFLECTORS = 2


def _lens_bulge(al, spec, centre, *, centre_sigma=None):
    kwargs = dict(centre_sigma=centre_sigma) if centre_sigma is not None else {}
    return mge(
        al,
        spec.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=tuple(centre),
        sigma_min=spec.pixel_scale / 10.0,
        **kwargs,
    )


def build_source_lp(leg, context, results):
    """Every deflector's MGE light and ``Isothermal`` mass (centre fixed to its light centre),
    one shear, an MGE source."""
    af, al = leg.af, leg.al
    spec = context.spec
    lenses = {}
    for i, centre in enumerate(spec.main_lens_centres):
        mass = af.Model(al.mp.Isothermal)
        mass.centre = tuple(centre)
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=UPPER_EINSTEIN_RADIUS)
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=REDSHIFT_LENS,
            bulge=_lens_bulge(al, spec, centre, centre_sigma=0.1),
            disk=None,
            point=None,
            mass=mass,
        )
    source_bulge = al.model_util.mge_model_from(
        mask_radius=spec.mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )
    model = af.Collection(
        galaxies=af.Collection(
            **lenses, source=af.Model(al.Galaxy, redshift=REDSHIFT_SOURCE, bulge=source_bulge)
        ),
        fields=af.Model(al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)),
    )
    return model, analysis(leg, context), nautilus(leg, context.stages["source_lp"])


def build_source_pix_1(leg, context, results):
    """The pixelized source on the ``source_lp[1]`` light, each ``Isothermal`` chained with its
    centre freed."""
    af, al = leg.af, leg.al
    lp = results["source_lp"]
    lenses = {}
    for i in range(n_main(lp.instance)):
        instance = getattr(lp.instance.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=instance.bulge,
            disk=instance.disk,
            point=instance.point,
            mass=al.util.chaining.mass_from(
                mass=af.Model(al.mp.Isothermal),
                mass_result=getattr(lp.model.galaxies, f"lens_{i}").mass,
                unfix_mass_centre=True,
            ),
        )
    meshes = context.meshes
    source = af.Model(
        al.Galaxy,
        redshift=lp.instance.galaxies.source.redshift,
        pixelization=pixelization(leg, meshes.mesh_init, meshes.regularization),
    )
    model = af.Collection(galaxies=af.Collection(**lenses, source=source), fields=lp.model.fields)
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, lp),
        positions_likelihood_list=[
            positions_likelihood(
                leg, context, lp, factor=3.0, stage="source_pix[1]", positions=False
            )
        ],
    )
    return model, built, nautilus(leg, context.stages["source_pix_1"])


def build_source_pix_2(leg, context, results):
    """The final pixelized source on ``source_pix[1]``'s adapt images, every deflector fixed."""
    af, al = leg.af, leg.al
    lp, pix1 = results["source_lp"], results["source_pix_1"]
    lenses = {}
    for i in range(n_main(pix1.instance)):
        instance = getattr(lp.instance.galaxies, f"lens_{i}")
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
        redshift=lp.instance.galaxies.source.redshift,
        pixelization=pixelization(leg, meshes.mesh, meshes.regularization),
    )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source), fields=pix1.instance.fields
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, pix1),
        positions_likelihood_list=[
            positions_likelihood(
                leg, context, pix1, factor=3.0, stage="source_pix[2]", positions=False
            )
        ],
    )
    return model, built, nautilus(leg, context.stages["source_pix_2"])


def build_light(leg, context, results):
    """A fresh MGE per deflector, centred on its ``source_pix[1]`` mass, mass and source fixed."""
    af, al = leg.af, leg.al
    pix1, pix2 = results["source_pix_1"], results["source_pix_2"]
    lenses = {}
    for i in range(n_main(pix1.instance)):
        instance = getattr(pix1.instance.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=_lens_bulge(al, context.spec, instance.mass.centre),
            disk=None,
            point=None,
            mass=instance.mass,
        )
    source = al.util.chaining.source_custom_model_from(result=pix2, source_is_model=False)
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source), fields=pix1.instance.fields
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, pix1),
        positions_likelihood_list=[
            positions_likelihood(leg, context, pix2, factor=3.0, stage="light[1]", positions=False)
        ],
    )
    return model, built, nautilus(leg, context.stages["light"])


def build_mass_total(leg, context, results):
    """Every deflector a ``PowerLaw`` chained from ``source_pix[1]``, the light fixed from
    ``light[1]``, the source from ``source_pix[2]``."""
    af, al = leg.af, leg.al
    pix1, pix2, light = results["source_pix_1"], results["source_pix_2"], results["light"]
    names = [f"lens_{i}" for i in range(n_main(pix1.instance))]
    lenses = {}
    for i, light_lens in enumerate(lens_light_galaxies(leg, light, names)[0]):
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
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=al.util.chaining.source_from(result=pix2)),
        fields=pix1.model.fields,
    )
    built = analysis(
        leg,
        context,
        adapt_images=adapt_images(leg, context, pix1),
        positions_likelihood_list=[
            positions_likelihood(
                leg, context, pix2, factor=3.0, stage="mass_total[1]", positions=False
            )
        ],
    )
    return model, built, nautilus(leg, context.stages["mass_total"])


def after_source_pix_1(leg, context, results):
    adaptive_pixelization_over_sampling(leg, context, results)


# ---------------------------------------------------------------------------
# The MGE analogue
# ---------------------------------------------------------------------------


def build_mge_light(leg, context, results):
    """A fresh MGE per deflector, centred on its mass; mass and the MGE source fixed."""
    af, al = leg.af, leg.al
    lp = results["source_lp"]
    lenses = {}
    for i in range(n_main(lp.instance)):
        instance = getattr(lp.instance.galaxies, f"lens_{i}")
        lenses[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=instance.redshift,
            bulge=_lens_bulge(al, context.spec, instance.mass.centre),
            disk=None,
            point=None,
            mass=instance.mass,
        )
    source = af.Model(
        al.Galaxy,
        redshift=lp.instance.galaxies.source.redshift,
        bulge=lp.instance.galaxies.source.bulge,
    )
    model = af.Collection(
        galaxies=af.Collection(**lenses, source=source), fields=lp.instance.fields
    )
    return model, analysis(leg, context), nautilus(leg, context.stages["light"])


def build_mge_mass_total(leg, context, results):
    """Every deflector a ``PowerLaw`` chained from ``source_lp[1]`` (centre freed), the light
    fixed from ``light[1]``, the source MGE free again."""
    af, al = leg.af, leg.al
    lp, light = results["source_lp"], results["light"]
    names = [f"lens_{i}" for i in range(n_main(light.instance))]
    lenses = {}
    for i, light_lens in enumerate(lens_light_galaxies(leg, light, names)[0]):
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
    source = af.Model(
        al.Galaxy,
        redshift=lp.instance.galaxies.source.redshift,
        bulge=lp.model.galaxies.source.bulge,
    )
    model = af.Collection(galaxies=af.Collection(**lenses, source=source), fields=lp.model.fields)
    return model, analysis(leg, context), nautilus(leg, context.stages["mass_total"])


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------

#: ``n_live`` per stage: ``multi_galaxy/slam.py``'s deflector-scaled formulas at two deflectors
#: (``source_lp[1]`` 150 + 50 n, ``source_pix[1]`` 150 + 50 (n - 1), ``source_pix[2]`` 75,
#: ``light[1]`` and ``mass_total[1]`` 150 + 100 (n - 1)).
N_LIVE = {
    "source_lp": 150 + 50 * N_DEFLECTORS,
    "source_pix_1": 150 + 50 * (N_DEFLECTORS - 1),
    "source_pix_2": 75,
    "light": 150 + 100 * (N_DEFLECTORS - 1),
    "mass_total": 150 + 100 * (N_DEFLECTORS - 1),
}

PIXELIZED_STAGES = (
    Stage(
        "source_lp",
        "source_lp[1]",
        N_LIVE["source_lp"],
        50,
        build_source_lp,
        evaluates_likelihood=False,
    ),
    Stage(
        "source_pix_1",
        "source_pix[1]",
        N_LIVE["source_pix_1"],
        20,
        build_source_pix_1,
        inversion_applies=True,
        after=after_source_pix_1,
    ),
    Stage(
        "source_pix_2",
        "source_pix[2]",
        N_LIVE["source_pix_2"],
        20,
        build_source_pix_2,
        inversion_applies=True,
    ),
    Stage("light", "light[1]", N_LIVE["light"], 20, build_light, inversion_applies=True),
    Stage(
        "mass_total",
        "mass_total[1]",
        N_LIVE["mass_total"],
        20,
        build_mass_total,
        inversion_applies=True,
    ),
)

MGE_STAGES = (
    Stage("source_lp", "source_lp[1]", N_LIVE["source_lp"], 50, build_source_lp),
    Stage("light", "light[1]", N_LIVE["light"], 20, build_mge_light),
    Stage("mass_total", "mass_total[1]", N_LIVE["mass_total"], 20, build_mge_mass_total),
)


def _family(likelihood_variant: str) -> Family:
    if likelihood_variant == "mge":
        stages, inversions = MGE_STAGES, ("dense",)
    else:
        stages, inversions = PIXELIZED_STAGES, ("dense", "sparse")
    return Family(
        dataset_type="multi_galaxy",
        likelihood_variant=likelihood_variant,
        stages=stages,
        load=lambda leg: load_galaxy_group(leg, "multi_galaxy", likelihood_variant, stages),
        inversions=inversions,
        model_fields=model_fields,
        title="Multi-galaxy SLaM run",
        light_fixed=True,
    )


FAMILY_RECTANGULAR = _family("rectangular")
FAMILY_DELAUNAY = _family("delaunay")
FAMILY_MGE = _family("mge")
