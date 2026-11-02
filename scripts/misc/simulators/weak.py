"""
Simulator: Weak-Lensing Shear Catalogues
========================================

Simulates the weak-lensing shear catalogue of one ``INSTRUMENTS`` preset into
``dataset/weak/<instrument>/`` (``dataset.json`` and ``tracer.json``, via ``al.output_to_json``).
Likelihood-lifecycle phase 2g (autolens_profiling#476), ported unchanged from
autolens_profiling's ``scripts/misc/simulators/weak.py``: the catalogue is byte-for-byte the one
the profiling weak likelihood cells fit.

The ``simple`` preset is the catalogue of the autolens_workspace ``scripts/weak/simulator.py``
and the autolens_visualization weak producer (``true_models/weak.py``): an elliptical isothermal
cluster-scale lens (Einstein radius 25", axis ratio 0.8 at 45 deg, z = 0.5), 1500 background
galaxies drawn uniformly in area in the 50"-200" annulus (``numpy.random.default_rng(1)``), plain
shear (not reduced), no per-galaxy redshifts, iid Gaussian shape noise sigma = 0.25 per component
(``SimulatorShearYX(noise_sigma=0.25, seed=1)``). 1500 galaxies over the annulus's 32.7 arcmin^2
is ~46 galaxies per arcmin^2, a deep ground / Euclid-like source density.

``INSTRUMENTS`` is the single source of truth for the weak runners; they import it, and
:func:`standard_model` / :func:`truth_vector` build the one model every profiling weak setup
evaluated.

Unlike the other simulators it writes no simulator-profiling summary: simulating 1500 shears is
milliseconds, and the dataset is what the cells need.

Usage
-----

    python scripts/misc/simulators/weak.py --instrument simple [--output-root <repo>]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

#: The weak catalogue presets. ``n_galaxies`` is the standard size of the weak cells (decision D3
#: on the 2g campaign page).
INSTRUMENTS = {
    "simple": {
        "lens_redshift": 0.5,
        "source_redshift": 1.0,
        "lens_centre": (0.0, 0.0),
        "lens_einstein_radius": 25.0,
        "lens_axis_ratio": 0.8,
        "lens_angle": 45.0,
        "n_galaxies": 1500,
        "radius_inner": 50.0,
        "radius_outer": 200.0,
        "noise_sigma": 0.25,
        "seed": 1,
    },
}

#: The likelihood variant the presets are fitted with (``results/weak/<variant>/``): the lens mass
#: profile, as an imaging variant names its source model (decision D2).
VARIANT = "isothermal"


def _inference_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_inference root (ruff.toml) not found")


def dataset_path(instrument: str, root: Path | None = None) -> Path:
    return (root or _inference_root()) / "dataset" / "weak" / instrument


def true_tracer(instrument: str):
    import autolens as al

    config = INSTRUMENTS[instrument]
    lens = al.Galaxy(
        redshift=config["lens_redshift"],
        mass=al.mp.Isothermal(
            centre=config["lens_centre"],
            einstein_radius=config["lens_einstein_radius"],
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=config["lens_axis_ratio"], angle=config["lens_angle"]
            ),
        ),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=config["source_redshift"])])


def positions(instrument: str):
    """The catalogue positions: uniform in area in the annulus, seeded."""
    import autolens as al
    import numpy as np

    config = INSTRUMENTS[instrument]
    rng = np.random.default_rng(config["seed"])
    n = config["n_galaxies"]
    radii = np.sqrt(rng.uniform(config["radius_inner"] ** 2, config["radius_outer"] ** 2, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return al.Grid2DIrregular(values=np.stack([radii * np.sin(phi), radii * np.cos(phi)], axis=1))


def truth_vector(instrument: str) -> list[float]:
    """The standard instance's parameter vector, in the order of :func:`standard_model`'s paths
    (centre_0, centre_1, ell_comps_0, ell_comps_1, einstein_radius)."""
    import autolens as al

    config = INSTRUMENTS[instrument]
    ell = al.convert.ell_comps_from(
        axis_ratio=config["lens_axis_ratio"], angle=config["lens_angle"]
    )
    return [
        float(config["lens_centre"][0]),
        float(config["lens_centre"][1]),
        float(ell[0]),
        float(ell[1]),
        float(config["lens_einstein_radius"]),
    ]


def standard_model(instrument: str):
    """The weak model: a free elliptical isothermal lens and an empty source galaxy.

    GaussianPrior(mean=truth, sigma=small) keeps the five parameters free (so a vmap batch and a
    gradient have dimensionality) while the prior medians equal the simulator truth.
    """
    import autofit as af
    import autolens as al

    config = INSTRUMENTS[instrument]
    truth = truth_vector(instrument)
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=truth[0], sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=truth[1], sigma=0.1)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=truth[2], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=truth[3], sigma=0.01)
    mass.einstein_radius = af.GaussianPrior(mean=truth[4], sigma=0.5)
    lens = af.Model(al.Galaxy, redshift=config["lens_redshift"], mass=mass)
    source = af.Model(al.Galaxy, redshift=config["source_redshift"])
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def simulate(instrument: str = "simple", output_root: Path | None = None) -> Path:
    """Simulate the preset's catalogue into ``dataset/weak/<instrument>/``; return the folder."""
    import autolens as al

    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{instrument}'. Choose from: {list(INSTRUMENTS)}")
    config = INSTRUMENTS[instrument]
    tracer = true_tracer(instrument)
    simulator = al.SimulatorShearYX(noise_sigma=config["noise_sigma"], seed=config["seed"])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=positions(instrument), name=instrument)
    out = dataset_path(instrument, output_root)
    out.mkdir(parents=True, exist_ok=True)
    al.output_to_json(obj=dataset, file_path=out / "dataset.json")
    al.output_to_json(obj=tracer, file_path=out / "tracer.json")
    print(f"  [weak simulator] {dataset.n_galaxies} galaxies -> {out}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--instrument", default="simple", choices=sorted(INSTRUMENTS))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)

    from autonerves import jax_wrapper  # noqa: F401

    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
        return 0

    simulate(args.instrument, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
