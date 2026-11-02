"""The multi_galaxy and group datasets — the dataset half of autolens_profiling's cell builder.

Vendored from ``autolens_profiling/scripts/misc/cell_builders/galaxy_group.py`` (likelihood-
lifecycle 2d, autolens_profiling#473): the dataset specs, the true galaxies and ``simulate``,
unchanged, so ``dataset/{multi_galaxy,group}/<instrument>/`` here is byte-for-byte the data the
profiling multi_galaxy / group likelihood cells fit. The model half (cell model, priors, iid
draws, pixelized presets) is not here; the facts a runner needs to mask and over-sample the data
the way those cells did (``mask_radius``, ``noise_scaled``, the over-sampling bins, the galaxy
centres) are fields of :data:`SPECS`. Nothing runs on import (stdlib only at module level).

Datasets mirror the workspace simulators and the autolens_visualization producers
(``autolens_workspace/scripts/{multi_galaxy,group}/simulator.py``,
``autolens_visualization/true_models/{multi_galaxy,group}.py``): the same galaxies, grid,
exposure, sky and over-sampling. The PSF is the instrument preset's
(``instruments/imaging.py``, 21 x 21), so the cells compare with the imaging cells of the same
instrument (decision D2 on the profiling campaign page).

- ``multi_galaxy`` at ``hst`` (0.05"/px, 200 x 200, 3.0" mask): two co-dominant lenses
  (Sersic light + Isothermal mass), an external shear, a light-only extra galaxy whose pixels are
  noise-scaled (``mask_extra_galaxies``) as ``multi_galaxy/modeling.py`` does, and a SersicCore
  source.
- ``group`` at ``euclid`` (0.1"/px, 250 x 250, 7.5" mask): one main lens (SersicSph +
  IsothermalSph), two extra galaxies (SersicSph + dPIEMassSph), a SersicCore source.

The two CLIs are ``group.py`` and ``multi_galaxy.py`` beside this module (the script name is the
dataset type, as ``_inference_cli.auto_simulate_if_missing`` requires).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

#: The dataset types this module builds.
DATASET_TYPES = ("multi_galaxy", "group")


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset type's simulation and masking (the workspace simulator, instrument PSF)."""

    dataset_type: str
    instrument: str
    pixel_scale: float
    shape_native: tuple[int, int]
    mask_radius: float
    psf_shape: tuple[int, int]
    psf_sigma: float
    exposure_time: float
    background_sky_level: float
    noise_seed: int
    main_lens_centres: tuple
    extra_galaxies_centres: tuple = ()
    #: (centre, radius) of the noise-scaled extra-galaxy region (multi_galaxy only).
    noise_scaled: tuple | None = None
    provenance: str = ""
    lp_sub_size_list: tuple = (4, 2, 2)
    lp_radial_list: tuple = (0.3, 0.6)
    simulate_sub_size_list: tuple = (32, 8, 2)
    notes: list = field(default_factory=list)

    @property
    def centres(self) -> list[tuple[float, float]]:
        """Every galaxy centre the light-profile over-sampling is centred on (workspace rule)."""
        return [tuple(c) for c in (*self.main_lens_centres, *self.extra_galaxies_centres)]

    @property
    def masked_pixels(self) -> int:
        """Image pixels inside the circular mask: pi r^2 / pixel_scale^2."""
        return int(math.pi * self.mask_radius**2 / self.pixel_scale**2)


SPECS = {
    "multi_galaxy": DatasetSpec(
        dataset_type="multi_galaxy",
        instrument="hst",
        pixel_scale=0.05,
        shape_native=(200, 200),
        mask_radius=3.0,
        psf_shape=(21, 21),
        psf_sigma=0.05,
        exposure_time=900.0,
        background_sky_level=0.1,
        noise_seed=1,
        main_lens_centres=((0.35, 0.25), (-0.35, -0.25)),
        noise_scaled=((2.2, 1.6), 0.9),
        provenance=(
            'autolens_workspace scripts/multi_galaxy/simulator.py (grid 200x200 at 0.05", '
            'galaxies, exposure 900 s, sky 0.1, mask_extra_galaxies 3 x 0.3") and '
            'modeling.py (3.0" mask, noise scaling, radial-bin over-sampling on both lenses); '
            "autolens_visualization true_models/multi_galaxy.py; PSF from instruments/imaging.py "
            'hst (21x21, sigma 0.05")'
        ),
    ),
    "group": DatasetSpec(
        dataset_type="group",
        instrument="euclid",
        pixel_scale=0.1,
        shape_native=(250, 250),
        mask_radius=7.5,
        psf_shape=(21, 21),
        psf_sigma=0.1,
        exposure_time=300.0,
        background_sky_level=0.1,
        noise_seed=1,
        main_lens_centres=((0.0, 0.0),),
        extra_galaxies_centres=((3.5, 2.5), (-4.4, -5.0)),
        provenance=(
            'autolens_workspace scripts/group/simulator.py (grid 250x250 at 0.1", galaxies, '
            'exposure 300 s, sky 0.1) and modeling.py (7.5" mask, radial-bin over-sampling on '
            "every galaxy); autolens_visualization true_models/group.py; PSF from "
            'instruments/imaging.py euclid (21x21, sigma 0.1")'
        ),
    ),
}

#: The instrument each dataset type is measured on (decision D2).
INSTRUMENTS = {dt: spec.instrument for dt, spec in SPECS.items()}


def spec_for(dataset_type: str, instrument: str | None = None) -> DatasetSpec:
    if dataset_type not in SPECS:
        raise KeyError(f"no galaxy/group cell for dataset type {dataset_type!r}")
    spec = SPECS[dataset_type]
    if instrument not in (None, spec.instrument):
        raise SystemExit(
            f"{dataset_type} is planned at {spec.instrument} only (decision D2); got {instrument!r}"
        )
    return spec


def dataset_path(dataset_type: str, root: Path) -> Path:
    spec = SPECS[dataset_type]
    return Path(root) / "dataset" / dataset_type / spec.instrument


# --------------------------------------------------------------------------- truth galaxies


def truth_galaxies(dataset_type: str):
    """``(galaxies, fields)`` of the simulator's true model (the visualization true models)."""
    import autolens as al

    if dataset_type == "multi_galaxy":
        lens_0 = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(0.35, 0.25),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
                intensity=1.2,
                effective_radius=0.6,
                sersic_index=4.0,
            ),
            mass=al.mp.Isothermal(
                centre=(0.30, 0.28),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
                einstein_radius=1.0,
            ),
        )
        lens_1 = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(-0.35, -0.25),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
                intensity=1.0,
                effective_radius=0.5,
                sersic_index=4.0,
            ),
            mass=al.mp.Isothermal(
                centre=(-0.31, -0.22),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
                einstein_radius=0.8,
            ),
        )
        extra = al.Galaxy(
            redshift=0.5,
            light=al.lp.ExponentialSph(centre=(2.2, 1.6), intensity=1.0, effective_radius=0.3),
        )
        source = al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=(0.0, 0.03),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=3.0,
                effective_radius=0.15,
                sersic_index=1.0,
            ),
        )
        field = al.MassField(redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05))
        return [lens_0, lens_1, extra, source], [field]

    if dataset_type == "group":
        lens_0 = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.SersicSph(
                centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
            ),
            mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
        )
        extras = [
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.SersicSph(
                    centre=centre, intensity=0.9, effective_radius=0.8, sersic_index=3.0
                ),
                mass=al.mp.dPIEMassSph(
                    centre=centre,
                    sigma=sigma,
                    r_core=0.0,
                    r_cut=10.0,
                    redshift_object=0.5,
                    redshift_source=1.0,
                ),
            )
            for centre, sigma in zip(SPECS["group"].extra_galaxies_centres, (212.0, 239.0))
        ]
        source = al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=(0.0, 0.1),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=3.0,
                effective_radius=0.4,
                sersic_index=1.0,
            ),
        )
        return [lens_0, *extras, source], []

    raise KeyError(dataset_type)


# --------------------------------------------------------------------------- simulation


def simulate(dataset_type: str, root: Path) -> Path:
    """Write ``dataset/<dataset_type>/<instrument>/`` (data, psf, noise map, tracer.json, centres).

    Deterministic (``noise_seed``): the same bytes of pixel data on every machine, so the dataset
    is regenerated on demand rather than tracked.
    """
    import autolens as al

    spec = SPECS[dataset_type]
    path = dataset_path(dataset_type, root)
    path.mkdir(parents=True, exist_ok=True)
    grid = al.Grid2D.uniform(shape_native=spec.shape_native, pixel_scales=spec.pixel_scale)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=list(spec.simulate_sub_size_list),
        radial_list=list(spec.lp_radial_list),
        centre_list=spec.centres,
    )
    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)
    psf = al.Convolver.from_gaussian(
        shape_native=spec.psf_shape, sigma=spec.psf_sigma, pixel_scales=grid.pixel_scales
    )
    simulator = al.SimulatorImaging(
        exposure_time=spec.exposure_time,
        psf=psf,
        background_sky_level=spec.background_sky_level,
        add_poisson_noise_to_data=True,
        noise_seed=spec.noise_seed,
    )
    galaxies, fields = truth_galaxies(dataset_type)
    tracer = al.Tracer(galaxies=galaxies, fields=fields) if fields else al.Tracer(galaxies=galaxies)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    import autolens.plot as aplt

    aplt.fits_imaging(
        dataset=dataset,
        data_path=path / "data.fits",
        psf_path=path / "psf.fits",
        noise_map_path=path / "noise_map.fits",
        overwrite=True,
    )
    al.output_to_json(obj=tracer, file_path=path / "tracer.json")
    al.output_to_json(
        obj=al.Grid2DIrregular([list(c) for c in spec.main_lens_centres]),
        file_path=path / "main_lens_centres.json",
    )
    if spec.extra_galaxies_centres:
        al.output_to_json(
            obj=al.Grid2DIrregular([list(c) for c in spec.extra_galaxies_centres]),
            file_path=path / "extra_galaxies_centres.json",
        )
    (path / "simulation.json").write_text(
        json.dumps(
            {
                "dataset_type": dataset_type,
                "instrument": spec.instrument,
                "provenance": spec.provenance,
                "shape_native": list(spec.shape_native),
                "pixel_scale": spec.pixel_scale,
                "psf": {"shape": list(spec.psf_shape), "sigma": spec.psf_sigma},
                "exposure_time": spec.exposure_time,
                "background_sky_level": spec.background_sky_level,
                "noise_seed": spec.noise_seed,
                "over_sampling": {
                    "sub_size_list": list(spec.simulate_sub_size_list),
                    "radial_list": list(spec.lp_radial_list),
                    "centres": [list(c) for c in spec.centres],
                },
            },
            indent=2,
        )
    )
    return path


# --------------------------------------------------------------------------- the CLI


def main(dataset_type: str, doc: str, argv=None) -> int:
    """The ``group.py`` / ``multi_galaxy.py`` CLI: simulate, then time it into
    ``results/misc/simulators/<dataset_type>_<instrument>_summary_v<version>.json``.

    The profiling builder times nothing (its datasets were a by-product of the likelihood cells),
    so the summary is one ``simulate`` phase: enough to catch a dataset that starts taking minutes.
    """
    import argparse
    import time

    spec = SPECS[dataset_type]
    parser = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    parser.add_argument(
        "--instrument",
        type=str,
        default=spec.instrument,
        choices=[spec.instrument],
        help=f"Instrument preset to simulate (default: {spec.instrument}; the only one, D2).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override the autolens_inference root that holds dataset/. "
            "Defaults to the repo root inferred from this file's location."
        ),
    )
    args = parser.parse_args(argv)
    spec_for(dataset_type, args.instrument)
    root = args.output_root if args.output_root is not None else _inference_root()

    start = time.perf_counter()
    path = simulate(dataset_type, root)
    elapsed = time.perf_counter() - start
    print(f"  [{dataset_type} simulator] {elapsed:.4f} s -> {path}")

    import autolens as al

    al_version = al.__version__
    results_dir = Path(root) / "results" / "misc" / "simulators"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{dataset_type}_{spec.instrument}_summary_v{al_version}.json"
    json_path.write_text(
        json.dumps(
            {
                "autolens_version": al_version,
                "type": dataset_type,
                "instrument": spec.instrument,
                "configuration": json.loads((path / "simulation.json").read_text()),
                "phases": {"simulate": elapsed},
            },
            indent=2,
        )
    )
    print(f"  Results saved to: {json_path}")
    return 0


def _inference_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_inference root (ruff.toml) not found")
