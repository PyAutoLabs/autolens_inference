"""
Simulator + Profiler: Instrument-Based Imaging Datasets
========================================================

Single dual-purpose module:

1. **Simulates** an imaging dataset for one of the ``INSTRUMENTS`` presets
   (``euclid`` / ``hst`` / ``jwst`` / ``ao``). Writes ``data.fits``,
   ``psf.fits``, ``noise_map.fits``, ``lensed_source.fits``,
   ``positions.json`` and ``tracer.json`` into
   ``dataset/imaging/<instrument>/``.

2. **Profiles** each phase (grid + over-sampling setup, PSF + simulator
   construction, eager + JIT ``image_2d_from``, ``via_tracer_from``,
   ``PointSolver.solve``, fits output) and writes
   ``results/simulators/imaging_<instrument>_summary_v<ver>.{json,png}``.

``INSTRUMENTS`` is the single source of truth for all imaging likelihood
scripts — they import it directly::

    from simulators.imaging import INSTRUMENTS

Usage
-----

    python simulators/imaging.py                       # hst (default)
    python simulators/imaging.py --instrument euclid
"""

import sys as _sys
from pathlib import Path as _Path


def _inference_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_inference root (ruff.toml) not found")


_misc_dir = str(_inference_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)


import sys
from pathlib import Path

# Soft-transition re-export — INSTRUMENTS now lives in `instruments/imaging.py`.
# Existing `from simulators.imaging import INSTRUMENTS` consumers keep working.
sys.path.insert(0, str(_inference_root()))
from instruments.imaging import INSTRUMENTS  # noqa: E402, F401

_REPO_ROOT = _inference_root()  # autolens_inference/


def simulate(instrument: str = "hst", output_root: Path | None = None) -> Path:
    """Simulate + profile the named imaging instrument. Returns the dataset dir."""
    import json
    import time
    from contextlib import contextmanager

    import jax
    import jax.numpy as jnp
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import autolens as al
    import autolens.plot as aplt
    import matplotlib.pyplot as plt

    if instrument not in INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. Choose from: {list(INSTRUMENTS.keys())}"
        )

    config = INSTRUMENTS[instrument]
    pixel_scale = config["pixel_scale"]
    mask_radius = config["mask_radius"]
    psf_shape = config["psf_shape"]
    psf_sigma = config["psf_sigma"]
    seed = config["seed"]

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "imaging" / instrument
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Grid size derived so the mask_radius circular mask fits in the image.
    shape_pixels = int(np.ceil(2 * mask_radius / pixel_scale))
    if shape_pixels % 2 == 0:
        shape_pixels += 1  # odd for symmetric centering

    class Timer:
        def __init__(self):
            self.records: list[tuple[str, float]] = []

        @contextmanager
        def section(self, label: str):
            start = time.perf_counter()
            yield
            elapsed = time.perf_counter() - start
            self.records.append((label, elapsed))
            print(f"  [{label}] {elapsed:.4f} s")

        def summary(self):
            print("\n" + "=" * 70)
            print("PROFILING SUMMARY")
            print("=" * 70)
            max_label = max(len(r[0]) for r in self.records)
            total = 0.0
            for label, elapsed in self.records:
                print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
                total += elapsed
            print("-" * 70)
            print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
            print("=" * 70)

    def block(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    def jit_profile(func, label, *args, n_repeats=10):
        jitted = jax.jit(func)
        with timer.section(f"{label}_lower"):
            lowered = jitted.lower(*args)
        with timer.section(f"{label}_compile"):
            compiled = lowered.compile()
        with timer.section(f"{label}_first_call"):
            result = compiled(*args)
            block(result)
        with timer.section(f"{label}_steady_x{n_repeats}"):
            for _ in range(n_repeats):
                result = compiled(*args)
                block(result)
        per_call = timer.records[-1][1] / n_repeats
        print(f"    -> per-call avg: {per_call:.6f} s")
        return compiled, result

    timer = Timer()

    print(f"\n--- Imaging simulator + profiler [{instrument}] ---")
    print(f"  pixel_scale:     {pixel_scale} arcsec/px")
    print(f"  grid_shape:      {shape_pixels} x {shape_pixels}")
    print(f"  mask_radius:     {mask_radius} arcsec")
    print(f"  psf_shape:       {psf_shape[0]} x {psf_shape[1]}")
    print(f"  output:          {dataset_path}")

    # === PART 1 — Setup ===

    print("\n--- PART 1: Setup ---")

    with timer.section("setup_grids"):
        grid = al.Grid2D.uniform(
            shape_native=(shape_pixels, shape_pixels), pixel_scales=pixel_scale
        )
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )
        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    with timer.section("setup_psf_simulator"):
        psf = al.Convolver.from_gaussian(
            shape_native=psf_shape,
            sigma=psf_sigma,
            pixel_scales=grid.pixel_scales,
        )
        simulator = al.SimulatorImaging(
            exposure_time=300.0,
            psf=psf,
            background_sky_level=0.1,
            add_poisson_noise_to_data=True,
            noise_seed=seed,
        )

    with timer.section("setup_galaxies"):
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(0.0, 0.0),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                intensity=2.0,
                effective_radius=0.6,
                sersic_index=3.0,
            ),
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.6,
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            ),
            shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
        )
        source_galaxy = al.Galaxy(
            redshift=1.0,
            bulge=al.lp.SersicCore(
                centre=(0.0, 0.0),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=4.0,
                effective_radius=0.1,
                sersic_index=1.0,
            ),
        )

    with timer.section("setup_tracer"):
        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    # === PART 2 — image_2d_from: eager + JIT ===

    print("\n--- PART 2: tracer.image_2d_from (eager + JIT) ---")

    with timer.section("image_2d_eager"):
        image_eager = tracer.image_2d_from(grid=grid)

    def _image_fn(grid_array):
        return tracer.image_2d_from(grid=grid, xp=jnp).array

    jnp_grid = jnp.asarray(grid.array)
    _, image_jit = jit_profile(_image_fn, "image_2d_jit", jnp_grid)

    np.testing.assert_allclose(
        np.asarray(image_eager.array),
        np.asarray(image_jit),
        rtol=1e-4,
        err_msg="imaging: eager vs JIT image_2d_from mismatch",
    )
    print("  eager ≡ JIT assertion PASSED")

    # === PART 3 — via_tracer_from ===

    print("\n--- PART 3: simulator.via_tracer_from ---")

    with timer.section("via_tracer_from"):
        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    # === PART 4 — solver.solve ===

    print("\n--- PART 4: solver.solve ---")

    with timer.section("solver_build"):
        solver = al.PointSolver.for_grid(
            grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
        )

    with timer.section("solver_solve_eager"):
        positions = solver.solve(tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre)

    # === PART 5 — outputs ===

    print("\n--- PART 5: outputs ---")

    with timer.section("output_fits"):
        aplt.fits_imaging(
            dataset=dataset,
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            overwrite=True,
        )

    # Lensed source image (PSF-convolved) — used as the ``adapt_image`` by
    # downstream likelihood scripts that profile ``RectangularBilinearAdaptImage`` and
    # ``image_mesh.Hilbert``.
    with timer.section("output_lensed_source"):
        lensed_source_unblurred = tracer.image_2d_list_from(grid=grid)[-1]
        lensed_source = psf.convolved_image_from(image=lensed_source_unblurred, blurring_image=None)
        al.output_to_fits(
            values=lensed_source.native_for_fits,
            file_path=dataset_path / "lensed_source.fits",
            overwrite=True,
        )

    with timer.section("output_json"):
        al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
        al.output_to_json(obj=positions, file_path=dataset_path / "positions.json")

    # === Summary ===

    al_version = al.__version__
    results_dir = root / "results" / "simulators"
    results_dir.mkdir(parents=True, exist_ok=True)

    phases = dict(timer.records)

    results_summary = {
        "autolens_version": al_version,
        "type": "imaging",
        "instrument": instrument,
        "configuration": {
            "grid_shape": [shape_pixels, shape_pixels],
            "pixel_scales": pixel_scale,
            "mask_radius": mask_radius,
            "psf_shape": list(psf_shape),
            "psf_sigma": psf_sigma,
            "over_sampling_sub_sizes": [32, 8, 2],
        },
        "phases": phases,
        "key_timings": {
            "image_2d_eager_s": phases.get("image_2d_eager"),
            "via_tracer_from_s": phases.get("via_tracer_from"),
            "solver_solve_eager_s": phases.get("solver_solve_eager"),
        },
    }

    json_path = results_dir / f"imaging_{instrument}_summary_v{al_version}.json"
    json_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\n  Results saved to: {json_path}")

    labels = [r[0] for r in timer.records]
    times = [r[1] for r in timer.records]
    colors = plt.cm.tab20.colors[: len(labels)]

    fig, ax = plt.subplots(figsize=(12, max(4.0, len(labels) * 0.45)))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.4f} s",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)", fontsize=11)
    fig.suptitle(
        f"Simulator Profiling: Imaging ({instrument})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"AutoLens v{al_version}  |  "
        f'{shape_pixels}×{shape_pixels} @ {pixel_scale}"/px  |  '
        f"PSF {psf_shape[0]}×{psf_shape[1]}",
        fontsize=9,
    )
    ax.margins(x=0.22)
    fig.tight_layout()
    chart_path = results_dir / f"imaging_{instrument}_summary_v{al_version}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Bar chart saved to: {chart_path}")

    timer.summary()

    return dataset_path


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from autonerves import jax_wrapper  # noqa: F401

    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--instrument",
        type=str,
        default="hst",
        choices=list(INSTRUMENTS.keys()),
        help="Instrument preset to simulate + profile (default: hst).",
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
    args = parser.parse_args()
    simulate(instrument=args.instrument, output_root=args.output_root)
