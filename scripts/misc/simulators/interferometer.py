"""
Simulator + Profiler: Instrument-Based Interferometer Datasets
==============================================================

Single dual-purpose module:

1. **Simulates** an interferometer dataset for one of the ``INSTRUMENTS``
   presets (``sma`` / ``alma`` / ``alma_high``). Writes ``data.fits``,
   ``noise_map.fits``, ``uv_wavelengths.fits``, ``lensed_source.fits``,
   ``positions.json`` and ``tracer.json`` into
   ``dataset/interferometer/<instrument>/``.

2. **Profiles** each phase of the simulator pipeline (grid setup, uv
   generation, simulator construction, eager + JIT ``image_2d_from``,
   ``via_tracer_from``, ``PointSolver.solve``, fits output) using a
   per-section timer and writes
   ``results/simulators/interferometer_<instrument>_summary_v<ver>.{json,png}``.

``INSTRUMENTS`` is the single source of truth for all interferometer
likelihood scripts — they import it directly::

    from simulators.interferometer import INSTRUMENTS

The auto-simulate hook in those scripts (via
``_inference_cli.auto_simulate_if_missing``) shells out to this script's
``--instrument`` CLI when the dataset is missing.

Usage
-----

    python simulators/interferometer.py                          # sma (default)
    python simulators/interferometer.py --instrument alma        # 1M vis ALMA
    python simulators/interferometer.py --instrument alma_high   # 10M vis
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

# Soft-transition re-export — INSTRUMENTS now lives in
# `instruments/interferometer.py`. Existing
# `from simulators.interferometer import INSTRUMENTS` consumers keep working.
sys.path.insert(0, str(_inference_root()))
from instruments.interferometer import (  # noqa: E402, F401
    INSTRUMENTS,
)
from instruments.interferometer import (
    TRANSFORMER_CLASS_NAME as _TRANSFORMER_CLASS,
)

_REPO_ROOT = _inference_root()  # autolens_inference/


def simulate(instrument: str = "sma", output_root: Path | None = None) -> Path:
    """Simulate + profile the named instrument. Returns the dataset directory."""
    # Lazy imports — only run when the function is invoked, so module import
    # by a likelihood script just to pick up INSTRUMENTS is side-effect-free.
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
    real_space_shape = config["real_space_shape"]
    n_visibilities = config["n_visibilities"]
    uv_scale = config["uv_scale"]
    noise_sigma = config["noise_sigma"]
    seed = config["seed"]
    transformer_choice = config.get("transformer", "dft").lower()
    transformer_chunk_size = config.get("transformer_chunk_size", None)
    if transformer_choice == "nufft":
        # Lambda inject so chunk_size flows into TransformerNUFFT.__init__
        # without needing a transformer_kwargs API in Interferometer.from_fits.
        def transformer_class(uv_wavelengths, real_space_mask):
            return al.TransformerNUFFT(
                uv_wavelengths=uv_wavelengths,
                real_space_mask=real_space_mask,
                chunk_size=transformer_chunk_size,
            )
    elif transformer_choice == "dft":
        transformer_class = al.TransformerDFT
    else:
        raise ValueError(f"Unknown transformer '{transformer_choice}'")

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "interferometer" / instrument
    dataset_path.mkdir(parents=True, exist_ok=True)

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

    print(f"\n--- Interferometer simulator + profiler [{instrument}] ---")
    print(f"  pixel_scale:     {pixel_scale} arcsec/px")
    print(f"  real_space_shape: {real_space_shape[0]} x {real_space_shape[1]}")
    print(f"  n_visibilities:  {n_visibilities:,}")
    print(f"  output:          {dataset_path}")

    # === PART 1 — Setup ===

    print("\n--- PART 1: Setup ---")

    with timer.section("setup_grids"):
        # Interferometer does not use over-sampling
        grid = al.Grid2D.uniform(shape_native=real_space_shape, pixel_scales=pixel_scale)

    with timer.section("setup_uv_wavelengths"):
        # Synthetic baselines drawn from a 2D isotropic Gaussian whose 3-sigma
        # envelope matches ``uv_scale``. Crude vs real instrument coverage but
        # sufficient for profiling. Seeded for reproducibility.
        rng = np.random.default_rng(seed)
        uv_wavelengths = rng.normal(loc=0.0, scale=uv_scale / 3.0, size=(n_visibilities, 2)).astype(
            np.float64
        )

    with timer.section("setup_simulator"):
        simulator = al.SimulatorInterferometer(
            uv_wavelengths=uv_wavelengths,
            exposure_time=300.0,
            noise_sigma=noise_sigma,
            transformer_class=transformer_class,
            noise_seed=seed,
        )

    with timer.section("setup_galaxies"):
        lens_galaxy = al.Galaxy(
            redshift=0.5,
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
                centre=(0.1, 0.1),
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
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
        err_msg="interferometer: eager vs JIT image_2d_from mismatch",
    )
    print("  eager ≡ JIT assertion PASSED")

    # === PART 3 — via_tracer_from (DFT bottleneck) ===

    print("\n--- PART 3: simulator.via_tracer_from (DFT) ---")

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
        aplt.fits_interferometer(
            dataset=dataset,
            data_path=dataset_path / "data.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
            overwrite=True,
        )

    # Lensed source image (real-space, pre-NUFFT) — used as the ``adapt_image``
    # by downstream interferometer + datacube likelihood scripts that profile
    # ``RectangularBilinearAdaptImage`` and ``image_mesh.Hilbert``. No PSF convolution
    # here; the visibility transform lives downstream in the likelihood path.
    with timer.section("output_lensed_source"):
        lensed_source = tracer.image_2d_list_from(grid=grid)[-1]
        # Save native 2D shape — al.output_to_fits writes whatever values it
        # is handed; passing .array gives the masked/slim 1D form which the
        # downstream `aa.Array2D.from_fits` loader can't reshape without
        # shape_native context. ``.native_for_fits`` returns the (N_y, N_x)
        # form expected by from_fits.
        al.output_to_fits(
            values=lensed_source.native_for_fits,
            file_path=dataset_path / "lensed_source.fits",
            overwrite=True,
        )

    with timer.section("output_positions"):
        al.output_to_json(obj=positions, file_path=dataset_path / "positions.json")

    with timer.section("output_tracer"):
        al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")

    # === Summary ===

    al_version = al.__version__
    results_dir = root / "results" / "simulators"
    results_dir.mkdir(parents=True, exist_ok=True)

    phases = dict(timer.records)

    results_summary = {
        "autolens_version": al_version,
        "type": "interferometer",
        "instrument": instrument,
        "configuration": {
            "grid_shape": list(real_space_shape),
            "pixel_scales": pixel_scale,
            "n_visibilities": n_visibilities,
            "uv_scale": uv_scale,
            "noise_sigma": noise_sigma,
            "transformer": _TRANSFORMER_CLASS[transformer_choice],
            "transformer_chunk_size": transformer_chunk_size,
        },
        "phases": phases,
        "key_timings": {
            "image_2d_eager_s": phases.get("image_2d_eager"),
            "via_tracer_from_s": phases.get("via_tracer_from"),
            "solver_solve_eager_s": phases.get("solver_solve_eager"),
        },
    }

    json_path = results_dir / f"interferometer_{instrument}_summary_v{al_version}.json"
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
        f"Simulator Profiling: Interferometer ({instrument})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"AutoLens v{al_version}  |  "
        f'{real_space_shape[0]}×{real_space_shape[1]} @ {pixel_scale}"/px  |  '
        f"{n_visibilities:,} visibilities (DFT)",
        fontsize=9,
    )
    ax.margins(x=0.22)
    fig.tight_layout()
    chart_path = results_dir / f"interferometer_{instrument}_summary_v{al_version}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Bar chart saved to: {chart_path}")

    timer.summary()

    return dataset_path


if __name__ == "__main__":
    # autonerves.jax_wrapper must be imported first (sets up env before jax
    # actually loads). Doing it here rather than at module level keeps the
    # module import side-effect-free for likelihood scripts that pull
    # ``INSTRUMENTS`` without wanting to trigger the autonerves shim.
    import argparse
    import os
    import sys

    from autonerves import jax_wrapper  # noqa: F401

    # Smoke gate — only meaningful when running as a script, not on import.
    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--instrument",
        type=str,
        default="sma",
        choices=list(INSTRUMENTS.keys()),
        help="Instrument preset to simulate + profile (default: sma).",
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
