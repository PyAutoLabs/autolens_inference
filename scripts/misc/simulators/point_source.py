"""
Simulator + Profiler: Point-Source Datasets
============================================

Single dual-purpose module:

1. **Simulates** a lensed ``PointDataset`` for one of the ``INSTRUMENTS``
   presets (currently just ``simple``; kept dict-shaped so quad / low-S/N
   variants can land later mirroring the imaging + interferometer pattern).
   Writes ``point_dataset_positions_only.{json,csv}``, ``data.fits``,
   ``psf.fits``, ``noise_map.fits`` and ``tracer.json`` into
   ``dataset/point_source/<instrument>/``.

2. **Profiles** each phase (grid + solver setup, eager + JIT
   ``solver.solve``, ``time_delays_from``, imaging-side
   ``via_tracer_from``, output) and writes
   ``results/simulators/point_source_<instrument>_summary_v<ver>.{json,png}``.

``INSTRUMENTS`` is the single source of truth for the point-source
likelihood scripts — they import it directly::

    from simulators.point_source import INSTRUMENTS

Usage
-----

    python simulators/point_source.py                    # simple (default)
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


from pathlib import Path

INSTRUMENTS = {
    "simple": {
        "lens_centre": (0.0, 0.0),
        "lens_einstein_radius": 1.6,
        "lens_ell_comps_axis_angle": (0.9, 45.0),
        "source_centre": (0.07, 0.07),
        "source_flux": 0.1,  # PointFlux truth anchor only — no flux data is simulated
        "source_intensity": 0.1,
        "source_effective_radius": 0.02,
        "source_radius_break": 0.025,
        "grid_shape": (200, 200),
        "pixel_scale": 0.05,
        "pixel_scale_precision": 0.001,
        "magnification_threshold": 0.1,
        "psf_shape": (11, 11),
        "psf_sigma": 0.1,
        "exposure_time": 300.0,
        "background_sky_level": 0.1,
        "position_noise_sigma": 0.005,
        "seed": 1,
    },
}

# #678 phase B evidence-campaign variants — all derived from "simple" (same lens,
# grid, noise, seed); ``simulate()`` interprets the extra keys AFTER solving
# positions (see the "output_point_datasets" section below).
INSTRUMENTS["simple_missing"] = {
    **INSTRUMENTS["simple"],
    # Drops solved-position index 0 — the lowest-|mu| (~6.8, the faintest of the
    # 4 "simple" images) counter-image — before writing. truth.json is identical
    # to "simple" (only the observed positions are perturbed).
    "variant": {"drop_image_index": 0},
}
INSTRUMENTS["simple_extra"] = {
    **INSTRUMENTS["simple"],
    # Appends a spurious (non-lensed) position with the same noise sigma before
    # writing. truth.json is identical to "simple".
    "variant": {"extra_position": [1.0, -1.2]},
}
INSTRUMENTS["near_caustic"] = {
    k: v for k, v in INSTRUMENTS["simple"].items() if k != "source_centre"
}
# No "source_centre" — simulate() resolves it from the lens's tangential caustic
# (see _source_centre_near_caustic) and writes the resolved value into both the
# tracer and truth.json.
INSTRUMENTS["near_caustic"]["source_centre_caustic_fraction"] = 0.95


_REPO_ROOT = _inference_root()  # autolens_inference/


def _source_centre_near_caustic(al, lens_galaxy, grid, fraction: float) -> tuple[float, float]:
    """Resolve a ``near_caustic`` source_centre: ``fraction`` of the way to the
    lens's tangential caustic along the +y=+x diagonal (#678 phase B).

    Uses ``al.LensCalc.from_tracer`` on a lens-only tracer (no source profile
    needed to trace the critical curve/caustic) to compute the tangential
    caustic on a fine evaluation grid, then picks the caustic point whose polar
    angle is closest to the +y=+x diagonal (45 degrees) — matching "simple"'s
    existing (0.07, 0.07) placement direction. Scaling by ``fraction`` (< 1)
    keeps the source inside the caustic (still a quad).
    """
    import numpy as np

    lens_only_tracer = al.Tracer(galaxies=[lens_galaxy])
    calc = al.LensCalc.from_tracer(tracer=lens_only_tracer, use_multi_plane=False)
    caustic_list = calc.tangential_caustic_list_from(grid=grid, pixel_scale=0.01)
    if not caustic_list:
        raise RuntimeError(
            "near_caustic: al.LensCalc.tangential_caustic_list_from found no "
            "tangential caustic curve for this lens model."
        )
    caustic = np.asarray(caustic_list[0])
    angle = np.arctan2(caustic[:, 0], caustic[:, 1])
    diagonal_idx = int(np.argmin(np.abs(angle - np.pi / 4)))
    caustic_point = caustic[diagonal_idx]
    return (float(fraction * caustic_point[0]), float(fraction * caustic_point[1]))


def simulate(instrument: str = "simple", output_root: Path | None = None) -> Path:
    """Simulate + profile the named point-source instrument. Returns the dataset dir."""
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

    # Copy so a resolved "near_caustic" source_centre never mutates the module-
    # level INSTRUMENTS preset across repeated in-process calls.
    config = dict(INSTRUMENTS[instrument])
    grid_shape = config["grid_shape"]
    pixel_scale = config["pixel_scale"]

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "point_source" / instrument
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

    print(f"\n--- Point-source simulator + profiler [{instrument}] ---")
    print(f"  grid_shape:      {grid_shape[0]} x {grid_shape[1]}")
    print(f"  pixel_scale:     {pixel_scale} arcsec/px")
    print(f"  output:          {dataset_path}")

    # === PART 1 — Setup ===

    print("\n--- PART 1: Setup ---")

    with timer.section("setup_grids"):
        grid = al.Grid2D.uniform(shape_native=grid_shape, pixel_scales=pixel_scale)

    with timer.section("setup_galaxies"):
        axis, angle = config["lens_ell_comps_axis_angle"]
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=config["lens_centre"],
                einstein_radius=config["lens_einstein_radius"],
                ell_comps=al.convert.ell_comps_from(axis_ratio=axis, angle=angle),
            ),
        )
        if "source_centre_caustic_fraction" in config:
            config["source_centre"] = _source_centre_near_caustic(
                al, lens_galaxy, grid, config["source_centre_caustic_fraction"]
            )
            print(f"  Resolved near_caustic source_centre: {config['source_centre']}")
        source_galaxy = al.Galaxy(
            redshift=1.0,
            light=al.lp.ExponentialCore(
                centre=config["source_centre"],
                intensity=config["source_intensity"],
                effective_radius=config["source_effective_radius"],
                radius_break=config["source_radius_break"],
            ),
            point_0=al.ps.Point(centre=config["source_centre"]),
        )

    with timer.section("setup_tracer"):
        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    with timer.section("solver_build"):
        solver = al.PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=config["pixel_scale_precision"],
            magnification_threshold=config["magnification_threshold"],
        )

    # === PART 2 — solver.solve (eager) ===

    print("\n--- PART 2: solver.solve (eager) ---")

    with timer.section("solver_solve_eager"):
        positions = solver.solve(
            tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
        )

    print(f"  Found {len(positions)} image positions")

    if "source_centre_caustic_fraction" in config:
        # Fallback per the phase-B spec: no dedicated "is this a quad" helper
        # exists, so assert directly on the solved image count instead.
        assert len(positions) >= 4, (
            f"near_caustic: expected >=4 images inside the caustic, got {len(positions)}"
        )

    # === PART 3 — solver.solve (JIT) ===

    print("\n--- PART 3: solver.solve (JIT) ---")

    @jax.jit
    def jitted_solve(source_plane_coordinate):
        return solver.solve(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            xp=jnp,
            remove_infinities=False,
        ).array

    src_coord = jnp.asarray(source_galaxy.point_0.centre)
    _, raw_jit = jit_profile(jitted_solve, "solver_jit", src_coord, n_repeats=10)

    raw_np = np.asarray(raw_jit)
    finite_mask = ~(np.isinf(raw_np).any(axis=1) | np.isnan(raw_np).any(axis=1))
    positions_jit = al.Grid2DIrregular(raw_np[finite_mask])

    np.testing.assert_allclose(
        np.sort(np.asarray(positions), axis=0),
        np.sort(np.asarray(positions_jit), axis=0),
        rtol=1e-4,
        err_msg="point_source: eager vs JIT solver.solve positions mismatch",
    )
    print("  eager ≡ JIT solver assertion PASSED")

    # === PART 4 — time_delays_from ===

    print("\n--- PART 4: tracer.time_delays_from ---")

    with timer.section("time_delays_from"):
        time_delays = tracer.time_delays_from(grid=positions)  # noqa: F841

    # === PART 5 — imaging via_tracer_from (PSF convolution side) ===

    print("\n--- PART 5: simulator.via_tracer_from (imaging) ---")

    with timer.section("setup_psf_simulator"):
        psf = al.Convolver.from_gaussian(
            shape_native=config["psf_shape"],
            sigma=config["psf_sigma"],
            pixel_scales=grid.pixel_scales,
        )
        imaging_simulator = al.SimulatorImaging(
            exposure_time=config["exposure_time"],
            psf=psf,
            background_sky_level=config["background_sky_level"],
            add_poisson_noise_to_data=True,
            noise_seed=config["seed"],
        )

    with timer.section("via_tracer_from"):
        imaging = imaging_simulator.via_tracer_from(tracer=tracer, grid=grid)

    # === PART 6 — outputs ===

    print("\n--- PART 6: outputs ---")

    with timer.section("output_point_datasets"):
        # Seeded position noise — bit-identical reruns.
        position_noise = config["position_noise_sigma"]
        rng = np.random.default_rng(seed=config["seed"])
        positions_with_noise = positions + rng.normal(
            loc=0.0, scale=position_noise, size=positions.shape
        )
        positions_with_noise = np.asarray(positions_with_noise)

        # #678 phase B variants — applied to the dataset positions + noise
        # AFTER solving, before writing. No-op for "simple" (no "variant" key).
        variant = config.get("variant")
        if variant is not None and "drop_image_index" in variant:
            positions_with_noise = np.delete(
                positions_with_noise, variant["drop_image_index"], axis=0
            )
        if variant is not None and "extra_position" in variant:
            extra_position = np.asarray([variant["extra_position"]], dtype=float)
            positions_with_noise = np.concatenate([positions_with_noise, extra_position], axis=0)

        positions_with_noise = al.Grid2DIrregular(values=positions_with_noise)
        dataset = al.PointDataset(
            name="point_0",
            positions=positions_with_noise,
            positions_noise_map=position_noise,
        )
        al.output_to_json(
            obj=dataset,
            file_path=dataset_path / "point_dataset_positions_only.json",
        )
        dataset.to_csv(
            file_path=dataset_path / "point_dataset_positions_only.csv",
        )

    with timer.section("output_fits"):
        aplt.fits_imaging(
            dataset=imaging,
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            overwrite=True,
        )

    with timer.section("output_json"):
        al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
        # truth.json — mirrors simulators/group4_mge.py's truth.json convention
        # (a single JSON-friendly record next to the dataset for recovery /
        # truth-anchoring). "simple_missing" / "simple_extra" write the SAME
        # truth as "simple" (only the observed positions are perturbed);
        # "near_caustic" writes the RESOLVED source_centre.
        truth_record = {
            "lens_centre": [float(c) for c in config["lens_centre"]],
            "lens_ell_comps": [
                float(c) for c in al.convert.ell_comps_from(axis_ratio=axis, angle=angle)
            ],
            "lens_einstein_radius": float(config["lens_einstein_radius"]),
            "source_centre": [float(c) for c in config["source_centre"]],
            "source_flux": float(config["source_flux"]),
            "position_noise_sigma": float(position_noise),
            "seed": int(config["seed"]),
        }
        (dataset_path / "truth.json").write_text(json.dumps(truth_record, indent=2))

    # === Summary ===

    al_version = al.__version__
    results_dir = root / "results" / "simulators"
    results_dir.mkdir(parents=True, exist_ok=True)

    phases = dict(timer.records)

    results_summary = {
        "autolens_version": al_version,
        "type": "point_source",
        "instrument": instrument,
        "configuration": {
            "grid_shape": list(grid_shape),
            "pixel_scales": pixel_scale,
            "pixel_scale_precision": config["pixel_scale_precision"],
            "magnification_threshold": config["magnification_threshold"],
            "source_centre": list(config["source_centre"]),
        },
        "phases": phases,
        "key_timings": {
            "solver_solve_eager_s": phases.get("solver_solve_eager"),
            "via_tracer_from_s": phases.get("via_tracer_from"),
            "time_delays_from_s": phases.get("time_delays_from"),
            "n_positions_found": len(positions),
        },
    }

    json_path = results_dir / f"point_source_{instrument}_summary_v{al_version}.json"
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
        f"Simulator Profiling: Point Source ({instrument})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"AutoLens v{al_version}  |  "
        f'{grid_shape[0]}×{grid_shape[1]} @ {pixel_scale}"/px  |  '
        f"{len(positions)} images found",
        fontsize=9,
    )
    ax.margins(x=0.22)
    fig.tight_layout()
    chart_path = results_dir / f"point_source_{instrument}_summary_v{al_version}.png"
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
        default="simple",
        choices=list(INSTRUMENTS.keys()),
        help="Instrument preset to simulate + profile (default: simple).",
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
