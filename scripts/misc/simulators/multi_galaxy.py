"""
Simulator: Multi-Galaxy Imaging (HST)
=====================================

Simulates the profiled multi_galaxy dataset into ``dataset/multi_galaxy/hst/``: ``data.fits``,
``noise_map.fits``, ``psf.fits``, ``tracer.json``, ``main_lens_centres.json`` and
``simulation.json`` (the simulation record). It is byte-for-byte the dataset the
autolens_profiling multi_galaxy likelihood cells fit (``cell_builders/galaxy_group.py``, #473);
the spec and the code live in ``_galaxy_group.py`` beside this script, and a runner reads its
mask radius, noise-scaled extra-galaxy region, over-sampling and galaxy centres from there::

    from simulators._galaxy_group import SPECS

    spec = SPECS["multi_galaxy"]

Times the simulation into ``results/misc/simulators/multi_galaxy_hst_summary_v<ver>.json``.

Usage
-----

    python scripts/misc/simulators/multi_galaxy.py --instrument hst [--output-root <repo>]
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


from simulators._galaxy_group import INSTRUMENTS, SPECS, main, simulate  # noqa: F401

if __name__ == "__main__":
    import os
    import sys

    from autonerves import jax_wrapper  # noqa: F401

    if os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
        sys.exit(0)

    raise SystemExit(main("multi_galaxy", __doc__))
