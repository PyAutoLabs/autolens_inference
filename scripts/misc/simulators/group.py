"""
Simulator: Group-Scale Imaging (Euclid)
=======================================

Simulates the profiled group dataset into ``dataset/group/euclid/``: ``data.fits``,
``noise_map.fits``, ``psf.fits``, ``tracer.json``, ``main_lens_centres.json``,
``extra_galaxies_centres.json`` and ``simulation.json`` (the simulation record). It is
byte-for-byte the dataset the autolens_profiling group likelihood cells fit
(``cell_builders/galaxy_group.py``, #473); the spec and the code live in ``_galaxy_group.py``
beside this script, and a runner reads its mask radius, over-sampling and galaxy centres from
there::

    from simulators._galaxy_group import SPECS

    spec = SPECS["group"]

Times the simulation into ``results/misc/simulators/group_euclid_summary_v<ver>.json``.

Usage
-----

    python scripts/misc/simulators/group.py --instrument euclid [--output-root <repo>]
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

    raise SystemExit(main("group", __doc__))
