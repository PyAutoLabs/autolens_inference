"""
SLaM rectangular run: SMA interferometer
========================================

The 4-stage workspace interferometer SLaM chain with the profiled standard-size rectangular
source (35 x 35 ``RectangularBilinearAdaptImage``, ``Adapt``) on the simulated SMA cell, the
``interferometer/rectangular`` likelihood. The chain lives in
``scripts/misc/slam/_interferometer.py``.

    python3 scripts/interferometer/rectangular/validation/baseline_slam.py --backend numba_cpu \
        --inversion sparse --transformer wtilde \
        --config-name local_numba_cpu_sparse_fp64_wtilde --seed 0
"""

import sys
from pathlib import Path

# A leaf named `nautilus.py` / `emcee.py` beside a script would shadow the sampler package of
# that name, and a script's own directory is first on sys.path, so it is dropped (as every leaf
# here does).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import run_family  # noqa: E402
from slam._interferometer import RECTANGULAR  # noqa: E402

from _inference_cli import MESH_PIXELS_STANDARD  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        run_family(
            RECTANGULAR,
            default_instrument="sma",
            default_mesh="rect",
            default_mesh_pixels=MESH_PIXELS_STANDARD["rect"],
        )
    )
