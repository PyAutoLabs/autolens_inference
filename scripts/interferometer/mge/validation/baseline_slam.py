"""
SLaM MGE run: SMA interferometer
================================

The interferometer MGE chain (``source_lp[1]`` -> ``mass_total[1]``) with the profiled linear
20-Gaussian MGE source on the simulated SMA cell, through the NUFFT: the
``interferometer/mge`` likelihood at both stages. The chain lives in
``scripts/misc/slam/_interferometer.py``.

    python3 scripts/interferometer/mge/validation/baseline_slam.py --backend jax_cpu \
        --inversion dense --transformer nufft \
        --config-name local_jax_cpu_dense_fp64_nufft --seed 0
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
from slam._interferometer import MGE  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_family(MGE, default_instrument="sma"))
