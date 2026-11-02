"""
SLaM DelaunayNN run: HST imaging
================================

The 5-stage imaging SLaM chain on the simulated HST cell with the ``al.mesh.DelaunayNN`` source
mesh (1250 Hilbert vertices plus the 30-point zeroed ring, ``AdaptSplit``): the Delaunay recipe
with only the interpolation changed, the ``imaging/delaunay_nn`` likelihood. Its own leaf
because its cost profile is its own (the wall gate keys a cell by the script it runs). The
chain, the flags and the result row live in ``scripts/misc/slam/_runner.py``.

    python3 scripts/imaging/delaunay_nn/validation/baseline_slam.py --backend numba_cpu \
        --inversion sparse --config-name local_numba_cpu_sparse_fp64 --seed 0
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

from slam._runner import run_slam  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        run_slam(dataset_class="imaging", default_instrument="hst", default_mesh="delaunay_nn")
    )
