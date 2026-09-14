"""
SLaM Delaunay variant: HST imaging
==================================

The same 5-stage imaging SLaM chain as ``hst.py``, with the two pixelized source
stages on ``al.mesh.Delaunay`` at 1250 vertices instead of the 28x28 (784-cell)
adaptive rectangular mesh — the ``delaunay_1250`` run variant, which lands in
its own ``results/slam/imaging/hst/delaunay_1250/`` tree and its own
``hst/slam5_delaunay_1250/seed<n>`` parity group rather than joining the base
run's.

    python3 scripts/imaging/slam/hst_delaunay.py --backend jax_gpu \
        --inversion dense --config-name hpc_a100_jax_gpu_dense_fp64 --seed 0

**Why this is a separate leaf and not a flag on ``hst.py``.** The wall gate
derives a submit's *cell* from the script path it invokes
(``scripts/misc/wall/check_submits.py``: ``scripts/imaging/slam/hst.py`` is the
cell ``imaging/slam/hst``), and ``scripts/misc/wall/rates.py`` keys its measured
step rates by that cell. A 1250-vertex Delaunay chain is a different cost
profile from a 784-cell rectangular one, so it must never share a rate row: a
``--time`` justified from the rectangular cell's rate would be a number measured
on a different model. Its own leaf gives it its own cell id,
``imaging/slam/hst_delaunay``, and the gate then demands its own measurement.

The vertex count is the family default (``--mesh delaunay`` resolves
``--mesh-pixels`` to 1250, ``_inference_cli.MESH_PIXELS_DEFAULT``);
``--mesh-pixels N`` overrides it, and a non-default count names its own variant
folder rather than overwriting this one.

See ``scripts/imaging/slam/README.md`` for the flags and the legs of the parity
row.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._runner import run_slam  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_slam(dataset_class="imaging", default_instrument="hst", default_mesh="delaunay"))
