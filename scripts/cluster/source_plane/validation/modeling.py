"""
Cluster modeling: simple, source-plane solved chi-squared
=========================================================

``autolens_workspace/scripts/cluster/modeling.py``'s one Nautilus search over a factor graph
of the two sources, with the profiled cluster model (2 main + 10 scaling dPIE, NFW host)
and fit (``al.FitPositionsSourceSolved``),
under the Optimal setups profiling measured. One stage, ``modeling[1]``, written as a one-stage
``stages.json`` that pairs with this likelihood's profiling record. This leaf is a name and a
family; the model, the flags and the row live in ``scripts/misc/slam/_cluster.py``.

    python3 scripts/cluster/source_plane/validation/modeling.py --instrument simple \
        --backend jax_cpu --inversion na --config-name local_jax_cpu_na_fp64 --seed 0
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._cluster import SOURCE_PLANE  # noqa: E402
from slam._core import run_family  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_family(SOURCE_PLANE, default_instrument="simple"))
