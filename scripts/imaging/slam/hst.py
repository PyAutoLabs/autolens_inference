"""
SLaM base run: HST imaging
==========================

The 5-stage imaging SLaM chain on the simulated HST cell, under any
``--backend`` x ``--inversion`` combination. This leaf is a name and a dataset
class; the chain, the flags and the result row all live in
``scripts/misc/slam/_runner.py``.

    python3 scripts/imaging/slam/hst.py --backend jax_cpu --inversion dense \
        --config-name local_jax_cpu_dense_fp64 --seed 0

See ``scripts/imaging/slam/README.md`` for the flags and the six legs of the
parity row.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._runner import run_slam  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_slam(dataset_class="imaging", default_instrument="hst"))
