"""
SLaM base run: HST imaging
==========================

The 5-stage imaging SLaM chain on the simulated HST cell, under any
``--backend`` x ``--inversion`` combination. This leaf is a name and a dataset
class; the chain, the flags and the result row all live in
``scripts/misc/slam/_runner.py``.

    python3 scripts/imaging/rectangular/validation/baseline_slam.py --backend jax_cpu --inversion dense \
        --config-name local_jax_cpu_dense_fp64 --seed 0

See ``scripts/imaging/slam/README.md`` for the flags and the six legs of the
parity row.
"""

import sys
from pathlib import Path

# This leaf's directory also holds `nautilus.py` and `emcee.py`, which would shadow
# the sampler packages of the same name for every import made by this process
# (`from nautilus import Sampler` inside PyAutoFit found this directory's leaf).
# Python puts the script's directory first on sys.path, so drop it.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._runner import run_slam  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_slam(dataset_class="imaging", default_instrument="hst"))
