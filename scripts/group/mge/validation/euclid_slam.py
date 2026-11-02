"""
SLaM MGE: Euclid group
======================

The ``group/mge`` likelihood's validation chain (likelihood-lifecycle 4c-2,
autolens_inference#54): ``group/features/multi_gaussian_expansion/slam.py``'s three stages on
the profiled euclid group dataset, every one an MGE linear solve (no mesh). The stages, which of
them evaluate the profiled likelihood, and the departures from the workspace are in
``scripts/misc/slam/_group.py``.

    python3 scripts/group/mge/validation/euclid_slam.py \
        --backend jax_cpu --inversion dense --config-name hpc_ral_jax_cpu_dense_fp64 --seed 0

The row lands under ``results/group/mge/euclid/validation/default/<setup_id>/seed_<n>/``. The
leaf is named for its instrument because the wall gate reads the leaf of a cell
(``group/mge/validation/euclid_slam``) as the instrument it runs.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import run_family  # noqa: E402
from slam._group import FAMILY_MGE  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_family(FAMILY_MGE, default_instrument="euclid"))
