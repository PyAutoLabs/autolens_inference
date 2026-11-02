"""
SLaM MGE: HST multi-galaxy
==========================

The ``multi_galaxy/mge`` likelihood's validation chain (likelihood-lifecycle 4c-2,
autolens_inference#54): the nearest analogue of a pure-MGE multi-galaxy pipeline (the workspace
has none): ``source_lp[1]`` -> ``light[1]`` -> ``mass_total[1]`` on ``multi_galaxy/slam.py``'s
models, on the profiled hst two-deflector dataset. The stages, which of them evaluate the
profiled likelihood, and the departures from the workspace are in
``scripts/misc/slam/_multi_galaxy.py``.

    python3 scripts/multi_galaxy/mge/validation/hst_slam.py \
        --backend jax_cpu --inversion dense --config-name hpc_ral_jax_cpu_dense_fp64 --seed 0

The row lands under ``results/multi_galaxy/mge/hst/validation/default/<setup_id>/seed_<n>/``.
The leaf is named for its instrument because the wall gate reads the leaf of a cell
(``multi_galaxy/mge/validation/hst_slam``) as the instrument it runs.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import run_family  # noqa: E402
from slam._multi_galaxy import FAMILY_MGE  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_family(FAMILY_MGE, default_instrument="hst"))
