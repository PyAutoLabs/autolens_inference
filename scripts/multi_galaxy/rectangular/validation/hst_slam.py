"""
SLaM rectangular: HST multi-galaxy
==================================

The ``multi_galaxy/rectangular`` likelihood's validation chain (likelihood-lifecycle 4c-2,
autolens_inference#54): ``multi_galaxy/slam.py``'s five stages on the profiled hst two-deflector
dataset, with the profiled rectangular mesh at 35 x 35 under ``Adapt``. The stages, which of
them evaluate the profiled likelihood, and the departures from the workspace are in
``scripts/misc/slam/_multi_galaxy.py``.

    python3 scripts/multi_galaxy/rectangular/validation/hst_slam.py \
        --backend numba_cpu --inversion sparse --config-name hpc_ral_numba_cpu_sparse_fp64 --seed 0

The row lands under
``results/multi_galaxy/rectangular/hst/validation/default/<setup_id>/seed_<n>/``. The leaf is
named for its instrument because the wall gate reads the leaf of a cell
(``multi_galaxy/rectangular/validation/hst_slam``) as the instrument it runs.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import run_family  # noqa: E402
from slam._multi_galaxy import FAMILY_RECTANGULAR  # noqa: E402

from _inference_cli import MESH_PIXELS_STANDARD  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        run_family(
            FAMILY_RECTANGULAR,
            default_instrument="hst",
            default_mesh="rect",
            default_mesh_pixels=MESH_PIXELS_STANDARD["rect"],
        )
    )
