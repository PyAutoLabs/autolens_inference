"""
SLaM Delaunay: HST multi-galaxy
===============================

The ``multi_galaxy/delaunay`` likelihood's validation chain (likelihood-lifecycle 4c-2,
autolens_inference#54): ``multi_galaxy/slam.py``'s five stages (the workspace has no Delaunay
variant) on the profiled hst two-deflector dataset, with the profiled ``al.mesh.Delaunay`` at
1250 Hilbert vertices under ``AdaptSplit``. The stages, which of them evaluate the profiled
likelihood, and the departures from the workspace are in ``scripts/misc/slam/_multi_galaxy.py``.

    python3 scripts/multi_galaxy/delaunay/validation/hst_slam.py \
        --backend numba_cpu --inversion sparse --config-name hpc_ral_numba_cpu_sparse_fp64 --seed 0

The row lands under
``results/multi_galaxy/delaunay/hst/validation/default/<setup_id>/seed_<n>/``. The leaf is named
for its instrument because the wall gate reads the leaf of a cell
(``multi_galaxy/delaunay/validation/hst_slam``) as the instrument it runs.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import run_family  # noqa: E402
from slam._multi_galaxy import FAMILY_DELAUNAY  # noqa: E402

from _inference_cli import MESH_PIXELS_STANDARD  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        run_family(
            FAMILY_DELAUNAY,
            default_instrument="hst",
            default_mesh="delaunay",
            default_mesh_pixels=MESH_PIXELS_STANDARD["delaunay"],
        )
    )
