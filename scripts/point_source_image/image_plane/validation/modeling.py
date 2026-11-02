"""
Point-source modeling: simple, image-plane solved chi-squared
=============================================================

``autolens_workspace/scripts/point_source/modeling.py``'s one Nautilus search with the
profiled image-plane fit (``Isothermal`` + ``PointSolved``, ``al.FitPositionsImagePairAllSolved``),
under the Optimal setups profiling measured. One stage, ``modeling[1]``, written as a one-stage
``stages.json`` that pairs with this likelihood's profiling record. This leaf is a name and a
family; the model, the flags and the row live in ``scripts/misc/slam/_point.py``.

    python3 scripts/point_source_image/image_plane/validation/modeling.py --instrument simple \
        --backend jax_cpu --inversion na --config-name local_jax_cpu_na_fp64 --seed 0
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam._core import run_family  # noqa: E402
from slam._point import IMAGE_PLANE  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_family(IMAGE_PLANE, default_instrument="simple"))
