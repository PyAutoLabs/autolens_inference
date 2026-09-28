"""
Nautilus search: simple point source, source-plane solved χ²
============================================================

One Nautilus fit of the L5 point-source model (``Isothermal`` + ``PointSolved``,
5 free parameters) to the ``simple`` dataset with ``al.FitPositionsSourceSolved``,
under ``--backend jax_cpu`` or ``jax_gpu``. The row it writes carries the
**admission bar** — ``per_call_s``, ``compile_s`` and ``likelihood_share`` beside
the sampler's ``wall_s`` and ``likelihood_evals`` — defined in
``scripts/misc/searches/_point_runner.py``, where the model, the flags and the
row all live. This leaf is a name and a dataset class.

    python3 scripts/point_source/searches/nautilus/simple_source_plane.py \
        --instrument simple --backend jax_cpu --inversion dense \
        --config-name local_jax_cpu_dense_fp64 --seed 0

``--inversion`` must be ``dense``: a point source has no inversion, and the
config-name grammar still carries the field.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from searches._point_runner import run_point_search  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_point_search(sampler="nautilus", default_instrument="simple"))
