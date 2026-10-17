"""
MultiStartAdam search: simple point source, source-plane solved χ²
==================================================================

The gradient-search sibling of ``baseline.py``: the same L5 point-source model
(``Isothermal`` + ``PointSolved``, 5 free parameters), priors and ``simple``
dataset, fitted with ``al.FitPositionsSourceSolved`` by ``af.MultiStartAdam`` at
the library defaults (48 starts, 300-step ceiling, auto-convergence). The search
differentiates the likelihood in the analysis's declared ``gradient_mode``
(``AnalysisPoint`` declares ``"forward"``).

The row carries the gradient admission bar — the batched value-and-gradient
cost per evaluation, compile time, steps to convergence, evaluations
(``total_steps * n_starts``), likelihood share — and the MAP point against truth
in units of the committed Nautilus posterior width. All of it is defined in
``scripts/misc/searches/_point_runner.py``; this leaf is a name and a dataset class.

    python3 scripts/point_source/simple/multi_start_adam.py \\
        --instrument simple --backend jax_cpu --inversion dense \\
        --config-name local_jax_cpu_dense_fp64 --seed 0
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from searches._point_runner import run_point_search  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_point_search(sampler="multi_start_adam", default_instrument="simple"))
