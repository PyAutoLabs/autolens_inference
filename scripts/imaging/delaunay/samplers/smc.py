"""Run smc on a prepared imaging/delaunay mass_total[1] problem."""

import sys
from pathlib import Path

# This leaf's directory also holds `nautilus.py` and `emcee.py`, which would shadow
# the sampler packages of the same name for every import made by this process
# (`from nautilus import Sampler` inside PyAutoFit found this directory's leaf).
# Python puts the script's directory first on sys.path, so drop it.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]

from experiments.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(setup_family=("imaging", "delaunay"), sampler="smc"))
