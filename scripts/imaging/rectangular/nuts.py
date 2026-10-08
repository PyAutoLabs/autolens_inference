"""Run nuts on a prepared imaging/rectangular mass_total[1] problem."""

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]

from experiments.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(setup_family=("imaging", "rectangular"), sampler="nuts"))
