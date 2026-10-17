"""Wave 1 of the HST clean-slate mass-sampler comparison: arm table, resolver, accounting.

PyAutoInsight campaign ``hst-imaging-mass-samplers``, task ``hst_mass_samplers_wave1``.
Phase 2 exports one frozen ``mass_total[1]`` problem per (config, base seed, light mode) to::

    output/prepared/cleanslate_2026_10_17/<config>/seed_<base seed>/<variant>/manifest.json

Wave 1 runs every sampler ``experiments/run.py`` supports today, cold, on the base-seed-0
problem of both light modes, at three sampler seeds. The arm table below is the single
place that mapping lives: the SLURM submits ask this module for each array index's
leaf, and the tests check the mapping once, here.

Arms are indexed **seed-major**: index = seed x 8 + variant x 4 + sampler, so the first
eight arms are sampler seed 0 of every (variant, sampler) and a ``%4`` throttle starts one
arm of each sampler first (each first arm is that sampler's probe on this cell).

Run from the repo root::

    python3 scripts/misc/experiments/wave.py --list --config hpc_a100_jax_gpu_dense_fp64
    python3 scripts/misc/experiments/wave.py --account --config hpc_a100_jax_gpu_dense_fp64
    python3 scripts/misc/experiments/wave.py --arm 5 --config hpc_a100_jax_gpu_dense_fp64 \\
        --base-seed 0 --hardware-id ral-a100-1gpu-4cpu --cores 4 --print-command

``--account`` reports each expected arm as ``row`` (a complete result row), ``failed``
(a row whose status is not complete, or whose identity does not match the arm) or
``missing`` (no row). It never infers success from a job state or an output folder.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())

WAVE_ID = "cleanslate_2026_10_17_wave1"
RUN_PREFIX = "cs1017"
PREPARED_ROOT = "output/prepared/cleanslate_2026_10_17"
RESULTS_DIR = "results/searches/experiments"

VARIANTS = ("slam_base", "slam_base_lightfixed")
SAMPLERS = ("nautilus", "emcee", "nuts", "smc")
SEEDS = (0, 1, 2)

#: Per-sampler settings, passed verbatim to the leaf. Nautilus is the control: the same
#: live points and batch as the base stage it re-runs.
SETTINGS = {
    "nautilus": ("--live", "150", "--batch", "20"),
    "emcee": ("--walkers", "50", "--steps", "2000"),
    "nuts": ("--chains", "1", "--warmup", "500", "--steps", "1000"),
    "smc": ("--particles", "256", "--steps", "1000"),
}

#: Every wave-1 arm starts cold; compilation is cold (fresh process), cache state is not
#: measured.
COMMON = ("--start", "cold", "--compilation", "cold", "--cache", "unknown")

CONFIG_GRAMMAR = re.compile(
    r"(local|hpc_a100)_(jax_cpu|numba_cpu|jax_gpu)_(dense|sparse)_(fp64|mp)"
)


@dataclass(frozen=True)
class Arm:
    index: int
    variant: str
    sampler: str
    seed: int

    def run_id(self, config: str) -> str:
        return f"{RUN_PREFIX}_{config}_{self.variant}_{self.sampler}_s{self.seed}"

    @property
    def leaf(self) -> str:
        return f"scripts/imaging/rectangular/{self.sampler}.py"


def arms() -> list[Arm]:
    """Every wave-1 arm, in array-index order (seed-major)."""
    out = []
    for seed in SEEDS:
        for variant in VARIANTS:
            for sampler in SAMPLERS:
                out.append(Arm(len(out), variant, sampler, seed))
    return out


def arm(index: int) -> Arm:
    table = arms()
    if not 0 <= index < len(table):
        raise ValueError(f"arm {index} is outside wave {WAVE_ID} (0-{len(table) - 1})")
    return table[index]


def check_config(config: str) -> str:
    if not CONFIG_GRAMMAR.fullmatch(config):
        raise ValueError(f"config {config!r} does not follow the config-name grammar")
    return config


def manifest_path(config: str, base_seed: int, variant: str) -> str:
    """The prepared problem an arm runs, relative to the repo root."""
    return f"{PREPARED_ROOT}/{check_config(config)}/seed_{base_seed}/{variant}/manifest.json"


def problem_id(root: Path, config: str, base_seed: int, variant: str) -> str:
    """The frozen problem's id, read from its manifest (the leaf re-verifies it)."""
    path = root / manifest_path(config, base_seed, variant)
    if not path.is_file():
        raise ValueError(f"prepared manifest missing: {path.relative_to(root)}")
    doc = json.loads(path.read_text())
    if doc.get("schema") != "prepared-inference-problem" or not doc.get("id"):
        raise ValueError(f"not a prepared problem manifest: {path.relative_to(root)}")
    return doc["id"]


def leaf_args(
    root: Path, index: int, config: str, base_seed: int, hardware_id: str, cores: int
) -> list[str]:
    """The exact argument list the arm's leaf is called with."""
    a = arm(index)
    if not hardware_id or cores <= 0:
        raise ValueError("an arm needs a hardware id and a positive core count")
    return [
        "--prepared",
        manifest_path(config, base_seed, a.variant),
        "--problem-id",
        problem_id(root, config, base_seed, a.variant),
        "--run-id",
        a.run_id(config),
        "--seed",
        str(a.seed),
        "--cores",
        str(cores),
        "--hardware-id",
        hardware_id,
        *COMMON,
        *SETTINGS[a.sampler],
    ]


def account(results: Path, config: str) -> list[dict]:
    """Each expected arm's state from committed/pulled rows: row, failed or missing."""
    out = []
    for a in arms():
        run_id = a.run_id(config)
        path = results / f"{run_id}.json"
        entry = dict(index=a.index, run_id=run_id, variant=a.variant, sampler=a.sampler)
        entry["seed"] = a.seed
        if not path.is_file():
            out.append(dict(entry, state="missing", detail="no result row"))
            continue
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            out.append(dict(entry, state="failed", detail=f"unreadable row: {exc}"))
            continue
        expected = dict(sampler=a.sampler, seed=a.seed, config_name=config, variant=a.variant)
        wrong = {k: row.get(k) for k, v in expected.items() if row.get(k) != v}
        status = str(row.get("status"))
        if wrong:
            out.append(dict(entry, state="failed", detail=f"identity mismatch: {wrong}"))
        elif status == "complete" and row.get("completed") is True:
            out.append(dict(entry, state="row", detail=f"problem {row.get('problem_id')}"))
        else:
            out.append(dict(entry, state="failed", detail=status))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="print every expected arm")
    mode.add_argument("--account", action="store_true", help="row / failed / missing per arm")
    mode.add_argument("--arm", type=int, help="resolve one array index")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--samplers",
        help="the submit's sampler order (space separated); refused unless it is this table's",
    )
    parser.add_argument(
        "--field", choices=("sampler", "sampler_index", "variant", "seed", "run_id", "manifest")
    )
    parser.add_argument("--print-args", action="store_true", help="the leaf's arguments")
    parser.add_argument("--print-command", action="store_true", help="the full leaf command")
    parser.add_argument("--hardware-id")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=ROOT / RESULTS_DIR)
    args = parser.parse_args(argv)

    try:
        config = check_config(args.config)
        if args.samplers is not None and tuple(args.samplers.split()) != SAMPLERS:
            raise ValueError(f"submit sampler order {args.samplers.split()} != {list(SAMPLERS)}")
        if args.list:
            print(f"# wave {WAVE_ID}  config {config}  base seed {args.base_seed}")
            for a in arms():
                path = manifest_path(config, args.base_seed, a.variant)
                print(f"{a.index:2d}  {a.run_id(config)}  {' '.join(SETTINGS[a.sampler])}  {path}")
            return 0
        if args.account:
            entries = account(args.results_dir, config)
            for e in entries:
                print(f"{e['index']:2d}  {e['state']:7s}  {e['run_id']}  {e['detail']}")
            counts = {
                s: sum(e["state"] == s for e in entries) for s in ("row", "failed", "missing")
            }
            print(
                f"# wave {WAVE_ID} {config}: {len(entries)} expected, "
                + ", ".join(f"{v} {k}" for k, v in counts.items())
            )
            return 0
        a = arm(args.arm)
        if args.field:
            value = {
                "sampler": a.sampler,
                "sampler_index": SAMPLERS.index(a.sampler),
                "variant": a.variant,
                "seed": a.seed,
                "run_id": a.run_id(config),
                "manifest": manifest_path(config, args.base_seed, a.variant),
            }[args.field]
            print(value)
            return 0
        if args.print_args or args.print_command:
            leaf = leaf_args(ROOT, args.arm, config, args.base_seed, args.hardware_id, args.cores)
            words = (["python3", a.leaf] if args.print_command else []) + leaf
            print(shlex.join(words))
            return 0
        parser.error("--arm needs --field, --print-args or --print-command")
    except ValueError as exc:
        print(f"wave.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
