"""Wave 1 of the HST clean-slate mass-sampler comparison: arm table, resolver, accounting.

PyAutoInsight campaign ``hst-imaging-mass-samplers``, task ``hst_mass_samplers_wave1``.
Phase 2 exports one frozen ``mass_total[1]`` problem per (config, base seed, light mode) to::

    output/autolens/prepared/<config>/seed_<base seed>/<variant>/manifest.json

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
(a row whose status is not complete, or whose identity does not match the arm),
``missing`` (no row) or ``deferred`` (the sampler is set aside for this wave, with the
reason, see ``DEFERRED``). It never infers success from a job state or an output folder.

Attempts. A run id is never reused (``run.py`` refuses an existing one), so re-running an
arm is a new **attempt**: ``--attempt N`` (N > 1) appends ``_a<N>`` to every run id.
``--account`` lists every attempt of every arm it finds (or up to ``--attempt``), and the
arm's live state is its latest complete row, else its latest attempt. ``--sampler-filter
nautilus,emcee`` restricts a submit to a subset: ``--arm`` then indexes the filtered arms
(in wave order), so ``--array=0-<n-1>`` covers exactly them. Attempt 1 of wave 1 (RAL array
412543) lost every Nautilus/Emcee row to a row-writer bug (tuple parameter names,
autolens_inference#33) and ran NUTS/SMC out of A100 memory; attempt 2 re-runs the twelve
Nautilus and Emcee arms with ``--attempt 2 --sampler-filter nautilus,emcee``.
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

# WAVE_ID and RUN_PREFIX name the wave and its run ids, not a folder: attempt 2 must match attempt 1.
WAVE_ID = "cleanslate_2026_10_17_wave1"
RUN_PREFIX = "cs1017"
PREPARED_ROOT = "output/autolens/prepared"
RESULTS_DIR = "results/misc/experiments"

VARIANTS = ("slam_base", "slam_base_light_fixed")
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

#: Samplers set aside for this wave, with the evidence. A deferred arm is never resolved
#: for a run and is accounted ``deferred``, never ``missing``.
DEFERRED = {
    "nuts": "A100 OOM 2026-10-17 (385 GiB NUTS, 48 GiB SMC): NUTS jit_log_l gradient "
    "asked for 385.32 GiB on an 80 GB A100 (RAL array 412543)",
    "smc": "A100 OOM 2026-10-17 (385 GiB NUTS, 48 GiB SMC): SMC 256-particle jit_scan "
    "asked for 48.45 GiB on an 80 GB A100 (RAL array 412543)",
}

ATTEMPT_SUFFIX = re.compile(r"_a([0-9]+)$")

CONFIG_GRAMMAR = re.compile(
    r"(local|hpc_a100|hpc_ral)_(jax_cpu|numba_cpu|jax_gpu)_(dense|sparse)_(fp64|mp)"
)


@dataclass(frozen=True)
class Arm:
    index: int
    variant: str
    sampler: str
    seed: int

    def run_id(self, config: str, attempt: int = 1) -> str:
        base = f"{RUN_PREFIX}_{config}_{self.variant}_{self.sampler}_s{self.seed}"
        return base if check_attempt(attempt) == 1 else f"{base}_a{attempt}"

    @property
    def deferred(self) -> str | None:
        return DEFERRED.get(self.sampler)

    @property
    def leaf(self) -> str:
        return f"scripts/imaging/rectangular/samplers/{self.sampler}.py"


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


def check_attempt(attempt: int) -> int:
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt must be a positive integer, not {attempt!r}")
    return attempt


def sampler_filter(text: str | None) -> tuple[str, ...]:
    """Parse ``nautilus,emcee`` / ``nautilus emcee``; empty or None means every sampler."""
    names = tuple(n for n in re.split(r"[,\s]+", text or "") if n)
    unknown = [n for n in names if n not in SAMPLERS]
    if unknown:
        raise ValueError(f"unknown sampler(s) {unknown} (wave samplers: {list(SAMPLERS)})")
    return names or SAMPLERS


def selection(samplers: tuple[str, ...] = SAMPLERS) -> list[Arm]:
    """The arms a filtered submit runs, in wave order; array index = position here."""
    return [a for a in arms() if a.sampler in samplers]


def select(position: int, samplers: tuple[str, ...] = SAMPLERS) -> Arm:
    """The arm at ``position`` of the filtered selection, refusing a deferred sampler."""
    table = selection(samplers)
    if not 0 <= position < len(table):
        raise ValueError(
            f"arm {position} is outside the {len(table)} selected arms of wave {WAVE_ID} "
            f"(samplers {list(samplers)})"
        )
    a = table[position]
    if a.deferred:
        raise ValueError(f"arm {a.index} ({a.sampler}) is deferred in {WAVE_ID}: {a.deferred}")
    return a


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
    root: Path,
    index: int,
    config: str,
    base_seed: int,
    hardware_id: str,
    cores: int,
    attempt: int = 1,
) -> list[str]:
    """The exact argument list the arm's leaf (wave index ``index``) is called with."""
    a = arm(index)
    if not hardware_id or cores <= 0:
        raise ValueError("an arm needs a hardware id and a positive core count")
    return [
        "--prepared",
        manifest_path(config, base_seed, a.variant),
        "--problem-id",
        problem_id(root, config, base_seed, a.variant),
        "--run-id",
        a.run_id(config, attempt),
        "--seed",
        str(a.seed),
        "--cores",
        str(cores),
        "--hardware-id",
        hardware_id,
        *COMMON,
        *SETTINGS[a.sampler],
    ]


def row_state(path: Path, a: Arm, config: str) -> tuple[str, str]:
    """``row`` / ``failed`` / ``missing`` for one attempt's row file."""
    if not path.is_file():
        return "missing", "no result row"
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return "failed", f"unreadable row: {exc}"
    expected = dict(sampler=a.sampler, seed=a.seed, config_name=config, variant=a.variant)
    wrong = {k: row.get(k) for k, v in expected.items() if row.get(k) != v}
    status = str(row.get("status"))
    if wrong:
        return "failed", f"identity mismatch: {wrong}"
    if status == "complete" and row.get("completed") is True:
        return "row", f"problem {row.get('problem_id')}"
    return "failed", status


def attempts_on_disk(results: Path, a: Arm, config: str) -> int:
    """The highest attempt of this arm with a row file (1 when there is none)."""
    base = a.run_id(config)
    found = [1]
    for path in results.glob(f"{base}_a*.json") if results.is_dir() else ():
        match = ATTEMPT_SUFFIX.search(path.stem)
        if match and path.stem == f"{base}_a{match.group(1)}":
            found.append(int(match.group(1)))
    return max(found)


def account(results: Path, config: str, attempt: int | None = None) -> list[dict]:
    """Each expected arm's state from committed/pulled rows, every attempt listed.

    ``attempt`` is the latest attempt expected (rows of attempts 1..attempt are looked
    for); ``None`` takes the highest found on disk per arm. The arm's ``state`` is
    ``deferred`` for a deferred sampler, else its latest ``row`` attempt, else the state
    of its latest attempt (``failed`` or ``missing``).
    """
    out = []
    for a in arms():
        last = (
            check_attempt(attempt) if attempt is not None else attempts_on_disk(results, a, config)
        )
        history = []
        for n in range(1, last + 1):
            run_id = a.run_id(config, n)
            state, detail = row_state(results / f"{run_id}.json", a, config)
            history.append(dict(attempt=n, run_id=run_id, state=state, detail=detail))
        live = next((h for h in reversed(history) if h["state"] == "row"), history[-1])
        entry = dict(index=a.index, variant=a.variant, sampler=a.sampler, seed=a.seed)
        entry.update(run_id=live["run_id"], attempt=live["attempt"], attempts=history)
        if a.deferred:
            entry.update(state="deferred", detail=a.deferred)
        else:
            entry.update(state=live["state"], detail=live["detail"])
        out.append(entry)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="print every expected arm")
    mode.add_argument(
        "--account", action="store_true", help="row / failed / missing / deferred per arm"
    )
    mode.add_argument(
        "--arm", type=int, help="resolve one array index (into --sampler-filter's arms)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--samplers",
        help="the submit's sampler order (space separated); refused unless it is this table's",
    )
    parser.add_argument(
        "--sampler-filter",
        help="run only these samplers' arms (comma or space separated); empty = all",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        help="attempt number: run ids gain _a<N> for N > 1 (default 1; --account: highest found)",
    )
    parser.add_argument(
        "--field",
        choices=("sampler", "sampler_index", "variant", "seed", "run_id", "manifest", "index"),
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
        samplers = sampler_filter(args.sampler_filter)
        attempt = check_attempt(args.attempt) if args.attempt is not None else None
        if args.list:
            print(
                f"# wave {WAVE_ID}  config {config}  base seed {args.base_seed}  "
                f"attempt {attempt or 1}  samplers {','.join(samplers)}"
            )
            for position, a in enumerate(selection(samplers)):
                path = manifest_path(config, args.base_seed, a.variant)
                note = f"  DEFERRED: {a.deferred}" if a.deferred else ""
                print(
                    f"{position:2d}  arm {a.index:2d}  {a.run_id(config, attempt or 1)}  "
                    f"{' '.join(SETTINGS[a.sampler])}  {path}{note}"
                )
            return 0
        if args.account:
            entries = [
                e for e in account(args.results_dir, config, attempt) if e["sampler"] in samplers
            ]
            for e in entries:
                print(f"{e['index']:2d}  {e['state']:8s}  {e['run_id']}  {e['detail']}")
                for h in e["attempts"]:
                    print(
                        f"      attempt {h['attempt']}  {h['state']:7s}  {h['run_id']}  {h['detail']}"
                    )
            counts = {
                s: sum(e["state"] == s for e in entries)
                for s in ("row", "failed", "missing", "deferred")
            }
            print(
                f"# wave {WAVE_ID} {config}: {len(entries)} expected, "
                + ", ".join(f"{v} {k}" for k, v in counts.items())
            )
            return 0
        a = select(args.arm, samplers)
        if args.field:
            value = {
                "sampler": a.sampler,
                "sampler_index": SAMPLERS.index(a.sampler),
                "variant": a.variant,
                "seed": a.seed,
                "run_id": a.run_id(config, attempt or 1),
                "manifest": manifest_path(config, args.base_seed, a.variant),
                "index": a.index,
            }[args.field]
            print(value)
            return 0
        if args.print_args or args.print_command:
            leaf = leaf_args(
                ROOT, a.index, config, args.base_seed, args.hardware_id, args.cores, attempt or 1
            )
            words = (["python3", a.leaf] if args.print_command else []) + leaf
            print(shlex.join(words))
            return 0
        parser.error("--arm needs --field, --print-args or --print-command")
    except ValueError as exc:
        print(f"wave.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
