"""Likelihood-lifecycle phase 4: the inference wall-time validation legs and their SLURM submits.

autolens_inference#51 (4a). Phase 4 checks the per-evaluation runtime estimates against real
Nautilus SLaM wall times on RAL, and measures CPU parallel efficiency. The human ruled on
2026-11-02 that the 56 rows already in ``dashboard/walltime_validation.json`` (release pin
2026.10.17.1, imaging/hst only) are adopted as day-zero evidence ("day zero may span
versions"), and that each likelihood gets one reference instrument; and that phase 4 runs
inference only under the Optimal setups ("lets not do these inference runs on Secondary
setups", 2026-11-02 — the Secondary CPU legs 4a generated are dropped). This module lists only
the GAPS around those rows, and renders one submit per leg::

    python scripts/misc/campaigns/validation_p4.py --list    # print the manifest
    python scripts/misc/campaigns/validation_p4.py --write   # (re)render the submits
    python scripts/misc/campaigns/validation_p4.py --check   # exit 1 if a submit is stale

The legs:

- ``imaging/delaunay/hst``, the Optimal GPU setup ``jax_gpu_dense_fp64`` on an A100, seeds
  0-1, ``--lens-light both`` (the default and light_fixed model variants from one chain;
  light_fixed is run only under an Optimal setup);
- ``imaging/delaunay/hst``, the Optimal CPU setup ``numba_cpu_sparse_fp64`` on a RAL CPU node,
  the default model variant the adopted light_fixed-only rows lack (human, 2026-11-02: "yes
  get the default legs too");
- a cores sweep of ``source_pix[1]`` alone on ``imaging/rectangular/hst`` under the Optimal
  CPU setup ``numba_cpu_sparse_fp64``, seed 0, at :data:`SWEEP_CORES` cores, one submit per
  core count (``--cpus-per-task`` cannot vary inside an array).

A chain leg is kept only for the seeds whose (likelihood, instrument, setup_id, host_class,
model_variant, seed) has no adopted row, so the generator never re-runs one. A CPU-backend
adopted row still carrying the ``hpc_a100`` config prefix ran on the ``ral`` partition (its
submit says so) and counts as ``hpc_ral_cpu`` here. A sweep leg is told apart by its results
tag ``cores_<n>``, which the exporter folds into ``sampler``.

Every leg runs on the libraries at the day-zero tag :data:`PIN`, the private checkout
:data:`PIN_BASE` (five repos at that tag, the layout ``activate.sh`` expects of a
``PYAUTO_LIB_BASE``; ``hpc/README.md``, "Running against a pinned release"). Each leg also
has its own PyAutoFit output root under :data:`OUTPUT_BASE`, never the shared
``output/autolens`` tree:

- PyAutoFit's output path omits the pin and the core count, so a leg in the shared tree
  would resume an earlier pin's partial fit (or a sweep leg the previous core count's);
- ``--stages-from`` resolves the loaded stage under the run's own output root, so the
  shared ``source_lp[1]`` is copied there from the jax_cpu dense chain before the runner
  starts. Each leg then loads its own copy, and two legs never race on one stage's
  ``search_internal/`` (412624_0, 2026-10-19).

Walls (``WALL-BASIS``, ``scripts/misc/wall/README.md``). Per stage the evaluations are the
most any adopted row of the same likelihood and model variant took (any setup, any seed),
and the per-evaluation cost is Pulse's per-call time of the matching runtime record in
``catalogue/pulse_lens_runtime_reference.json`` (the exporter's own pairing,
``match_profiling``). A leg whose cell and device have an adopted sibling is
``measured-wall``: each stage takes the larger of that derivation and the sibling's measured
wall, and the leg carries :data:`MEASURED_HEADROOM`. A leg without one is ``unmeasured``
(``probe-first: yes``) and offers the derivation as its wall at the gate's 3x floor, unless
3x would exceed :data:`MAX_TIME_S`; then it offers no wall and asks for that cap as a
containment. Refreshing either input changes the walls: rerun ``--write``.

Nothing here submits anything; a run needs the human's go.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = next(p for p in _HERE.parents if (p / "ruff.toml").exists())
for _path in (ROOT, ROOT / "scripts" / "misc", ROOT / "scripts" / "misc" / "tooling"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import export_inference_summary as summary  # noqa: E402  (stdlib-only)

GENERATOR = "scripts/misc/campaigns/validation_p4.py"
ISSUE = "autolens_inference#51"
#: The day-zero tag every new phase-4 leg runs on, and its private checkout on RAL.
PIN = "2026.10.25.1"
PIN_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
LIBRARIES = ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens")
AP_ROOT = "/mnt/ral/jnightin/inference/autolens_inference"
MAIL_USER = "james.w.nightingale@durham.ac.uk"
#: The adopted rows (2026.10.17.1) the gaps are measured against.
ADOPTED = "dashboard/walltime_validation.json"
#: Every leg's own PyAutoFit output root sits under here (relative to AP_ROOT).
OUTPUT_BASE = "output/p4_validation"
#: The chain whose completed ``source_lp[1]`` the sharing legs load (human decision
#: 2026-10-17; every clean-slate numba and Delaunay leg shares it).
SHARED_CONFIG = "hpc_a100_jax_cpu_dense_fp64"
INSTRUMENT = "hst"
SEEDS = (0, 1)
SWEEP_CORES = (1, 2, 4, 8, 16, 32)
SWEEP_SEED = 0
#: The chain's stages, in order (``_runner.STAGE_SEARCH_NAMES``).
STAGES = ("source_lp[1]", "source_pix[1]", "source_pix[2]", "light[1]", "mass_total[1]")
STAGE_FLAGS = {"source_pix[1]": "source_pix_1"}
#: The run's source pixels, which picks the profiling size bin (28x28 rectangular mesh,
#: 1250 Hilbert vertices).
SOURCE_PIXELS = {"rectangular": 784, "delaunay": 1250}
RUN_VARIANT = {"rectangular": "slam_base", "delaunay": "slam_delaunay"}
#: The wall gate's cell of each likelihood's leaf (``check_submits.LEGACY_CELLS``).
CELL = {"rectangular": "imaging/slam/hst", "delaunay": "imaging/slam/hst_delaunay"}
MEASURED_HEADROOM = 1.5
UNMEASURED_HEADROOM = 3.0
ROUND_S = 6 * 3600
#: The longest --time asked of RAL by any submit in this repo (five days).
MAX_TIME_S = 5 * 86400
CHAIN_CORES = 8
#: --mem (GB): the clean-slate requests for the same cells (Delaunay 96, rectangular
#: jax_cpu sparse 64); a sweep leg asks 4 GB per Nautilus worker, at least 64.
CHAIN_MEM_GB = {"delaunay": 96, "rectangular": 64}


@dataclass(frozen=True)
class Target:
    """One (likelihood, setup) the phase wants on its reference host, and the variants."""

    likelihood: str
    setup_id: str
    role: str
    model_variants: tuple[str, ...]
    share_source_lp: bool

    def __post_init__(self):
        if "light_fixed" in self.model_variants and not self.role.startswith("Optimal"):
            raise ValueError(f"{self}: light_fixed is run only under an Optimal setup")


#: The chain gaps, Optimal setups only (human, 2026-11-02). The Delaunay legs share
#: ``source_lp[1]``, as every Delaunay leg has.
TARGETS = (
    Target("delaunay", "jax_gpu_dense_fp64", "Optimal GPU", ("default", "light_fixed"), True),
    Target("delaunay", "numba_cpu_sparse_fp64", "Optimal CPU", ("default", "light_fixed"), True),
)
SWEEP = Target("rectangular", "numba_cpu_sparse_fp64", "Optimal CPU", ("default",), True)


def host_class_of(setup_id: str) -> str:
    return "hpc_a100" if setup_id.split("_")[1] == "gpu" else "hpc_ral_cpu"


@dataclass(frozen=True)
class Leg:
    """One submit: a chain (an array over seeds) or one core count of the sweep."""

    likelihood: str
    setup_id: str
    role: str
    model_variants: tuple[str, ...]
    seeds: tuple[int, ...]
    share_source_lp: bool
    cores: int = CHAIN_CORES
    stage: str | None = None
    results_tag: str | None = None

    @property
    def host_class(self) -> str:
        return host_class_of(self.setup_id)

    @property
    def on_gpu(self) -> bool:
        return self.host_class == "hpc_a100"

    @property
    def backend(self) -> str:
        return "_".join(self.setup_id.split("_")[:2])

    @property
    def inversion(self) -> str:
        return self.setup_id.split("_")[2]

    @property
    def config_name(self) -> str:
        return f"{'hpc_a100' if self.on_gpu else 'hpc_ral'}_{self.setup_id}"

    @property
    def device(self) -> str:
        return "a100" if self.on_gpu else f"ral_{self.backend}"

    @property
    def cell(self) -> str:
        return CELL[self.likelihood]

    @property
    def script(self) -> str:
        return f"scripts/imaging/{self.likelihood}/validation/baseline_slam.py"

    @property
    def lens_light(self) -> str:
        return "both" if "light_fixed" in self.model_variants else "linear"

    @property
    def short(self) -> str:
        return self.setup_id.removesuffix("_fp64")

    @property
    def name(self) -> str:
        if self.results_tag:
            return f"submit_p4_cores_{self.likelihood}_{INSTRUMENT}_{self.short}_{self.results_tag}"
        seeds = "" if self.seeds == SEEDS else "_seed_" + "_".join(map(str, self.seeds))
        return f"submit_p4_validation_{self.likelihood}_{INSTRUMENT}_{self.short}{seeds}"

    @property
    def path(self) -> str:
        return f"hpc/{'batch_gpu' if self.on_gpu else 'batch_cpu'}/{self.name}"

    @property
    def output_dir(self) -> str:
        return f"{OUTPUT_BASE}/{self.name.removeprefix('submit_p4_')}"

    @property
    def mem_gb(self) -> int:
        if self.results_tag:
            return max(64, 4 * self.cores)
        return CHAIN_MEM_GB[self.likelihood]

    def sampler(self, stage: str) -> str:
        suffix = f":{self.results_tag}" if self.results_tag else ""
        return f"nautilus:{stage}{suffix}"

    def fitted_stages(self, model_variant: str) -> tuple[str, ...]:
        """The stages this leg fits for one model variant (a light_fixed run adds only its
        own ``mass_total[1]``; stages 1-4 are the default chain's)."""
        if self.stage:
            return (self.stage,)
        if model_variant != "default":
            return ("mass_total[1]",)
        return STAGES[1:] if self.share_source_lp else STAGES

    def keys(self) -> list[tuple]:
        """Every walltime-validation key this leg writes, as (likelihood, instrument,
        setup_id, host_class, model_variant, seed, sampler)."""
        return [
            (
                self.likelihood,
                INSTRUMENT,
                self.setup_id,
                self.host_class,
                variant,
                seed,
                self.sampler(stage),
            )
            for variant in self.model_variants
            for seed in self.seeds
            for stage in self.fitted_stages(variant)
        ]


# ---------------------------------------------------------------------------
# The adopted rows
# ---------------------------------------------------------------------------


def adopted_records(root: Path = ROOT) -> list[dict]:
    return json.loads((root / ADOPTED).read_text())["records"]


def host_of(record: dict) -> str:
    """A record's host, with a CPU row's kept ``hpc_a100`` prefix read as the RAL CPU node
    its submit ran it on (``--partition=ral``)."""
    if record["host_class"] == "hpc_a100" and "_cpu_" in record["setup_id"]:
        return "hpc_ral_cpu"
    return record["host_class"]


def _is_sweep(record: dict) -> bool:
    return ":cores_" in record["sampler"]


def chain_key(record: dict, host: str | None = None) -> tuple:
    return (
        record["likelihood"]["variant"],
        record["instrument"],
        record["setup_id"],
        host or host_of(record),
        record["model_variant"],
        record["seed"],
    )


def planned_legs(records: list[dict] | None = None) -> list[Leg]:
    """Every phase-4 leg the adopted rows do not already cover."""
    records = adopted_records() if records is None else records
    have = {chain_key(r) for r in records if not _is_sweep(r)}
    swept = {(chain_key(r), r["sampler"]) for r in records if _is_sweep(r)}
    legs = []
    for t in TARGETS:
        host = host_class_of(t.setup_id)
        groups: dict[tuple[str, ...], list[int]] = {}
        for seed in SEEDS:
            # A chain re-run re-writes the default stages, so a seed whose default exists
            # is never re-run (a missing light_fixed alone cannot be filled without one).
            if (t.likelihood, INSTRUMENT, t.setup_id, host, "default", seed) in have:
                continue
            missing = tuple(
                v
                for v in t.model_variants
                if (t.likelihood, INSTRUMENT, t.setup_id, host, v, seed) not in have
            )
            groups.setdefault(missing, []).append(seed)
        for variants, seeds in groups.items():
            legs.append(
                Leg(t.likelihood, t.setup_id, t.role, variants, tuple(seeds), t.share_source_lp)
            )
    stage = "source_pix[1]"
    for n in SWEEP_CORES:
        leg = Leg(
            SWEEP.likelihood,
            SWEEP.setup_id,
            SWEEP.role,
            SWEEP.model_variants,
            (SWEEP_SEED,),
            SWEEP.share_source_lp,
            cores=n,
            stage=stage,
            results_tag=f"cores_{n}",
        )
        if not any((key[:6], key[6]) in swept for key in leg.keys()):
            legs.append(leg)
    return legs


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageWall:
    model_variant: str
    stage: str
    evals: int
    evals_ref: str
    derived_s: float | None
    per_call_ref: str | None
    measured_s: float | None
    measured_ref: str | None

    @property
    def wall_s(self) -> float:
        return max(w for w in (self.derived_s, self.measured_s) if w is not None)


def references(root: Path = ROOT) -> list[dict]:
    return json.loads((root / summary.PULSE_REFERENCE).read_text())["records"]


def per_call(leg: Leg, refs: list[dict]) -> tuple[float, str]:
    """Pulse's per-call time for the leg's likelihood and setup (the exporter's pairing).
    Profiling has no light_fixed record, so every stage is priced at the default model's."""
    item = {
        "likelihood": {"dataset_type": "imaging", "variant": leg.likelihood},
        "instrument": INSTRUMENT,
        "setup_id": leg.setup_id,
        "host_class": leg.host_class,
        "model_variant": "default",
    }
    ref, reason = summary.match_profiling(item, refs, SOURCE_PIXELS[leg.likelihood])
    if ref is None:
        raise SystemExit(f"{leg.name}: {reason}")
    return ref["per_call_s"], ref["id"]


def stage_walls(leg: Leg, records: list[dict], refs: list[dict]) -> list[StageWall]:
    """One estimate per (model variant, stage) the leg fits."""
    rows = [r for r in records if not _is_sweep(r) and r["likelihood"]["variant"] == leg.likelihood]
    call_s, call_ref = per_call(leg, refs)
    # numba's per-call time is one thread; Nautilus forks `cores` workers. A JAX per-call
    # (vmap.per_call) already prices one batched evaluation on the whole device.
    parallel = leg.cores if leg.backend == "numba_cpu" else 1
    out = []
    for variant in leg.model_variants:
        for stage in leg.fitted_stages(variant):
            same = [r for r in rows if r["model_variant"] == variant and r["stage"] == stage]
            counted = [r for r in same if r["likelihood_evals"]]
            if not counted:
                raise SystemExit(f"{leg.name}: no adopted row counts the evaluations of {stage}")
            most = max(counted, key=lambda r: (r["likelihood_evals"], r["id"]))
            derived = derived_ref = None
            if stage != "source_lp[1]":  # no inversion: no profiling likelihood prices it
                derived = call_s * most["likelihood_evals"] / parallel
                derived_ref = call_ref
            measured = measured_ref = None
            siblings = [
                r
                for r in same
                if r["setup_id"] == leg.setup_id
                and host_of(r) == leg.host_class
                and isinstance(r["wall_s"], (int, float))
            ]
            if siblings:
                sib = max(siblings, key=lambda r: (r["wall_s"], r["id"]))
                # A sibling at c cores bounds an n-core run from above at c/n of its wall
                # when n < c (efficiency <= 1), and at its own wall when n >= c.
                scale = max(1.0, (sib["cores"] or leg.cores) / leg.cores)
                measured, measured_ref = sib["wall_s"] * scale, sib["id"]
            if derived is None and measured is None:
                raise SystemExit(f"{leg.name}: nothing prices {variant} {stage}")
            out.append(
                StageWall(
                    variant,
                    stage,
                    most["likelihood_evals"],
                    most["id"],
                    derived,
                    derived_ref,
                    measured,
                    measured_ref,
                )
            )
    return out


@dataclass(frozen=True)
class Budget:
    source: str
    wall_s: int | None
    estimate_s: int
    time_s: int
    headroom: float
    refs: tuple[str, ...]


def budget(leg: Leg, records: list[dict], refs: list[dict]) -> tuple[Budget, list[StageWall]]:
    walls = stage_walls(leg, records, refs)
    estimate = math.ceil(sum(w.wall_s for w in walls))
    measured = all(w.measured_s is not None for w in walls)
    if measured:
        jobs = sorted({_job_of(w.measured_ref, records) for w in walls})
        time_s = _round_up(estimate * MEASURED_HEADROOM)
        if time_s > MAX_TIME_S:
            raise SystemExit(f"{leg.name}: a measured wall of {estimate} s exceeds the cap")
        return Budget(
            "measured-wall", estimate, estimate, time_s, MEASURED_HEADROOM, tuple(jobs)
        ), walls
    time_s = _round_up(estimate * UNMEASURED_HEADROOM)
    if time_s > MAX_TIME_S:
        return Budget("unmeasured", None, estimate, MAX_TIME_S, UNMEASURED_HEADROOM, ()), walls
    return Budget("unmeasured", estimate, estimate, time_s, UNMEASURED_HEADROOM, ()), walls


def _job_of(record_id: str, records: list[dict]) -> str:
    return next(r["slurm_job_id"] for r in records if r["id"] == record_id)


def _round_up(seconds: float) -> int:
    return max(ROUND_S, math.ceil(seconds / ROUND_S) * ROUND_S)


def slurm_time(seconds: int) -> str:
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{clock}" if days else clock


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _para(text: str, indent: str = "# ", hang: str | None = None) -> list[str]:
    """One comment paragraph, wrapped to the header width."""
    return textwrap.wrap(
        text,
        width=94,
        initial_indent=indent,
        subsequent_indent=hang or indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wall_block(leg: Leg, b: Budget, walls: list[StageWall]) -> list[str]:
    pricing = (
        f"Pulse's single-thread per-call time over {leg.cores} Nautilus worker(s)"
        if leg.backend == "numba_cpu"
        else "Pulse's per-call time of one batched JAX evaluation"
    )
    prose = (
        f"This submit runs ONE cell, `{leg.cell}`. Per stage, the evaluations are the most any "
        f"adopted row of this likelihood and model variant took, priced at {pricing} "
        "(catalogue/pulse_lens_runtime_reference.json)."
    )
    if b.source == "measured-wall":
        prose += (
            " An adopted sibling on this cell and device measured each stage, and each stage takes "
            "the larger of the two"
            + (
                f": the 8-core sibling's wall x 8/{leg.cores} below 8 cores (efficiency <= 1), "
                "its own wall at or above."
                if leg.results_tag
                else "."
            )
        )
    else:
        prose += (
            " No adopted row measured this cell on this device, so the row is `unmeasured` and "
            "the first arm IS the probe: read each stage's `wall_s / likelihood_evals` from its row."
        )
    lines = [
        "# WALL-BASIS: — one row per cell this submit runs. See ../../scripts/misc/wall/README.md.",
        *_para(prose, "#   "),
    ]
    for w in walls:
        parts = [f"{w.evals} evals ({w.evals_ref})"]
        if w.derived_s is not None:
            parts.append(f"derived {w.derived_s:.0f} s ({w.per_call_ref})")
        if w.measured_s is not None:
            parts.append(f"measured {w.measured_s:.0f} s ({w.measured_ref})")
        lines += _para(f"- {w.model_variant} {w.stage} = " + "; ".join(parts), "#   ", "#     ")
    if b.wall_s is None:
        closing = (
            f"The estimate is {b.estimate_s} s ({b.estimate_s / 3600:.0f} h); 3x of it exceeds "
            f"the {slurm_time(MAX_TIME_S)} cap, so no wall is offered and --time is that cap, "
            "a CONTAINMENT, not a derived budget."
        )
    else:
        closing = (
            f"The estimate is {b.estimate_s} s ({b.estimate_s / 3600:.1f} h); --time clears "
            f"{b.headroom:g}x of it."
        )
    lines += _para(closing, "#   ")
    evals = sum(w.evals for w in walls)
    row = f"#   lanes: 1  steps: {evals}  source: {b.source}"
    if b.source == "measured-wall":
        row += f"  wall: {b.wall_s}  ref: {','.join(b.refs)}  headroom: {b.headroom:g}"
    else:
        row += "  probe-first: yes"
        if b.wall_s is not None:
            row += f"  wall: {b.wall_s}"
    lines += [f"#   cell: {leg.cell}  device: {leg.device}  precision: fp64", row]
    return lines


def _describe(leg: Leg) -> list[str]:
    seeds = ", ".join(map(str, leg.seeds))
    if leg.results_tag:
        return _para(
            f"Likelihood-lifecycle phase 4 ({ISSUE}): the RAL cores sweep at {leg.cores} "
            f"core(s). `imaging/{leg.likelihood}/{INSTRUMENT}` under the {leg.role} setup "
            f"`{leg.setup_id}` (`{leg.config_name}`), seed {seeds}, ONE pixelized stage: "
            f"`--stages source_pix_1` fits `{leg.stage}` and stops, with `source_lp[1]` loaded "
            f"from `{SHARED_CONFIG}` (`--stages-from`), the stage the adopted 8-core row loaded "
            f"too. The row carries `--results-tag {leg.results_tag}`, which the exporter folds "
            "into `sampler` and computes `parallel_efficiency` from, against the 1-core leg. "
            "`--cpus-per-task` IS the core count (`--cores $SLURM_CPUS_PER_TASK`), one submit "
            "per count because an array cannot vary it. No `--export-prepared`: the leg never "
            "reaches mass_total[1]."
        )
    shared = (
        f"with `source_lp[1]` loaded from `{SHARED_CONFIG}` (`--stages-from`)"
        if leg.share_source_lp
        else "every stage fitted here, as the adopted sibling's was"
    )
    return _para(
        f"Likelihood-lifecycle phase 4 ({ISSUE}): a wall-time validation gap leg. "
        f"`imaging/{leg.likelihood}/{INSTRUMENT}` under the {leg.role} setup `{leg.setup_id}` "
        f"(`{leg.config_name}`), seed(s) {seeds} as the array, model variant(s) "
        f"{', '.join(leg.model_variants)} (`--lens-light {leg.lens_light}`). The full SLaM "
        f"chain, {shared}."
    )


def render(leg: Leg, records: list[dict], refs: list[dict]) -> str:
    b, walls = budget(leg, records, refs)
    array = len(leg.seeds) > 1 or not leg.results_tag
    seed_expr = "$SLURM_ARRAY_TASK_ID" if array else str(leg.seeds[0])
    lines = [
        "#!/bin/bash -l",
        "#",
        *_describe(leg),
        "#",
        *_para(
            f"GENERATED by {GENERATOR} --write; edit the generator, not this file "
            "(scripts/misc/test/test_validation_p4.py fails on a hand edit). A run needs the "
            "human's go."
        ),
        "#",
    ]
    if not leg.on_gpu:
        lines += [
            "# `--partition=ral` ONLY. CPU jobs never go on `gpu`. There is NO `cpu` partition on RAL.",
            "#",
        ]
    lines += _para(
        f"LIBRARIES. The day-zero tag {PIN}. PYAUTO_LIB_BASE is the private checkout of the "
        f"five libraries at that tag, {PIN_BASE}, in the layout activate.sh expects of a pin "
        'directory, so only PYTHONPATH moves (hpc/README.md, "Running against a pinned '
        'release"). The job refuses to run (exit 5) if a checkout there is not at the tag.'
    )
    output = (
        f"OUTPUT. This leg's own PyAutoFit output root, `{leg.output_dir}/`: the output path "
        "omits the pin and the core count, so the shared tree would resume another run's fit."
    )
    if leg.share_source_lp:
        output += (
            " `--stages-from` resolves the loaded stage under that root, so the job waits for "
            f"the `{SHARED_CONFIG}` chain's `source_lp[1]` and copies it in once; the leg then "
            "loads its own copy and never races another job on one stage's `search_internal/`. "
            "If the day-zero libraries hash the copy to another identifier, the runner exits 3."
        )
    lines += ["#", *_para(output)]
    lines += ["#", *_wall_block(leg, b, walls), "#"]
    sbatch = [
        f"#SBATCH -J {leg.name.removeprefix('submit_')}",
        f"#SBATCH --partition={'gpu' if leg.on_gpu else 'ral'}",
    ]
    if leg.on_gpu:
        sbatch.append("#SBATCH --gres=gpu:1")
    sbatch += [
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={leg.cores}",
        f"#SBATCH --mem={leg.mem_gb}gb",
        f"#SBATCH --time={slurm_time(b.time_s)}",
    ]
    if array:
        seeds = leg.seeds
        contiguous = len(seeds) > 1 and list(seeds) == list(range(seeds[0], seeds[-1] + 1))
        spec = f"{seeds[0]}-{seeds[-1]}" if contiguous else ",".join(map(str, seeds))
        sbatch.append(f"#SBATCH --array={spec}")
    sbatch += [
        "#SBATCH -o output/output.%A_%a.out" if array else "#SBATCH -o output/output.%j.out",
        "#SBATCH -e error/error.%A_%a.err" if array else "#SBATCH -e error/error.%j.err",
        "#SBATCH --mail-type=END,FAIL",
        f"#SBATCH --mail-user={MAIL_USER}",
    ]
    lines += sbatch
    lines += [
        "",
        f"export AP_ROOT={AP_ROOT}",
        "",
        "# The release pin: exported BEFORE activate.sh, which reads PYAUTO_LIB_BASE.",
        f"export PYAUTO_LIB_BASE={PIN_BASE}",
        f"export PYAUTO_RELEASE_PIN={PIN}",
        "source $AP_ROOT/activate.sh",
        "",
        f"for repo in {' '.join(LIBRARIES)}; do",
        '    tag=$(git -C "$PYAUTO_LIB_BASE/$repo" describe --tags --exact-match HEAD 2>/dev/null || echo none)',
        '    if [ "$tag" != "$PYAUTO_RELEASE_PIN" ]; then',
        '        echo "ERROR: $PYAUTO_LIB_BASE/$repo is at $tag, not $PYAUTO_RELEASE_PIN. Refusing to run." >&2',
        "        exit 5",
        "    fi",
        "done",
        "",
    ]
    if leg.on_gpu:
        lines += [
            "# JAX_PLATFORMS and JAX_ENABLE_X64 are exported EXPLICITLY: without the first this job can",
            '# silently fall back to CPU and report a "GPU" wall clock; without the second an fp64 config',
            '# runs in fp32. The driver also asserts jax.default_backend() == "gpu" and exits 2.',
            "export JAX_PLATFORM_NAME=cuda",
            "export JAX_PLATFORMS=cuda,cpu",
            "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "export JAX_ENABLE_X64=True",
            "",
            "# NPROC throttles JAX's CPU thread pool. Unset, this job takes the whole node.",
            "export NPROC=$SLURM_CPUS_PER_TASK",
            "",
            "# Proof of what the job actually got.",
            "nvidia-smi",
        ]
    else:
        lines += [
            "export JAX_PLATFORMS=cpu",
            "export JAX_ENABLE_X64=True",
            "",
            "# NPROC throttles JAX's CPU thread pool. Unset, this job takes the whole node.",
            "export NPROC=$SLURM_CPUS_PER_TASK",
        ]
        if leg.backend == "numba_cpu":
            lines += [
                "",
                "# PYAUTO_DISABLE_JAX is the process-wide belt to use_jax=False's braces.",
                "export PYAUTO_DISABLE_JAX=1",
                "",
                "# One BLAS / OpenMP thread per process (autolens_inference#38): Nautilus forks",
                "# --cores workers, and unpinned pools oversubscribe the node. Exported before",
                "# python3, because numpy/scipy read them at import.",
                "export OPENBLAS_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export OMP_NUM_THREADS=1",
                "export VECLIB_MAXIMUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
            ]
    lines += [
        "",
        f"SEED={seed_expr}",
        f"OUTPUT_ROOT=$AP_ROOT/{leg.output_dir}",
    ]
    shared_prefix = f"autolens/imaging/{INSTRUMENT}/slam/slam_base/{SHARED_CONFIG}/seed_$SEED"
    if leg.share_source_lp:
        lines += [
            f"SHARED_SEED_DIR=$AP_ROOT/output/{shared_prefix}",
            f"LOCAL_SHARED_DIR=$OUTPUT_ROOT/{shared_prefix}",
        ]
    lines += [
        "",
        'echo "=========================================="',
        "date",
        f'echo "Cell:       {leg.cell}"',
        f'echo "Instrument: {INSTRUMENT}"',
        f'echo "Config:     {leg.config_name}"',
        f'echo "Light:      {leg.lens_light} ({", ".join(leg.model_variants)})"',
    ]
    if leg.results_tag:
        lines.append(f'echo "Sweep:      {leg.stage} only, {leg.results_tag}"')
    if leg.share_source_lp:
        lines.append(f'echo "Shared:     source_lp[1] from {SHARED_CONFIG} ($SHARED_SEED_DIR)"')
    lines += [
        'echo "Pin:        $PYAUTO_RELEASE_PIN ($PYAUTO_LIB_BASE)"',
        'echo "Output:     $OUTPUT_ROOT"',
        'echo "Seed:       $SEED"',
        "",
        "cd $AP_ROOT",
        "",
    ]
    if leg.share_source_lp:
        lines += [
            "# In-job wait: 5-minute polls, 8 h cap, exit 4 on timeout. Then this leg's own copy.",
            'bash hpc/wait_for_stage.sh "$SHARED_SEED_DIR" source_lp 300 28800',
            'if [ ! -d "$LOCAL_SHARED_DIR/source_lp[1]" ]; then',
            '    mkdir -p "$LOCAL_SHARED_DIR"',
            '    cp -r "$SHARED_SEED_DIR/source_lp[1]" "$LOCAL_SHARED_DIR/"',
            "fi",
            "",
        ]
    flags = [
        f"--instrument {INSTRUMENT}",
        f"--backend {leg.backend}",
        f"--inversion {leg.inversion}",
    ]
    if leg.likelihood == "delaunay":
        flags.append("--mesh-pixels 1250")
    flags += [
        "--seed $SEED",
        f"--config-name {leg.config_name}",
        f"--lens-light {leg.lens_light}",
    ]
    if leg.stage:
        flags.append(f"--stages {STAGE_FLAGS[leg.stage]}")
    if leg.share_source_lp:
        flags.append(f"--stages-from {SHARED_CONFIG}:source_lp")
    if leg.results_tag:
        flags.append(f"--results-tag {leg.results_tag}")
    flags.append("--output-dir $OUTPUT_ROOT")
    if not leg.stage:
        # A single-mode run exports into the directory given; a `both` run adds <variant>/.
        prepared = f"{leg.output_dir}/prepared/seed_$SEED"
        if leg.lens_light != "both":
            prepared += f"/{RUN_VARIANT[leg.likelihood]}"
        flags.append(f"--export-prepared {prepared}")
    flags.append("--cores $SLURM_CPUS_PER_TASK")
    lines.append(f"python3 {leg.script} \\")
    lines += [f"    {flag} \\" for flag in flags[:-1]]
    lines += [f"    {flags[-1]}", "", 'echo "Finished."', "date", ""]
    return "\n".join(lines)


def submits(root: Path = ROOT) -> dict[str, str]:
    """{repo-relative submit path: rendered text}."""
    records, refs = adopted_records(root), references(root)
    return {leg.path: render(leg, records, refs) for leg in planned_legs(records)}


_MARK = re.compile(rf"^# GENERATED by {re.escape(GENERATOR)} ", re.M)


def generated_paths(root: Path = ROOT) -> set[str]:
    """Every committed submit carrying this generator's header."""
    return {
        p.relative_to(root).as_posix()
        for sub in ("batch_gpu", "batch_cpu")
        for p in (root / "hpc" / sub).glob("submit_*")
        if _MARK.search(p.read_text())
    }


def write_or_check(rendered: dict[str, str], root: Path, *, write: bool) -> int:
    """``--write`` renders every submit and removes orphans; ``--check`` lists the stale ones."""
    stale = []
    for rel in sorted(generated_paths(root) - set(rendered)):
        if write:
            (root / rel).unlink()
            print(f"removed {rel} (no longer a gap)")
        else:
            stale.append(rel)
    for rel, text in rendered.items():
        path = root / rel
        if write:
            path.write_text(text)
            path.chmod(0o755)
            print(f"wrote {rel}")
        elif not path.is_file() or path.read_text() != text:
            stale.append(rel)
    for rel in stale:
        print(f"stale: {rel} (run {GENERATOR} --write)")
    return 1 if stale else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print the manifest")
    group.add_argument("--write", action="store_true", help="(re)render the submits")
    group.add_argument("--check", action="store_true", help="exit 1 if a submit is stale")
    args = parser.parse_args(argv)
    if args.list:
        records, refs = adopted_records(), references()
        legs = planned_legs(records)
        for leg in legs:
            b, _ = budget(leg, records, refs)
            print(
                f"{leg.path}: imaging/{leg.likelihood}/{INSTRUMENT} {leg.setup_id}@{leg.host_class}"
                f" {'+'.join(leg.model_variants)} seeds {','.join(map(str, leg.seeds))}"
                f" cores {leg.cores}{' ' + leg.results_tag if leg.results_tag else ''}"
                f" -- {b.source}, estimate {b.estimate_s} s, --time {slurm_time(b.time_s)}"
            )
        print(f"{len(legs)} legs")
        return 0
    return write_or_check(submits(), ROOT, write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
