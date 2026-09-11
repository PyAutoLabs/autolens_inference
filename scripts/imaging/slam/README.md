# `scripts/imaging/slam`

Full **SLaM pipeline** runs on imaging data — the science chains as they are actually run, end
to end, stage by stage.

The first leaf landed in phase 3: [`hst.py`](hst.py), the backend-parameterised base-run
driver, exercised by the phase-4 Cortex task `slam_hst_base`.

## Running a leg

```bash
python3 scripts/imaging/slam/hst.py \
    --backend {jax_cpu,numba_cpu,jax_gpu} --inversion {dense,sparse} \
    --config-name <config> [--seed 0] [--cores N] [--stages source_lp] [--output-dir DIR]
```

- The **instrument is a flag** (`--instrument hst`, the default), never a directory. The HST
  cell is the simulated dataset from `scripts/misc/simulators/imaging.py`, auto-simulated on
  first use so RAL needs no data transfer.
- The **backend is a column**: the same leaf runs every leg of a parity row, and the legs
  differ only in the flags above and the resulting `--config-name`. The driver refuses a
  `--config-name` whose backend / inversion / precision fields disagree with the flags — a
  config name that lies about its run makes the whole parity row a lie.
- `--cores` defaults to `SLURM_CPUS_PER_TASK`, else `os.cpu_count()`. `--stages <name>` stops
  the chain after that stage (used to measure a per-backend rate from a `source_lp` leg).
- `--output-dir` overrides the **PyAutoFit run-output root** for this leaf (default
  `output/slam/imaging/hst/<config_name>/seed_<n>/`); the result row always lands under
  `results/`.
- `--backend jax_gpu` asserts `jax.default_backend() == "gpu"` and exits 2 otherwise. There is
  no silent CPU fallback: a "GPU" leg that ran on CPU reports a wall clock that is nothing of
  the kind.

## What a leg writes

- One result row per `(config_name, seed)`:
  `results/slam/imaging/hst/<config_name>/stages_seed<n>.json`, schema version 1, with a
  `stages` list of five rows (`source_lp[1]`, `source_pix[1]`, `source_pix[2]`, `light[1]`,
  `mass_total[1]`) carrying wall clock, reject-inclusive likelihood evals, log evidence,
  posterior (median + 1σ), `truth_delta_sigma` against `dataset/imaging/hst/tracer.json`,
  `positions_info_present`, `completed` and `resumed`. A wall-per-stage PNG sits beside it.
- The full PyAutoFit output tree under `output/` (gitignored — see `AGENTS.md`, "Outputs are
  KEPT"). Budget ~150 MB per 5-stage lens per leg.

`build_readme.py` flattens those stage lists into the root README's `slam` table and renders a
**parity view** per (target, seed): the `mass_total[1]` posterior of every config side by side,
each cell the difference from the alphabetically-first config in units of that reference leg's
1σ.

## The chain, and where it departs from the workspace

The chain mirrors `autolens_workspace/scripts/guides/modeling/slam_start_here.py`. Two
deliberate departures, both recorded in the driver's module docstring:

1. A **positions likelihood is attached to all four pixelized stages**, not just
   `source_pix[1]` and `mass_total[1]`. A mesh stage with no `positions.info` beside it is not
   citable here.
2. **`log_evidence_err` is always `null`** — nautilus 1.0.5 exposes no `log_z_err`, and a
   derived one would be an invention.

Everything else — flags, the config-name grammar and the backend facts that make a leg honest —
is in [`../../../_inference_cli.py`](../../../_inference_cli.py),
[`../../misc/slam/_runner.py`](../../misc/slam/_runner.py) and
[`../../../AGENTS.md`](../../../AGENTS.md).
