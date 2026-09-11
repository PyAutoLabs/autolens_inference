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

## The six legs of the parity row

One cell — `imaging/slam/hst` — run six ways. The backend is a column, never a reason to
split the table.

| leg | `--backend` | `--inversion` | `--config-name` | where it runs |
|---|---|---|---|---|
| 1 | `jax_gpu` | `dense` | `hpc_a100_jax_gpu_dense_fp64` | RAL `gpu` (A100) |
| 2 | `jax_gpu` | `sparse` | `hpc_a100_jax_gpu_sparse_fp64` | RAL `gpu` (A100) |
| 3 | `jax_cpu` | `dense` | `hpc_a100_jax_cpu_dense_fp64` | RAL `ral` (8 cores) |
| 4 | `jax_cpu` | `sparse` | `hpc_a100_jax_cpu_sparse_fp64` | RAL `ral` (8 cores) |
| 5 | `numba_cpu` | `dense` | `hpc_a100_numba_cpu_dense_fp64` | RAL `ral` (8 cores) |
| 6 | `numba_cpu` | `sparse` | `hpc_a100_numba_cpu_sparse_fp64` | RAL `ral` (8 cores) |

**`hpc_a100_` on a CPU leg is not a typo.** The config-name grammar
(`_inference_cli.CONFIG_NAME_RE`) offers exactly two "where" tokens, `local` and `hpc_a100`,
and the second means *RAL*, not *the A100 partition*. The device is already carried by the
backend token, so legs 3–6 are honestly named; the alternative would be a third token that
says the same thing twice.

Each leg is one submit under `hpc/batch_{gpu,cpu}/submit_slam_hst_<backend>_<inversion>`, each
an `--array=0-1` (one seed per array task), each sized by its own `# WALL-BASIS:` block.
`hpc/sync submit --gpu|--cpu <submit filename>` sends one; `hpc/sync jobs` / `tail` / `pull`
bring it back. The submit names, the seed convention and the memory request are tabulated
in [`../../../hpc/README.md`](../../../hpc/README.md).

**All four HST CPU legs and both GPU legs are RAL jobs, not laptop jobs, and that is a
measurement rather than a preference.** See the memory section below.

## Memory: the HST cell does not fit on a laptop under JAX

The HST cell is **15,361 masked pixels**, and the cost is in the Nautilus batch rather than
the dataset. Measured on 2026-09-11, on the `source_pix[1]` stage with `n_batch=20`:

| leg | RSS |
|---|---|
| `jax_cpu` × dense | **~13 GB** at `source_pix[1]` |
| `jax_cpu` × sparse | **~21.6 GB** at `source_pix[1]` |
| `numba_cpu` × either | not the constraint — there is no vmap at all: with `use_jax=False` Nautilus takes the `number_of_cores` multiprocessing path. The production `source_lp[1]` probe held ~6.3 GB across its nine processes, and the `PYAUTO_TEST_MODE` chain reached `mass_total[1]` on the same 15 GB machine. Its `source_pix[1]` figure at production settings is **not** measured. |

Both JAX-CPU figures OOM a 15 GB laptop, which has three consequences worth stating plainly:

1. The two `jax_cpu` HST legs of the parity row exist **only** as RAL jobs.
2. The **local rate measurements are `--stages source_lp` only** — the parametric opening
   stage builds no inversion and holds no mesh, so it fits in RAM and still measures the
   backend's seconds per likelihood evaluation.
3. The CI witness (`.github/workflows/profile.yml`) runs `--instrument euclid`, not `hst`:
   a GitHub-hosted runner has 7–16 GB and the euclid cell exercises the identical code path
   on a smaller grid. The local `PYAUTO_TEST_MODE` witness for JAX on HST is expected to
   OOM and is not evidence of a bug.

Every submit asks for `--mem=64gb` — the template's figure, kept because it clears the
worst measured leg by 3x and a leg killed by the OOM killer at stage two costs a night.

## Rates, and what they are allowed to justify

Measured 2026-09-11, and narrower than they look — each is `source_lp[1]` **only**, at
production settings, on the HST cell (15,361 masked pixels, 17 free parameters, nautilus
`n_live=200` / `n_batch=50`, seed 0, dense, fp64):

| device key | s / likelihood eval | how it was measured |
|---|---|---|
| `laptop_numba_cpu` | **0.06831** | partial run, 17,950 evals over 1,226 s, 8 cores, `DESKTOP-H143S82` |
| `laptop_jax_cpu` | **0.04925** | partial run, 25,500 evals over 1,256 s, 8 cores, same host |
| `a100` | *not measured* | RAL job **342695** (`submit_slam_hst_rate_jax_gpu`) was still `PENDING(Priority)` when phase 3 shipped |

Both local probes were stopped before `source_lp[1]` finished — nautilus was still in its
exploration phase — so **`results/` carries no production row yet**; these are rates, not
results. The environment is part of the measurement and is recorded in `PROVENANCE`: the
session's ambient `XLA_FLAGS=--xla_disable_hlo_passes=constant_folding` was removed with
`env -u XLA_FLAGS`, and `OMP_NUM_THREADS=1` was kept (correct for the numba leg, which takes
nautilus's `number_of_cores` multiprocessing path — kernel threading under multiprocessing
costs rather than pays; the JAX legs are sized by `NPROC` instead).

The device key names the **host** as well as the backend on purpose. The submits declare
`ral_numba_cpu`, `ral_jax_cpu` and `a100`, so a laptop rate cannot be cited by a RAL submit
even by accident: the gate simply does not match the key.

`wall/rates.py` holds these rows and `wall/check_submits.py` gates every submit's `--time`
against them. Read [`../../misc/wall/README.md`](../../misc/wall/README.md) before citing
one: the rule is that **a `--time` justification never crosses cells**, and the corollary
here is that a `source_lp` rate is the *cheapest* stage of this cell. Four of the five
stages are pixelized and per-eval-inversion bound at many times that cost, so the six
production submits all declare `source: unmeasured  probe-first: yes` — their first run is
itself the probe that will replace the declaration with `measured-wall`. The measured rows
are cited by one submit only: `submit_slam_hst_rate_jax_gpu`, the `--stages source_lp` probe
they were measured on.

## The chain, and where it departs from the workspace

The chain mirrors `autolens_workspace/scripts/guides/modeling/slam_start_here.py`. Four
deliberate decisions, all recorded in the driver's module docstring — the first two are
departures from the workspace script, the last two are choices this repo had to make
because the workspace script never faced them:

1. A **positions likelihood is attached to all four pixelized stages**, not just
   `source_pix[1]` and `mass_total[1]`. A mesh stage with no `positions.info` beside it is not
   citable here.
2. **`log_evidence_err` is always `null`** — nautilus 1.0.5 exposes no `log_z_err`, and a
   derived one would be an invention. The row carries a sibling
   `log_evidence_err_note` saying so, so a reader never has to guess whether the field is
   missing or zero.
3. **The over-sample map before `source_pix[2]` is kept** (it is the workspace default, and
   dropping it would make this chain a different pipeline from the one science runs), and
   the **sparse operator is re-applied immediately after it**. `Imaging.apply_over_sampling`
   rebuilds the `Imaging` object from scratch and silently drops the sparse operator, so a
   sparse leg that did not re-apply would run its last three stages dense while still
   calling itself sparse — a parity row that lies.
4. **`--output-dir` moves the PyAutoFit run-output root, not the results directory.** A
   leg's bulk state is the run tree (~150 MB per 5-stage lens), and that is what a RAL job
   or a scratch disk needs to redirect. The small result row always lands under
   `results/slam/<dataset_class>/<instrument>/<config_name>/`, so a pulled run and a local
   one are read from the same place.

Everything else — flags, the config-name grammar and the backend facts that make a leg honest —
is in [`../../../_inference_cli.py`](../../../_inference_cli.py),
[`../../misc/slam/_runner.py`](../../misc/slam/_runner.py) and
[`../../../AGENTS.md`](../../../AGENTS.md).
