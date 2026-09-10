# autolens_inference — Agent Instructions

This repo is the single home for **PyAutoLens inference measurement**: which non-linear
search — gradient-free or gradient-based — finds the right lens model, how reliably, and
at what cost, on CPU and on A100 GPUs. It is the sibling of
[`autolens_profiling`](https://github.com/PyAutoLabs/autolens_profiling), which owns
likelihood **timing**; this repo owns everything downstream of one likelihood call:
samplers, pipelines, convergence, reliability, and the wall-clock price of an answer.

It is a collection of standalone scripts, **not** an installable package — there is no
`pyproject.toml`. These are the canonical, agent-agnostic instructions. `README.md` is the
human-facing overview; `CORTEX.md` says where the rulings of record live.

Science runs, tasks and rulings are managed by
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex) under the `projects.yaml` key
`autolens_inference`, **not** by PyAutoMind. A question about what to run and whether to
believe the answer is a Cortex task; a change to the code here is a Mind development task.

## Nothing is inherited

This repo restarted **from scratch on 2026-09-10**. It is the successor to the retired
Cortex project `inference_programme`, and none of that programme's material came across:
no baselines, no tolerance tables, no target-id code, no result JSONs, no searches
framework, no notes. **The retired programme's evidence is never cited here** — not in a
result, not in a wall-clock justification, not in a ruling. Every number this project
stands on is measured in this repo, by a run it can name. If you find yourself reaching
for a number from `inference_programme`, the honest move is to measure it again.

## Repository Structure

Scripts are laid out **dataset-class first, task second**, mirroring the
`autolens_workspace*` taxonomy:

```
scripts/
  <dataset_class>/          imaging/ interferometer/ point_source/ — one folder per
                            PyAutoLens dataset family. Imaging is the priority.
    slam/                   Full SLaM pipeline runs (the base-run tier)
    searches/<sampler>/     Single-search runs, one folder per sampler
  misc/                     Dataset-agnostic material + each task's shared framework:
                            misc/simulators/ (dataset simulators), misc/tooling/
                            (build_readme.py), misc/wall/ (SLURM --time budget gate),
                            misc/test/ (the pytest suite the lint gate runs)
_inference_cli.py           Shared CLI / JSON / auto-simulate helper imported by every leaf
instruments/                Instrument definitions (pixel scale, shape) used to frame results
config/                     PyAutoConf overrides for runs launched from this repo
hpc/                        hpc/sync (laptop-side RAL driver) + SLURM submit scripts
results/                    Committed result rows: results/slam/, results/searches/
output/                     PyAutoFit run output — KEPT here (see below), gitignored
wiki/project/               The Cortex ledger (state.md) + its journal-entry template
```

**The instrument is a flag, never a directory.** `--instrument hst` selects the preset;
there is no `scripts/imaging/slam/hst/`. Instruments are columns in a table, and putting
one in a path forces the tree to be re-cut every time a preset is added.

**Import model.** Leaves sit several levels below the repo root, so each finds the root by
walking up to the directory containing `ruff.toml` (a depth-proof sentinel) and puts both
the **repo root** and **`scripts/misc/`** on `sys.path`. That keeps the shared libraries
importable by their top-level names with no per-file path math: `_inference_cli` and
`instruments` (repo root), `simulators` / `wall` (under `scripts/misc/`).

## Config names and parity rows

Every run carries a `--config-name` built from a fixed grammar:

```
{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}
```

- **where** — `local` (this laptop) or `hpc_a100` (RAL, `gpu` partition).
- **backend** — `jax_cpu`, `numba_cpu`, or `jax_gpu`.
- **inversion** — `dense` (mapping matrix) or `sparse` (w-tilde operator).
- **precision** — `fp64`, or `mp` for the targeted mixed-precision paths.

A **parity row** is the unit this repo reasons in: a set of runs that share one *target* —
same dataset, same model, same pipeline, same seed — and differ **only** in the config
name. The backend is a **column, never a reason to split a table**. If two rows cannot sit
side by side under one target because their setups differ in something the config name
does not carry, that difference is a bug in the run, not a new table: fix the run.
The first parity row is the 5-stage HST SLaM chain across
numba-CPU / JAX-CPU / A100 × dense / sparse.

## Backend facts (get these wrong and the parity row is a lie)

- **`use_jax=`** is a keyword on each `Analysis`. It defaults to `True`. The
  `numba_cpu` leg must pass `use_jax=False` explicitly on *every* Analysis a pipeline
  builds — one missed stage silently makes that stage a JAX run.
- **`PYAUTO_DISABLE_JAX=1`** disables JAX process-wide. It is the belt to `use_jax=False`'s
  braces on a numba leg; it is not a substitute, because it does not change what an
  `Analysis` reports about itself.
- **Sparse is two different calls.** Under JAX (on any device) it is
  `dataset.apply_sparse_operator()`; under numba it is
  `dataset.apply_sparse_operator_cpu()`. Calling the JAX one on a numba leg does not
  error — it produces a leg that is neither.
- **`JAX_PLATFORMS` and `JAX_ENABLE_X64` must be exported explicitly in every sbatch
  script.** A GPU job that does not set `JAX_PLATFORMS=cuda,cpu` can silently fall back to
  CPU and report a "GPU" wall clock that is nothing of the kind, and a job that does not
  set `JAX_ENABLE_X64=True` runs an `fp64` config in fp32.
- **`NPROC=$SLURM_CPUS_PER_TASK`** in every sbatch script: `NPROC` throttles JAX's CPU
  thread pool, and an unset one lets a job take the whole node.

## Outputs are KEPT

Unlike `autolens_profiling`, this repo **keeps its run output**. Inference is judged on
the posterior, the samples and the convergence history, not on a single number, so the
`config/general.yaml` shipped here sets:

```yaml
hpc:
  hpc_mode: false        # do NOT delete the unzipped output tree
output:
  remove_files: false    # keep every per-stage file
  samples_to_csv: true
```

Budget roughly **150 MB per 5-stage lens per leg**. `output/` is gitignored: it is bulk
run state, kept on disk and pulled from RAL, never committed. What *is* committed is the
small result row under `results/slam/` or `results/searches/`.

## RAL, and getting runs back

The RAL copy of this repo is a **git clone** at `/mnt/ral/jnightin/autolens_inference`, not
an rsync target.

- **Code goes over with `git pull` on the login node.** There is deliberately no
  `hpc/sync push`; running it prints the real procedure and exits non-zero.
- **Results and output come back with `hpc/sync pull`** (`cp hpc/sync.conf.example
  hpc/sync.conf` first; `sync.conf` is gitignored). `output/` and `results/` map 1:1 onto
  the local checkout, and RAL's `hpc/batch_{gpu,cpu}/{output,error}` land under
  `logs/{output,error}`.
- **Results are committed locally**, from this checkout, by a human reading them — never
  by a job on RAL.
- **Partitions:** the A100s are on `--partition=gpu`. The CPU partition is `ral`. There is
  **no `cpu` partition on RAL**; a submit that asks for one is rejected.
- The PyAuto* libraries resolve from the shared checkouts on `PYTHONPATH`
  (`source activate.sh`) and are updated with `HPCPullPyAuto` — never pip-installed.

Full detail in [`hpc/README.md`](hpc/README.md).

## Testing

The PR gate is `.github/workflows/lint.yml` on Python 3.12 (every PR + push to `main`).
Its headline lint is **ruff**, not black:

```bash
ruff check .
ruff format --check .
```

The same job also runs `scripts/misc/tooling/build_readme.py --check` (dashboard
idempotence), `scripts/misc/wall/check_submits.py --check` (every submit's `--time` is
justified per cell), `pytest scripts/misc/test -q`, a `lychee` link-rot check over the
`README.md` files, and a **smoke** leg that runs each simulator under
`AUTOLENS_INFERENCE_SMOKE=1`. Every runnable script reads that variable at module top and
exits 0 straight after the import + setup section, so the smoke catches import-graph
breakage without running an inference. None of it produces result artifacts.

`.github/workflows/profile.yml` is `workflow_dispatch`-only and does nothing yet — it is
the placeholder for the phase-3 backend-parameterised driver.

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at
writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python3 scripts/misc/simulators/imaging.py
```

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted
region. **Never produce a whole-file write unless you have read the entire current file** —
a whole-file write from a header skim silently deletes every section below the header.

## Related Repos

- `../autolens_profiling` — the sibling repo that owns likelihood **timing**. A question
  about how long one likelihood call takes belongs there, not here.
- `../PyAutoLens` — the library being exercised (plus `../PyAutoGalaxy`, `../PyAutoArray`,
  `../PyAutoFit`, `../PyAutoNerves` on `PYTHONPATH`).
- `../PyAutoCortex` — the science ledger: tasks, runs and rulings for this project.
- `../autolens_workspace` — user-facing science scripts and tutorials.

## Task Workflows

When adding or updating a script, keep `ruff check .` and `ruff format --check .` clean
(the PR gate), write the result row under `results/`, and do not commit machine-specific
absolute paths. Flag any change that affects the source libraries or `autolens_workspace`
in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->
