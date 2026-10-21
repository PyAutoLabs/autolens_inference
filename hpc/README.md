# hpc

Driving the RAL HPC from the laptop, and getting runs back.

## Layout

```
sync                    laptop-side driver (below)
sync.conf.example       template; sync.conf is gitignored
wait_for_stage.sh       in-job wait for another chain's completed SLaM stage (--stages-from)
batch_gpu/
  template              the shape every A100 submit is copied from
  submit_<name>         one submit per run
  output/   error/      SLURM stdout/stderr (gitignored; .gitkeep tracked)
batch_cpu/
  template              the shape every CPU submit is copied from
  submit_<name>
  output/   error/
```

## Partitions

| | partition | what |
|---|---|---|
| GPU | `--partition=gpu` | the A100 nodes. `--gres=gpu:1`, `--cpus-per-task=4`. |
| CPU | `--partition=ral` | the general nodes. `--cpus-per-task=8`. |

**There is no `cpu` partition on RAL.** A submit that asks for one is rejected by SLURM.
Check with `sinfo` before inventing a partition name.

## Running

On the login node:

```bash
source activate.sh          # repo-root helper: venv + PYTHONPATH at the canonical checkouts
sbatch hpc/batch_gpu/submit_<name>
```

The PyAuto* libraries resolve from sibling source checkouts on `PYTHONPATH` — never
pip-install them into the venv (`HPCPullPyAuto` is the update story).

## Running against a pinned release

The shared mirror `/mnt/ral/jnightin/PyAuto` follows `main` and other campaigns' jobs
read it live, so **it is never pulled for a measurement that must name its release**. A
submit that needs a fixed release points `PYTHONPATH` at a private *pin directory*
instead: `activate.sh` reads `PYAUTO_LIB_BASE` (default: the shared mirror) for the five
library checkouts, and keeps the shared venv (`/mnt/ral/jnightin/PyAuto/PyAuto`, third-party
dependencies only) and the shared cache root, so a pin directory needs nothing but the
five checkouts.

Create the 2026.10.17.1 pin once, on the login node (documented here; not run by any
task — the tags carry **no** `v` prefix):

```bash
PIN=/mnt/ral/jnightin/inference/pins/2026.10.17.1
mkdir -p "$PIN" && cd "$PIN"
for repo in PyAutoNerves PyAutoFit PyAutoArray PyAutoGalaxy PyAutoLens; do
    git clone https://github.com/PyAutoLabs/$repo.git
    git -C $repo checkout 2026.10.17.1
done
```

A pinned submit exports, **before** `source activate.sh`:

```bash
export PYAUTO_LIB_BASE=/mnt/ral/jnightin/inference/pins/2026.10.17.1
export PYAUTO_RELEASE_PIN=2026.10.17.1
```

`PYAUTO_RELEASE_PIN` is recorded in every SLaM result row (schema v3, `release_pin`) as a
*claim*; the row's `library_revisions` — the HEAD commits of the checkouts the process
actually imported — is the *evidence*, and a reader checks one against the other. For
2026.10.17.1 a pinned row's `library_revisions` must equal the tag commits:

| repo | tag `2026.10.17.1` |
|---|---|
| PyAutoNerves | `42b4d31` |
| PyAutoFit | `f4d4f60` |
| PyAutoArray | `9993b42` |
| PyAutoGalaxy | `f555e31` |
| PyAutoLens | `02f9402` |

A pin directory is never pulled, rebased or edited after creation; a new release gets a new
directory. Rows also record `slurm_job_id` (`<array job>_<task>` inside an array).

## `hpc/sync` — the laptop-side driver

Copy the config first:

```bash
cp hpc/sync.conf.example hpc/sync.conf   # then edit; sync.conf is gitignored
```

| verb | what it does |
|------|--------------|
| `hpc/sync pull` | Download batch logs, then run outputs and results |
| `hpc/sync logs` | Batch logs only — small and fast, use mid-run |
| `hpc/sync status` | Dry run: what a pull would transfer |
| `hpc/sync submit [--gpu\|--cpu] [--dependency <spec>] <name>` | `sbatch` a submit script, from `hpc/batch_<type>/`; `--dependency` is passed through as `sbatch --dependency=<spec>` |
| `hpc/sync jobs` / `sacct` / `cancel <id>` | `squeue` / `sacct` / `scancel` |
| `hpc/sync tail [gpu\|cpu]` | Stream the newest live `.out` |
| `hpc/sync du` | Remote disk usage (`-d1` — never a bare recursive `du`; RAL's NFS is slow) |
| `hpc/sync check` | Verify SSH, remote paths, `sbatch`, and the local pull root |

`submit` runs `sbatch` **from `hpc/batch_<type>/`**, not the project root, because the
submit scripts' `#SBATCH -o` / `-e` paths are relative. Submitting from the project root
makes SLURM fail to open the error file and kill the batch step at t=1s with ExitCode 0:53
and no `.err` to read.

### Where a pull lands: `LOCAL_PULL_ROOT`

**This checkout**, by default — unlike `autolens_profiling`, whose pulls go to a separate
mirror. This repo keeps its output (`config/general.yaml` sets `hpc_mode: false` and
`remove_files: false`; see `AGENTS.md`, "Outputs are KEPT"), `output/` is gitignored here
precisely so a pull can land in it, and a pulled result row is read in place and committed
by a human from this checkout. That commit is the only way a result ever enters git — a
job on RAL never commits.

| on RAL | lands at |
|--------|----------|
| `output/` | `$LOCAL_PULL_ROOT/output/` |
| `results/` | `$LOCAL_PULL_ROOT/results/` |
| `hpc/batch_gpu/{output,error}/` | `$LOCAL_PULL_ROOT/logs/gpu/{output,error}/` |
| `hpc/batch_cpu/{output,error}/` | `$LOCAL_PULL_ROOT/logs/cpu/{output,error}/` |

Budget roughly **150 MB per 5-stage lens per leg**, and a parity row is six legs.

`search_internal/` is excluded from every pull: it is sampler state (Nautilus
`checkpoint.hdf5`, live points), it is large, and it is not needed to read a result. A run
that needs its checkpoint is read on RAL.

`PYAUTO_PULL_DIRS` appends extra roots to the pull, space-separated — the Cortex check-in
passes the `output*` roots its tasks declare, so a run written to a custom
`PYAUTO_OUTPUT_DIR` is pulled without editing this script. It only ever appends; the
project's own roots always pull.

### The pull manifest: `.cortex/pull.json`

Because `search_internal/` never comes back, a checkpoint that is still growing on RAL is
invisible from the laptop — and "is this run alive?" is exactly what a stalled overnight
sweep needs answered. So a real `pull` ends by writing `$LOCAL_PULL_ROOT/.cortex/pull.json`:
one `find` over the ControlMaster mux records the size and mtime of every
`checkpoint.hdf5` under `output/` on RAL, keyed by run directory relative to the pull root.
PyAutoCortex `collect` reads it to score the checkpoint leg of a run's evidence.

`hpc/sync status` is a dry run and does not write the manifest; a failed gather writes it
anyway with a `gather_error` key rather than failing the pull; `.cortex/` is gitignored.

`pull` tolerates rsync exit 23/24 (files vanished or partially transferred), which is
normal while jobs are still writing, and skips any remote directory that does not exist
yet rather than aborting.

### There is no `push`

`hpc/sync push` prints the real procedure and exits non-zero. The RAL copy of this repo is
a **git clone** with local uncommitted state; rsyncing a laptop tree over it would clobber
that and leave the working tree disagreeing with its own HEAD. Code goes to RAL with git
on the login node:

```bash
ssh euclid_jump
cd /mnt/ral/jnightin/inference/autolens_inference
git status        # look before you pull — the checkout is often dirty
git pull
```

The PyAuto* libraries are a separate story again: resolved from the shared checkouts on
`PYTHONPATH` and updated with `HPCPullPyAuto`, never pip-installed and never rsynced.

### Chaining a submit on another job: `--dependency`

`hpc/sync submit --dependency <spec> <name>` hands `<spec>` to `sbatch` verbatim as
`--dependency=<spec>`, so a job that reads another job's products can be queued before
those products exist and starts only once they do:

```bash
hpc/sync submit --dependency afterok:<base job id> submit_hst_cleanslate_mass_samplers_jax_gpu_dense
```

`afterok:<id>` on an array job waits for **every** task of that array to exit 0; a single
failed task leaves the dependent job pending forever with reason
`DependencyNeverSatisfied`, which `scancel` clears. `afterok:<id>_<task>` waits for one
task only. The spec is never written into a submit script: the job id it names is only
known at submit time, and a hard-coded id would chain a resubmission onto a stale job.
Read the dependency state back with `squeue -j <id>` (reason `Dependency` while waiting).

## A crashed job FAILS: the SLURM exit-code guard

Every submit ends the same way:

```bash
python3 scripts/... --instrument hst ...

echo "Finished."
date
```

A bash script exits with the status of its **last** command, so that `date` would make
every job exit `0` no matter what Python did — a job that died four seconds in on a CUDA
init failure gets recorded by SLURM as `COMPLETED 0:0` in a suspiciously fast time, and
the only tells are a traceback in the `.err` and a result file that never appears. That is
a failure that reports success.

The guard lives in the repo-root **`activate.sh`**, which every submit already sources:

```bash
if [ -n "${SLURM_JOB_ID:-}" ]; then
    set -eE
    trap '...; exit ${rc}' ERR
    export PYTHONUNBUFFERED=1
fi
```

- **Scoped to a SLURM job.** `activate.sh` is also sourced in interactive login-node
  shells, where `errexit` would close the terminal on the first typo.
- **No `pipefail`.** No submit pipes anything into `python3`, so it would add risk without
  covering the failure the guard exists for.
- **Deliberately-tolerated failures are unaffected** — `errexit` does not fire on a
  command whose status is consumed by `||`.
- **Nothing per-submit to remember.** A new submit inherits the guard by sourcing
  `activate.sh`, which it must do anyway to get the venv and `PYTHONPATH`.
- **`PYTHONUNBUFFERED=1`** for the same reason: a SLURM `.out` is a file, not a tty, so
  Python block-buffers stdout at 8 KiB and a six-hour job's progress lines sit in an
  unflushed buffer that the wall-clock kill then discards.

Read `sacct` states as real — but keep reading `.err`: a run can still fail *by producing
wrong numbers*, which no exit code catches.

## Environment every submit must export

Getting one of these wrong does not error; it produces a leg that is quietly not the leg
its config name claims:

| variable | why |
|---|---|
| `JAX_PLATFORMS=cuda,cpu` (GPU) / `cpu` (CPU) | a GPU job without it can silently fall back to CPU and report a "GPU" wall clock that is nothing of the kind |
| `JAX_ENABLE_X64=True` | without it, an `fp64` config runs in fp32 |
| `NPROC=$SLURM_CPUS_PER_TASK` | `NPROC` throttles JAX's CPU thread pool; unset, a job takes the whole node |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` (GPU) | stops XLA grabbing the whole card up front |

Cache locations (`NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`,
`JAX_COMPILATION_CACHE_DIR`, CUDA/Triton, astropy) are **not** a submit's job: `activate.sh`
points every unset one under `/mnt/ral/jnightin/.cache` (`$PYAUTO_HPC_CACHE`). Never send
them to `$HOME` or `/tmp` — on RAL both sit on each node's small root disk, and filling it
breaks the node (RAL admin, 2026-09-25). A submit may still set its own
`JAX_COMPILATION_CACHE_DIR` (it wins; an empty value disables the cache).

`PYAUTO_DISABLE_JAX` is deliberately **left to the script**, not set by the template: it
is process-wide and belongs next to the `--backend numba_cpu` flag that motivates it.

## Wall clock: `# WALL-BASIS:` is mandatory

Every submit — and both `template` files — must carry a `# WALL-BASIS:` block above its
`#SBATCH` stanza, with **one row per cell it runs**.
`scripts/misc/wall/check_submits.py --check` gates it on every PR.

The rule it enforces: **never carry a `--time` justification across cells.** A submit that
cited a parametric step rate for an array of mostly pixelized arms lost 35 of its 39 arms
at ~12% of budget — an entire overnight A100 block. An array submit is sized by its
**slowest** cell, never its fastest.

`scripts/misc/wall/rates.py` started **empty** in this repo: it inherits no rates from the
retired `inference_programme`. Its first two rows landed 2026-09-11, and both are the
`source_lp[1]` stage of `imaging/slam/hst` measured on a laptop — which is why **all nine
submits here still declare `source: unmeasured  probe-first: yes`**. The four completed
A100 base legs did not change that: they are a measured *wall* for `imaging/slam/hst`, and
citing them on `imaging/slam/hst_delaunay` would be the cross-cell carry this gate exists
to stop — the Delaunay submits name them as context for their containment and still
declare themselves unmeasured. A parametric rate
cannot size a mostly-pixelized chain, and a laptop rate cannot size a RAL node; the device
keys (`laptop_numba_cpu`, `laptop_jax_cpu` vs `ral_numba_cpu`, `ral_jax_cpu`, `a100`) make
that a mismatch the gate catches rather than a judgement call. A truncated arm still
measures s/step, so the first production arm to run is the measurement that replaces those
declarations with `source: measured-wall`. The block's grammar and the three `source:` kinds
are in [`../scripts/misc/wall/README.md`](../scripts/misc/wall/README.md).

## The submits that exist today

Thirteen run the SLaM chain, over **two cells**. Eleven run `imaging/slam/hst` (`scripts/imaging/rectangular/validation/baseline_slam.py`, the
backend-parameterised SLaM driver): six are the six legs of the first parity row, the
seventh is the rate probe that had to run before any of them could be sized, and four are
the 2026-10-17 **clean slate** (below). Two run
`imaging/slam/hst_delaunay` (`scripts/imaging/delaunay/validation/baseline_slam.py`) — the same five-stage
chain on the Delaunay run variant (`delaunay_1250` then; `slam_delaunay` from 2026-10-19). Two more, the clean-slate wave-1 arrays (below),
run the four sampler leaves of `scripts/imaging/rectangular/samplers/` on frozen mass problems.

| submit | cell | partition | array | what it runs |
|---|---|---|---|---|
| `batch_gpu/submit_slam_hst_rate_jax_gpu` | `imaging/slam/hst` | `gpu` | — | **the rate probe**: `--stages source_lp`, one stage, to measure A100 s/eval |
| `batch_gpu/submit_slam_hst_jax_gpu_dense` | `imaging/slam/hst` | `gpu` | `0-1` | full chain, `jax_gpu` × dense |
| `batch_gpu/submit_slam_hst_jax_gpu_sparse` | `imaging/slam/hst` | `gpu` | `0-1` | full chain, `jax_gpu` × sparse |
| `batch_cpu/submit_slam_hst_jax_cpu_dense` | `imaging/slam/hst` | `ral` | `0-1` | full chain, `jax_cpu` × dense |
| `batch_cpu/submit_slam_hst_jax_cpu_sparse` | `imaging/slam/hst` | `ral` | `0-1` | full chain, `jax_cpu` × sparse |
| `batch_cpu/submit_slam_hst_numba_cpu_dense` | `imaging/slam/hst` | `ral` | `0-1` | full chain, `numba_cpu` × dense |
| `batch_cpu/submit_slam_hst_numba_cpu_sparse` | `imaging/slam/hst` | `ral` | `0-1` | full chain, `numba_cpu` × sparse |
| `batch_gpu/submit_slam_delaunay1250_hst_jax_gpu_dense` | `imaging/slam/hst_delaunay` | `gpu` | `0-1` | full chain on `delaunay_1250`, `jax_gpu` × dense |
| `batch_gpu/submit_slam_delaunay1250_hst_jax_gpu_sparse` | `imaging/slam/hst_delaunay` | `gpu` | `0-1` | full chain on `delaunay_1250`, `jax_gpu` × sparse |
| `batch_gpu/submit_hst_cleanslate_jax_gpu_dense` | `imaging/slam/hst` | `gpu` | `0-1` | clean slate: pinned 2026.10.17.1, `jax_gpu` × dense, `--lens-light both` |
| `batch_gpu/submit_hst_cleanslate_jax_gpu_sparse` | `imaging/slam/hst` | `gpu` | `0-1` | clean slate: pinned 2026.10.17.1, `jax_gpu` × sparse, `--lens-light both` |
| `batch_cpu/submit_hst_cleanslate_jax_cpu_dense` | `imaging/slam/hst` | `ral` | `0-1` | clean slate: pinned 2026.10.17.1, `jax_cpu` × dense, `--lens-light both` (48 GB) |
| `batch_cpu/submit_hst_cleanslate_jax_cpu_sparse` | `imaging/slam/hst` | `ral` | `0-1` | clean slate: pinned 2026.10.17.1, `jax_cpu` × sparse, `--lens-light both` (64 GB) |
| `batch_cpu/submit_hst_cleanslate_numba_cpu_dense` | `imaging/slam/hst` | `ral` | `0-1` | clean slate: pinned 2026.10.17.1, `numba_cpu` × dense, `--lens-light both`, `source_lp[1]` shared from the jax_cpu dense chain (in-job wait) (48 GB) |
| `batch_cpu/submit_hst_cleanslate_numba_cpu_sparse` | `imaging/slam/hst` | `ral` | `0-1` | clean slate **redo** (#38): `hpc_ral_numba_cpu_sparse_fp64`, BLAS/OpenMP pools pinned to 1 thread, pinned 2026.10.17.1, `--lens-light both`, `source_lp[1]` shared from the jax_cpu dense chain (64 GB) |
| `batch_cpu/submit_hst_cleanslate_numba_cpu_sparse_delaunay` | `imaging/slam/hst_delaunay` | `ral` | `0-1` | the same redo on `slam_delaunay` (+ `slam_delaunay_light_fixed`), the standard SLaM Delaunay chain (96 GB) |
| `batch_gpu/submit_hst_cleanslate_jax_gpu_dense_delaunay` | `imaging/slam/hst_delaunay` | `gpu` | `0-1` | the A100 Delaunay legs: `jax_gpu` × dense on `slam_delaunay` (+ `slam_delaunay_light_fixed`), `source_lp[1]` shared from the jax_cpu dense chain (96 GB) |
| `batch_gpu/submit_hst_cleanslate_jax_gpu_sparse_delaunay` | `imaging/slam/hst_delaunay` | `gpu` | `0-1` | the same on `jax_gpu` × sparse; submit with `--dependency=after:<dense Delaunay job>+15` (two jobs loading one shared stage at once race on its `search_internal/`) |
| `batch_gpu/submit_hst_cleanslate_mass_samplers_jax_gpu_dense` | `imaging/rectangular/{nautilus,emcee}` | `gpu` | `0-11%4` | clean-slate wave 1 attempt 2: Nautilus and Emcee × seeds 0-2 × both light modes on the frozen A100 dense problems (NUTS/SMC deferred, A100 OOM) |
| `batch_cpu/submit_hst_cleanslate_mass_samplers_jax_cpu_dense` | `imaging/rectangular/{nautilus,emcee}` | `ral` | `0-11%6` | the same 12 arms on the frozen RAL CPU dense problems; waits for the CPU base rows |

### The clean-slate submits (2026-10-17)

PyAutoInsight campaign `hst-imaging-mass-samplers`, phase 1 (autolens_inference#24). One
job per (backend, inversion), seeds 0-1 as the array, and `--lens-light both`: stages 1-4
run once and `mass_total[1]` twice, writing a `slam_base` (linear lens light) and a
`slam_base_light_fixed` (lens light fixed to ordinary profiles) row per seed. Each runs
against the pinned 2026.10.17.1 checkouts (above). The clean slate is the starting point of
this repo's runs, so it writes the plain layout (autolens_inference#35):

- run tree `output/autolens/imaging/hst/slam/<variant>/<config_name>/seed_<n>/` (dataset
  class, instrument, then pipeline);
- rows `results/imaging/<rectangular|delaunay>/hst/validation/<model_variant>/<setup_id>/seed_<n>/stages.json`;
- frozen `mass_total[1]` problems (`--export-prepared`)
  `output/autolens/prepared/<config_name>/seed_<n>/<variant>/`, and the wave's experiment
  runs and source snapshots under `output/autolens/experiments/` and
  `output/autolens/experiment_sources/`.

The 2026-09-24 trees were moved aside on RAL and their rows archived under
`archive/2026/09/2026-09-24_superseded_by_cleanslate/`, so nothing older is resumed or
overwritten. The campaign's first arrays wrote a dated `output/cleanslate_2026_10_17/` root
with `--results-tag cleanslate_2026_10_17`; `scripts/misc/flatten_cleanslate.py` moves those
trees and rows onto the plain layout once their arrays complete (its docstring has the order:
RAL, then the laptop mirror).

The CPU pair is `--partition=ral` only — never `gpu`, `ral,gpu` or `gpu,ral`.

The numba_cpu pair (autolens_inference#31) shares the jax_cpu dense chain's `source_lp[1]`
(`--stages-from hpc_a100_jax_cpu_dense_fp64:source_lp`; the rule is in
`scripts/imaging/slam/README.md`, "Shared stages"). Before the runner starts, each job runs
`hpc/wait_for_stage.sh <jax_cpu dense seed dir>`. The helper polls `_runner.py
--stage-complete` every 5 minutes, capped at 8 h, and on timeout exits 4 without fitting. The
helper is a poll, not a cell, so it stays out of the submit's `python3 scripts/...` lines,
which are what the wall gate reads. The pair can be submitted before the jax_cpu stage
finishes. If the jax_cpu array is cancelled, cancel these too, or they exit 4 after 8 h.

**The numba thread caps and `hpc_ral` (2026-10-19, autolens_inference#38).** The 10-17
numba legs ran `source_pix[1]` at 2.6 s wall per evaluation against ~0.06-0.08 s expected:
nothing set `OPENBLAS/MKL/OMP/VECLIB/NUMEXPR_NUM_THREADS`, so every forked Nautilus worker
ran a BLAS pool the size of the node. Every numba submit now exports the five as `1` before
`python3` (the runner also sets them for `numba_cpu`), and the sparse reruns carry the
`hpc_ral_` config prefix — a RAL CPU node on the `ral` partition, never an A100 (`hpc_a100_`
stays on the GPU legs and on the rows already written). The superseded
`hpc_a100_numba_cpu_*` trees and rows are set aside, not deleted, by
`scripts/misc/rename_light_fixed.py --aside` (rows to
`archive/2026/10/2026-10-19_numba_thread_oversubscription/`). The same script renames the
old `_lightfixed` variant folders, rows and prepared problems to `_light_fixed`. The
Delaunay rerun shares the `slam_base` tree's `source_lp[1]`: the runner resolves a shared
parametric stage under `slam_base` whatever the mesh.

### The clean-slate wave-1 sampler submits (2026-10-17)

Phase 3 wave 1 of the same campaign (autolens_inference#27). Two submits, `jax_gpu` and
`jax_cpu` dense, each a 24-arm array over the frozen `mass_total[1]` problems phase 2 exports
for base seed 0: every sampler the experiment runner supports (`nautilus`, `emcee`, `nuts`,
`smc`), cold, at sampler seeds 0-2, on both light modes. The arm table is data in
[`../scripts/misc/experiments/wave.py`](../scripts/misc/experiments/wave.py); the submit asks
it for each array index's sampler and leaf arguments, so the mapping lives (and is tested) in
one place. `python3 scripts/misc/experiments/wave.py --list --config <config>` prints the arms
and run ids; `--account --config <config>` reports each as `row`, `failed` or `missing` from
`results/misc/experiments/`.

Each submit runs four cells (one per sampler leaf), all `unmeasured` with `probe-first: yes`:
the `%4` / `%6` throttles start one arm of every sampler first. The GPU array is submitted
`--dependency afterok:<GPU dense base array>`; the CPU array waits for the CPU base rows
(days) and is submitted the same way on the CPU dense base array.

**Attempts and the sampler filter (autolens_inference#33).** Attempt 1 of the GPU array
(412543, all 24 arms) wrote no rows: Nautilus and Emcee lost theirs to a row-writer bug after
completing, NUTS and SMC ran out of A100 memory and are now deferred in `wave.py`. Each submit
reads two environment variables, with defaults in the file:

- `PYAUTO_WAVE_ATTEMPT` (GPU default `2`, CPU default `1`): run ids gain `_a<N>` for N > 1, so
  a re-run never collides with an earlier attempt's run id (which `run.py` refuses);
- `PYAUTO_WAVE_SAMPLERS` (default `nautilus,emcee`): the arms the array covers. The array
  index is a position in that filtered list, in wave order, so `--array` is `0-<n-1>` for its
  `n` arms (`0-11` for Nautilus and Emcee); a deferred sampler is refused.

`hpc/sync submit` runs `sbatch` over ssh, which does not carry laptop variables, so the
defaults in the file are what a submit runs; a later attempt edits them in a commit. The GPU
attempt-2 array needs no dependency (the problems exist):
`hpc/sync submit submit_hst_cleanslate_mass_samplers_jax_gpu_dense`.

### Why the second cell exists

`hst_delaunay.py` is not a copy of `hst.py`; it is the same runner with
`default_mesh="delaunay"`, and it exists as a separate **file** because the wall gate reads
a submit's cell from the script path it invokes, and `wall/rates.py` keys its measured step
rates by that cell. **Cell id = cost profile = rate key.** A 1250-vertex Delaunay chain
costs a different amount per evaluation from a 784-cell rectangular one, so a `--time`
justified from the rectangular cell's rate would be a number measured on a different model
— the exact carry that killed 35 of 39 arms of an overnight A100 block. Its own leaf gives
it its own cell id and forces it to be measured on its own terms.

The mesh is a **run variant**, not a config: `--mesh` / `--mesh-pixels` name a directory
level between the instrument and the config name
(`results/imaging/<rectangular|delaunay>/hst/validation/<model_variant>/<setup_id>/`) and a field of the target id, so
these two submits' rows never join the base run's parity row. The config name still says
only which backend ran. See
[`../scripts/imaging/slam/README.md`](../scripts/imaging/slam/README.md).

Submit one with `hpc/sync submit --gpu|--cpu <full submit filename>` — the argument is the
file's name including the `submit_` prefix, because `sbatch` is handed it verbatim.

### The seed convention

`--array=0-1` and `--seed $SLURM_ARRAY_TASK_ID`. There is no `SEEDS=(...)` indirection
here because the seeds *are* 0 and 1; a submit whose seeds are not `0..N` reads them from
a bash array indexed by `SLURM_ARRAY_TASK_ID`, as described below. The seed reaches the
result filename (`stages_seed<n>.json`) and the PyAutoFit output path (`seed_<n>/`), so
the two arms never overwrite each other. `seed` is also a PyAutoFit *identifier* field
while `--config-name` is not, which is why the config name lives in the `path_prefix`:
without that, two backends at one seed would hash to the same identifier and silently
resume each other's fit.

### Memory: why the base submits all ask for `--mem=64gb`

The HST cell is **15,361 masked pixels**, and the memory cost is in the Nautilus batch,
not the dataset. Measured on 2026-09-11, a `jax_cpu` leg's vmap over `n_batch=20` at
`source_pix[1]` needs roughly:

| leg | RSS at `source_pix[1]` |
|---|---|
| `jax_cpu` × dense | ~13 GB |
| `jax_cpu` × sparse | ~21.6 GB |

Both OOM a 15 GB laptop, which is why the two JAX-CPU HST legs of the parity row exist
only as RAL jobs and why the local rate measurements are `--stages source_lp` only. The
`numba_cpu` legs have no vmap at all — with `use_jax=False` Nautilus takes the
`number_of_cores` multiprocessing path — and the production `source_lp[1]` probe held
~6.3 GB across nine processes; their pixelized-stage figure is **not** measured, so they
keep the same request rather than a smaller one guessed from a mechanism. `--mem=64gb` is the template's figure and it stays: it clears the worst measured
leg by 3x, and a leg killed by the OOM killer at stage two costs a whole night.

The clean-slate `jax_cpu` pair sizes from the same two figures: 48 GB for dense (~3.7x
its 13 GB) and 64 GB for sparse (~3x its 21.6 GB).

The two `delaunay_1250` submits scale those same measured figures by the mesh: 1250
vertices against 784 rectangular cells is ~1.6x, so ~21 GB dense and ~35 GB sparse. Dense
therefore keeps `--mem=64gb` (~3x its expected peak) and **sparse asks for `--mem=96gb`**
(~2.7x). The `gpu` nodes carry 885 GB, so the request is not what makes a job queue —
under-asking is the only expensive mistake here.

## Array submits (repeated-seed campaigns)

A submit that declares `#SBATCH --array=0-4` runs one **seed per array task** — the shape
a reliability measurement needs, since one run tells you a search *can* find the answer
and only a spread of seeds says how often it does. Read the seed from a `SEEDS=(...)` bash
array indexed by `SLURM_ARRAY_TASK_ID` and pass it as `--seed`; use the `%A_%a`
(job_array) pattern for stdout/stderr so each task's log is separate.

Every arm must land in its own result row **and** its own PyAutoFit output directory.
`--config-name` carries the tier and seed, which keeps the result filename distinct.
