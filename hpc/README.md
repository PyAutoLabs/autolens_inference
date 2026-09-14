# hpc

Driving the RAL HPC from the laptop, and getting runs back.

## Layout

```
sync                    laptop-side driver (below)
sync.conf.example       template; sync.conf is gitignored
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
| `hpc/sync submit [--gpu\|--cpu] <name>` | `sbatch` a submit script, from `hpc/batch_<type>/` |
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
cd /mnt/ral/jnightin/autolens_inference
git status        # look before you pull — the checkout is often dirty
git pull
```

The PyAuto* libraries are a separate story again: resolved from the shared checkouts on
`PYTHONPATH` and updated with `HPCPullPyAuto`, never pip-installed and never rsynced.

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
| `NUMBA_CACHE_DIR` / `MPLCONFIGDIR` | compute nodes cannot write the default cache locations |

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

Nine, over **two cells**. Seven run `imaging/slam/hst` (`scripts/imaging/slam/hst.py`, the
backend-parameterised SLaM driver): six are the six legs of the first parity row, and the
seventh is the rate probe that had to run before any of them could be sized. Two run
`imaging/slam/hst_delaunay` (`scripts/imaging/slam/hst_delaunay.py`) — the same five-stage
chain on the `delaunay_1250` run variant.

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
(`results/slam/imaging/hst/<variant>/<config_name>/`) and a field of the target id, so
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
