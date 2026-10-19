# `scripts/imaging/searches`

Single-**search** runs on imaging data: one sampler, one model, one seed. Where `slam/` asks
"what does the production chain cost and does it agree across backends", this asks "does
*this* search find the right answer, how often, and how fast".

Empty in phase 1. Leaves are added one sampler at a time, each in its own directory:

```
scripts/imaging/searches/<sampler>/<target>.py
```

Nautilus first, because it is what production runs; the gradient-based optimisers and
samplers follow once the base run exists to compare them against.

## Rules that apply to every leaf here

- **The instrument is a flag** (`--instrument`), never a directory.
- **A reliability claim needs seeds.** One run tells you a search *can* find the answer;
  it takes a spread of `--seed` values to say how often it does.
- **Result rows go to `results/searches/`**, one per run, carrying `target`,
  `sampler`, `config_name`, `instrument`, `wall_s` and the evidence — the columns the
  README auto-table renders.
- **No submit without a wall-clock basis.** See
  [`../../misc/wall/README.md`](../../misc/wall/README.md): an unmeasured cell declares
  itself as one and probes first.

## Clean-slate wave 1: four samplers on one frozen mass problem (2026-10-17)

The HST sampler comparison of PyAutoInsight campaign `hst-imaging-mass-samplers` does not
add leaves here: it runs the sampler leaves of `scripts/imaging/rectangular/` (`nautilus`,
`emcee`, `nuts`, `smc`, one runner, `scripts/misc/experiments/run.py`) on the frozen
`mass_total[1]` problems the clean-slate base exports. Wave 1
(`cleanslate_2026_10_17_wave1`, autolens_inference#27) is:

- **Problems:** base seed 0, both light modes (`slam_base`, `slam_base_lightfixed`), from
  `output/autolens/prepared/<config>/seed_0/<variant>/manifest.json`; the problem id is
  read from the manifest at run time and the leaf re-verifies it. Each arm's PyAutoFit run
  lands under `output/autolens/experiments/<run id>/` (source snapshots under
  `output/autolens/experiment_sources/`). The wave and run-id names keep the dated
  `cleanslate_2026_10_17` / `cs1017` spelling; only the folders were flattened (#35).
- **Samplers, all `--start cold`:** `nautilus --live 150 --batch 20` (the control: the
  base stage's own settings), `emcee --walkers 50 --steps 2000`,
  `nuts --chains 1 --warmup 500 --steps 1000`, `smc --particles 256 --steps 1000`.
- **Seeds:** sampler seeds 0, 1, 2 per arm: 4 x 3 x 2 = 24 arms per config.
- **Run ids:** `cs1017_<config>_<variant>_<sampler>_s<seed>`; rows land in
  `results/searches/experiments/<run id>.json`, catalogued under their `problem_id` with
  no cross-problem or cross-sampler ranking.
- **Where:** `hpc_a100_jax_gpu_dense_fp64` on the A100 (submitted with
  `--dependency afterok:` on the GPU dense base array) and `hpc_a100_jax_cpu_dense_fp64`
  on `ral`, which **waits for the CPU base rows** (days) before it is submitted.

The arm table is data in
[`../../misc/experiments/wave.py`](../../misc/experiments/wave.py):
`--list --config <config>` prints every arm, `--account --config <config>` reports each as
`row`, `failed`, `missing` or `deferred` and never infers success. Wave 2 (NSS, Dynesty,
Zeus) needs runner support first.

**Attempt 1 failed (RAL array 412543, 2026-10-17); attempt 2 re-runs Nautilus and Emcee.**
Every Nautilus and Emcee arm completed its fit (the PyAutoFit trees exist under
`output/autolens/experiments/cs1017_*` once flattened) and then lost its row: `run.py` keyed
`max_likelihood_parameters` by `samples.names`, which PyAutoFit returns as *tuples* of dotted
aliases, and the JSON write refused them (autolens_inference#33; now keyed through the SLaM
runner's `posterior_path`, and a diagnostics failure after a completed fit is recorded as
`complete; diagnostics failed: ...` rather than losing the row). NUTS (`jit_log_l`, 385 GiB)
and SMC (256-particle `jit_scan`, 48 GiB) ran out of memory on the 80 GB A100: both are
**deferred** in wave 1 (`wave.py` `DEFERRED`, reason "A100 OOM 2026-10-17 (385 GiB NUTS,
48 GiB SMC)") pending a memory-feasible formulation (the sparse-operator problem
`hpc_a100_jax_gpu_sparse_fp64`, fewer particles, or a CPU problem with host memory).
A re-run of an arm is a new attempt: `--attempt N` appends `_a<N>` to the run id (run ids are
never reused), and `--sampler-filter nautilus,emcee` restricts a submit to the twelve
Nautilus and Emcee arms. `--account` lists every attempt per arm; the arm's live state is its
latest complete row.
