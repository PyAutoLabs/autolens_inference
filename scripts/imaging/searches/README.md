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
  `output/prepared/cleanslate_2026_10_17/<config>/seed_0/<variant>/manifest.json`; the
  problem id is read from the manifest at run time and the leaf re-verifies it.
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
`row`, `failed` or `missing` and never infers success. Wave 2 (NSS, Dynesty, Zeus) needs
runner support first.
