# autolens_inference

Non-linear search and inference benchmarking for strong lensing with
[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) — the proving ground for *which
search finds the right lens model, how reliably, and at what cost*, across gradient-free
and gradient-based samplers, on CPU and on A100 GPUs.

It is the sibling of [autolens_profiling](https://github.com/PyAutoLabs/autolens_profiling),
which owns likelihood **timing**. This repo owns everything downstream of one likelihood
call: samplers, pipelines, convergence, reliability, and the wall-clock price of an answer.

> **Rulings of record live in [PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex)**
> (`projects.yaml`, row `autolens_inference`) — see [`CORTEX.md`](CORTEX.md). The project
> ledger is [`wiki/project/state.md`](wiki/project/state.md).

## Vision

Fitting a lens is not one likelihood call; it is tens of thousands of them, arranged by a
search that may or may not find the right answer. This repo measures that arrangement.

**What is measured:**

- **Pipelines** — the SLaM chains as they are actually run for science, end to end, stage
  by stage.
- **Searches** — one sampler on one model: Nautilus first, then the gradient-based
  optimisers and samplers, then whatever earns a place next to them.
- **Reliability** — not "did it finish" but "did it find the right answer, from a cold
  start, more often than not".

**Configurations covered.** Every run carries a `--config-name` from one grammar:

```
{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}
```

A **parity row** is a set of runs sharing one target — same dataset, same model, same
pipeline, same seed — that differ *only* in the config name. The backend is a column, never
a reason to split a table.

**Dataset framing.** Results are framed by astronomy instrument (HST first, then Euclid,
ALMA, …) rather than by raw pixel counts, so the headline numbers map onto a real
observing programme. The instrument is a `--instrument` flag, never a directory.

## Where this project is

Born **2026-09-10**. Nothing is inherited from the retired `inference_programme`: no
baselines, no notes, no result JSONs. The first science task is the base run — the 5-stage
HST SLaM chain under numba-CPU, JAX-CPU and A100, dense and sparse — which is what
everything later is improvement *against*.

The driver that runs it landed **2026-09-11** (phase 3):
[`scripts/imaging/slam/hst.py`](scripts/imaging/slam/README.md), one leaf over one chain,
switched by `--backend` and `--inversion`, with six SLURM submits behind it. The base run
itself is phase 4 and has not been run, so the tables below are still empty and
`scripts/misc/wall/rates.py` holds only the two `source_lp[1]` rates the submits had to be
sized against. The ledger is [`wiki/project/state.md`](wiki/project/state.md).

## Pipeline runs

<!-- BEGIN auto-table:slam -->
| Target | Variant | Stage | Config | Seed | Wall | Evals | log Z | Version |
|---|---|---|---|---|---|---|---|---|
| `hst/slam5/seed0` | `slam_base` | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 923.3 s | 95,950 | 31,968.39 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 808.2 s | 92,300 | 31,968.28 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 761.9 s | 32,820 | 31,418.34 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1930.8 s | 33,140 | 31,422.23 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 131.5 s | 3,720 | 31,618.43 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 178.6 s | 3,260 | 31,615.93 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 237.3 s | 9,620 | 31,611.07 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 267.5 s | 11,320 | 31,608.94 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 516.6 s | 20,080 | 31,592.70 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 800.3 s | 20,780 | 31,589.52 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 834.8 s | 96,150 | 31,968.40 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 798.2 s | 96,150 | 31,968.53 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 748.7 s | 33,000 | 31,437.89 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1904.3 s | 33,020 | 31,432.37 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 362.3 s | 13,800 | 31,620.54 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 187.8 s | 3,580 | 31,618.61 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 267.4 s | 12,200 | 31,618.61 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 253.9 s | 9,980 | 31,611.35 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 650.2 s | 25,820 | 31,596.11 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 868.1 s | 21,640 | 31,592.44 | 2026.8.17.1 |

**Parity — `hst/slam5/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.605 ± 0.0041 | -0.03σ |
| `slope` | 2.054 ± 0.051 | -0.04σ |
| `shear_magnitude` | 0.07295 ± 0.0025 | +0.00σ |
| `centre_0` | -0.0001093 ± 0.00058 | -0.31σ |
| `centre_1` | -0.0005402 ± 0.00044 | -0.00σ |
| `ell_comps_0` | 0.05506 ± 0.0024 | +0.06σ |
| `ell_comps_1` | -0.0007526 ± 0.00093 | +0.10σ |
| `gamma_1` | 0.05205 ± 0.0027 | -0.01σ |
| `gamma_2` | 0.05106 ± 0.00098 | +0.21σ |

**Parity — `hst/slam5/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.601 ± 0.0039 | +0.66σ |
| `slope` | 2.008 ± 0.049 | +0.67σ |
| `shear_magnitude` | 0.07161 ± 0.0022 | +0.36σ |
| `centre_0` | -0.0008513 ± 0.00031 | +2.22σ |
| `centre_1` | -8.889e-05 ± 0.00032 | -1.49σ |
| `ell_comps_0` | 0.05365 ± 0.0022 | +0.54σ |
| `ell_comps_1` | 0.001321 ± 0.00081 | -2.58σ |
| `gamma_1` | 0.05075 ± 0.0022 | +0.24σ |
| `gamma_2` | 0.05052 ± 0.00093 | +0.65σ |
<!-- END auto-table:slam -->

## Search runs

<!-- BEGIN auto-table:searches -->
_No search runs yet — results land under `results/searches/`._
<!-- END auto-table:searches -->

The tables above are auto-generated by `scripts/misc/tooling/build_readme.py` from the
result rows under `results/` — never edit them by hand. Run
`python scripts/misc/tooling/build_readme.py` after committing a result and commit the
refreshed table; CI checks idempotence with `--check`.

A pipeline row is one **stage** of one leg, because a SLaM chain that agrees on its final
answer while disagreeing at `source_pix[2]` has not agreed. Under the stage table,
`slam` also renders a **parity view** per (target, seed): the `mass_total[1]` posterior of
every config side by side, each cell the difference from the alphabetically-first config of
that seed in units of *that reference leg's* 1σ. It is the table the base-run question is
actually asked in — a backend is a column, and a column that drifts by more than a fraction
of a sigma is either a real numerical difference worth naming or a bug in one leg.

## Layout

```
scripts/<dataset_class>/<task>/<leaf>.py   imaging|interferometer|point_source × slam|searches
scripts/misc/                              shared framework: simulators, tooling, wall, test
_inference_cli.py                          shared CLI / JSON / auto-simulate helper
instruments/                               instrument presets (pixel scale, shape)
config/                                    PyAutoConf overrides — outputs are KEPT here
hpc/                                       hpc/sync (RAL driver) + SLURM templates
results/{slam,searches}/                   committed result rows
wiki/project/state.md                      the Cortex ledger
```

## Running

```bash
source activate.sh                                    # on RAL; locally use the workspace env
python scripts/misc/simulators/imaging.py --instrument hst
```

Operational detail — backend facts, the config grammar, the RAL story, the testing gate —
is in [`AGENTS.md`](AGENTS.md). RAL specifics are in [`hpc/README.md`](hpc/README.md).
