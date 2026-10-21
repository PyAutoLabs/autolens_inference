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
{local,hpc_a100,hpc_ral}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}
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
[`scripts/imaging/rectangular/baseline_slam.py`](scripts/imaging/slam/README.md), one leaf over one chain,
switched by `--backend` and `--inversion`, with six SLURM submits behind it. The first four
legs came back on **2026-09-14** — A100, dense and sparse, seeds 0 and 1, all five stages —
and are the rows in the table below. They are one backend of the parity row: the CPU legs
have not run, so there is still nothing to compare them *to*.
`scripts/misc/wall/rates.py` still holds only the two `source_lp[1]` rates the submits had
to be sized against, because a completed wall on one cell is not a step rate for another.
A **run variant** level now sits between the instrument and the config name
(`slam_base`, `slam_delaunay`) so that a second mesh is a second experiment rather than a
seventh column. The ledger is [`wiki/project/state.md`](wiki/project/state.md).

## Pipeline runs

<!-- BEGIN auto-table:slam -->
| Target | Variant | Light | Stage | Config | Seed | Wall | Evals | log Z | Version |
|---|---|---|---|---|---|---|---|---|---|
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 3367.1 s | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_cpu_sparse_fp64` | 0 | 3472.0 s | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 714.9 s | 92,550 | 31,968.56 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 715.0 s | 92,550 | 31,968.56 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 10.61 h | 34,540 | 31,432.71 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_cpu_sparse_fp64` | 0 | 23.09 h | 34,540 | 31,432.71 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 700.9 s | 33,020 | 31,430.46 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1786.2 s | 32,820 | 31,430.63 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.38 h | 33,660 | 31,432.41 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 1.61 h | 5,560 | 31,625.61 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_cpu_sparse_fp64` | 0 | 1.76 h | 3,820 | 31,621.63 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 181.1 s | 7,800 | 31,623.71 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 298.1 s | 8,140 | 31,619.29 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 579.8 s | 5,660 | 31,620.87 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 1.51 h | 11,080 | 31,620.87 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_jax_cpu_sparse_fp64` | 0 | 1.10 h | 10,880 | 31,614.34 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 234.5 s | 11,600 | 31,620.71 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 221.9 s | 11,100 | 31,617.94 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1275.3 s | 10,520 | 31,617.29 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 9.01 h | 30,400 | 31,596.30 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_cpu_sparse_fp64` | 0 | 10.01 h | 21,720 | 31,594.55 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 558.6 s | 23,680 | 31,601.91 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 826.9 s | 23,060 | 31,596.51 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2605.2 s | 22,920 | 31,599.97 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 3410.5 s | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 712.4 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 719.9 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 10.73 h | 34,180 | 31,424.34 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 732.6 s | 34,520 | 31,416.54 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1648.6 s | 30,360 | 31,416.66 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.35 h | 33,200 | 31,424.34 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 1.12 h | 3,720 | 31,619.06 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 130.3 s | 5,000 | 31,619.36 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 226.3 s | 6,000 | 31,625.54 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 939.2 s | 9,460 | 31,617.40 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 1.58 h | 10,620 | 31,613.51 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 219.8 s | 10,100 | 31,612.66 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 196.7 s | 9,200 | 31,621.46 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1478.9 s | 12,000 | 31,616.15 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 6.70 h | 22,020 | 31,592.95 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 525.0 s | 21,640 | 31,591.75 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 811.5 s | 22,980 | 31,598.45 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2562.1 s | 22,240 | 31,596.13 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 3367.1 s | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 714.9 s | 92,550 | 31,968.56 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 715.0 s | 92,550 | 31,968.56 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 10.61 h | 34,540 | 31,432.71 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 700.9 s | 33,020 | 31,430.46 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1786.2 s | 32,820 | 31,430.63 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.38 h | 33,660 | 31,432.41 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 1.61 h | 5,560 | 31,625.61 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 181.1 s | 7,800 | 31,623.71 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 298.1 s | 8,140 | 31,619.29 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 579.8 s | 5,660 | 31,620.87 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | light[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 1.51 h | 11,080 | 31,620.87 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 234.5 s | 11,600 | 31,620.71 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 221.9 s | 11,100 | 31,617.94 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1275.3 s | 10,520 | 31,617.29 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_cpu_dense_fp64` | 0 | 6.04 h | 23,780 | 31,597.63 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 490.5 s | 22,820 | 31,600.99 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1624.9 s | 48,700 | -43,567.08 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed0` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2329.4 s | 24,340 | 31,598.85 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 3410.5 s | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 712.4 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 719.9 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 10.73 h | 34,180 | 31,424.34 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 732.6 s | 34,520 | 31,416.54 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1648.6 s | 30,360 | 31,416.66 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.35 h | 33,200 | 31,424.34 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 1.12 h | 3,720 | 31,619.06 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 130.3 s | 5,000 | 31,619.36 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 226.3 s | 6,000 | 31,625.54 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 939.2 s | 9,460 | 31,617.40 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | light[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 1.58 h | 10,620 | 31,613.51 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 219.8 s | 10,100 | 31,612.66 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 196.7 s | 9,200 | 31,621.46 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1478.9 s | 12,000 | 31,616.15 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_cpu_dense_fp64` | 1 | 5.58 h | 20,940 | 31,592.62 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 449.7 s | 20,780 | 31,592.66 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1396.9 s | 41,880 | -43,201.59 | 2026.8.17.1 |
| `hst/slam5_slam_base_light_fixed/seed1` | `slam_base_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2076.2 s | 21,960 | 31,595.85 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.50 h | 53,920 | 31,552.29 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 453.9 s | 2,120 | 31,584.21 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2269.4 s | 10,180 | 31,573.79 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2.40 h | 38,660 | 31,554.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.50 h | 53,920 | 31,552.29 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 453.9 s | 2,120 | 31,584.21 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2269.4 s | 10,180 | 31,573.79 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.27 h | 34,060 | 31,557.08 | 2026.8.17.1 |

**Parity — `hst/slam5/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_cpu_dense_fp64` (ref) | `hpc_a100_jax_cpu_sparse_fp64` | `hpc_a100_jax_gpu_dense_fp64` | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|---|---|---|
| `einstein_radius` | 1.609 ± 0.0026 | -0.75σ | -4.36σ | -1.94σ | -2.40σ |
| `slope` | 2.101 ± 0.032 | -0.81σ | -4.30σ | -1.92σ | -2.41σ |
| `shear_magnitude` | 0.07639 ± 0.0016 | -1.50σ | -4.11σ | -2.29σ | -2.77σ |
| `centre_0` | -0.0007246 ± 0.00032 | +1.77σ | -0.58σ | -0.31σ | +0.19σ |
| `centre_1` | -0.0003957 ± 0.00035 | -0.65σ | +0.30σ | -0.46σ | +0.22σ |
| `ell_comps_0` | 0.05729 ± 0.0019 | -0.57σ | -2.83σ | -1.78σ | -1.83σ |
| `ell_comps_1` | 0.003949 ± 0.0036 | -1.35σ | -0.58σ | -0.87σ | -1.01σ |
| `gamma_1` | 0.05615 ± 0.0016 | -2.04σ | -4.62σ | -2.47σ | -3.18σ |
| `gamma_2` | 0.05177 ± 0.00077 | -0.21σ | -2.62σ | -1.61σ | -1.57σ |

**Parity — `hst/slam5/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_cpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_dense_fp64` | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|---|---|
| `einstein_radius` | 1.604 ± 0.0034 | +1.03σ | -0.28σ | -0.42σ |
| `slope` | 2.047 ± 0.043 | +1.00σ | -0.23σ | -0.42σ |
| `shear_magnitude` | 0.07246 ± 0.0023 | +1.07σ | -0.01σ | -0.14σ |
| `centre_0` | -0.000236 ± 0.00047 | -0.25σ | -1.39σ | -0.93σ |
| `centre_1` | -0.0004575 ± 0.00045 | -0.41σ | -0.10σ | +0.37σ |
| `ell_comps_0` | 0.05453 ± 0.0024 | +1.06σ | -0.34σ | -0.17σ |
| `ell_comps_1` | -0.0008996 ± 0.001 | +0.22σ | +1.70σ | +1.01σ |
| `gamma_1` | 0.05156 ± 0.0023 | +1.02σ | +0.22σ | -0.07σ |
| `gamma_2` | 0.05091 ± 0.001 | +1.02σ | -0.51σ | -0.23σ |

**Parity — `hst/slam5_slam_base_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_cpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_dense_fp64` | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|---|---|
| `einstein_radius` | 1.603 ± 0.0033 | -0.89σ | +273.72σ | +0.23σ |
| `slope` | 2.027 ± 0.042 | -0.85σ | +22.37σ | +0.25σ |
| `shear_magnitude` | 0.07349 ± 0.0025 | -0.94σ | +16.24σ | -0.33σ |
| `centre_0` | -0.0005266 ± 0.00049 | -0.68σ | -0.05σ | -0.23σ |
| `centre_1` | -0.0004655 ± 0.00039 | +0.18σ | -0.38σ | +0.33σ |
| `ell_comps_0` | 0.05394 ± 0.0019 | -0.45σ | +276.16σ | +0.29σ |
| `ell_comps_1` | 0.00415 ± 0.0041 | -0.51σ | -0.85σ | -0.96σ |
| `gamma_1` | 0.0534 ± 0.0029 | -1.05σ | +16.04σ | -0.55σ |
| `gamma_2` | 0.05039 ± 0.0008 | -0.26σ | +6.56σ | +0.59σ |

**Parity — `hst/slam5_slam_base_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_cpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_dense_fp64` | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|---|---|
| `einstein_radius` | 1.603 ± 0.0029 | -0.62σ | +285.82σ | -0.40σ |
| `slope` | 2.031 ± 0.037 | -0.63σ | +25.02σ | -0.37σ |
| `shear_magnitude` | 0.07168 ± 0.002 | -0.49σ | +21.03σ | -0.03σ |
| `centre_0` | -0.0001717 ± 0.0005 | -0.07σ | -5.06σ | -1.00σ |
| `centre_1` | -0.000415 ± 0.00045 | -0.04σ | +3.46σ | +0.38σ |
| `ell_comps_0` | 0.05365 ± 0.0022 | -0.26σ | +234.12σ | -0.09σ |
| `ell_comps_1` | -0.0009049 ± 0.001 | +0.03σ | +5.51σ | +1.06σ |
| `gamma_1` | 0.05087 ± 0.002 | -0.61σ | +24.13σ | -0.02σ |
| `gamma_2` | 0.05052 ± 0.00098 | -0.20σ | +4.14σ | -0.13σ |

**Parity — `hst/slam5_slam_delaunay/seed1` (seed 1), `mass_total[1]`**

_Only one config has run for this target and seed — a parity view needs at least two legs._

**Parity — `hst/slam5_slam_delaunay_light_fixed/seed1` (seed 1), `mass_total[1]`**

_Only one config has run for this target and seed — a parity view needs at least two legs._

**Chain walls (fitted stages only)**

| Target | Seed | Config | Chain wall | Excludes |
|---|---|---|---|---|
| `hst/slam5/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 2.62 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 2.74 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_base_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 2.54 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_base_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 2.60 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 6.66 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 5.53 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
<!-- END auto-table:slam -->

## Search runs

<!-- BEGIN auto-table:searches -->
| Target | Sampler | Instrument | Config | Seed | Wall | Evals | Per call | Share | Log evidence | Version |
|---|---|---|---|---|---|---|---|---|---|---|
| `simple/source_plane_solved/seed0` | multi_start_adam | simple | `hpc_a100_jax_cpu_dense_fp64` | 0 | 9.3 s | 14,400 | 5.2 µs | 0.8% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed1` | multi_start_adam | simple | `hpc_a100_jax_cpu_dense_fp64` | 1 | 9.2 s | 14,400 | 5.2 µs | 0.8% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed2` | multi_start_adam | simple | `hpc_a100_jax_cpu_dense_fp64` | 2 | 7.5 s | 14,400 | 5.1 µs | 1.0% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed3` | multi_start_adam | simple | `hpc_a100_jax_cpu_dense_fp64` | 3 | 7.6 s | 14,400 | 4.7 µs | 0.9% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed4` | multi_start_adam | simple | `hpc_a100_jax_cpu_dense_fp64` | 4 | 7.1 s | 12,432 | 5.1 µs | 0.9% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed0` | multi_start_adam | simple | `hpc_a100_jax_gpu_dense_fp64` | 0 | 37.0 s | 14,400 | 8.3 µs | 0.3% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed1` | multi_start_adam | simple | `hpc_a100_jax_gpu_dense_fp64` | 1 | 19.2 s | 14,400 | 7.6 µs | 0.6% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed2` | multi_start_adam | simple | `hpc_a100_jax_gpu_dense_fp64` | 2 | 19.5 s | 14,400 | 7.1 µs | 0.5% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed3` | multi_start_adam | simple | `hpc_a100_jax_gpu_dense_fp64` | 3 | 20.3 s | 14,400 | 7.1 µs | 0.5% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed4` | multi_start_adam | simple | `hpc_a100_jax_gpu_dense_fp64` | 4 | 18.6 s | 12,432 | 7.3 µs | 0.5% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed0` | multi_start_adam_n3000 | simple | `hpc_a100_jax_cpu_dense_fp64` | 0 | 7.8 s | 33,792 | 5.0 µs | 2.2% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed1` | multi_start_adam_n3000 | simple | `hpc_a100_jax_cpu_dense_fp64` | 1 | 8.8 s | 52,848 | 5.0 µs | 3.0% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed2` | multi_start_adam_n3000 | simple | `hpc_a100_jax_cpu_dense_fp64` | 2 | 9.4 s | 68,736 | 5.1 µs | 3.7% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed3` | multi_start_adam_n3000 | simple | `hpc_a100_jax_cpu_dense_fp64` | 3 | 8.8 s | 61,488 | 5.1 µs | 3.6% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed4` | multi_start_adam_n3000 | simple | `hpc_a100_jax_cpu_dense_fp64` | 4 | 7.4 s | 12,432 | 4.8 µs | 0.8% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed0` | multi_start_adam_n3000 | simple | `hpc_a100_jax_gpu_dense_fp64` | 0 | 20.7 s | 33,792 | 7.2 µs | 1.2% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed1` | multi_start_adam_n3000 | simple | `hpc_a100_jax_gpu_dense_fp64` | 1 | 22.0 s | 52,848 | 7.3 µs | 1.8% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed2` | multi_start_adam_n3000 | simple | `hpc_a100_jax_gpu_dense_fp64` | 2 | 24.2 s | 68,736 | 8.0 µs | 2.3% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed3` | multi_start_adam_n3000 | simple | `hpc_a100_jax_gpu_dense_fp64` | 3 | 23.5 s | 61,488 | 7.0 µs | 1.8% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed4` | multi_start_adam_n3000 | simple | `hpc_a100_jax_gpu_dense_fp64` | 4 | 19.1 s | 12,432 | 7.9 µs | 0.5% | — | 2026.10.17.1 |
| `simple/source_plane_solved/seed0` | nautilus | simple | `hpc_a100_jax_cpu_dense_fp64` | 0 | 56.6 s | 4,850 | 4.7 µs | 0.041% | -7.15 | 2026.8.17.1 |
| `simple/source_plane_solved/seed1` | nautilus | simple | `hpc_a100_jax_cpu_dense_fp64` | 1 | 54.2 s | 4,850 | 5.0 µs | 0.045% | -7.16 | 2026.8.17.1 |
| `simple/source_plane_solved/seed2` | nautilus | simple | `hpc_a100_jax_cpu_dense_fp64` | 2 | 53.8 s | 4,800 | 4.8 µs | 0.043% | -7.13 | 2026.8.17.1 |
| `simple/source_plane_solved/seed3` | nautilus | simple | `hpc_a100_jax_cpu_dense_fp64` | 3 | 50.4 s | 4,700 | 4.6 µs | 0.043% | -7.14 | 2026.8.17.1 |
| `simple/source_plane_solved/seed4` | nautilus | simple | `hpc_a100_jax_cpu_dense_fp64` | 4 | 58.5 s | 4,750 | 5.0 µs | 0.041% | -7.17 | 2026.8.17.1 |
| `simple/source_plane_solved/seed0` | nautilus | simple | `local_jax_cpu_dense_fp64` | 0 | 60.4 s | 4,850 | 7.2 µs | 0.058% | -7.15 | 2026.8.17.1 |
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

## Setup catalogue

[`dashboard/catalogue.json`](dashboard/catalogue.json) publishes inference-summary@2
alongside the retained v1 summary. Stable setup IDs identify HST rectangular,
HST Delaunay and simple point-source families independently of runnable paths.
Original record IDs, archived results and output directories are preserved.
No historical initialization, frozen priors or scientific acceptance is inferred.

Runnable baselines now live at:

- [`scripts/imaging/rectangular/baseline_slam.py`](scripts/imaging/rectangular/baseline_slam.py)
- [`scripts/imaging/delaunay/baseline_slam.py`](scripts/imaging/delaunay/baseline_slam.py)
- [`scripts/point_source/simple/baseline.py`](scripts/point_source/simple/baseline.py)

The wall gate maps these leaves to their existing measured cell identities;
this migration creates no measurements or new timing justification.
Run `python scripts/misc/tooling/export_inference_summary.py` to regenerate both
feeds. Baseline export writes portable manifest metadata under tracked `prepared/<problem-id>/manifest.json` alongside local bulk snapshots in `output/autolens/prepared/`. Commit that metadata with the result rows so CI and fresh clones retain the problem declaration. These manifests attach
exact dataset/model/prior identities to their source baseline; unpublished
artifact paths do not promise publicly downloadable samples.

Export the completed baseline's frozen `mass_total[1]` problem with
`--export-prepared output/autolens/prepared/hst-rectangular` on the rectangular baseline
command. Then investigate its shared problem with:

```bash
python scripts/misc/experiments/run.py --prepared output/autolens/prepared/hst-rectangular/manifest.json --problem-id <manifest-id> --run-id hst-rectangular-nautilus-cold-seed0 --sampler nautilus --start cold --hardware-id laptop-cpu-8cores-single-process
```

Replace `<manifest-id>` with the manifest's exact `id` field and set `--hardware-id` to the actual host/device, allocated resources and concurrency.
Each imaging setup folder also provides `nautilus.py`, `emcee.py`, `nuts.py`
and `smc.py` investigation leaves. They bind the shared runner to the named
setup family and sampler and refuse a mismatched prepared problem. For example,
a prepared JAX Delaunay baseline can be investigated with:

```bash
python scripts/imaging/delaunay/smc.py --prepared output/autolens/prepared/hst-delaunay/manifest.json --problem-id <manifest-id> --run-id hst-delaunay-smc-cold-seed0 --start cold --hardware-id laptop-cpu-8cores-single-process
```

Use `python scripts/misc/experiments/run.py --help` for supported start modes
and samplers. Cold/warm/resume sampler initialization is recorded independently
of compilation and cache conditions. Frozen priors are never narrowed to create
a warm start. Executors verify manifest identities and artifact hashes before
reuse; unavailable historical artifacts remain unknown.
