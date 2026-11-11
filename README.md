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
{local,hpc_a100,hpc_ral}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse,na}_{fp64,mp}[_{nufft,wtilde}]
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
[`scripts/imaging/rectangular/validation/baseline_slam.py`](scripts/imaging/slam/README.md), one leaf over one chain,
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
| `cluster/image_plane/simple/modeling_image_plane/seed0` | `modeling_image_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 0 | 318.2 s | 20,550 | -3.05 | 2026.8.17.1 |
| `cluster/image_plane/simple/modeling_image_plane/seed0` | `modeling_image_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 0 | 597.7 s | 20,550 | -3.05 | 2026.8.17.1 |
| `cluster/image_plane/simple/modeling_image_plane/seed1` | `modeling_image_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 1 | 334.9 s | 21,900 | -2.84 | 2026.8.17.1 |
| `cluster/image_plane/simple/modeling_image_plane/seed1` | `modeling_image_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 1 | 330.2 s | 21,900 | -2.84 | 2026.8.17.1 |
| `cluster/source_plane/simple/modeling_source_plane/seed0` | `modeling_source_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 0 | 140.2 s | 20,700 | 5.73 | 2026.8.17.1 |
| `cluster/source_plane/simple/modeling_source_plane/seed0` | `modeling_source_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 0 | 119.2 s | 20,700 | 5.73 | 2026.8.17.1 |
| `cluster/source_plane/simple/modeling_source_plane/seed1` | `modeling_source_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 1 | 165.3 s | 24,300 | 5.62 | 2026.8.17.1 |
| `cluster/source_plane/simple/modeling_source_plane/seed1` | `modeling_source_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 1 | 145.6 s | 23,850 | 5.87 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 318.4 s | 36,500 | 32,540.96 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.21 h | 37,000 | 32,540.93 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1979.1 s | 43,140 | 32,120.97 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.14 h | 38,740 | 32,148.00 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 82.3 s | 2,020 | 32,182.59 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 455.3 s | 1,900 | 32,169.62 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1752.5 s | 58,540 | 32,908.68 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.18 h | 52,060 | 32,884.94 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2168.9 s | 47,260 | 32,950.75 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2.89 h | 39,200 | 32,932.97 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 481.7 s | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.14 h | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 319.0 s | 37,350 | 32,510.62 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.22 h | 37,450 | 32,555.82 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1974.2 s | 44,060 | 32,121.56 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.17 h | 38,860 | 32,136.61 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 411.3 s | 21,020 | 32,160.41 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 462.1 s | 1,920 | 32,186.57 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1557.6 s | 51,700 | 32,900.43 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.17 h | 51,400 | 32,897.03 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1521.2 s | 33,500 | 32,951.98 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.00 h | 42,260 | 32,932.41 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 439.9 s | 39,550 | -134,831.38 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.08 h | 42,650 | -134,831.35 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 318.4 s | 36,500 | 32,540.96 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.21 h | 37,000 | 32,540.93 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1979.1 s | 43,140 | 32,120.97 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.14 h | 38,740 | 32,148.00 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 82.3 s | 2,020 | 32,182.59 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 455.3 s | 1,900 | 32,169.62 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1752.5 s | 58,540 | 32,908.68 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.18 h | 52,060 | 32,884.94 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1771.8 s | 46,760 | 32,943.29 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2.25 h | 47,880 | 32,919.75 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 481.7 s | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.14 h | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 319.0 s | 37,350 | 32,510.62 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.22 h | 37,450 | 32,555.82 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1974.2 s | 44,060 | 32,121.56 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.17 h | 38,860 | 32,136.61 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 411.3 s | 21,020 | 32,160.41 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 462.1 s | 1,920 | 32,186.57 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1557.6 s | 51,700 | 32,900.43 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.17 h | 51,400 | 32,897.03 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1125.7 s | 29,800 | 32,940.57 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.97 h | 41,580 | 32,924.38 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 439.9 s | 39,550 | -134,831.38 | 2026.8.17.1 |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.08 h | 42,650 | -134,831.35 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 768.6 s | 93,950 | 33,451.76 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1.40 h | 94,100 | 33,451.77 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed0` | `slam_mge` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 143.1 s | 12,100 | 33,527.39 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed0` | `slam_mge` | linear | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 460.0 s | 12,140 | 33,526.90 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 411.3 s | 47,460 | 33,489.66 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1684.2 s | 48,940 | 33,489.64 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 750.7 s | 93,250 | 33,451.81 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1.39 h | 94,150 | 33,451.80 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed1` | `slam_mge` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 138.2 s | 12,220 | 33,527.81 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed1` | `slam_mge` | linear | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 457.1 s | 12,160 | 33,527.55 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 430.9 s | 49,420 | 33,489.64 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1737.1 s | 50,480 | 33,489.75 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 768.6 s | 93,950 | 33,451.76 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1.40 h | 94,100 | 33,451.77 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 143.1 s | 12,100 | 33,527.39 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 460.0 s | 12,140 | 33,526.90 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 407.5 s | 49,000 | 33,488.52 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1619.5 s | 51,480 | 33,488.53 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 750.7 s | 93,250 | 33,451.81 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1.39 h | 94,150 | 33,451.80 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 138.2 s | 12,220 | 33,527.81 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 457.1 s | 12,160 | 33,527.55 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 417.9 s | 49,920 | 33,488.58 | 2026.8.17.1 |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1495.5 s | 47,360 | 33,488.73 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 338.1 s | 36,500 | 32,540.96 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.21 h | 37,000 | 32,540.93 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1304.7 s | 33,480 | 32,236.93 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.09 h | 32,840 | 32,236.68 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 159.2 s | 3,280 | 32,270.28 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 979.5 s | 3,480 | 32,272.43 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 929.2 s | 34,060 | 32,994.26 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.44 h | 36,000 | 32,999.04 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 866.5 s | 22,060 | 33,047.94 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.96 h | 22,460 | 33,050.82 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 478.4 s | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.14 h | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 332.3 s | 37,350 | 32,510.62 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.43 h | 37,450 | 32,555.82 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1292.0 s | 33,260 | 32,212.93 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.84 h | 33,820 | 32,248.84 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 153.2 s | 3,120 | 32,246.32 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1334.8 s | 3,700 | 32,282.21 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 931.5 s | 34,480 | 33,000.17 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.17 h | 34,100 | 32,999.65 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 889.9 s | 22,600 | 33,049.03 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2.27 h | 22,200 | 33,047.11 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 447.7 s | 39,550 | -134,831.38 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.29 h | 42,650 | -134,831.35 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 338.1 s | 36,500 | 32,540.96 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.21 h | 37,000 | 32,540.93 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1304.7 s | 33,480 | 32,236.93 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.09 h | 32,840 | 32,236.68 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 159.2 s | 3,280 | 32,270.28 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 979.5 s | 3,480 | 32,272.43 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 929.2 s | 34,060 | 32,994.26 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.44 h | 36,000 | 32,999.04 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 773.0 s | 22,660 | 33,037.03 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.02 h | 21,820 | 33,038.85 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 478.4 s | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.14 h | 45,500 | -134,831.39 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 332.3 s | 37,350 | 32,510.62 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.43 h | 37,450 | 32,555.82 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1292.0 s | 33,260 | 32,212.93 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.84 h | 33,820 | 32,248.84 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 153.2 s | 3,120 | 32,246.32 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1334.8 s | 3,700 | 32,282.21 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 931.5 s | 34,480 | 33,000.17 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.17 h | 34,100 | 32,999.65 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 767.4 s | 22,300 | 33,038.45 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.13 h | 22,180 | 33,036.50 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_lp[0] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 447.7 s | 39,550 | -134,831.38 | 2026.8.17.1 |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_lp[0] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.29 h | 42,650 | -134,831.35 | 2026.8.17.1 |
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
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2064.6 s | 50,480 | 31,566.58 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.35 h | 53,240 | 31,561.94 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 80.0 s | 2,120 | 31,497.84 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 490.5 s | 2,180 | 31,454.88 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 374.8 s | 12,080 | 31,486.76 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2888.1 s | 11,920 | 31,443.70 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1274.3 s | 31,480 | 31,471.39 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.93 h | 31,360 | 31,426.95 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2464.9 s | 58,380 | 31,552.86 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.50 h | 53,920 | 31,552.29 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 84.6 s | 2,200 | 31,493.17 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 453.9 s | 2,120 | 31,584.21 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 387.9 s | 12,260 | 31,483.48 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2269.4 s | 10,180 | 31,573.79 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1048.4 s | 25,180 | 31,472.18 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2.40 h | 38,660 | 31,554.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2064.6 s | 50,480 | 31,566.58 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.35 h | 53,240 | 31,561.94 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 80.0 s | 2,120 | 31,497.84 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 490.5 s | 2,180 | 31,454.88 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 374.8 s | 12,080 | 31,486.76 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2888.1 s | 11,920 | 31,443.70 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 963.3 s | 26,620 | 31,471.75 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.18 h | 31,140 | 31,426.86 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2464.9 s | 58,380 | 31,552.86 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.50 h | 53,920 | 31,552.29 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 84.6 s | 2,200 | 31,493.17 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 453.9 s | 2,120 | 31,584.21 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 387.9 s | 12,260 | 31,483.48 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2269.4 s | 10,180 | 31,573.79 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 820.1 s | 23,020 | 31,472.48 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.27 h | 34,060 | 31,557.08 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2205.9 s | 38,320 | 31,505.77 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.82 h | 36,780 | 31,507.15 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 92.0 s | 2,280 | 31,391.04 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 597.4 s | 2,100 | 31,449.61 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 379.5 s | 11,440 | 31,380.25 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3176.2 s | 10,660 | 31,438.76 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1083.4 s | 21,620 | 31,400.21 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed0` | `slam_delaunay_nn` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2.00 h | 24,100 | 31,451.57 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1876.1 s | 32,280 | 31,480.96 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.48 h | 31,880 | 31,481.08 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 88.8 s | 2,260 | 31,399.09 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 669.9 s | 2,200 | 31,460.77 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 452.2 s | 13,080 | 31,388.80 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3054.3 s | 10,140 | 31,450.64 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1073.0 s | 21,340 | 31,382.31 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn/seed1` | `slam_delaunay_nn` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.78 h | 22,920 | 31,460.08 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | shared from `hpc_a100_jax_cpu_dense_fp64` | 92,550 | 31,968.38 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2205.9 s | 38,320 | 31,505.77 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.82 h | 36,780 | 31,507.15 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 92.0 s | 2,280 | 31,391.04 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 597.4 s | 2,100 | 31,449.61 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 379.5 s | 11,440 | 31,380.25 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3176.2 s | 10,660 | 31,438.76 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 873.8 s | 20,100 | 31,399.81 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | `slam_delaunay_nn_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.05 h | 21,860 | 31,448.01 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | shared from `hpc_a100_jax_cpu_dense_fp64` | 91,250 | 31,968.23 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1876.1 s | 32,280 | 31,480.96 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.48 h | 31,880 | 31,481.08 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 88.8 s | 2,260 | 31,399.09 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 669.9 s | 2,200 | 31,460.77 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 452.2 s | 13,080 | 31,388.80 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3054.3 s | 10,140 | 31,450.64 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 850.4 s | 19,580 | 31,380.77 | 2026.8.17.1 |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | `slam_delaunay_nn_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.04 h | 20,760 | 31,457.24 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 477.9 s | 61,900 | 31,965.17 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 2163.7 s | 61,750 | 31,965.15 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed0` | `slam_mge` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 66.5 s | 6,320 | 32,038.12 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed0` | `slam_mge` | linear | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 135.8 s | 6,120 | 32,038.30 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 402.2 s | 45,800 | 31,986.24 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1374.0 s | 44,980 | 31,986.32 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 480.5 s | 63,500 | 31,965.14 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 2225.3 s | 63,450 | 31,965.14 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed1` | `slam_mge` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 71.4 s | 6,200 | 32,038.33 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed1` | `slam_mge` | linear | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 137.1 s | 6,180 | 32,037.83 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 451.7 s | 48,760 | 31,985.97 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1451.1 s | 47,480 | 31,986.43 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 477.9 s | 61,900 | 31,965.17 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 2163.7 s | 61,750 | 31,965.15 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 66.5 s | 6,320 | 32,038.12 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 135.8 s | 6,120 | 32,038.30 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 434.9 s | 48,020 | 31,985.88 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1388.8 s | 47,700 | 31,986.04 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 480.5 s | 63,500 | 31,965.14 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 2225.3 s | 63,450 | 31,965.14 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 71.4 s | 6,200 | 32,038.33 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 137.1 s | 6,180 | 32,037.83 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 372.3 s | 43,080 | 31,985.98 | 2026.8.17.1 |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1302.7 s | 44,480 | 31,985.98 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 43.8 s | 4,300 | -3,154.28 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 77.5 s | 4,150 | -3,154.29 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 185.7 s | 6,660 | -3,163.82 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 1206.6 s | 14,380 | -3,174.89 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 34.4 s | 1,660 | -3,157.30 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 573.0 s | 8,040 | -3,158.39 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 164.4 s | 6,900 | -3,163.72 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 333.3 s | 5,280 | -3,162.72 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 34.5 s | 4,000 | -3,154.27 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 1 | 78.8 s | 4,200 | -3,154.31 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 205.8 s | 7,800 | -3,164.25 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 1 | 1553.8 s | 15,340 | -3,164.99 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 32.0 s | 1,660 | -3,156.96 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 1 | 1004.9 s | 12,560 | -3,158.28 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 220.3 s | 8,560 | -3,164.82 | 2026.8.17.1 |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 1 | 1994.4 s | 21,960 | -3,164.46 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64_nufft` | 0 | 49.9 s | 5,150 | -3,153.94 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64_nufft` | 0 | 736.6 s | 5,150 | -3,153.94 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64_nufft` | 0 | 38.5 s | 3,960 | -3,154.04 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64_nufft` | 0 | 562.5 s | 3,960 | -3,154.07 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64_nufft` | 1 | 42.6 s | 4,900 | -3,153.93 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64_nufft` | 1 | 693.0 s | 4,900 | -3,153.93 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64_nufft` | 1 | 44.5 s | 4,580 | -3,154.02 | 2026.8.17.1 |
| `interferometer/mge/sma/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64_nufft` | 1 | 598.4 s | 4,200 | -3,154.06 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 39.2 s | 4,300 | -3,154.28 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 78.7 s | 4,150 | -3,154.29 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 169.2 s | 9,400 | -3,164.42 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 2354.7 s | 26,920 | -3,175.18 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 50.2 s | 2,760 | -3,157.75 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 2700.6 s | 36,620 | -3,157.79 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 0 | 182.2 s | 9,500 | -3,163.04 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | `slam_rectangular` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 0 | 1371.2 s | 12,240 | -3,160.50 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 35.8 s | 4,000 | -3,154.27 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 199.9 s | 11,580 | -3,164.94 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 33.3 s | 1,800 | -3,157.57 | 2026.8.17.1 |
| `interferometer/rectangular/sma/slam_rectangular/seed1` | `slam_rectangular` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 1 | 198.8 s | 11,260 | -3,163.93 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 875.8 s | 103,700 | 74,675.42 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2.75 h | 99,300 | 74,675.80 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2.52 h | 124,660 | 73,810.01 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 15.58 h | 112,500 | 73,749.16 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 695.5 s | 11,100 | 73,570.74 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.28 h | 9,700 | 73,691.99 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2919.7 s | 44,060 | 73,587.77 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.08 h | 32,320 | 73,695.83 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1.32 h | 60,400 | 73,578.25 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 5.12 h | 46,100 | 73,708.03 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 841.3 s | 96,500 | 74,676.07 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2.58 h | 94,650 | 74,675.77 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2.23 h | 109,720 | 73,778.18 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 17.34 h | 121,000 | 73,765.58 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 679.9 s | 10,960 | 73,741.14 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.40 h | 11,520 | 73,726.34 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2228.0 s | 33,760 | 73,742.87 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.65 h | 55,360 | 73,727.58 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 3504.1 s | 46,580 | 73,755.28 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.77 h | 60,620 | 73,723.28 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 875.8 s | 103,700 | 74,675.42 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2.75 h | 99,300 | 74,675.80 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2.52 h | 124,660 | 73,810.01 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 15.58 h | 112,500 | 73,749.16 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 695.5 s | 11,100 | 73,570.74 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 1.28 h | 9,700 | 73,691.99 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2919.7 s | 44,060 | 73,587.77 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.08 h | 32,320 | 73,695.83 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 3577.4 s | 56,840 | 73,584.32 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.09 h | 43,180 | 73,707.16 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 841.3 s | 96,500 | 74,676.07 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2.58 h | 94,650 | 74,675.77 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2.23 h | 109,720 | 73,778.18 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 17.34 h | 121,000 | 73,765.58 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 679.9 s | 10,960 | 73,741.14 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 1.40 h | 11,520 | 73,726.34 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2228.0 s | 33,760 | 73,742.87 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.65 h | 55,360 | 73,727.58 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 3064.3 s | 48,080 | 73,753.31 | 2026.8.17.1 |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 5.26 h | 75,740 | 73,730.81 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1777.9 s | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 6.17 h | 264,400 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 3161.0 s | 82,540 | 11,653.16 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 6.00 h | 88,840 | 11,668.52 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 99.1 s | 2,380 | 11,778.23 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 451.1 s | 2,600 | 11,743.83 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1431.5 s | 46,820 | 11,774.48 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.59 h | 50,600 | 11,736.26 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1.10 h | 98,140 | 11,765.14 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.97 h | 83,000 | 11,730.15 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1910.5 s | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.03 h | 260,550 | 7,984.36 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1.07 h | 99,200 | 11,654.72 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.04 h | 88,620 | 11,659.68 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 101.3 s | 2,520 | 11,781.43 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 398.7 s | 2,060 | 11,798.18 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1449.5 s | 47,100 | 11,776.13 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.49 h | 50,200 | 11,791.58 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 3213.7 s | 79,180 | 11,767.79 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | `slam_delaunay` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.78 h | 81,840 | 11,789.55 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1777.9 s | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 6.17 h | 264,400 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 3161.0 s | 82,540 | 11,653.16 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 6.00 h | 88,840 | 11,668.52 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 99.1 s | 2,380 | 11,778.23 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 451.1 s | 2,600 | 11,743.83 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1431.5 s | 46,820 | 11,774.48 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.59 h | 50,600 | 11,736.26 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 3494.6 s | 103,000 | 11,763.28 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.36 h | 81,180 | 11,724.41 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1910.5 s | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.03 h | 260,550 | 7,984.36 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1.07 h | 99,200 | 11,654.72 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 6.04 h | 88,620 | 11,659.68 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 101.3 s | 2,520 | 11,781.43 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 398.7 s | 2,060 | 11,798.18 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1449.5 s | 47,100 | 11,776.13 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.49 h | 50,200 | 11,791.58 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2630.3 s | 76,200 | 11,763.92 | 2026.8.17.1 |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | `slam_delaunay_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.16 h | 80,520 | 11,784.02 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1774.3 s | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed0` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 4.26 h | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed0` | `slam_mge` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 644.2 s | 65,520 | 8,064.36 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed0` | `slam_mge` | linear | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 2937.3 s | 65,520 | 8,064.36 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1294.8 s | 131,460 | 12,413.52 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed0` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1.16 h | 131,160 | 12,413.55 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1873.1 s | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed1` | `slam_mge` | linear | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 4.24 h | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed1` | `slam_mge` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 731.7 s | 73,040 | 8,066.34 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed1` | `slam_mge` | linear | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 2995.1 s | 73,040 | 8,066.34 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1306.9 s | 133,340 | 12,410.22 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge/seed1` | `slam_mge` | linear | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 1.06 h | 131,700 | 12,410.19 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1774.3 s | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 4.26 h | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 644.2 s | 65,520 | 8,064.36 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 2937.3 s | 65,520 | 8,064.36 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1287.3 s | 133,980 | 11,850.03 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 1.13 h | 135,700 | 11,849.99 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1873.1 s | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | source_lp[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 4.24 h | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 731.7 s | 73,040 | 8,066.34 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | light[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 2995.1 s | 73,040 | 8,066.34 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1231.2 s | 132,420 | 11,848.18 | 2026.8.17.1 |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | `slam_mge_light_fixed` | fixed | mass_total[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 3271.3 s | 132,420 | 11,848.18 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1783.0 s | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.85 h | 245,300 | 7,984.57 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2084.6 s | 68,240 | 11,660.07 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.98 h | 69,680 | 11,661.67 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 174.1 s | 4,000 | 11,857.81 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 732.8 s | 3,760 | 11,858.79 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1359.3 s | 52,100 | 11,857.87 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.46 h | 47,840 | 11,860.56 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2152.2 s | 66,960 | 11,844.66 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | `slam_rectangular` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.90 h | 68,400 | 11,844.64 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1884.7 s | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 5.64 h | 284,750 | 7,984.54 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2183.5 s | 69,320 | 11,654.38 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.81 h | 68,660 | 11,656.30 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 145.4 s | 3,540 | 11,859.32 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 749.9 s | 3,840 | 11,853.63 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1106.2 s | 42,000 | 11,860.67 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.84 h | 53,640 | 11,858.41 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2053.2 s | 64,180 | 11,843.35 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | `slam_rectangular` | linear | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.67 h | 65,860 | 11,842.98 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1783.0 s | 249,950 | 7,984.56 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.85 h | 245,300 | 7,984.57 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 2084.6 s | 68,240 | 11,660.07 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 4.98 h | 69,680 | 11,661.67 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 174.1 s | 4,000 | 11,857.81 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 732.8 s | 3,760 | 11,858.79 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1359.3 s | 52,100 | 11,857.87 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 3.46 h | 47,840 | 11,860.56 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1873.1 s | 65,460 | 11,841.73 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 0 | 2.34 h | 64,640 | 11,842.64 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1884.7 s | 258,100 | 7,984.71 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_lp[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 5.64 h | 284,750 | 7,984.54 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2183.5 s | 69,320 | 11,654.38 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 4.81 h | 68,660 | 11,656.30 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 145.4 s | 3,540 | 11,859.32 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | source_pix[2] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 749.9 s | 3,840 | 11,853.63 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1106.2 s | 42,000 | 11,860.67 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | light[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 3.84 h | 53,640 | 11,858.41 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1867.5 s | 66,000 | 11,841.69 | 2026.8.17.1 |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | `slam_rectangular_light_fixed` | fixed | mass_total[1] | `hpc_ral_numba_cpu_sparse_fp64` | 1 | 2.41 h | 65,160 | 11,840.81 | 2026.8.17.1 |
| `point_source_image/image_plane/simple/modeling_image_plane/seed0` | `modeling_image_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 0 | 58.5 s | 5,250 | 4.29 | 2026.8.17.1 |
| `point_source_image/image_plane/simple/modeling_image_plane/seed0` | `modeling_image_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 0 | 55.5 s | 5,250 | 4.29 | 2026.8.17.1 |
| `point_source_image/image_plane/simple/modeling_image_plane/seed1` | `modeling_image_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 1 | 49.0 s | 5,000 | 4.31 | 2026.8.17.1 |
| `point_source_image/image_plane/simple/modeling_image_plane/seed1` | `modeling_image_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 1 | 50.5 s | 5,000 | 4.31 | 2026.8.17.1 |
| `point_source_source/source_plane/simple/modeling_source_plane/seed0` | `modeling_source_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 0 | 43.1 s | 4,850 | -7.15 | 2026.8.17.1 |
| `point_source_source/source_plane/simple/modeling_source_plane/seed0` | `modeling_source_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 0 | 34.9 s | 4,850 | -7.15 | 2026.8.17.1 |
| `point_source_source/source_plane/simple/modeling_source_plane/seed1` | `modeling_source_plane` | linear | modeling[1] | `hpc_a100_jax_gpu_na_fp64` | 1 | 43.8 s | 4,850 | -7.16 | 2026.8.17.1 |
| `point_source_source/source_plane/simple/modeling_source_plane/seed1` | `modeling_source_plane` | linear | modeling[1] | `hpc_ral_jax_cpu_na_fp64` | 1 | 42.8 s | 4,850 | -7.16 | 2026.8.17.1 |
| `weak/isothermal/simple/modeling_isothermal/seed0` | `modeling_isothermal` | linear | modeling[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 13.6 s | 2,500 | -93.00 | 2026.8.17.1 |
| `weak/isothermal/simple/modeling_isothermal/seed0` | `modeling_isothermal` | linear | modeling[1] | `hpc_ral_jax_cpu_dense_fp64` | 0 | 12.4 s | 2,500 | -93.00 | 2026.8.17.1 |
| `weak/isothermal/simple/modeling_isothermal/seed1` | `modeling_isothermal` | linear | modeling[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 12.9 s | 2,500 | -93.01 | 2026.8.17.1 |
| `weak/isothermal/simple/modeling_isothermal/seed1` | `modeling_isothermal` | linear | modeling[1] | `hpc_ral_jax_cpu_dense_fp64` | 1 | 12.0 s | 2,500 | -93.01 | 2026.8.17.1 |

**Parity — `group/delaunay/euclid/slam_delaunay/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.004 ± 0.0014 | +1.65σ |
| `slope` | 1.961 ± 0.022 | -0.67σ |
| `shear_magnitude` | 0.0008326 ± 0.00051 | +0.37σ |
| `centre_0` | 0.0002312 ± 0.002 | +1.08σ |
| `centre_1` | 0.003683 ± 0.0018 | +1.29σ |
| `ell_comps_0` | -0.001887 ± 0.0012 | +0.99σ |
| `ell_comps_1` | -0.0004373 ± 0.00096 | -0.56σ |
| `extra_galaxies.0.mass.sigma` | 208.3 ± 2.1 | -0.96σ |
| `extra_galaxies.1.mass.sigma` | 233.6 ± 2.4 | -0.20σ |
| `gamma_1` | -0.0002491 ± 0.00052 | -0.49σ |
| `gamma_2` | -0.0005764 ± 0.00053 | +2.19σ |

**Parity — `group/delaunay/euclid/slam_delaunay/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.007 ± 0.0028 | +0.99σ |
| `slope` | 1.922 ± 0.022 | +1.61σ |
| `shear_magnitude` | 0.001204 ± 0.00072 | +0.02σ |
| `centre_0` | 0.003536 ± 0.003 | +1.44σ |
| `centre_1` | 0.005473 ± 0.0025 | +0.89σ |
| `ell_comps_0` | -0.0007128 ± 0.0024 | +1.20σ |
| `ell_comps_1` | -0.0001486 ± 0.0012 | -0.36σ |
| `extra_galaxies.0.mass.sigma` | 204 ± 2.2 | +1.25σ |
| `extra_galaxies.1.mass.sigma` | 230.8 ± 2.7 | +1.05σ |
| `gamma_1` | -3.61e-05 ± 0.00063 | -0.05σ |
| `gamma_2` | 0.0009342 ± 0.001 | +0.09σ |

**Parity — `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.004 ± 0.0013 | +2.10σ |
| `slope` | 2 ± 0.0094 | -1.25σ |
| `shear_magnitude` | 0.001575 ± 0.00046 | -1.87σ |
| `centre_0` | 0.002941 ± 0.0014 | +1.87σ |
| `centre_1` | 0.007127 ± 0.0013 | +0.99σ |
| `ell_comps_0` | -0.005308 ± 0.0013 | +1.49σ |
| `ell_comps_1` | -0.0006308 ± 0.00079 | +0.67σ |
| `extra_galaxies.0.mass.sigma` | 211.8 ± 0.87 | -2.00σ |
| `extra_galaxies.1.mass.sigma` | 238.3 ± 0.93 | +0.30σ |
| `gamma_1` | -0.0004478 ± 0.0004 | +0.91σ |
| `gamma_2` | -0.001459 ± 0.00048 | +3.38σ |

**Parity — `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.007 ± 0.0024 | +0.96σ |
| `slope` | 1.96 ± 0.016 | +0.44σ |
| `shear_magnitude` | 0.0009331 ± 0.00053 | +0.27σ |
| `centre_0` | 0.006002 ± 0.0023 | +0.82σ |
| `centre_1` | 0.008065 ± 0.0022 | +0.94σ |
| `ell_comps_0` | -0.002918 ± 0.0016 | +0.63σ |
| `ell_comps_1` | -1.185e-05 ± 0.0011 | -0.69σ |
| `extra_galaxies.0.mass.sigma` | 207.5 ± 1.7 | +0.13σ |
| `extra_galaxies.1.mass.sigma` | 235.7 ± 1.6 | +0.25σ |
| `gamma_1` | 1.333e-05 ± 0.0006 | -0.54σ |
| `gamma_2` | 0.000515 ± 0.00082 | +0.40σ |

**Parity — `group/mge/euclid/slam_mge/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 4.003 ± 0.002 | -0.03σ |
| `slope` | 1.982 ± 0.02 | +0.00σ |
| `shear_magnitude` | 0.0008174 ± 0.00044 | +0.02σ |
| `extra_galaxies.0.mass.sigma` | 210.1 ± 1.9 | +0.01σ |
| `extra_galaxies.1.mass.sigma` | 235.8 ± 2.2 | +0.04σ |
| `galaxies.lens_0.mass.centre_0` | 0.0005103 ± 0.0021 | +0.03σ |
| `galaxies.lens_0.mass.centre_1` | 0.001409 ± 0.002 | +0.04σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.001539 ± 0.0016 | -0.05σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0008521 ± 0.0009 | -0.02σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | 0.004873 ± 0.0026 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.1034 ± 0.0032 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09435 ± 0.0024 | -0.05σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05753 ± 0.0018 | +0.06σ |
| `gamma_1` | 0.000485 ± 0.00046 | -0.06σ |
| `gamma_2` | 0.000131 ± 0.0007 | +0.01σ |

**Parity — `group/mge/euclid/slam_mge/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 4.003 ± 0.002 | +0.01σ |
| `slope` | 1.98 ± 0.019 | +0.09σ |
| `shear_magnitude` | 0.0008271 ± 0.00045 | +0.01σ |
| `extra_galaxies.0.mass.sigma` | 209.9 ± 1.8 | +0.09σ |
| `extra_galaxies.1.mass.sigma` | 235.6 ± 2.1 | +0.09σ |
| `galaxies.lens_0.mass.centre_0` | 0.0004088 ± 0.002 | +0.03σ |
| `galaxies.lens_0.mass.centre_1` | 0.001535 ± 0.0021 | -0.03σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.001591 ± 0.0015 | -0.06σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0008676 ± 0.00089 | -0.04σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | 0.00479 ± 0.0027 | +0.04σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.1033 ± 0.0033 | +0.06σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09418 ± 0.0024 | +0.06σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05745 ± 0.0018 | -0.02σ |
| `gamma_1` | 0.0004752 ± 0.00046 | -0.03σ |
| `gamma_2` | 0.0001694 ± 0.0007 | -0.05σ |

**Parity — `group/mge/euclid/slam_mge_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 4.003 ± 0.002 | +0.03σ |
| `slope` | 1.992 ± 0.015 | -0.31σ |
| `shear_magnitude` | 0.0008338 ± 0.00044 | -0.04σ |
| `extra_galaxies.0.mass.sigma` | 210.9 ± 1.4 | -0.28σ |
| `extra_galaxies.1.mass.sigma` | 237 ± 1.6 | -0.35σ |
| `galaxies.lens_0.mass.centre_0` | 0.0006196 ± 0.002 | +0.00σ |
| `galaxies.lens_0.mass.centre_1` | 0.001496 ± 0.002 | -0.01σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.001428 ± 0.0015 | -0.00σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.000923 ± 0.00094 | -0.08σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | 0.004565 ± 0.0024 | +0.02σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.1039 ± 0.0031 | -0.11σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09487 ± 0.0022 | -0.17σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05763 ± 0.0018 | +0.12σ |
| `gamma_1` | 0.000492 ± 0.00048 | -0.06σ |
| `gamma_2` | 7.74e-05 ± 0.00068 | -0.02σ |

**Parity — `group/mge/euclid/slam_mge_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 4.003 ± 0.0019 | -0.08σ |
| `slope` | 1.988 ± 0.015 | -0.21σ |
| `shear_magnitude` | 0.0007921 ± 0.00044 | +0.02σ |
| `extra_galaxies.0.mass.sigma` | 210.6 ± 1.4 | -0.15σ |
| `extra_galaxies.1.mass.sigma` | 236.5 ± 1.5 | -0.25σ |
| `galaxies.lens_0.mass.centre_0` | 0.0004156 ± 0.002 | +0.06σ |
| `galaxies.lens_0.mass.centre_1` | 0.001316 ± 0.002 | +0.17σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.001401 ± 0.0015 | -0.07σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0008249 ± 0.00091 | -0.02σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | 0.004774 ± 0.0025 | +0.11σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.1037 ± 0.0032 | +0.05σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.0948 ± 0.0022 | -0.09σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.0575 ± 0.0017 | +0.06σ |
| `gamma_1` | 0.0004531 ± 0.00048 | -0.00σ |
| `gamma_2` | 3.016e-05 ± 0.00069 | -0.03σ |

**Parity — `group/rectangular/euclid/slam_rectangular/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.005 ± 0.0023 | +0.07σ |
| `slope` | 1.961 ± 0.023 | -0.42σ |
| `shear_magnitude` | 0.0009027 ± 0.00055 | -0.07σ |
| `centre_0` | 0.001978 ± 0.0023 | -0.19σ |
| `centre_1` | 0.003747 ± 0.0023 | -0.21σ |
| `ell_comps_0` | -0.0009967 ± 0.0022 | +0.20σ |
| `ell_comps_1` | -0.0005639 ± 0.0011 | +0.17σ |
| `extra_galaxies.0.mass.sigma` | 207.9 ± 2.3 | -0.36σ |
| `extra_galaxies.1.mass.sigma` | 233.8 ± 2.6 | -0.38σ |
| `gamma_1` | -0.0002393 ± 0.00061 | +0.13σ |
| `gamma_2` | -0.0003147 ± 0.0009 | +0.26σ |

**Parity — `group/rectangular/euclid/slam_rectangular/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.005 ± 0.0023 | +0.19σ |
| `slope` | 1.953 ± 0.023 | -0.07σ |
| `shear_magnitude` | 0.000846 ± 0.00053 | +0.01σ |
| `centre_0` | 0.002024 ± 0.0024 | -0.04σ |
| `centre_1` | 0.003478 ± 0.0024 | +0.13σ |
| `ell_comps_0` | -0.0005751 ± 0.002 | -0.03σ |
| `ell_comps_1` | -0.0002569 ± 0.0012 | -0.16σ |
| `extra_galaxies.0.mass.sigma` | 207.2 ± 2.2 | -0.12σ |
| `extra_galaxies.1.mass.sigma` | 233 ± 2.7 | -0.07σ |
| `gamma_1` | -9.592e-05 ± 0.00061 | -0.15σ |
| `gamma_2` | -0.0001165 ± 0.00086 | +0.07σ |

**Parity — `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.004 ± 0.002 | -0.02σ |
| `slope` | 2.007 ± 0.014 | -0.37σ |
| `shear_magnitude` | 0.001147 ± 0.00058 | -0.14σ |
| `centre_0` | 0.004406 ± 0.0023 | -0.22σ |
| `centre_1` | 0.007315 ± 0.0022 | -0.06σ |
| `ell_comps_0` | -0.004186 ± 0.0017 | +0.19σ |
| `ell_comps_1` | -0.0005593 ± 0.0011 | +0.17σ |
| `extra_galaxies.0.mass.sigma` | 212.3 ± 1.3 | -0.35σ |
| `extra_galaxies.1.mass.sigma` | 239.5 ± 1.6 | -0.32σ |
| `gamma_1` | -0.0002535 ± 0.00053 | +0.09σ |
| `gamma_2` | -0.0009665 ± 0.00065 | +0.22σ |

**Parity — `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 4.005 ± 0.0021 | +0.18σ |
| `slope` | 2.003 ± 0.014 | +0.00σ |
| `shear_magnitude` | 0.001035 ± 0.00056 | -0.05σ |
| `centre_0` | 0.004597 ± 0.0025 | +0.05σ |
| `centre_1` | 0.00691 ± 0.0023 | +0.27σ |
| `ell_comps_0` | -0.003508 ± 0.0019 | -0.05σ |
| `ell_comps_1` | -0.0004864 ± 0.0011 | +0.03σ |
| `extra_galaxies.0.mass.sigma` | 211.8 ± 1.4 | -0.03σ |
| `extra_galaxies.1.mass.sigma` | 239.1 ± 1.6 | +0.02σ |
| `gamma_1` | -0.0002202 ± 0.00055 | -0.00σ |
| `gamma_2` | -0.0007857 ± 0.00074 | +0.00σ |

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

**Parity — `hst/slam5_slam_delaunay/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.603 ± 0.00063 | +20.11σ |
| `slope` | 2.015 ± 0.0079 | +21.77σ |
| `shear_magnitude` | 0.07251 ± 0.00042 | +16.92σ |
| `centre_0` | -0.0004431 ± 0.00036 | +0.11σ |
| `centre_1` | -0.0006257 ± 0.00035 | +1.69σ |
| `ell_comps_0` | 0.05526 ± 0.00067 | +8.77σ |
| `ell_comps_1` | 0.00236 ± 0.00082 | -1.77σ |
| `gamma_1` | 0.05122 ± 0.00057 | +14.35σ |
| `gamma_2` | 0.05131 ± 0.00034 | +4.82σ |

**Parity — `hst/slam5_slam_delaunay/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.603 ± 0.00097 | +5.84σ |
| `slope` | 2.027 ± 0.013 | +5.09σ |
| `shear_magnitude` | 0.07231 ± 0.00078 | +5.56σ |
| `centre_0` | -0.0007392 ± 0.00045 | +0.08σ |
| `centre_1` | 2.525e-05 ± 0.00038 | +1.74σ |
| `ell_comps_0` | 0.05435 ± 0.00093 | +5.52σ |
| `ell_comps_1` | 0.0006525 ± 0.0013 | +1.25σ |
| `gamma_1` | 0.05131 ± 0.00091 | +4.24σ |
| `gamma_2` | 0.0509 ± 0.00047 | +4.97σ |

**Parity — `hst/slam5_slam_delaunay_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.602 ± 0.0016 | +0.20σ |
| `slope` | 2.011 ± 0.019 | +0.34σ |
| `shear_magnitude` | 0.07233 ± 0.00088 | -0.20σ |
| `centre_0` | -0.0004888 ± 0.00041 | +0.78σ |
| `centre_1` | -0.0006013 ± 0.00042 | +1.09σ |
| `ell_comps_0` | 0.05506 ± 0.00093 | -0.60σ |
| `ell_comps_1` | 0.002423 ± 0.00094 | -1.77σ |
| `gamma_1` | 0.05102 ± 0.0011 | +0.00σ |
| `gamma_2` | 0.05122 ± 0.00045 | -0.43σ |

**Parity — `hst/slam5_slam_delaunay_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.596 ± 0.0029 | +1.91σ |
| `slope` | 1.943 ± 0.036 | +2.01σ |
| `shear_magnitude` | 0.06894 ± 0.0015 | +2.72σ |
| `centre_0` | -0.0006218 ± 0.00045 | +0.31σ |
| `centre_1` | -0.0002107 ± 0.00042 | +1.61σ |
| `ell_comps_0` | 0.05213 ± 0.0013 | +3.23σ |
| `ell_comps_1` | 0.001058 ± 0.0012 | +0.92σ |
| `gamma_1` | 0.04741 ± 0.0017 | +2.35σ |
| `gamma_2` | 0.05007 ± 0.00063 | +3.03σ |

**Parity — `hst/slam5_slam_delaunay_nn/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.6 ± 0.0031 | -4.00σ |
| `slope` | 1.981 ± 0.036 | -3.77σ |
| `shear_magnitude` | 0.07369 ± 0.0016 | -5.58σ |
| `centre_0` | -0.0001885 ± 0.00033 | -1.44σ |
| `centre_1` | -0.0007094 ± 0.00029 | +3.87σ |
| `ell_comps_0` | 0.05514 ± 0.0015 | -3.59σ |
| `ell_comps_1` | 0.01016 ± 0.00088 | -11.82σ |
| `gamma_1` | 0.05303 ± 0.0017 | -6.31σ |
| `gamma_2` | 0.05121 ± 0.00065 | -3.48σ |

**Parity — `hst/slam5_slam_delaunay_nn/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.595 ± 0.0025 | -1.94σ |
| `slope` | 1.932 ± 0.031 | -1.70σ |
| `shear_magnitude` | 0.06704 ± 0.0014 | +0.49σ |
| `centre_0` | -0.0003469 ± 0.00054 | -1.07σ |
| `centre_1` | 4.752e-05 ± 0.00045 | +0.15σ |
| `ell_comps_0` | 0.05033 ± 0.0016 | +2.01σ |
| `ell_comps_1` | -0.001695 ± 0.00083 | +6.02σ |
| `gamma_1` | 0.04561 ± 0.0015 | -0.53σ |
| `gamma_2` | 0.04911 ± 0.00087 | +1.89σ |

**Parity — `hst/slam5_slam_delaunay_nn_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.596 ± 0.0031 | -1.57σ |
| `slope` | 1.944 ± 0.035 | -1.56σ |
| `shear_magnitude` | 0.07207 ± 0.0017 | -3.02σ |
| `centre_0` | -0.00017 ± 0.0004 | -1.26σ |
| `centre_1` | -0.0006182 ± 0.00037 | +2.71σ |
| `ell_comps_0` | 0.05397 ± 0.0015 | -1.74σ |
| `ell_comps_1` | 0.009738 ± 0.0012 | -7.34σ |
| `gamma_1` | 0.05118 ± 0.0018 | -3.55σ |
| `gamma_2` | 0.0507 ± 0.00071 | -1.28σ |

**Parity — `hst/slam5_slam_delaunay_nn_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.597 ± 0.0022 | -1.01σ |
| `slope` | 1.95 ± 0.028 | -0.98σ |
| `shear_magnitude` | 0.06794 ± 0.0014 | +1.30σ |
| `centre_0` | -0.0004413 ± 0.00051 | -1.09σ |
| `centre_1` | 0.0001632 ± 0.00049 | -0.12σ |
| `ell_comps_0` | 0.05103 ± 0.0016 | +2.45σ |
| `ell_comps_1` | -0.001646 ± 0.00089 | +5.97σ |
| `gamma_1` | 0.0466 ± 0.0014 | +0.40σ |
| `gamma_2` | 0.04947 ± 0.00086 | +2.21σ |

**Parity — `imaging/mge/hst/slam_mge/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 1.598 ± 0.0035 | -0.08σ |
| `slope` | 1.971 ± 0.043 | -0.10σ |
| `shear_magnitude` | 0.06956 ± 0.002 | -0.09σ |
| `galaxies.lens.mass.centre_0` | -7.512e-05 ± 0.00028 | +0.06σ |
| `galaxies.lens.mass.centre_1` | -0.0001886 ± 0.00027 | -0.05σ |
| `galaxies.lens.mass.ell_comps_0` | 0.05185 ± 0.0015 | -0.03σ |
| `galaxies.lens.mass.ell_comps_1` | 0.0005038 ± 0.00098 | -0.07σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.0002939 ± 0.00024 | +0.01σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | -0.0002679 ± 0.00021 | -0.03σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09798 ± 0.0022 | +0.04σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05594 ± 0.0023 | +0.03σ |
| `gamma_1` | 0.04855 ± 0.0022 | -0.09σ |
| `gamma_2` | 0.04973 ± 0.00069 | -0.02σ |

**Parity — `imaging/mge/hst/slam_mge/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 1.598 ± 0.0035 | -0.36σ |
| `slope` | 1.975 ± 0.043 | -0.34σ |
| `shear_magnitude` | 0.06974 ± 0.0019 | -0.34σ |
| `galaxies.lens.mass.centre_0` | -5.97e-05 ± 0.00028 | -0.06σ |
| `galaxies.lens.mass.centre_1` | -0.0002031 ± 0.00027 | -0.06σ |
| `galaxies.lens.mass.ell_comps_0` | 0.05204 ± 0.0015 | -0.28σ |
| `galaxies.lens.mass.ell_comps_1` | 0.0004953 ± 0.00088 | -0.06σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.0002845 ± 0.00023 | -0.04σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | -0.0002585 ± 0.0002 | -0.07σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09796 ± 0.002 | +0.05σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05619 ± 0.0022 | +0.27σ |
| `gamma_1` | 0.04877 ± 0.0022 | -0.34σ |
| `gamma_2` | 0.0498 ± 0.00068 | -0.27σ |

**Parity — `imaging/mge/hst/slam_mge_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 1.599 ± 0.0029 | -0.29σ |
| `slope` | 1.98 ± 0.036 | -0.29σ |
| `shear_magnitude` | 0.06992 ± 0.0017 | -0.23σ |
| `galaxies.lens.mass.centre_0` | -6.688e-05 ± 0.00033 | -0.02σ |
| `galaxies.lens.mass.centre_1` | -0.000212 ± 0.00029 | -0.02σ |
| `galaxies.lens.mass.ell_comps_0` | 0.05218 ± 0.0014 | -0.19σ |
| `galaxies.lens.mass.ell_comps_1` | 0.0004292 ± 0.00097 | +0.03σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.0002977 ± 0.00025 | +0.01σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | -0.0002712 ± 0.00021 | +0.03σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09795 ± 0.0023 | +0.07σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05615 ± 0.0023 | +0.11σ |
| `gamma_1` | 0.04901 ± 0.0018 | -0.25σ |
| `gamma_2` | 0.04986 ± 0.00064 | -0.16σ |

**Parity — `imaging/mge/hst/slam_mge_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `einstein_radius` | 1.598 ± 0.003 | +0.13σ |
| `slope` | 1.972 ± 0.037 | +0.13σ |
| `shear_magnitude` | 0.06962 ± 0.0017 | +0.08σ |
| `galaxies.lens.mass.centre_0` | -6.294e-05 ± 0.00031 | -0.01σ |
| `galaxies.lens.mass.centre_1` | -0.0001983 ± 0.00028 | -0.09σ |
| `galaxies.lens.mass.ell_comps_0` | 0.05197 ± 0.0014 | +0.10σ |
| `galaxies.lens.mass.ell_comps_1` | 0.0004826 ± 0.00098 | -0.02σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.0002994 ± 0.00024 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | -0.0002634 ± 0.00022 | -0.07σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09811 ± 0.0022 | -0.04σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05604 ± 0.0022 | -0.07σ |
| `gamma_1` | 0.0487 ± 0.0019 | +0.10σ |
| `gamma_2` | 0.04978 ± 0.00063 | +0.06σ |

**Parity — `interferometer/delaunay/sma/slam_delaunay/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_sparse_fp64_wtilde` (ref) | `hpc_ral_numba_cpu_sparse_fp64_wtilde` |
|---|---|---|
| `einstein_radius` | 0.8883 ± 0.078 | +28.92σ |
| `slope` | 1.783 ± 0.26 | -0.05σ |
| `shear_magnitude` | 0.2033 ± 0.084 | -0.33σ |
| `centre_0` | -0.0407 ± 0.088 | +0.55σ |
| `centre_1` | 0.01821 ± 0.089 | -0.01σ |
| `ell_comps_0` | 0.006612 ± 0.17 | -0.35σ |
| `ell_comps_1` | -0.0008865 ± 0.18 | -0.38σ |
| `gamma_1` | 0.06409 ± 0.15 | -1.30σ |
| `gamma_2` | -0.1303 ± 0.12 | +0.40σ |

**Parity — `interferometer/delaunay/sma/slam_delaunay/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_sparse_fp64_wtilde` (ref) | `hpc_ral_numba_cpu_sparse_fp64_wtilde` |
|---|---|---|
| `einstein_radius` | 0.8044 ± 0.092 | +4.97σ |
| `slope` | 1.692 ± 0.23 | +0.02σ |
| `shear_magnitude` | 0.2056 ± 0.08 | +0.03σ |
| `centre_0` | 0.02432 ± 0.061 | -0.61σ |
| `centre_1` | -0.01798 ± 0.079 | +0.24σ |
| `ell_comps_0` | 0.04965 ± 0.19 | -0.03σ |
| `ell_comps_1` | -0.01326 ± 0.2 | +0.03σ |
| `gamma_1` | 0.03461 ± 0.13 | +0.13σ |
| `gamma_2` | -0.1433 ± 0.14 | +0.78σ |

**Parity — `interferometer/mge/sma/slam_mge/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64_nufft` (ref) | `hpc_ral_jax_cpu_dense_fp64_nufft` |
|---|---|---|
| `einstein_radius` | 1.247 ± 2.5 | -0.03σ |
| `slope` | 2.255 ± 0.52 | +0.06σ |
| `shear_magnitude` | 0.2365 ± 0.091 | +0.00σ |
| `galaxies.lens.mass.centre_0` | 0.004494 ± 0.093 | -0.03σ |
| `galaxies.lens.mass.centre_1` | 0.01022 ± 0.093 | -0.05σ |
| `galaxies.lens.mass.ell_comps_0` | -0.0052 ± 0.31 | +0.01σ |
| `galaxies.lens.mass.ell_comps_1` | 0.01363 ± 0.29 | +0.01σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | 0.04969 ± 0.28 | -0.01σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | -0.04825 ± 0.3 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | -0.02361 ± 0.27 | -0.01σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.01323 ± 0.31 | -0.02σ |
| `gamma_1` | 0.009352 ± 0.21 | +0.00σ |
| `gamma_2` | -0.03953 ± 0.19 | -0.00σ |

**Parity — `interferometer/mge/sma/slam_mge/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64_nufft` (ref) | `hpc_ral_jax_cpu_dense_fp64_nufft` |
|---|---|---|
| `einstein_radius` | 1.367 ± 2.5 | +0.02σ |
| `slope` | 2.239 ± 0.5 | +0.01σ |
| `shear_magnitude` | 0.2403 ± 0.09 | +0.01σ |
| `galaxies.lens.mass.centre_0` | -8.37e-05 ± 0.096 | -0.02σ |
| `galaxies.lens.mass.centre_1` | 0.0001432 ± 0.1 | +0.00σ |
| `galaxies.lens.mass.ell_comps_0` | -0.01638 ± 0.29 | -0.00σ |
| `galaxies.lens.mass.ell_comps_1` | 0.007863 ± 0.29 | -0.03σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | 0.0228 ± 0.28 | -0.02σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | -0.04258 ± 0.31 | +0.02σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | -0.05468 ± 0.29 | -0.01σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.03139 ± 0.29 | -0.02σ |
| `gamma_1` | 0.00813 ± 0.21 | -0.03σ |
| `gamma_2` | -0.01867 ± 0.2 | -0.01σ |

**Parity — `interferometer/rectangular/sma/slam_rectangular/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_sparse_fp64_wtilde` (ref) | `hpc_ral_numba_cpu_sparse_fp64_wtilde` |
|---|---|---|
| `einstein_radius` | 0.8889 ± 0.2 | +7.96σ |
| `slope` | 1.949 ± 0.33 | -0.95σ |
| `shear_magnitude` | 0.213 ± 0.089 | +0.69σ |
| `centre_0` | 0.006164 ± 0.074 | +0.34σ |
| `centre_1` | -0.01853 ± 0.093 | +0.27σ |
| `ell_comps_0` | -0.03681 ± 0.22 | -0.33σ |
| `ell_comps_1` | 0.06292 ± 0.24 | -0.13σ |
| `gamma_1` | 0.04248 ± 0.19 | -1.20σ |
| `gamma_2` | -0.05376 ± 0.17 | -0.80σ |

**Parity — `interferometer/rectangular/sma/slam_rectangular/seed1` (seed 1), `mass_total[1]`**

_Only one config has run for this target and seed — a parity view needs at least two legs._

**Parity — `multi_dataset/delaunay/hst/slam_delaunay/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.602 ± 0.00083 | -5.35σ |
| `slope` | 2.023 ± 0.0095 | -5.52σ |
| `shear_magnitude` | 0.07283 ± 0.0005 | -4.66σ |
| `centre_0` | -8.718e-05 ± 0.00011 | -7.37σ |
| `centre_1` | -0.0002471 ± 0.00017 | +1.91σ |
| `ell_comps_0` | 0.05665 ± 0.00055 | -5.16σ |
| `ell_comps_1` | -0.0006815 ± 0.00013 | +15.07σ |
| `gamma_1` | 0.05082 ± 0.00048 | -3.90σ |
| `gamma_2` | 0.05214 ± 0.00028 | -4.93σ |
| `grid_offset_0` | -0.0009424 ± 0.00013 | +11.32σ |
| `grid_offset_1` | -4.1e-05 ± 0.00015 | +1.24σ |

**Parity — `multi_dataset/delaunay/hst/slam_delaunay/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.611 ± 0.0011 | -0.96σ |
| `slope` | 2.119 ± 0.013 | -0.51σ |
| `shear_magnitude` | 0.07886 ± 0.00059 | -3.55σ |
| `centre_0` | -0.0003045 ± 0.0002 | +3.65σ |
| `centre_1` | -0.0003586 ± 0.0002 | +2.09σ |
| `ell_comps_0` | 0.06146 ± 0.00072 | -2.51σ |
| `ell_comps_1` | 0.006232 ± 0.0003 | -19.60σ |
| `gamma_1` | 0.05777 ± 0.00062 | -3.62σ |
| `gamma_2` | 0.05365 ± 0.00026 | -2.46σ |
| `grid_offset_0` | 0.000345 ± 0.00018 | -2.91σ |
| `grid_offset_1` | -0.0002356 ± 0.00017 | +4.53σ |

**Parity — `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.586 ± 0.0022 | +5.10σ |
| `slope` | 1.834 ± 0.023 | +5.81σ |
| `shear_magnitude` | 0.06391 ± 0.001 | +6.28σ |
| `centre_0` | 5.522e-05 ± 0.00026 | -3.64σ |
| `centre_1` | -0.0002402 ± 0.00039 | +0.89σ |
| `ell_comps_0` | 0.04918 ± 0.00078 | +5.74σ |
| `ell_comps_1` | 0.0006893 ± 0.00029 | +2.12σ |
| `gamma_1` | 0.04162 ± 0.0012 | +6.08σ |
| `gamma_2` | 0.04859 ± 0.00045 | +4.57σ |
| `grid_offset_0` | -0.0009837 ± 0.00015 | +9.90σ |
| `grid_offset_1` | -0.0001117 ± 0.0002 | +1.31σ |

**Parity — `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `einstein_radius` | 1.609 ± 0.0017 | -5.50σ |
| `slope` | 2.101 ± 0.02 | -5.59σ |
| `shear_magnitude` | 0.07806 ± 0.00088 | -7.59σ |
| `centre_0` | -0.0003237 ± 0.00021 | +0.18σ |
| `centre_1` | -0.0003458 ± 0.00021 | +1.85σ |
| `ell_comps_0` | 0.06055 ± 0.00096 | -5.49σ |
| `ell_comps_1` | 0.006063 ± 0.00034 | -14.15σ |
| `gamma_1` | 0.05696 ± 0.00095 | -7.88σ |
| `gamma_2` | 0.05334 ± 0.00033 | -5.42σ |
| `grid_offset_0` | 0.0003738 ± 0.00019 | -2.92σ |
| `grid_offset_1` | -0.0002539 ± 0.00018 | +5.02σ |

**Parity — `multi_galaxy/delaunay/hst/slam_delaunay/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.06779 ± 0.0012 | +2.07σ |
| `galaxies.lens_0.mass.centre_0` | 0.2986 ± 0.0041 | -1.40σ |
| `galaxies.lens_0.mass.centre_1` | 0.2896 ± 0.0037 | -4.21σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9675 ± 0.011 | +3.63σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08356 ± 0.0036 | -0.49σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0002222 ± 0.0034 | +0.49σ |
| `galaxies.lens_0.mass.slope` | 2.013 ± 0.016 | -3.42σ |
| `galaxies.lens_1.mass.centre_0` | -0.2945 ± 0.0058 | -3.19σ |
| `galaxies.lens_1.mass.centre_1` | -0.2185 ± 0.0047 | -0.81σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.8119 ± 0.013 | -3.61σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.08853 ± 0.0038 | -2.98σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05826 ± 0.0033 | -0.60σ |
| `galaxies.lens_1.mass.slope` | 1.96 ± 0.012 | +3.40σ |
| `gamma_1` | 0.04743 ± 0.0013 | +1.00σ |
| `gamma_2` | 0.04837 ± 0.0014 | +1.66σ |

**Parity — `multi_galaxy/delaunay/hst/slam_delaunay/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.0699 ± 0.0013 | +0.04σ |
| `galaxies.lens_0.mass.centre_0` | 0.2954 ± 0.0042 | -1.33σ |
| `galaxies.lens_0.mass.centre_1` | 0.2767 ± 0.004 | -0.62σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9849 ± 0.009 | +2.65σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08485 ± 0.0034 | -0.74σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.003748 ± 0.003 | -1.63σ |
| `galaxies.lens_0.mass.slope` | 1.967 ± 0.012 | -0.91σ |
| `galaxies.lens_1.mass.centre_0` | -0.3065 ± 0.0051 | -0.79σ |
| `galaxies.lens_1.mass.centre_1` | -0.2164 ± 0.0045 | -1.78σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.7741 ± 0.011 | -1.09σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09289 ± 0.0032 | -1.79σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05829 ± 0.0025 | -1.19σ |
| `galaxies.lens_1.mass.slope` | 1.974 ± 0.01 | +2.55σ |
| `gamma_1` | 0.04822 ± 0.0012 | -0.67σ |
| `gamma_2` | 0.05059 ± 0.0013 | +0.62σ |

**Parity — `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.07168 ± 0.00077 | -1.96σ |
| `galaxies.lens_0.mass.centre_0` | 0.2952 ± 0.0026 | +2.29σ |
| `galaxies.lens_0.mass.centre_1` | 0.2757 ± 0.0025 | +1.90σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9935 ± 0.0069 | +2.14σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08663 ± 0.0027 | -3.24σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.004572 ± 0.0019 | -1.53σ |
| `galaxies.lens_0.mass.slope` | 1.99 ± 0.0084 | +1.25σ |
| `galaxies.lens_1.mass.centre_0` | -0.3038 ± 0.0032 | -3.25σ |
| `galaxies.lens_1.mass.centre_1` | -0.2142 ± 0.0039 | -2.51σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.7898 ± 0.0072 | +1.29σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09262 ± 0.0026 | -3.26σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05559 ± 0.0016 | -1.12σ |
| `galaxies.lens_1.mass.slope` | 1.988 ± 0.0078 | +2.71σ |
| `gamma_1` | 0.05014 ± 0.00063 | +0.58σ |
| `gamma_2` | 0.0512 ± 0.00094 | -2.71σ |

**Parity — `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.07063 ± 0.0011 | -0.55σ |
| `galaxies.lens_0.mass.centre_0` | 0.2981 ± 0.0036 | -1.36σ |
| `galaxies.lens_0.mass.centre_1` | 0.2773 ± 0.0036 | -0.16σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9875 ± 0.0088 | +2.53σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08522 ± 0.003 | -1.39σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.004354 ± 0.0028 | -2.14σ |
| `galaxies.lens_0.mass.slope` | 1.99 ± 0.011 | -0.83σ |
| `galaxies.lens_1.mass.centre_0` | -0.3057 ± 0.0045 | -0.96σ |
| `galaxies.lens_1.mass.centre_1` | -0.2148 ± 0.0042 | -2.37σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.791 ± 0.0095 | -1.13σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09268 ± 0.003 | -1.87σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.057 ± 0.0023 | -1.19σ |
| `galaxies.lens_1.mass.slope` | 1.98 ± 0.0099 | +2.31σ |
| `gamma_1` | 0.04945 ± 0.001 | -0.97σ |
| `gamma_2` | 0.0504 ± 0.0011 | +0.04σ |

**Parity — `multi_galaxy/mge/hst/slam_mge/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `shear_magnitude` | 0.07048 ± 0.001 | +0.03σ |
| `galaxies.lens_0.mass.centre_0` | 0.3037 ± 0.0036 | +0.00σ |
| `galaxies.lens_0.mass.centre_1` | 0.2821 ± 0.0034 | -0.02σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9986 ± 0.0092 | +0.11σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.07959 ± 0.0029 | -0.02σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.002542 ± 0.0024 | +0.04σ |
| `galaxies.lens_0.mass.slope` | 2.025 ± 0.012 | +0.00σ |
| `galaxies.lens_1.mass.centre_0` | -0.3112 ± 0.0042 | -0.05σ |
| `galaxies.lens_1.mass.centre_1` | -0.2184 ± 0.0039 | -0.01σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.8162 ± 0.0097 | +0.00σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09566 ± 0.0029 | -0.06σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05811 ± 0.0019 | +0.01σ |
| `galaxies.lens_1.mass.slope` | 1.999 ± 0.011 | +0.09σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.0038 ± 0.0022 | +0.03σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.02668 ± 0.0017 | +0.04σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09508 ± 0.0014 | +0.01σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05538 ± 0.0014 | -0.02σ |
| `gamma_1` | 0.05079 ± 0.00097 | +0.06σ |
| `gamma_2` | 0.04884 ± 0.0011 | -0.00σ |

**Parity — `multi_galaxy/mge/hst/slam_mge/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `shear_magnitude` | 0.07039 ± 0.001 | -0.01σ |
| `galaxies.lens_0.mass.centre_0` | 0.3034 ± 0.0036 | -0.01σ |
| `galaxies.lens_0.mass.centre_1` | 0.2825 ± 0.0033 | -0.05σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9995 ± 0.0094 | -0.03σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.07955 ± 0.0029 | -0.01σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.002156 ± 0.0025 | +0.01σ |
| `galaxies.lens_0.mass.slope` | 2.025 ± 0.013 | -0.04σ |
| `galaxies.lens_1.mass.centre_0` | -0.3108 ± 0.0041 | -0.05σ |
| `galaxies.lens_1.mass.centre_1` | -0.2191 ± 0.0039 | +0.04σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.8166 ± 0.01 | -0.06σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09587 ± 0.003 | -0.01σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05838 ± 0.002 | +0.02σ |
| `galaxies.lens_1.mass.slope` | 2 ± 0.012 | -0.05σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.003771 ± 0.0022 | +0.03σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.02669 ± 0.0017 | +0.03σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09507 ± 0.0014 | +0.02σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.05538 ± 0.0015 | +0.01σ |
| `gamma_1` | 0.05069 ± 0.00099 | +0.01σ |
| `gamma_2` | 0.0488 ± 0.0011 | +0.02σ |

**Parity — `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `shear_magnitude` | 0.06813 ± 0.001 | -0.01σ |
| `galaxies.lens_0.mass.centre_0` | 0.3511 ± 0.0027 | +0.01σ |
| `galaxies.lens_0.mass.centre_1` | 0.3068 ± 0.0029 | +0.03σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9389 ± 0.0077 | -0.00σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.06159 ± 0.0027 | -0.01σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.009127 ± 0.0031 | +0.01σ |
| `galaxies.lens_0.mass.slope` | 2.201 ± 0.0099 | -0.01σ |
| `galaxies.lens_1.mass.centre_0` | -0.3093 ± 0.0034 | -0.02σ |
| `galaxies.lens_1.mass.centre_1` | -0.2073 ± 0.0033 | +0.02σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.9449 ± 0.0064 | +0.03σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.08304 ± 0.0022 | -0.01σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.04358 ± 0.0018 | +0.02σ |
| `galaxies.lens_1.mass.slope` | 1.947 ± 0.0093 | +0.01σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.03894 ± 0.0016 | -0.02σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.001149 ± 0.0012 | -0.02σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09539 ± 0.0013 | +0.04σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.06209 ± 0.0014 | -0.02σ |
| `gamma_1` | 0.05553 ± 0.00097 | +0.01σ |
| `gamma_2` | 0.03947 ± 0.00082 | -0.02σ |

**Parity — `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_jax_cpu_dense_fp64` |
|---|---|---|
| `shear_magnitude` | 0.06822 ± 0.00099 | +0.00σ |
| `galaxies.lens_0.mass.centre_0` | 0.3516 ± 0.0028 | -0.00σ |
| `galaxies.lens_0.mass.centre_1` | 0.307 ± 0.0029 | -0.00σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9389 ± 0.0079 | -0.00σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.06126 ± 0.0029 | +0.00σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.00941 ± 0.003 | +0.00σ |
| `galaxies.lens_0.mass.slope` | 2.203 ± 0.01 | +0.00σ |
| `galaxies.lens_1.mass.centre_0` | -0.3097 ± 0.0033 | +0.00σ |
| `galaxies.lens_1.mass.centre_1` | -0.2071 ± 0.0034 | +0.00σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.9461 ± 0.0065 | +0.00σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.08308 ± 0.0023 | +0.00σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.04303 ± 0.0018 | -0.00σ |
| `galaxies.lens_1.mass.slope` | 1.947 ± 0.0093 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.centre_0` | -0.03926 ± 0.0017 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.centre_1` | 0.0009533 ± 0.0012 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_0` | 0.09554 ± 0.0013 | -0.00σ |
| `galaxies.source.bulge.profile_list.0.ell_comps_1` | -0.06197 ± 0.0014 | -0.00σ |
| `gamma_1` | 0.05573 ± 0.00095 | -0.00σ |
| `gamma_2` | 0.03933 ± 0.00086 | +0.00σ |

**Parity — `multi_galaxy/rectangular/hst/slam_rectangular/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.069 ± 0.0011 | -0.06σ |
| `galaxies.lens_0.mass.centre_0` | 0.2994 ± 0.003 | -0.28σ |
| `galaxies.lens_0.mass.centre_1` | 0.2837 ± 0.0033 | +0.28σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9885 ± 0.0087 | -0.07σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08078 ± 0.0026 | +0.43σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0008899 ± 0.0026 | -0.33σ |
| `galaxies.lens_0.mass.slope` | 1.984 ± 0.012 | +0.35σ |
| `galaxies.lens_1.mass.centre_0` | -0.308 ± 0.0037 | +0.53σ |
| `galaxies.lens_1.mass.centre_1` | -0.2229 ± 0.0036 | -0.10σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.7882 ± 0.0094 | +0.26σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09527 ± 0.0029 | +0.21σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05728 ± 0.0024 | -0.09σ |
| `galaxies.lens_1.mass.slope` | 1.982 ± 0.01 | -0.15σ |
| `gamma_1` | 0.04843 ± 0.001 | -0.20σ |
| `gamma_2` | 0.04906 ± 0.0011 | +0.16σ |

**Parity — `multi_galaxy/rectangular/hst/slam_rectangular/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.06901 ± 0.001 | -0.25σ |
| `galaxies.lens_0.mass.centre_0` | 0.2995 ± 0.0027 | -0.07σ |
| `galaxies.lens_0.mass.centre_1` | 0.2849 ± 0.0031 | +0.16σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9861 ± 0.009 | -0.23σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08174 ± 0.0024 | +0.02σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0005105 ± 0.0025 | -0.15σ |
| `galaxies.lens_0.mass.slope` | 1.99 ± 0.011 | -0.24σ |
| `galaxies.lens_1.mass.centre_0` | -0.3061 ± 0.0037 | +0.17σ |
| `galaxies.lens_1.mass.centre_1` | -0.2223 ± 0.0033 | -0.13σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.7933 ± 0.0085 | -0.21σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09397 ± 0.003 | +0.08σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.0569 ± 0.0022 | -0.05σ |
| `galaxies.lens_1.mass.slope` | 1.98 ± 0.01 | -0.30σ |
| `gamma_1` | 0.04842 ± 0.00097 | -0.34σ |
| `gamma_2` | 0.04914 ± 0.00094 | -0.04σ |

**Parity — `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.06953 ± 0.00091 | -0.29σ |
| `galaxies.lens_0.mass.centre_0` | 0.3003 ± 0.0025 | -0.57σ |
| `galaxies.lens_0.mass.centre_1` | 0.2835 ± 0.0027 | +0.30σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9916 ± 0.0075 | -0.11σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08111 ± 0.0023 | +0.33σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.00131 ± 0.0025 | -0.41σ |
| `galaxies.lens_0.mass.slope` | 1.995 ± 0.01 | -0.03σ |
| `galaxies.lens_1.mass.centre_0` | -0.3081 ± 0.0036 | +0.53σ |
| `galaxies.lens_1.mass.centre_1` | -0.222 ± 0.0034 | -0.26σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.7961 ± 0.0075 | -0.12σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09544 ± 0.0028 | +0.18σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05679 ± 0.0023 | -0.28σ |
| `galaxies.lens_1.mass.slope` | 1.987 ± 0.0091 | -0.22σ |
| `gamma_1` | 0.04921 ± 0.00099 | -0.53σ |
| `gamma_2` | 0.04907 ± 0.00086 | +0.21σ |

**Parity — `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_ral_numba_cpu_sparse_fp64` |
|---|---|---|
| `shear_magnitude` | 0.06916 ± 0.00087 | -0.14σ |
| `galaxies.lens_0.mass.centre_0` | 0.2997 ± 0.0023 | -0.20σ |
| `galaxies.lens_0.mass.centre_1` | 0.2851 ± 0.0025 | +0.01σ |
| `galaxies.lens_0.mass.einstein_radius` | 0.9883 ± 0.0077 | -0.13σ |
| `galaxies.lens_0.mass.ell_comps_0` | 0.08135 ± 0.0024 | +0.18σ |
| `galaxies.lens_0.mass.ell_comps_1` | 0.0004321 ± 0.0023 | -0.06σ |
| `galaxies.lens_0.mass.slope` | 1.996 ± 0.0097 | -0.29σ |
| `galaxies.lens_1.mass.centre_0` | -0.3062 ± 0.0034 | +0.13σ |
| `galaxies.lens_1.mass.centre_1` | -0.2223 ± 0.0032 | -0.09σ |
| `galaxies.lens_1.mass.einstein_radius` | 0.797 ± 0.0073 | -0.29σ |
| `galaxies.lens_1.mass.ell_comps_0` | -0.09433 ± 0.0028 | +0.03σ |
| `galaxies.lens_1.mass.ell_comps_1` | -0.05673 ± 0.0021 | -0.19σ |
| `galaxies.lens_1.mass.slope` | 1.983 ± 0.0089 | -0.23σ |
| `gamma_1` | 0.04877 ± 0.0009 | -0.33σ |
| `gamma_2` | 0.04902 ± 0.00084 | +0.12σ |

**Chain walls (fitted stages only)**

| Target | Seed | Config | Chain wall | Excludes |
|---|---|---|---|---|
| `cluster/image_plane/simple/modeling_image_plane/seed0` | 0 | `hpc_a100_jax_gpu_na_fp64` | 318.2 s | all stages fitted |
| `cluster/image_plane/simple/modeling_image_plane/seed0` | 0 | `hpc_ral_jax_cpu_na_fp64` | 597.7 s | all stages fitted |
| `cluster/image_plane/simple/modeling_image_plane/seed1` | 1 | `hpc_a100_jax_gpu_na_fp64` | 334.9 s | all stages fitted |
| `cluster/image_plane/simple/modeling_image_plane/seed1` | 1 | `hpc_ral_jax_cpu_na_fp64` | 330.2 s | all stages fitted |
| `cluster/source_plane/simple/modeling_source_plane/seed0` | 0 | `hpc_a100_jax_gpu_na_fp64` | 140.2 s | all stages fitted |
| `cluster/source_plane/simple/modeling_source_plane/seed0` | 0 | `hpc_ral_jax_cpu_na_fp64` | 119.2 s | all stages fitted |
| `cluster/source_plane/simple/modeling_source_plane/seed1` | 1 | `hpc_a100_jax_gpu_na_fp64` | 165.3 s | all stages fitted |
| `cluster/source_plane/simple/modeling_source_plane/seed1` | 1 | `hpc_ral_jax_cpu_na_fp64` | 145.6 s | all stages fitted |
| `group/delaunay/euclid/slam_delaunay/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.88 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 12.69 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.73 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 12.76 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.77 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 12.05 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.62 h | all stages fitted |
| `group/delaunay/euclid/slam_delaunay_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 11.73 h | all stages fitted |
| `group/mge/euclid/slam_mge/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1323.0 s | all stages fitted |
| `group/mge/euclid/slam_mge/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 1.99 h | all stages fitted |
| `group/mge/euclid/slam_mge/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1319.8 s | all stages fitted |
| `group/mge/euclid/slam_mge/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 2.00 h | all stages fitted |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1319.2 s | all stages fitted |
| `group/mge/euclid/slam_mge_light_fixed/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 1.98 h | all stages fitted |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1306.7 s | all stages fitted |
| `group/mge/euclid/slam_mge_light_fixed/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 1.93 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.13 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 11.11 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.12 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 13.38 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.11 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 10.17 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.09 h | all stages fitted |
| `group/rectangular/euclid/slam_rectangular_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 12.24 h | all stages fitted |
| `hst/slam5/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 2.62 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 2.74 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_base_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 2.54 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_base_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 2.60 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.05 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 6.21 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.11 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 6.66 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 3482.7 s | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 5.47 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.04 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 5.53 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.04 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 6.88 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 3490.1 s | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 6.29 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 3551.2 s | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 5.93 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 3267.4 s | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `hst/slam5_slam_delaunay_nn_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 5.55 h | excludes source_lp[1] (shared from `hpc_a100_jax_cpu_dense_fp64`) |
| `imaging/mge/hst/slam_mge/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 946.6 s | all stages fitted |
| `imaging/mge/hst/slam_mge/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 1.02 h | all stages fitted |
| `imaging/mge/hst/slam_mge/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1003.5 s | all stages fitted |
| `imaging/mge/hst/slam_mge/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 1.06 h | all stages fitted |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 979.3 s | all stages fitted |
| `imaging/mge/hst/slam_mge_light_fixed/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 1.02 h | all stages fitted |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 924.1 s | all stages fitted |
| `imaging/mge/hst/slam_mge_light_fixed/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 1.02 h | all stages fitted |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | 0 | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 428.3 s | all stages fitted |
| `interferometer/delaunay/sma/slam_delaunay/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 2190.4 s | all stages fitted |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | 1 | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 492.7 s | all stages fitted |
| `interferometer/delaunay/sma/slam_delaunay/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 1.29 h | all stages fitted |
| `interferometer/mge/sma/slam_mge/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64_nufft` | 88.4 s | all stages fitted |
| `interferometer/mge/sma/slam_mge/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64_nufft` | 1299.1 s | all stages fitted |
| `interferometer/mge/sma/slam_mge/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64_nufft` | 87.0 s | all stages fitted |
| `interferometer/mge/sma/slam_mge/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64_nufft` | 1291.4 s | all stages fitted |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | 0 | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 440.7 s | all stages fitted |
| `interferometer/rectangular/sma/slam_rectangular/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64_wtilde` | 1.81 h | all stages fitted |
| `interferometer/rectangular/sma/slam_rectangular/seed1` | 1 | `hpc_a100_jax_gpu_sparse_fp64_wtilde` | 467.7 s | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 5.09 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 28.81 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 4.24 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 34.74 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 4.76 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 26.78 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 4.12 h | all stages fitted |
| `multi_dataset/delaunay/hst/slam_delaunay_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 33.23 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 2.89 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 20.87 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 2.92 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 20.44 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 2.77 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 19.25 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 2.76 h | all stages fitted |
| `multi_galaxy/delaunay/hst/slam_delaunay_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 18.83 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.03 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 6.23 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.09 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 6.13 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 1.03 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 6.20 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 1.07 h | all stages fitted |
| `multi_galaxy/mge/hst/slam_mge_light_fixed/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 5.98 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 2.10 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 18.39 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 2.05 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 19.17 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 2.02 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed0` | 0 | `hpc_ral_numba_cpu_sparse_fp64` | 15.83 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 2.00 h | all stages fitted |
| `multi_galaxy/rectangular/hst/slam_rectangular_light_fixed/seed1` | 1 | `hpc_ral_numba_cpu_sparse_fp64` | 16.92 h | all stages fitted |
| `point_source_image/image_plane/simple/modeling_image_plane/seed0` | 0 | `hpc_a100_jax_gpu_na_fp64` | 58.5 s | all stages fitted |
| `point_source_image/image_plane/simple/modeling_image_plane/seed0` | 0 | `hpc_ral_jax_cpu_na_fp64` | 55.5 s | all stages fitted |
| `point_source_image/image_plane/simple/modeling_image_plane/seed1` | 1 | `hpc_a100_jax_gpu_na_fp64` | 49.0 s | all stages fitted |
| `point_source_image/image_plane/simple/modeling_image_plane/seed1` | 1 | `hpc_ral_jax_cpu_na_fp64` | 50.5 s | all stages fitted |
| `point_source_source/source_plane/simple/modeling_source_plane/seed0` | 0 | `hpc_a100_jax_gpu_na_fp64` | 43.1 s | all stages fitted |
| `point_source_source/source_plane/simple/modeling_source_plane/seed0` | 0 | `hpc_ral_jax_cpu_na_fp64` | 34.9 s | all stages fitted |
| `point_source_source/source_plane/simple/modeling_source_plane/seed1` | 1 | `hpc_a100_jax_gpu_na_fp64` | 43.8 s | all stages fitted |
| `point_source_source/source_plane/simple/modeling_source_plane/seed1` | 1 | `hpc_ral_jax_cpu_na_fp64` | 42.8 s | all stages fitted |
| `weak/isothermal/simple/modeling_isothermal/seed0` | 0 | `hpc_a100_jax_gpu_dense_fp64` | 13.6 s | all stages fitted |
| `weak/isothermal/simple/modeling_isothermal/seed0` | 0 | `hpc_ral_jax_cpu_dense_fp64` | 12.4 s | all stages fitted |
| `weak/isothermal/simple/modeling_isothermal/seed1` | 1 | `hpc_a100_jax_gpu_dense_fp64` | 12.9 s | all stages fitted |
| `weak/isothermal/simple/modeling_isothermal/seed1` | 1 | `hpc_ral_jax_cpu_dense_fp64` | 12.0 s | all stages fitted |
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

- [`scripts/imaging/rectangular/validation/baseline_slam.py`](scripts/imaging/rectangular/validation/baseline_slam.py)
- [`scripts/imaging/delaunay/validation/baseline_slam.py`](scripts/imaging/delaunay/validation/baseline_slam.py)
- [`scripts/point_source_source/source_plane/samplers/baseline.py`](scripts/point_source_source/source_plane/samplers/baseline.py)

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
python scripts/imaging/delaunay/samplers/smc.py --prepared output/autolens/prepared/hst-delaunay/manifest.json --problem-id <manifest-id> --run-id hst-delaunay-smc-cold-seed0 --start cold --hardware-id laptop-cpu-8cores-single-process
```

Use `python scripts/misc/experiments/run.py --help` for supported start modes
and samplers. Cold/warm/resume sampler initialization is recorded independently
of compilation and cache conditions. Frozen priors are never narrowed to create
a warm start. Executors verify manifest identities and artifact hashes before
reuse; unavailable historical artifacts remain unknown.
