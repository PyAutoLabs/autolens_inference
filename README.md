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
[`scripts/imaging/rectangular/baseline_slam.py`](scripts/imaging/slam/README.md), one leaf over one chain,
switched by `--backend` and `--inversion`, with six SLURM submits behind it. The first four
legs came back on **2026-09-14** — A100, dense and sparse, seeds 0 and 1, all five stages —
and are the rows in the table below. They are one backend of the parity row: the CPU legs
have not run, so there is still nothing to compare them *to*.
`scripts/misc/wall/rates.py` still holds only the two `source_lp[1]` rates the submits had
to be sized against, because a completed wall on one cell is not a step rate for another.
A **run variant** level now sits between the instrument and the config name
(`slam_base`, `delaunay_1250`) so that a second mesh is a second experiment rather than a
seventh column. The ledger is [`wiki/project/state.md`](wiki/project/state.md).

## Pipeline runs

<!-- BEGIN auto-table:slam -->
| Target | Variant | Light | Stage | Config | Seed | Wall | Evals | log Z | Version |
|---|---|---|---|---|---|---|---|---|---|
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 771.5 s | 89,350 | 31,968.39 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 766.9 s | 89,350 | 31,968.39 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_lp[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 2.21 h | 91,900 | 31,968.50 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 755.8 s | 33,920 | 31,421.13 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1856.5 s | 33,640 | 31,420.79 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 24.42 h | 33,520 | 31,422.48 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 137.6 s | 3,840 | 31,614.40 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 238.3 s | 5,380 | 31,621.78 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | source_pix[2] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 1.79 h | 3,740 | 31,617.05 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 241.4 s | 10,280 | 31,608.58 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 231.2 s | 9,760 | 31,618.44 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | light[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 6.19 h | 13,000 | 31,610.92 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 562.1 s | 22,600 | 31,589.22 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 837.2 s | 22,900 | 31,597.52 | 2026.8.17.1 |
| `hst/slam5/seed0` | `slam_base` | linear | mass_total[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 12.28 h | 23,460 | 31,592.05 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 774.5 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 777.9 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_lp[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 2.13 h | 88,100 | 31,968.36 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 693.2 s | 30,560 | 31,416.31 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1837.8 s | 33,200 | 31,416.49 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 23.56 h | 32,280 | 31,425.11 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 137.0 s | 3,940 | 31,616.05 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 182.3 s | 3,620 | 31,621.15 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | source_pix[2] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 4.25 h | 9,000 | 31,620.90 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 239.7 s | 9,980 | 31,609.43 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 292.0 s | 14,080 | 31,613.56 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | light[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 5.51 h | 10,900 | 31,617.60 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 521.1 s | 21,040 | 31,590.06 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 824.5 s | 22,360 | 31,594.05 | 2026.8.17.1 |
| `hst/slam5/seed1` | `slam_base` | linear | mass_total[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 13.91 h | 25,500 | 31,595.97 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 751.0 s | 89,350 | 31,968.39 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 751.3 s | 89,350 | 31,968.39 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_lp[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 2.19 h | 91,900 | 31,968.50 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1877.9 s | 43,300 | 31,543.19 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1.32 h | 51,420 | 31,542.90 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_pix[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 96.20 h | 50,020 | 31,545.50 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 109.4 s | 2,260 | 31,526.51 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 109.5 s | 2,200 | 31,551.70 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | source_pix[2] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 3.21 h | 2,140 | 31,544.10 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 435.2 s | 12,880 | 31,516.58 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 346.3 s | 10,340 | 31,541.50 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | light[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 16.96 h | 11,080 | 31,533.97 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 0 | 1216.9 s | 27,940 | 31,499.14 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 0 | 1491.9 s | 27,180 | 31,531.15 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed0` | `delaunay_1250` | linear | mass_total[1] | `hpc_a100_numba_cpu_sparse_fp64` | 0 | 96.01 h | 39,580 | 31,515.84 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_lp[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 773.9 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_lp[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 757.2 s | 87,700 | 31,968.14 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_lp[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 1.88 h | 90,650 | 31,968.42 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_pix[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 2384.0 s | 55,840 | 31,546.08 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_pix[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1.41 h | 55,720 | 31,545.51 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_pix[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 88.18 h | 48,000 | 31,563.06 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_pix[2] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 109.6 s | 2,180 | 31,550.64 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_pix[2] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 111.3 s | 2,300 | 31,564.08 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | source_pix[2] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 3.23 h | 2,240 | 31,568.86 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | light[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 405.0 s | 11,320 | 31,539.88 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | light[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 368.4 s | 10,860 | 31,553.15 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | light[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 17.08 h | 11,620 | 31,556.74 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | mass_total[1] | `hpc_a100_jax_gpu_dense_fp64` | 1 | 1537.6 s | 35,980 | 31,528.44 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | mass_total[1] | `hpc_a100_jax_gpu_sparse_fp64` | 1 | 1699.2 s | 29,920 | 31,538.40 | 2026.8.17.1 |
| `hst/slam5_delaunay_1250/seed1` | `delaunay_1250` | linear | mass_total[1] | `hpc_a100_numba_cpu_sparse_fp64` | 1 | 100.15 h | 39,680 | 31,544.76 | 2026.8.17.1 |

**Parity — `hst/slam5/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_a100_numba_cpu_sparse_fp64` |
|---|---|---|---|
| `einstein_radius` | 1.605 ± 0.0035 | -0.33σ | -0.25σ |
| `slope` | 2.057 ± 0.044 | -0.21σ | -0.22σ |
| `shear_magnitude` | 0.0733 ± 0.0022 | +0.78σ | -0.30σ |
| `centre_0` | -0.0004065 ± 0.00045 | -0.97σ | +0.05σ |
| `centre_1` | -0.0005637 ± 0.00045 | +0.39σ | +0.00σ |
| `ell_comps_0` | 0.05554 ± 0.0022 | -0.22σ | -0.40σ |
| `ell_comps_1` | -0.0003968 ± 0.001 | +8.14σ | +0.03σ |
| `gamma_1` | 0.05228 ± 0.0023 | +1.25σ | -0.20σ |
| `gamma_2` | 0.05135 ± 0.00088 | -0.63σ | -0.45σ |

**Parity — `hst/slam5/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_a100_numba_cpu_sparse_fp64` |
|---|---|---|---|
| `einstein_radius` | 1.6 ± 0.0031 | +1.30σ | +2.20σ |
| `slope` | 1.991 ± 0.04 | +1.28σ | +2.21σ |
| `shear_magnitude` | 0.07018 ± 0.002 | +1.13σ | +2.27σ |
| `centre_0` | -0.0005879 ± 0.00046 | +0.63σ | -0.16σ |
| `centre_1` | -0.0003854 ± 0.00048 | -0.09σ | -0.41σ |
| `ell_comps_0` | 0.05306 ± 0.0018 | +1.00σ | +1.94σ |
| `ell_comps_1` | -0.000944 ± 0.0012 | +0.01σ | +1.38σ |
| `gamma_1` | 0.04883 ± 0.0021 | +1.13σ | +2.39σ |
| `gamma_2` | 0.05038 ± 0.00083 | +0.95σ | +1.60σ |

**Parity — `hst/slam5_delaunay_1250/seed0` (seed 0), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_a100_numba_cpu_sparse_fp64` |
|---|---|---|---|
| `einstein_radius` | 1.61 ± 0.0028 | +2.01σ | +1.52σ |
| `slope` | 2.105 ± 0.034 | +2.24σ | +1.59σ |
| `shear_magnitude` | 0.07773 ± 0.0015 | +1.91σ | +2.08σ |
| `centre_0` | -0.0007015 ± 0.00055 | +0.13σ | +0.60σ |
| `centre_1` | 0.0001288 ± 0.00039 | -0.68σ | -1.60σ |
| `ell_comps_0` | 0.06029 ± 0.0017 | +1.04σ | +1.72σ |
| `ell_comps_1` | 0.005179 ± 0.0014 | -0.18σ | +2.99σ |
| `gamma_1` | 0.05675 ± 0.0017 | +2.22σ | +2.14σ |
| `gamma_2` | 0.05315 ± 0.00053 | +0.21σ | +1.35σ |

**Parity — `hst/slam5_delaunay_1250/seed1` (seed 1), `mass_total[1]`**

| Parameter | `hpc_a100_jax_gpu_dense_fp64` (ref) | `hpc_a100_jax_gpu_sparse_fp64` | `hpc_a100_numba_cpu_sparse_fp64` |
|---|---|---|---|
| `einstein_radius` | 1.597 ± 0.002 | +3.01σ | +9.95σ |
| `slope` | 1.955 ± 0.024 | +3.13σ | +9.94σ |
| `shear_magnitude` | 0.0692 ± 0.0012 | +3.80σ | +10.17σ |
| `centre_0` | -0.0002983 ± 0.0003 | +0.15σ | +0.91σ |
| `centre_1` | 0.0006144 ± 0.00022 | -3.64σ | -3.44σ |
| `ell_comps_0` | 0.05332 ± 0.00098 | +3.62σ | +10.21σ |
| `ell_comps_1` | -0.0008448 ± 0.00042 | +7.07σ | +19.10σ |
| `gamma_1` | 0.04698 ± 0.0014 | +3.76σ | +10.45σ |
| `gamma_2` | 0.05071 ± 0.00047 | +3.03σ | +5.93σ |
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
feeds. Baseline export writes portable manifest metadata under tracked `prepared/<problem-id>/manifest.json` alongside local bulk snapshots in `output/prepared/`. Commit that metadata with the result rows so CI and fresh clones retain the problem declaration. These manifests attach
exact dataset/model/prior identities to their source baseline; unpublished
artifact paths do not promise publicly downloadable samples.

Export the completed baseline's frozen `mass_total[1]` problem with
`--export-prepared output/prepared/hst-rectangular` on the rectangular baseline
command. Then investigate its shared problem with:

```bash
python scripts/misc/experiments/run.py --prepared output/prepared/hst-rectangular/manifest.json --problem-id <manifest-id> --run-id hst-rectangular-nautilus-cold-seed0 --sampler nautilus --start cold --hardware-id laptop-cpu-8cores-single-process
```

Replace `<manifest-id>` with the manifest's exact `id` field and set `--hardware-id` to the actual host/device, allocated resources and concurrency.
Each imaging setup folder also provides `nautilus.py`, `emcee.py`, `nuts.py`
and `smc.py` investigation leaves. They bind the shared runner to the named
setup family and sampler and refuse a mismatched prepared problem. For example,
a prepared JAX Delaunay baseline can be investigated with:

```bash
python scripts/imaging/delaunay/smc.py --prepared output/prepared/hst-delaunay/manifest.json --problem-id <manifest-id> --run-id hst-delaunay-smc-cold-seed0 --start cold --hardware-id laptop-cpu-8cores-single-process
```

Use `python scripts/misc/experiments/run.py --help` for supported start modes
and samplers. Cold/warm/resume sampler initialization is recorded independently
of compilation and cache conditions. Frozen priors are never narrowed to create
a warm start. Executors verify manifest identities and artifact hashes before
reuse; unavailable historical artifacts remain unknown.
