# Phase 4 — backend parity at `mass_total[1]` (laid out for human rulings)

Likelihood-lifecycle phase 4e ([autolens_inference#59](https://github.com/PyAutoLabs/autolens_inference/issues/59)).
This page lays out the best fit each backend reached, so the human can rule on parity. **It
computes no verdict.** A ruling goes into `PARITY_RULINGS` in
[`export_inference_summary.py`](../../scripts/misc/tooling/export_inference_summary.py) only on
the human's word; until then every record's `parity_status` stays `not_assessed`.

**What is tabulated.** Every committed, non-archived validation row
(`results/<dataset_type>/<variant>/<instrument>/validation/<model_variant>/<setup>/seed_<n>/stages.json`)
without a results tag, so the cores-sweep legs, which fit `source_pix[1]` only, are left out. For each
row the table gives `max_log_likelihood` and `log_evidence` of the parity stage `mass_total[1]`
(the exporter's `PARITY_STAGE`). The point-source, cluster and weak likelihoods fit one search
and have no `mass_total[1]`, so their one stage, `modeling[1]`, is shown in its place.
"Pin" is the row's `release_pin`. The 414189 rows are marked `+ Array#677`: they ran on the
pin plus PyAutoArray#677 (the exporter's `PATCHED_RUNS`). The spread is the highest
`max log L` minus the lowest among the legs of one (likelihood, instrument, model variant,
seed).

Rows of legs still running on 2026-11-05 are not here (see
[state.md](state.md), 2026-11-05 entry). Most numba-CPU legs are among them, so many
pixelized families have only their A100 leg so far.

`build_readme.py`'s parity view in the [README](../../README.md#pipeline-runs) is the
companion table: it compares posterior medians (Δσ against the first config) for the SLaM
families, but it does not show max log L, and it has no view for the single-search
likelihoods.

## Spread across setups, per seed

| Likelihood | Instr. | Model variant | Stage | Seed | Legs | Spread of max log L | Highest | Lowest |
|---|---|---|---|---|---:|---:|---|---|
| cluster/image_plane | simple | default | modeling[1] | 0 | 2 | 0.00 | jax_gpu_na_fp64 (2026.10.25.1) | jax_cpu_na_fp64 (2026.10.25.1) |
| cluster/image_plane | simple | default | modeling[1] | 1 | 2 | 0.00 | jax_gpu_na_fp64 (2026.10.25.1) | jax_cpu_na_fp64 (2026.10.25.1) |
| cluster/source_plane | simple | default | modeling[1] | 0 | 2 | 0.00 | jax_gpu_na_fp64 (2026.10.25.1) | jax_cpu_na_fp64 (2026.10.25.1) |
| cluster/source_plane | simple | default | modeling[1] | 1 | 2 | 0.11 | jax_gpu_na_fp64 (2026.10.25.1) | jax_cpu_na_fp64 (2026.10.25.1) |
| group/delaunay | euclid | default | mass_total[1] | 0 | 1 | — (one leg) | | |
| group/delaunay | euclid | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| group/delaunay | euclid | light_fixed | mass_total[1] | 0 | 1 | — (one leg) | | |
| group/delaunay | euclid | light_fixed | mass_total[1] | 1 | 1 | — (one leg) | | |
| group/mge | euclid | default | mass_total[1] | 0 | 2 | 0.68 | jax_gpu_dense_fp64 (2026.10.25.1) | jax_cpu_dense_fp64 (2026.10.25.1) |
| group/mge | euclid | default | mass_total[1] | 1 | 2 | 0.30 | jax_cpu_dense_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| group/mge | euclid | light_fixed | mass_total[1] | 0 | 2 | 0.14 | jax_cpu_dense_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| group/mge | euclid | light_fixed | mass_total[1] | 1 | 2 | 0.10 | jax_cpu_dense_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| group/rectangular | euclid | default | mass_total[1] | 0 | 1 | — (one leg) | | |
| group/rectangular | euclid | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| group/rectangular | euclid | light_fixed | mass_total[1] | 0 | 1 | — (one leg) | | |
| group/rectangular | euclid | light_fixed | mass_total[1] | 1 | 1 | — (one leg) | | |
| imaging/delaunay | hst | default | mass_total[1] | 0 | 2 | 43.35 | jax_gpu_dense_fp64 (2026.10.25.1) | numba_cpu_sparse_fp64 (2026.10.25.1) |
| imaging/delaunay | hst | default | mass_total[1] | 1 | 2 | 87.82 | numba_cpu_sparse_fp64 (2026.10.17.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/delaunay | hst | light_fixed | mass_total[1] | 0 | 2 | 43.38 | jax_gpu_dense_fp64 (2026.10.25.1) | numba_cpu_sparse_fp64 (2026.10.25.1) |
| imaging/delaunay | hst | light_fixed | mass_total[1] | 1 | 2 | 90.37 | numba_cpu_sparse_fp64 (2026.10.17.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/delaunay_nn | hst | default | mass_total[1] | 0 | 2 | 55.58 | numba_cpu_sparse_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/delaunay_nn | hst | default | mass_total[1] | 1 | 2 | 79.05 | numba_cpu_sparse_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/delaunay_nn | hst | light_fixed | mass_total[1] | 0 | 2 | 49.35 | numba_cpu_sparse_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/delaunay_nn | hst | light_fixed | mass_total[1] | 1 | 2 | 76.48 | numba_cpu_sparse_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/mge | hst | default | mass_total[1] | 0 | 2 | 0.18 | jax_cpu_dense_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/mge | hst | default | mass_total[1] | 1 | 2 | 0.08 | jax_gpu_dense_fp64 (2026.10.25.1) | jax_cpu_dense_fp64 (2026.10.25.1) |
| imaging/mge | hst | light_fixed | mass_total[1] | 0 | 2 | 0.27 | jax_cpu_dense_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| imaging/mge | hst | light_fixed | mass_total[1] | 1 | 2 | 0.09 | jax_gpu_dense_fp64 (2026.10.25.1) | jax_cpu_dense_fp64 (2026.10.25.1) |
| imaging/rectangular | hst | default | mass_total[1] | 0 | 5 | 11.27 | jax_gpu_dense_fp64 (2026.10.17.1) | jax_cpu_sparse_fp64 (2026.10.17.1) |
| imaging/rectangular | hst | default | mass_total[1] | 1 | 4 | 7.40 | jax_gpu_sparse_fp64 (2026.10.17.1) | jax_gpu_dense_fp64 (2026.10.17.1) |
| imaging/rectangular | hst | light_fixed | mass_total[1] | 0 | 4 | 75,156.58 | jax_gpu_dense_fp64 (2026.10.17.1) | jax_gpu_sparse_fp64 (2026.10.17.1) |
| imaging/rectangular | hst | light_fixed | mass_total[1] | 1 | 4 | 74,779.92 | numba_cpu_sparse_fp64 (2026.10.17.1) | jax_gpu_sparse_fp64 (2026.10.17.1) |
| interferometer/delaunay | sma | default | mass_total[1] | 0 | 2 | 0.22 | numba_cpu_sparse_fp64_wtilde (2026.10.25.1) | jax_gpu_sparse_fp64_wtilde (2026.10.25.1) |
| interferometer/delaunay | sma | default | mass_total[1] | 1 | 2 | 3.16 | numba_cpu_sparse_fp64_wtilde (2026.10.25.1) | jax_gpu_sparse_fp64_wtilde (2026.10.25.1) |
| interferometer/mge | sma | default | mass_total[1] | 0 | 2 | 0.19 | jax_cpu_dense_fp64_nufft (2026.10.25.1) | jax_gpu_dense_fp64_nufft (2026.10.25.1) |
| interferometer/mge | sma | default | mass_total[1] | 1 | 2 | 0.09 | jax_cpu_dense_fp64_nufft (2026.10.25.1) | jax_gpu_dense_fp64_nufft (2026.10.25.1) |
| interferometer/rectangular | sma | default | mass_total[1] | 0 | 2 | 3.99 | numba_cpu_sparse_fp64_wtilde (2026.10.25.1) | jax_gpu_sparse_fp64_wtilde (2026.10.25.1) |
| interferometer/rectangular | sma | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_dataset/delaunay | hst | default | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_dataset/delaunay | hst | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_dataset/delaunay | hst | light_fixed | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_dataset/delaunay | hst | light_fixed | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_galaxy/delaunay | hst | default | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_galaxy/delaunay | hst | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_galaxy/delaunay | hst | light_fixed | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_galaxy/delaunay | hst | light_fixed | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_galaxy/mge | hst | default | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_galaxy/mge | hst | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_galaxy/mge | hst | light_fixed | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_galaxy/mge | hst | light_fixed | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_galaxy/rectangular | hst | default | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_galaxy/rectangular | hst | default | mass_total[1] | 1 | 1 | — (one leg) | | |
| multi_galaxy/rectangular | hst | light_fixed | mass_total[1] | 0 | 1 | — (one leg) | | |
| multi_galaxy/rectangular | hst | light_fixed | mass_total[1] | 1 | 1 | — (one leg) | | |
| point_source_image/image_plane | simple | default | modeling[1] | 0 | 2 | 0.06 | jax_gpu_na_fp64 (2026.10.25.1) | jax_cpu_na_fp64 (2026.10.25.1) |
| point_source_image/image_plane | simple | default | modeling[1] | 1 | 2 | 0.00 | jax_cpu_na_fp64 (2026.10.25.1) | jax_gpu_na_fp64 (2026.10.25.1) |
| point_source_source/source_plane | simple | default | modeling[1] | 0 | 2 | 0.00 | jax_cpu_na_fp64 (2026.10.25.1) | jax_gpu_na_fp64 (2026.10.25.1) |
| point_source_source/source_plane | simple | default | modeling[1] | 1 | 2 | 0.00 | jax_gpu_na_fp64 (2026.10.25.1) | jax_cpu_na_fp64 (2026.10.25.1) |
| weak/isothermal | simple | default | modeling[1] | 0 | 2 | 0.00 | jax_cpu_dense_fp64 (2026.10.25.1) | jax_gpu_dense_fp64 (2026.10.25.1) |
| weak/isothermal | simple | default | modeling[1] | 1 | 2 | 0.00 | jax_gpu_dense_fp64 (2026.10.25.1) | jax_cpu_dense_fp64 (2026.10.25.1) |

## Gaps worth a look (observations, not verdicts)

- **imaging/delaunay (hst), seed 1: the adopted numba row sits about 88 (default) and 90
  (light_fixed) log L above the A100 row.** The two rows are on different pins: numba on
  2026.10.17.1 (412624_1, adopted), A100 on 2026.10.25.1 (413924_1). Seed 0, where both legs are on
  2026.10.25.1, shows a gap of about 43 the other way: the A100 row (31,526.6) is above numba
  (31,483.2, 414149_0). Between the two numba rows on different pins and seeds, max log L moves by
  about 130 (31,613.9 against 31,483.2).
- **imaging/delaunay_nn (hst): the numba rows (pin + Array#677) sit 49–79 log L above the A100
  rows (pin)** at both seeds and in both model variants. The setups also differ in inversion:
  numba sparse against A100 dense.
- **imaging/rectangular (hst) light_fixed: jax_gpu_sparse is about 75,000 log L below every other
  leg.** This is the existing `diverged` ruling of 2026-10-18 (`PARITY_RULINGS`), unchanged.
- **Interferometer (sma): numba wtilde against A100 wtilde is within 0.2–4 log L.** MGE (jax CPU
  against A100 NUFFT) is within 0.2.
- **MGE (imaging, group, interferometer), point source, cluster and weak: CPU and A100 agree to 0.7
  log L or better**, mostly to 0.1–0.3.
- **multi_dataset/delaunay: seeds 0 and 1 differ by about 175 log L** (73,659 against 73,833) on
  the one backend that has run (A100). Other likelihoods' seeds differ by a few log L at most.
- **multi_galaxy/mge: default and light_fixed differ by about 560 log L** (12,516 against 11,954).
  The other multi_galaxy meshes differ by 1–2 log L between variants. No second backend has run
  yet (414362, jax CPU, is running).

## Every leg

| Likelihood | Instr. | Model variant | Stage | Setup | Host | Pin | Slurm job | Seed | max log L | log Z |
|---|---|---|---|---|---|---|---|---|---:|---:|
| cluster/image_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414145_0 | 0 | 34.61 | -3.05 |
| cluster/image_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413966_0 | 0 | 34.61 | -3.05 |
| cluster/image_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414145_1 | 1 | 34.50 | -2.84 |
| cluster/image_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413966_1 | 1 | 34.50 | -2.84 |
| cluster/source_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414147_0 | 0 | 43.07 | 5.73 |
| cluster/source_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413967_0 | 0 | 43.07 | 5.73 |
| cluster/source_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414147_1 | 1 | 42.96 | 5.87 |
| cluster/source_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413967_1 | 1 | 43.07 | 5.62 |
| group/delaunay | euclid | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413969_0 | 0 | 33,018.61 | 32,950.75 |
| group/delaunay | euclid | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413969_1 | 1 | 33,011.54 | 32,951.98 |
| group/delaunay | euclid | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413969_0 | 0 | 33,010.93 | 32,943.29 |
| group/delaunay | euclid | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413969_1 | 1 | 33,000.02 | 32,940.57 |
| group/mge | euclid | default | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414152_0 | 0 | 33,569.09 | 33,489.64 |
| group/mge | euclid | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413970_0 | 0 | 33,569.77 | 33,489.66 |
| group/mge | euclid | default | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414152_1 | 1 | 33,569.66 | 33,489.75 |
| group/mge | euclid | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413970_1 | 1 | 33,569.36 | 33,489.64 |
| group/mge | euclid | light_fixed | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414152_0 | 0 | 33,569.11 | 33,488.53 |
| group/mge | euclid | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413970_0 | 0 | 33,568.96 | 33,488.52 |
| group/mge | euclid | light_fixed | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414152_1 | 1 | 33,569.03 | 33,488.73 |
| group/mge | euclid | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413970_1 | 1 | 33,568.92 | 33,488.58 |
| group/rectangular | euclid | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413971_0 | 0 | 33,105.31 | 33,047.94 |
| group/rectangular | euclid | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413971_1 | 1 | 33,106.85 | 33,049.03 |
| group/rectangular | euclid | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413971_0 | 0 | 33,095.21 | 33,037.03 |
| group/rectangular | euclid | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413971_1 | 1 | 33,096.72 | 33,038.45 |
| imaging/delaunay | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413924_0 | 0 | 31,526.60 | 31,471.39 |
| imaging/delaunay | hst | default | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.25.1 | 414149_0 | 0 | 31,483.24 | 31,426.95 |
| imaging/delaunay | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413924_1 | 1 | 31,526.03 | 31,472.18 |
| imaging/delaunay | hst | default | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412624_1 | 1 | 31,613.85 | 31,554.38 |
| imaging/delaunay | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413924_0 | 0 | 31,526.58 | 31,471.75 |
| imaging/delaunay | hst | light_fixed | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.25.1 | 414149_0 | 0 | 31,483.20 | 31,426.86 |
| imaging/delaunay | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413924_1 | 1 | 31,525.26 | 31,472.48 |
| imaging/delaunay | hst | light_fixed | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412624_1 | 1 | 31,615.63 | 31,557.08 |
| imaging/delaunay_nn | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413968_0 | 0 | 31,453.85 | 31,400.21 |
| imaging/delaunay_nn | hst | default | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.25.1 + Array#677 | 414189_0 | 0 | 31,509.43 | 31,451.57 |
| imaging/delaunay_nn | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413968_1 | 1 | 31,434.88 | 31,382.31 |
| imaging/delaunay_nn | hst | default | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.25.1 + Array#677 | 414189_1 | 1 | 31,513.93 | 31,460.08 |
| imaging/delaunay_nn | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413968_0 | 0 | 31,452.24 | 31,399.81 |
| imaging/delaunay_nn | hst | light_fixed | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.25.1 + Array#677 | 414189_0 | 0 | 31,501.59 | 31,448.01 |
| imaging/delaunay_nn | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413968_1 | 1 | 31,434.28 | 31,380.77 |
| imaging/delaunay_nn | hst | light_fixed | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.25.1 + Array#677 | 414189_1 | 1 | 31,510.76 | 31,457.24 |
| imaging/mge | hst | default | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414157_0 | 0 | 32,062.47 | 31,986.32 |
| imaging/mge | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413975_0 | 0 | 32,062.28 | 31,986.24 |
| imaging/mge | hst | default | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414157_1 | 1 | 32,062.45 | 31,986.43 |
| imaging/mge | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413975_1 | 1 | 32,062.53 | 31,985.97 |
| imaging/mge | hst | light_fixed | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414157_0 | 0 | 32,062.43 | 31,986.04 |
| imaging/mge | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413975_0 | 0 | 32,062.17 | 31,985.88 |
| imaging/mge | hst | light_fixed | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414157_1 | 1 | 32,062.08 | 31,985.98 |
| imaging/mge | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413975_1 | 1 | 32,062.17 | 31,985.98 |
| imaging/rectangular | hst | default | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.17.1 | 412539_0 | 0 | 31,652.26 | 31,596.30 |
| imaging/rectangular | hst | default | mass_total[1] | jax_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412541_0 | 0 | 31,645.87 | 31,594.55 |
| imaging/rectangular | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.17.1 | 412535_0 | 0 | 31,657.14 | 31,601.91 |
| imaging/rectangular | hst | default | mass_total[1] | jax_gpu_sparse_fp64 | A100 | 2026.10.17.1 | 412537_0 | 0 | 31,649.20 | 31,596.51 |
| imaging/rectangular | hst | default | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412622_0 | 0 | 31,652.16 | 31,599.97 |
| imaging/rectangular | hst | default | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.17.1 | 412539_1 | 1 | 31,644.40 | 31,592.95 |
| imaging/rectangular | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.17.1 | 412535_1 | 1 | 31,644.25 | 31,591.75 |
| imaging/rectangular | hst | default | mass_total[1] | jax_gpu_sparse_fp64 | A100 | 2026.10.17.1 | 412537_1 | 1 | 31,651.65 | 31,598.45 |
| imaging/rectangular | hst | default | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412622_1 | 1 | 31,648.57 | 31,596.13 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.17.1 | 412539_0 | 0 | 31,651.35 | 31,597.63 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.17.1 | 412535_0 | 0 | 31,654.85 | 31,600.99 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | jax_gpu_sparse_fp64 | A100 | 2026.10.17.1 | 412537_0 | 0 | -43,501.73 | -43,567.08 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412622_0 | 0 | 31,652.06 | 31,598.85 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.17.1 | 412539_1 | 1 | 31,644.03 | 31,592.62 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.17.1 | 412535_1 | 1 | 31,643.25 | 31,592.66 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | jax_gpu_sparse_fp64 | A100 | 2026.10.17.1 | 412537_1 | 1 | -43,131.69 | -43,201.59 |
| imaging/rectangular | hst | light_fixed | mass_total[1] | numba_cpu_sparse_fp64 | RAL CPU | 2026.10.17.1 | 412622_1 | 1 | 31,648.23 | 31,595.85 |
| interferometer/delaunay | sma | default | mass_total[1] | jax_gpu_sparse_fp64_wtilde | A100 | 2026.10.25.1 | 413972_0 | 0 | -3,155.06 | -3,163.72 |
| interferometer/delaunay | sma | default | mass_total[1] | numba_cpu_sparse_fp64_wtilde | RAL CPU | 2026.10.25.1 | 414154_0 | 0 | -3,154.84 | -3,162.72 |
| interferometer/delaunay | sma | default | mass_total[1] | jax_gpu_sparse_fp64_wtilde | A100 | 2026.10.25.1 | 413972_1 | 1 | -3,155.01 | -3,164.82 |
| interferometer/delaunay | sma | default | mass_total[1] | numba_cpu_sparse_fp64_wtilde | RAL CPU | 2026.10.25.1 | 414154_1 | 1 | -3,151.85 | -3,164.46 |
| interferometer/mge | sma | default | mass_total[1] | jax_cpu_dense_fp64_nufft | RAL CPU | 2026.10.25.1 | 414155_0 | 0 | -3,146.79 | -3,154.07 |
| interferometer/mge | sma | default | mass_total[1] | jax_gpu_dense_fp64_nufft | A100 | 2026.10.25.1 | 413973_0 | 0 | -3,146.97 | -3,154.04 |
| interferometer/mge | sma | default | mass_total[1] | jax_cpu_dense_fp64_nufft | RAL CPU | 2026.10.25.1 | 414155_1 | 1 | -3,146.51 | -3,154.06 |
| interferometer/mge | sma | default | mass_total[1] | jax_gpu_dense_fp64_nufft | A100 | 2026.10.25.1 | 413973_1 | 1 | -3,146.60 | -3,154.02 |
| interferometer/rectangular | sma | default | mass_total[1] | jax_gpu_sparse_fp64_wtilde | A100 | 2026.10.25.1 | 413974_0 | 0 | -3,155.07 | -3,163.04 |
| interferometer/rectangular | sma | default | mass_total[1] | numba_cpu_sparse_fp64_wtilde | RAL CPU | 2026.10.25.1 | 414156_0 | 0 | -3,151.08 | -3,160.50 |
| interferometer/rectangular | sma | default | mass_total[1] | jax_gpu_sparse_fp64_wtilde | A100 | 2026.10.25.1 | 413974_1 | 1 | -3,155.07 | -3,163.93 |
| multi_dataset/delaunay | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413976_0 | 0 | 73,658.77 | 73,578.25 |
| multi_dataset/delaunay | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413976_1 | 1 | 73,833.47 | 73,755.28 |
| multi_dataset/delaunay | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413976_0 | 0 | 73,663.19 | 73,584.32 |
| multi_dataset/delaunay | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413976_1 | 1 | 73,831.76 | 73,753.31 |
| multi_galaxy/delaunay | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413977_0 | 0 | 11,845.19 | 11,765.14 |
| multi_galaxy/delaunay | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413977_1 | 1 | 11,846.39 | 11,767.79 |
| multi_galaxy/delaunay | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413977_0 | 0 | 11,845.89 | 11,763.28 |
| multi_galaxy/delaunay | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413977_1 | 1 | 11,843.91 | 11,763.92 |
| multi_galaxy/mge | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413978_0 | 0 | 12,515.57 | 12,413.52 |
| multi_galaxy/mge | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413978_1 | 1 | 12,511.98 | 12,410.22 |
| multi_galaxy/mge | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413978_0 | 0 | 11,954.48 | 11,850.03 |
| multi_galaxy/mge | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413978_1 | 1 | 11,952.14 | 11,848.18 |
| multi_galaxy/rectangular | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413979_0 | 0 | 11,921.80 | 11,844.66 |
| multi_galaxy/rectangular | hst | default | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413979_1 | 1 | 11,921.03 | 11,843.35 |
| multi_galaxy/rectangular | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413979_0 | 0 | 11,920.12 | 11,841.73 |
| multi_galaxy/rectangular | hst | light_fixed | mass_total[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413979_1 | 1 | 11,920.09 | 11,841.69 |
| point_source_image/image_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414162_0 | 0 | 28.13 | 4.29 |
| point_source_image/image_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413980_0 | 0 | 28.20 | 4.29 |
| point_source_image/image_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414162_1 | 1 | 28.18 | 4.31 |
| point_source_image/image_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413980_1 | 1 | 28.18 | 4.31 |
| point_source_source/source_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414163_0 | 0 | 16.49 | -7.15 |
| point_source_source/source_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413981_0 | 0 | 16.49 | -7.15 |
| point_source_source/source_plane | simple | default | modeling[1] | jax_cpu_na_fp64 | RAL CPU | 2026.10.25.1 | 414163_1 | 1 | 16.47 | -7.16 |
| point_source_source/source_plane | simple | default | modeling[1] | jax_gpu_na_fp64 | A100 | 2026.10.25.1 | 413981_1 | 1 | 16.47 | -7.16 |
| weak/isothermal | simple | default | modeling[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414180_0 | 0 | -84.18 | -93.00 |
| weak/isothermal | simple | default | modeling[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413982_0 | 0 | -84.18 | -93.00 |
| weak/isothermal | simple | default | modeling[1] | jax_cpu_dense_fp64 | RAL CPU | 2026.10.25.1 | 414180_1 | 1 | -84.19 | -93.01 |
| weak/isothermal | simple | default | modeling[1] | jax_gpu_dense_fp64 | A100 | 2026.10.25.1 | 413982_1 | 1 | -84.19 | -93.01 |
