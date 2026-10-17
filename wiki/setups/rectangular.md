# HST rectangular imaging

Stable setup: `imaging/rectangular/hst`.
Run `python scripts/imaging/rectangular/baseline_slam.py --instrument hst
--backend jax_cpu --inversion dense --config-name local_jax_cpu_dense_fp64 --seed 0`
(as one shell command). Scientific builders remain shared in `scripts/misc/slam/`.

The project catalogue preserves historical stages and archived evidence.
No selected reference is declared until a current complete baseline has explicit
scientific acceptance and frozen dataset, model and prior identities.
The original result/output variant identities remain unchanged.

The lens-light mode is a run variant of this setup, never a config:
`--lens-light linear` (default) is `slam_base`; `--lens-light fixed` fits
`mass_total[1]` with the lens light fixed to ordinary profiles at the intensities
`light[1]`'s maximum-likelihood fit solved for, as the variant
`slam_base_lightfixed`; `--lens-light both` runs stages 1 to 4 once and the mass
stage twice, writing one row per variant. Both variants bind to this setup id, so
a prepared mass problem of either mode runs under the rectangular sampler leaves
(`--export-prepared DIR` writes `DIR/<variant>/` under `both`).

Clean slate (2026-10-17, PyAutoInsight `hst-imaging-mass-samplers` phase 1): the
four `hpc/batch_*/submit_hst_cleanslate_*` submits re-measure the HST base on the
pinned release 2026.10.17.1 (`jax_gpu` and `jax_cpu`, dense and sparse, seeds 0
and 1, both light modes) into `output/cleanslate_2026_10_17/` and result rows
tagged `cleanslate_2026_10_17`. Schema v3 rows record `library_revisions`,
`release_pin` and `slurm_job_id`.
