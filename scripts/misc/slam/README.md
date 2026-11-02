# `slam` — the validation drivers: one core, one module per family

Every `scripts/<dataset_type>/<variant>/validation/` leaf is a short shim over a driver here.
The drivers write one `stages.json` row per `(variant, config_name, seed)` under
`results/<dataset_type>/<variant>/<instrument>/validation/<model_variant>/<setup_id>/seed_<n>/`,
which `scripts/misc/tooling/export_inference_summary.py` reads into the inference summary and
pairs, stage by stage, with the profiling runtime records in `walltime_validation.json`.

| Module | What it owns |
|--------|--------------|
| [`_core.py`](./_core.py) | Everything that is not a dataset: the flag checks (config-name grammar, `--stages-from`), the backend environment, the stage-row readers, posterior and truth helpers, the PyAutoFit output prefix and the results path, row provenance, shared-stage loading, the `stages.json` writer — and `run_family`, the generic driver. |
| [`_runner.py`](./_runner.py) | The imaging family: the five-stage SLaM chain (`rect`, `delaunay`, `delaunay_nn` meshes), its lens-light modes and prepared export, and `run_slam`. It re-exports the core under the names it always had. |
| [`_imaging_mge.py`](./_imaging_mge.py) | 4c-1: `imaging/mge`, the three-stage workspace MGE SLaM chain (`source_lp[1]` -> `light[1]` -> `mass_total[1]`), every stage the 20 + 20 Gaussian MGE; `--lens-light fixed\|both` refits `mass_total[1]` (`Family.light_fixed`). |
| [`_interferometer.py`](./_interferometer.py) | 4c-1: `interferometer/{rectangular,delaunay,mge}` on `sma`. The 4-stage workspace interferometer SLaM chain with the profiled 35 x 35 rectangular or 1250-vertex Delaunay source in the three pixelized stages (`source_lp[1]` declares `null`), and the MGE chain (`source_lp[1]` -> `mass_total[1]`, a 20-Gaussian source). `--transformer nufft` (dense) / `wtilde` (sparse). |
| [`_datacube.py`](./_datacube.py) | 4c-1: `datacube/delaunay` on `sma`, one Nautilus search over a factor graph of the profiled 34-channel cube, the interferometer Delaunay preset per channel. |
| [`_point.py`](./_point.py) | 4c-3: `point_source_source/source_plane` and `point_source_image/image_plane`, the one `modeling[1]` search of `autolens_workspace/scripts/point_source/modeling.py` with the profiled solved fit class; also the setup refusal the other single-search families share. |
| [`_cluster.py`](./_cluster.py) | 4c-3: `cluster/source_plane` and `cluster/image_plane`, `scripts/cluster/modeling.py`'s one factor-graph search with the profiled 2 main + 10 scaling dPIE + NFW model (image plane: the nearest analogue, no workspace pipeline). |
| [`_weak.py`](./_weak.py) | 4c-3: `weak/isothermal`, `scripts/weak/modeling.py`'s one search of a free elliptical `Isothermal`. |
| [`_group.py`](./_group.py) | 4c-2: `group/{rectangular,delaunay,mge}` on `euclid`. `group/slam.py` and its pixelization variant (six stages; the profiled 35 x 35 rectangular or 1250-vertex Delaunay source in the four pixelized stages, both `source_lp` stages declare `null`) and its MGE variant (three); `--lens-light fixed\|both` refits `mass_total[1]` (`Family.light_fixed`). Also the masked-dataset loader and stage helpers `_multi_galaxy.py` shares. |
| [`_multi_galaxy.py`](./_multi_galaxy.py) | 4c-2: `multi_galaxy/{rectangular,delaunay,mge}` on `hst`. `multi_galaxy/slam.py`'s five stages on each mesh, and the three-stage MGE analogue; `Family.light_fixed`. |
| [`_multi_dataset.py`](./_multi_dataset.py) | 4c-2: `multi_dataset/delaunay` on `hst`. `multi_dataset/features/slam/simultaneous.py` without its subhalo stages, a `FactorGraphModel` over the hst dataset and its 0.5-scaled copy; `Family.light_fixed`. |

## Adding a family

A family is one likelihood's chain, as the profiling catalogue names the likelihood
(`interferometer/mge`, `point_source_image/image_plane`, `weak/isothermal`). Mirror the
`autolens_workspace` pipeline of that dataset type, with the profiled cell's mesh,
regularization and size in the stages whose likelihood is the one timed.

1. **Module** — `scripts/misc/slam/_<dataset_type>.py`, holding a `load(leg)` that reads the
   dataset (`leg.dataset_path`, auto-simulated first) into a context object, one
   `_core.Stage` per search whose `build(leg, context, results)` returns
   `(model, analysis, search)`, and the `_core.Family` that ties them together:

   ```python
   FAMILY = Family(
       dataset_type="interferometer",
       likelihood_variant="mge",
       stages=(Stage("source_lp", "source_lp[1]", 200, 50, build_source_lp), ...),
       load=load,
       transformers=("nufft",),
       inversions=("dense",),
   )
   ```

   Build every search with `leg.nautilus(stage)` and pass `use_jax=leg.use_jax` to every
   `Analysis`. The imaging pixelization helpers (`_runner.mesh_models`, `_adapt_images`,
   `mesh_fields`) take any masked dataset; a pixelized family records `mesh`, `mesh_pixels`
   and `mesh_shape` through `model_fields`, which is what the exporter's size comparison
   reads. Run it at the profiled standard size (`_inference_cli.MESH_PIXELS_STANDARD`:
   rectangular 35 x 35, 1250 Delaunay vertices).
2. **Leaf** — `scripts/<dataset_type>/<variant>/validation/<name>.py`: the preamble of
   [`baseline_slam.py`](../../imaging/rectangular/validation/baseline_slam.py), then
   `sys.exit(run_family(FAMILY, default_instrument="sma"))`.
3. **Witness** — a `PYAUTO_TEST_MODE=1` run of the leaf writes a row; the exporter must
   resolve it to the family's likelihood and setup id.

A single-search family (point source, cluster, weak lensing, the datacube) is a `Family` with
one `Stage` and writes the same `stages.json`, with that one stage.

## What a family row declares

- `likelihood: {dataset_type, variant}` and `model_variant` on the row — the exporter files the
  row by these, never by its variant name.
- `likelihood_evaluated` on every stage: the likelihood block when the stage's evaluations are
  that likelihood, `null` when they are none of the profiled ones (`Stage(...,
  evaluates_likelihood=False)`, e.g. a parametric `source_lp[1]` ahead of a pixelized chain).
  The exporter pairs each stage with the profiling record of the likelihood it declares, so an
  MGE, point, cluster or weak stage pairs although it solves no inversion.
- `transformer` (`nufft` / `wtilde`, else `null`), the fifth field of the config name.
- `lens_light_mode` (`linear` / `fixed`), only on a family with lens light (`Family.light_fixed`):
  its last stage is fitted once per `--lens-light` mode, and a `fixed` row is the `light_fixed`
  model variant, as the imaging chain's `mass_total[1]`.

The imaging rows write none of these and keep their old resolution: the likelihood from the
mesh or variant name, and a stage with `inversion_applies: false` never paired.

## Config names

`{local,hpc_a100,hpc_ral}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse,na}_{fp64,mp}[_{nufft,wtilde}]`
— `na` for a likelihood with no linear inversion (`--inversion na`), and the transformer suffix
for a visibility-space one (`--transformer nufft` with dense, `wtilde` with sparse). The setup
id is the name less its first field, exactly as the Pulse reference spells it
(`jax_cpu_na_fp64`, `jax_cpu_dense_fp64_nufft`, `numba_cpu_sparse_fp64_wtilde`). Every field
is checked against its flag (`_core.validate_config_name`), and a family accepts only the
setups its likelihood has (`Family.inversions` / `Family.transformers`).
