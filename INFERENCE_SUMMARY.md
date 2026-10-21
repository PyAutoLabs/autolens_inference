# Inference exchange

PyAutoInsight reads `dashboard/summary.json`, schema `inference-summary`, version 1
(`inference-summary@1`); PyAutoPulse reads `dashboard/walltime_validation.json`
(`walltime-validation@1`, below). Generate with
`python scripts/misc/tooling/export_inference_summary.py`; `--check` verifies the
committed snapshot. The exporter imports only the Python standard library.

The source revision is the latest commit affecting results or the exporter, not
an ensuing generated-summary commit. Generation time is that commit's timestamp;
it is **not** measurement freshness. Legacy results have no measurement date, so
`evidence_updated_at` stays null. Publication dispatch separately identifies the
commit containing the published summary, allowing a reader to pin its receipt.

All current result JSONs under `results/` (except `results/misc/`, apart from
`results/misc/experiments/`) and the dated `archive/YYYY/MM/<slug>/` rows are scanned, including
failed/stopped/partial rows and every seed. This project's own archived results
are included with `archived: true`; they are not current baselines. Simulator
summaries and the retired inference programme are excluded from the source scope.
Malformed/unsupported rows remain visible in `coverage.excluded`. No expected-run
manifest exists, so absent/unrecorded jobs cannot be counted as successes or failures.

A pipeline has a parent record plus stable stage records referring to that parent.
A stage's completed marker describes execution; scientific convergence and
acceptance remain `not_assessed`. Cortex remains the conclusions of record.
Unknown provenance is explicit. Declared target, seed, model and stage are retained,
but no automatic comparison group, scientific threshold, winner or best seed is
invented. Original result evidence remains the detailed source of record.

## Clocks

- `setup_s`: null. Setup is not separately measured.
- `sampling_s`: source `wall_s`, the sampler's `samples_info.json` clock. It includes
  sampler overhead and is not pure likelihood or pure sampling time.
- `total_s`: source `total_wall_s`, elapsed `search.fit`. It excludes the earlier
  compilation probe and dataset/model construction, so it is not end-to-end job time.
- `compile_s`: a separate likelihood warm-up probe. It is not added to `total_s`.

Parent pipeline clocks are null: clocks are not summed or subtracted to infer
setup, and no non-overlap assumption is made. Resumed-run flags remain in configuration.

## Setup fields

Every record carries five additive fields (no version bump):

- `setup_mode`: `explicit` for every existing row — its setup was named by
  `--backend`, `--inversion` and `--config-name`, never chosen automatically.
- `setup_resolved`: `{backend, device, operator, precision, transformer, threads}`
  from the config name (`jax`/`numba`, `cpu`/`gpu`, `dense`/`sparse`, `fp64`/`mixed`);
  `transformer` is `na` for imaging and point source; `threads` is the row's pinned
  `omp_num_threads`, else null.
- `setup_id`: `<backend>_<device>_<operator>_<precision>`, the Pulse grammar.
- `host_class`: provenance, from the config prefix (`hpc_a100`, `hpc_ral` ->
  `hpc_ral_cpu`, `local` -> `local_laptop`). An `hpc_a100_*` CPU-backend row is
  relabelled `hpc_ral_cpu` only when its own `host` names a RAL compute node;
  otherwise the prefix is kept and `unknown_reasons.host_class` says so.
- `model_variant`: `default` or `light_fixed` (the run variant minus its base).

Records also carry `likelihood {dataset_type, variant}`. In `dashboard/catalogue.json`
`setup_id` stays the stable family id (`imaging/rectangular/hst`, PyAutoInsight v2);
the compute setup is `setup_resolved`. The family id is now read from `likelihood`
and the instrument rather than guessed from paths.

## Wall-time validation (`dashboard/walltime_validation.json`)

A second artefact, `walltime-validation@1` (PyAutoPulse REFERENCE.md), built from the
current SLaM validation rows. Each record is one **full-fit** Nautilus stage: one
stage of one chain for one (likelihood, instrument, setup_id, host_class,
model_variant, seed). The Pulse key has no stage field, so the stage is folded into
`sampler` as `nautilus:<stage>` and repeated in `stage`; `evidence_path` names the row.

Clocks and derived numbers, all computed here (Pulse computes nothing):

- `wall_s`: the stage's `wall_s`, the sampler clock. It includes sampler overhead,
  so `implied_ms_per_eval = wall_s / likelihood_evals * 1000` is an upper bound on
  the per-evaluation cost, never a per-likelihood measurement.
- `predicted_ms_per_eval`, `reference_runtime_record_id`: from the committed extract
  `catalogue/pulse_lens_runtime_reference.json` of the Pulse `lens` snapshot
  (refresh with `--refresh-pulse-reference <PyAutoPulse>/snapshots/lens.json`).
  A runtime record matches when its dataset, model and instrument equal the
  record's likelihood and instrument and it determines the same `setup_id`
  (backend, device, operator, precision); the metric is `vmap.per_call`, else
  `single_jit_block`, converted to ms. A legacy record that omits the operator never
  matches. No match leaves both null and adds a limitation. Today no record matches.
- `ratio_measured_over_predicted`: implied over predicted, null without a prediction.
- `parallel_efficiency`: null until a 1-core row of the same key exists.
- `parity_status`: `not_assessed` unless `PARITY_RULINGS` in the exporter declares a
  ruling of record (`parity_basis` quotes it). The one ruling today is the
  2026-10-18 `jax_gpu_sparse_fp64` light_fixed `mass_total[1]` divergence.

Omitted, and counted in `limitations`: stages loaded from another config
(`--stages-from`), light_fixed stages identical to the default run's (one
`--lens-light both` job fits stages 1-4 once), incomplete stages and archived rows.
`--check` covers both artefacts.

## Samples and access

Raw samples stay in project/storage homes. A safe relative `output/` path is only
an access hint, not proof that files still exist or are publicly available.
Absolute/traversing paths are omitted. Availability remains `unknown`; the exporter
does not scan local disks and accidentally claim a different public availability
when run on CI. There is no fabricated GitHub archive link.

## Publication and refresh

`inference-summary.yml` publishes on relevant main-branch changes or manual dispatch,
then sends `insight-refresh` to PyAutoLabs/PyAutoInsight using `PAT_PYAUTOLABS`.
The sender runs only after successful summary push; its payload names the exact
published `source_commit` and the summary's `summary_source_commit`. A missing
secret fails visibly. No compute is submitted and no scientific status changes.
