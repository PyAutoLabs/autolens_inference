# Inference exchange

PyAutoInsight reads `dashboard/summary.json`, schema `inference-summary`, version 1
(`inference-summary@1`). Generate with
`python scripts/misc/tooling/export_inference_summary.py`; `--check` verifies the
committed snapshot. The exporter imports only the Python standard library.

The source revision is the latest commit affecting results or the exporter, not
an ensuing generated-summary commit. Generation time is that commit's timestamp;
it is **not** measurement freshness. Legacy results have no measurement date, so
`evidence_updated_at` stays null. Publication dispatch separately identifies the
commit containing the published summary, allowing a reader to pin its receipt.

All current `results/searches` and `results/slam` JSONs are scanned, including
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
