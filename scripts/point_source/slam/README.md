# `scripts/point_source/slam`

Full **SLaM pipeline** runs on point source data — the science chains as they are actually run, end
to end, stage by stage.

Empty in phase 1. The first leaf lands in phase 3 (the backend-parameterised driver) and is
exercised by the phase-4 Cortex task `slam_hst_base`.

## Shape a leaf takes

One file per target, named for what it fits, never for its backend:

```
scripts/point_source/slam/<target>.py --instrument <inst> --config-name <config> \
    --backend {jax_cpu,numba_cpu,jax_gpu} --inversion {dense,sparse} --seed <n>
```

- The **instrument is a flag**, never a directory.
- The **backend is a column**: the same leaf runs every leg of a parity row, and the legs
  differ only in the flags above and the resulting `--config-name`.
- Each stage writes one result row under `results/slam/`, and the full PyAutoFit output
  tree is kept under `output/` (gitignored — see `AGENTS.md`, "Outputs are KEPT").

Flags, the config-name grammar and the backend facts that make a leg honest are in
[`../../../_inference_cli.py`](../../../_inference_cli.py) and
[`../../../AGENTS.md`](../../../AGENTS.md).
