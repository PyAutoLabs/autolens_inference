# autolens_inference — project state

The Cortex ledger for this repository (`PyAutoCortex/projects.yaml`, row
`autolens_inference`, `ledger: wiki/project/state.md`). Commentary, not the register:
where the ledger and a Cortex ruling disagree, the ruling counts.

## Science goal

**The base run.** Does the 5-stage HST SLaM chain give the same answer under every backend
we can run it on — numba-CPU, JAX-CPU and A100 GPU, each with the dense and the sparse
inversion — and what does each of those answers cost?

"The same answer" is the whole question. The six legs share a dataset, a model, a pipeline
and a seed; they differ only in the config name
(`{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}`), so any
disagreement between them is either a real numerical difference worth naming or a bug in
one of the legs. Until that row exists and agrees, nothing measured on top of it means
anything: it is the baseline every later inference improvement is measured *against*.

Beyond it, in order: which searches — with gradients and without — find that answer
faster and more reliably, from a cold start, across seeds; then the same questions for
interferometer and point-source data.

## Where we are

**Born 2026-09-10 — phase 3 of the `autolens-inference` epic shipped 2026-09-11 (PR
pending).**

- Phase 1 — **done**: the repo exists, is registered in the Mind, the Cortex, the Heart,
  the Brain and the org profile, has the profiling-shaped skeleton, and has a working
  `hpc/sync` link to a clone on RAL.
- Phase 2 — **done**: the retired `inference_programme` material in `autolens_profiling`
  was disposed of.
- Phase 3 — **shipped, PR pending**: the backend-parameterised SLaM driver
  (`scripts/misc/slam/_runner.py` + the `scripts/imaging/slam/hst.py` leaf), the per-stage
  result writer and its parity view, the six production submits plus the A100 rate probe,
  and the first measured rows in `wall/rates.py`.
- Phase 4 — **next**: the first Cortex task, `slam_hst_base` — the base run above, six
  legs × two seeds.

The first numbers measured in this repo landed with phase 3. They are **rates, not
results**: one stage (`source_lp[1]`) per backend, measured to size the submits. No parity
row exists yet, so nothing here answers the science question above.

## Journal

Dated entries, newest last, one per piece of work. Template:
[`_template.md`](_template.md).

### 2026-09-10 — repo born

**What ran.** Nothing scientific. The repo was created on github.com, cloned into the
workspace, and given the skeleton: root `ruff.toml` sentinel, `_inference_cli.py`,
`instruments/`, the three simulators, the README auto-table renderer, the SLURM `--time`
budget gate, `hpc/sync` and the `gpu` / `ral` submit templates, lint + profile workflows.

**What we learned.** Nothing yet — by construction. The one decision worth recording is
that **nothing is inherited**: the retired `inference_programme`'s baselines, notes,
tolerance tables, target code and run JSONs do not come across, and its evidence is never
cited here. The human does not trust those runs, and a benchmark repo that quotes numbers
it cannot reproduce is worse than one with no numbers at all.

**Next.** Phase 2 (dispose of the ancestor material), then phase 3 (the driver).

### 2026-09-11 — the SLaM driver, and the first rates measured here

**What ran.** No science. Three rate probes on the cell `imaging/slam/hst`, each
`--stages source_lp` at production settings (seed 0, dense inversion, fp64, no
`PYAUTO_TEST_MODE`) — the parametric MGE stage that opens the chain, chosen because it
builds no inversion and so is the only stage of this cell a laptop can hold.

| leg | host | outcome |
|---|---|---|
| `numba_cpu`, 8 cores | DESKTOP-H143S82 (WSL2, 15 GB) | **partial**: 17,950 evals / 1,226 s = **0.06831 s/eval** |
| `jax_cpu`, 8 cores | DESKTOP-H143S82 | **partial**: 25,500 evals / 1,256 s = **0.04925 s/eval** |
| `jax_gpu` (A100) | RAL, job **342695**, `submit_slam_hst_rate_jax_gpu` | **not measured** — PENDING(Priority) behind seven of our own 12-hour `gpu` jobs and three foreign jobs with ~1d22h left on `euclid-ral-gpu-2` |

Both local probes were stopped before `source_lp[1]` finished, so **`results/` still holds
no production row** and nothing here is a result. They were run **sequentially, never
concurrently**, and in matched windows (1,226 s and 1,256 s): eight cores shared between two
legs would have measured contention rather than either backend. On that basis `jax_cpu` was
**1.39× faster per evaluation** than `numba_cpu` on this stage — an observation about
`source_lp[1]`, not about the chain. The two rows in
`scripts/misc/wall/rates.py` are the first numbers this repo has ever measured, and their
`PROVENANCE` entries say exactly what they may not be used for.

**What we learned.**

1. **The HST cell does not fit on this laptop under JAX, and that is a measurement, not an
   inconvenience.** 15,361 masked pixels; the cost is the nautilus batch, not the dataset.
   At `source_pix[1]` with `n_batch=20`, `jax_cpu` needs **~13 GB** RSS with the dense
   inversion and **~21.6 GB** with the sparse one. A 15 GB machine OOMs on both. So the two
   JAX-CPU legs of the parity row exist only as RAL jobs, the local rate probes are
   `--stages source_lp` only, and the CI witness runs `--instrument euclid`.
2. **`source_lp[1]` is more expensive than the chain's shape suggests.** 17 free
   parameters, and after ~21 minutes neither probe had left nautilus's exploration phase:
   the numba leg was at shell 21 having spent 90 evaluations per live point, the jax leg at
   shell 31 and **128 evaluations per live point**. 128 × the chain's 725 live points =
   **92,800 evaluations** is therefore a *floor* on the whole chain — and the four pixelized
   stages are per-eval-inversion bound on top of that. Every production submit consequently
   ships `source: unmeasured  probe-first: yes` with **no `wall:` at all**: a floor
   multiplied by a rate measured on other hardware for a different kind of stage is an
   invention. `--time` is a stated containment (12 h on the A100 legs, 5 days on the CPU
   legs), not a derived budget.
3. **A rate is per host, and the table now says so in the key.** The measured device tokens
   are `laptop_numba_cpu` and `laptop_jax_cpu`; the submits declare `ral_numba_cpu`,
   `ral_jax_cpu` and `a100`. A laptop rate therefore cannot be cited by a RAL submit even
   by accident — the gate does not match the key.
4. **A cell id now carries its task directory** (`imaging/slam/hst`, the invoked script's
   path below `scripts/` without the `.py`). This repo names a leaf for the target it runs
   rather than for the task, so the old `<dataset>/<leaf>` cut called this cell `imaging/hst`
   — the same id a future `scripts/imaging/searches/<sampler>/hst.py` would claim. Two
   pipelines sharing one rate row is the carry the gate exists to prevent.

**Departures from the workspace chain**, both recorded in the driver's module docstring and
in `scripts/imaging/slam/README.md`: a **positions likelihood on all four pixelized stages**
(the workspace attaches one to `source_pix[1]` and `mass_total[1]` only; a mesh stage with
no `positions.info` beside it is not citable here, and the phase-4 witness reads
`positions_info_present` on every one), and **`log_evidence_err: null`** with a sibling note
— nautilus 1.0.5 exposes no `log_z_err`, and a derived one would be an invention.

**Next.** Re-check RAL job 342695 for the A100 rate, then submit the six production legs
(`hpc/sync submit --gpu|--cpu submit_slam_hst_*`) and open the phase-4 Cortex task
`slam_hst_base` against them.
