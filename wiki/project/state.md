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

**Born 2026-09-10 — phase 1 of the `autolens-inference` epic.**

- Phase 1 (this): the repo exists, is registered in the Mind, the Cortex, the Heart, the
  Brain and the org profile, has the profiling-shaped skeleton, and has a working
  `hpc/sync` link to a clone on RAL.
- Phase 2: dispose of the retired `inference_programme` material still sitting in
  `autolens_profiling`.
- Phase 3: the backend-parameterised SLaM driver and per-stage result writer.
- Phase 4: the first Cortex task, `slam_hst_base` — the base run above.

Nothing has been measured here yet. `results/` is empty, and `scripts/misc/wall/rates.py`
starts empty for the same reason: this repo inherits no evidence from its ancestor, so
every submit says `source: unmeasured  probe-first: yes` until a rate is measured here.

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
