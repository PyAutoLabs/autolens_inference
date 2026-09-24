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

**2026-09-24 — every existing SLaM HST row is archived; the A100 legs and the numba-CPU sparse leg are
re-running on the sped-up library mains.** The four `slam_base` and four `delaunay_1250` A100 rows (plus the
`342695` rate-probe row) were measured before the autolens_profiling likelihood speedups
landed, so they now describe code nobody runs. They live on, unchanged, under
`results/archive/2026-09-24_pre_likelihood_speedup/` (and their PyAutoFit trees under
`output/archive/…`, locally and on RAL); the dashboard reads only `results/slam/` and
`results/searches/`, so they are no longer live rows. The live tree is empty until the
re-runs land — see the 2026-09-24 journal entry for the job ids.

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
- Phase 4 — **running**: the first Cortex task, `slam_hst_base` — the base run above, six
  legs × two seeds. **The two A100 legs are in**: `jax_gpu` × {dense, sparse} × seeds 0 and
  1, all five stages, `status: complete`, committed under
  `results/slam/imaging/hst/slam_base/`. The four CPU legs have not run.

`results/` therefore holds this repo's first real measurements — four complete chains, not
rates. They are **one backend of the parity row**, so they still do not answer the science
question above: with nothing to compare against, "the same answer under every backend" has
no second term. What they do establish is that the chain runs end to end on an A100 and
recovers the truth (worst `truth_delta_sigma` at `mass_total[1]` across the four legs:
1.33σ), and what a chain costs there.

The cell also grew a **run-variant level** (`results/.../<instrument>/<variant>/<config>/`):
the base run is `slam_base`, and the mesh under test is `delaunay_1250`, whose two A100
legs are being submitted — both inversion routes proven end to end in test mode, on a
Hilbert-drawn mesh of 1250 interior vertices plus a 30-point zeroed edge ring, and asking for
8 CPUs because this mesh's host callback deadlocks a narrow thread pool.

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

### 2026-09-14 — the A100 base legs land, and the cell grows a variant level

**What ran.** The first four production legs of the base run, on RAL's `gpu` partition:
`scripts/imaging/slam/hst.py --backend jax_gpu --inversion {dense,sparse}`, seeds 0 and 1,
fp64, all five stages, `status: complete`, no stage resumed. Search wall per stage
(seed 0 / seed 1), and reject-inclusive likelihood evaluations:

| stage | n_live/n_batch | dense wall | dense evals | sparse wall | sparse evals |
|---|---|---|---|---|---|
| `source_lp[1]` | 200 / 50 | 923 s / 835 s | 95,950 / 96,150 | 808 s / 798 s | 92,300 / 96,150 |
| `source_pix[1]` | 150 / 20 | 762 s / 749 s | 32,820 / 33,000 | **1,931 s / 1,904 s** | 33,140 / 33,020 |
| `source_pix[2]` | 75 / 20 | 132 s / 362 s | 3,720 / 13,800 | 179 s / 188 s | 3,260 / 3,580 |
| `light[1]` | 150 / 20 | 237 s / 267 s | 9,620 / 12,200 | 268 s / 254 s | 11,320 / 9,980 |
| `mass_total[1]` | 150 / 20 | 517 s / 650 s | 20,080 / 25,820 | 800 s / 868 s | 20,780 / 21,640 |
| **chain** | | **2,571 s / 2,863 s** | 162,190 / 180,970 | **3,985 s / 4,012 s** | 160,800 / 164,370 |

Wall including visualisation and output is 4,144 s / 4,485 s dense and 5,567 s / 5,576 s
sparse — a run is roughly 60% search and 40% writing pictures of itself. `mass_total[1]`
log evidence: **31,592.70** and **31,596.11** dense, **31,589.52** and **31,592.44** sparse
(nautilus 1.0.5 exposes no `log_z_err`, so the row carries `null` and says why). The worst
`truth_delta_sigma` on any of the four is 1.33σ (`gamma_2`), and every pixelized stage has
its `positions.info` beside it.

**What we learned.**

1. **The chain costs ~43 minutes of search on an A100, and the sparse operator buys nothing
   here.** Sparse is 1.4–1.55× *slower* over the chain, and the whole difference is one
   stage: `source_pix[1]` costs 1,931 s sparse against 762 s dense for the same ~33,000
   evaluations — 2.5×. This is a single-dataset, single-mesh observation on the largest
   pixelized stage; it is not a ruling about the operator, and the CPU legs may well say the
   opposite, since the operator exists to avoid a dense mapping matrix that a GPU holds
   comfortably.
2. **The 92,800-evaluation floor was a floor on the wrong thing.** It was derived as 128
   evaluations per live point over the chain's 725 live points; in the event `source_lp[1]`
   *alone* cost 92,300–96,150, and the chain 160,800–180,970. The floor held, but only by
   accident of its own conservatism — the honest reading is that the parametric opening
   stage is over half the chain's evaluations and under a third of its wall.
3. **These are one backend, not a parity row.** Four legs, one device, one mesh, two seeds.
   Seed-to-seed spread on `mass_total[1]` log evidence is 3.4 nats dense and 2.9 sparse, and
   dense-vs-sparse at fixed seed is 3.2 and 3.7 — so the two inversions already agree to
   about the size of the seed noise, which is the first thing the parity row exists to check
   and the only part of it that can be checked yet. The four CPU legs (`ral` partition, 5-day
   containment) have not run.
4. **A second experiment on this cell needed a level in the path.** Rows group into a parity
   row by the payload's `target`, and a target was `<instrument>/slam5/seed<n>` — the mesh
   was nowhere in it. A Delaunay run would therefore have landed in the base run's directory
   *and* its parity group, and been rendered as a seventh column of a table whose whole
   premise is that the columns differ only in backend. So the results path, the PyAutoFit
   `path_prefix` and the target id all gained a `<variant>` segment between the instrument
   and the config name: `slam_base` for the workspace-default 28x28 rectangular mesh (the
   name the rows above were written under), `delaunay_1250` for the mesh under test. Result
   schema v2 carries `variant`, `mesh`, `mesh_pixels`, `mesh_areas_factor` and
   `regularization`; v1 rows still read, and an absent `variant` means `slam_base`.
5. **The Delaunay variant's model is constrained at both ends, and neither constraint is a
   preference.** `al.reg.Adapt` — what the rectangular legs run under — takes its pixel
   neighbours from a `scipy.spatial.Delaunay` call on the *traced* source grid and raises
   `TracerArrayConversionError` under jit on this mesh family, so the pairing is
   `al.reg.AdaptSplit`; and it is the **class**, not
   `_inference_cli.delaunay_regularization()`'s fixed-coefficient instance, so the
   coefficients stay free exactly as `Adapt`'s are on the rectangular legs. Comparability is
   the point: a variant that quietly drops free parameters is not the same experiment run
   differently.
6. **The mesh itself has to be an instance, and the smoke is what found it.**
   `af.Model(al.mesh.Delaunay, pixels=1250, zeroed_pixels=0)` asks PyAutoFit to give every
   unpinned constructor argument a prior, and `areas_factor` has no entry in any
   `config/priors` tree in this stack: both test-mode legs died at the first `search.fit` of
   `source_pix[1]` with `ConfigException: No prior config found for class: Delaunay …
   areas_factor`. The fix is what production does — pass the instance
   (`al.mesh.Delaunay(pixels=1250, zeroed_pixels=0, areas_factor=0.5)`), with `areas_factor`
   stated explicitly at the library default and recorded in the row rather than inherited
   silently. **Do not re-introduce `af.Model` here**: the mesh has no free parameters, and
   wrapping it only invites PyAutoFit to invent some. The regularization stays a class; the
   mesh is an instance; the asymmetry is real and this is why.
7. **A different mesh is a different cell, not a flag.** `scripts/imaging/slam/hst_delaunay.py`
   exists because the wall gate reads a submit's cell from the script path it invokes and
   `wall/rates.py` keys step rates by that cell. A 1250-vertex Delaunay chain is a different
   cost profile from a 784-cell rectangular one, so sharing `imaging/slam/hst` would let a
   `--time` be justified by a rate measured on another model — the carry that cost 35 of 39
   arms of an overnight A100 block. (The gate's instrument check learned that a leaf may
   carry a variant suffix: `imaging/slam/hst_delaunay` runs `--instrument hst`. Only that
   check was relaxed; the rate key keeps the whole leaf.)

**Both Delaunay routes were run end to end before either submit was cleared, and it took
four attempts to get there.** The legs ran on the euclid cell under `PYAUTO_TEST_MODE` —
the HST cell does not fit on this laptop under JAX, and the euclid cell is the same code
path on a smaller grid, which is why CI uses it. Every failure is worth keeping, because
three of the four are properties of the mesh family rather than accidents:

1. `ConfigException: No prior config found for class: Delaunay … areas_factor` at the first
   `search.fit` of `source_pix[1]` — item 6 above; the mesh had been wrapped in `af.Model`.
2. With the mesh an instance: `MeshException: The mesh Delaunay was not given an image-plane
   mesh grid`. **`al.mesh.Delaunay` does not place its own vertices**; they are drawn by an
   *image mesh* from the S/N-capped source adapt image and reach the fit only through
   `adapt_images`. `Delaunay.pixels` is documented as *a description of that grid rather
   than a control over it*. The answer is the production group-SLaM recipe verbatim
   (`autolens_workspace/scripts/group/slam.py`) at 1250 interior vertices:
   `al.image_mesh.Hilbert(pixels=1250, weight_power=3.5, weight_floor=0.01)` on that stage's
   own capped adapt image, a 30-point circle ring appended at the mask radius, handed over
   as `AdaptImages(galaxy_name_image_plane_mesh_grid_dict={source: grid})`, and
   `al.mesh.Delaunay(pixels=1250, zeroed_pixels=30, areas_factor=0.5)`. The Hilbert weights
   are production's, **not** the library defaults (0.0 / 0.0), which would draw a mesh that
   does not adapt to the source at all. **The edge ring and `zeroed_pixels` are one number
   written twice**: `total_pixels` is `pixels + zeroed_pixels`, the grid is 1250 + 30 = 1280,
   and changing one without the other breaks the accounting silently. The grid is rebuilt at
   every pixelized stage from that stage's own adapt image, as production does — `light[1]`
   and `mass_total[1]` inherit the source pixelization and need it too.
3. **A Delaunay leg needs a wider CPU thread pool than a rectangular one of the same size,
   and that is a property of the mesh family.** The Delaunay interpolator reaches qhull
   through a `jax.pure_callback`, so the fit calls back into the host *while* an XLA
   computation is in flight; a thread pool narrower than that callback depth deadlocks — the
   pool waits on the callback the pool has to run. At `--cores 2` the sparse leg sat at
   `source_pix[1]` with all 32 threads in `futex_wait` and **zero CPU for 22 minutes**, while
   autofit's `jax_compile` heartbeat kept printing "still compiling, Ns elapsed" — that line
   is a timer thread and proves only that a timer is alive, never that work is happening.
   Judge a suspected hang on CPU time, not on the heartbeat. At `--cores 8` both legs ran.
   The rectangular legs have no callback at all, which is why nothing like this was ever seen
   on the base run — and why the two Delaunay submits ask for `--cpus-per-task=8` where the
   base legs ask for 4.
4. **Two legs do not fit in 15 GB at once.** Run concurrently, the dense leg was OOM-killed
   by the kernel at `source_pix[1]` (`anon-rss` 6.7 GB) with no traceback, its log simply
   stopping. Run one at a time, both complete. The euclid smoke is a one-leg-at-a-time job on
   this laptop.

**The jit compile is a line item, and one stage dominates it.** In `PYAUTO_TEST_MODE` every
stage makes a *single* likelihood call, so a stage's wall clock essentially *is* its compile.
On the small euclid cell `source_pix[1]` cost **1,471.7 s dense and 2,058.7 s sparse** — 25
and 34 minutes to compile one stage, against seconds on the rectangular legs, which have no
qhull callback to trace around. The other three pixelized stages compile in 70-170 s
(per-stage: 49.6 / 22.2 / 30.9 / 43.1 s dense, 85.8 / 31.3 / 54.1 / 85.7 s sparse), so the
cost is concentrated in one stage and is paid once per stage rather than per evaluation — but
the A100 legs pay it on a cell of 15,361 masked pixels with a graph never compiled on that
backend, which is why their `--time` is a 24 h containment rather than the base legs' 12 h.

Run alone at `--cores 8`, each leg ran all five stages to `status: complete` and wrote a
schema-v2 row at the variant path carrying the whole recipe (`mesh_pixels`,
`mesh_zeroed_pixels`, `mesh_areas_factor`, `image_mesh`, its weight power and floor, the ring
size, `mesh_shape: null`): a row saying only "delaunay, 1250" could not be reproduced,
because what places those vertices is the image mesh and its weights. Measured there, and
worth carrying to the A100 because **the compile is a real line item and every pixelized
stage pays it**: per-stage jit compile 49.6 / 22.2 / 30.9 / 43.1 s dense and 85.8 / 31.3 /
54.1 / 85.7 s sparse, against seconds for the rectangular legs; peak RSS 9.3 GB dense and
6.5 GB sparse on the *euclid* cell. `source_pix[1]` has **10 free parameters** on both, the
same as the rectangular branch (mass 5 + shear 2 + 3 regularization coefficients) — the mesh
contributes none on either side, which is what keeps the two runs comparable. `source_pix[2]`
does differ, 3 against the rectangular branch's 5: the rectangular *image* mesh has free
`weight_power` and `weight_floor`, where the Hilbert weights here are fixed at 3.5 / 0.01.

**Next.** Submit the two A100 Delaunay legs — `submit_slam_delaunay1250_hst_jax_gpu_{dense,
sparse}`, seeds 0–1, `--mem` 64gb/96gb, `--cpus-per-task=8`, 24 h containment, both declaring
`source: unmeasured  probe-first: yes` because nothing has been measured on this cell and
the base run's walls belong to a different one. Then the four CPU legs, which are what turn
four completed runs into a parity row.

### 2026-09-24 — pre-speedup rows archived, legs re-submitted on the new mains

**What ran.** Nothing new measured yet. Every SLaM HST row this repo held was archived, and
the runs re-submitted on RAL against today's library mains (PyAutoArray `3de624b5`,
PyAutoLens `86054bbc1`, PyAutoFit `dd9fbe0aa`, PyAutoGalaxy `70a61e26`, PyAutoNerves
`1fa613a`, identical on the laptop and the RAL mirror), which carry the likelihood speedups
landed through the autolens_profiling campaigns.

Archived to `results/archive/2026-09-24_pre_likelihood_speedup/slam/imaging/hst/` (PyAutoFit
trees to the same path under `output/archive/`, locally and on RAL — moved, never deleted):

| variant / config | seeds | status | what it was |
|---|---|---|---|
| `slam_base/hpc_a100_jax_gpu_dense_fp64` | 0, 1 | complete | 2026-09-14 entry above |
| `slam_base/hpc_a100_jax_gpu_sparse_fp64` | 0, 1 | complete | 2026-09-14 entry above |
| `slam_base/hpc_a100_jax_gpu_dense_fp64__rate_probe_342695` | 0 | stopped_early | the A100 `source_lp[1]` rate probe, job 342695 |
| `delaunay_1250/hpc_a100_jax_gpu_dense_fp64` | 0, 1 | complete | RAL job 343143; never committed live |
| `delaunay_1250/hpc_a100_jax_gpu_sparse_fp64` | 0, 1 | complete | RAL job 343145; never committed live |

For the record, since the Delaunay rows were never journalled: search wall over the chain
4,770 s / 5,095 s dense and 9,112 s / 8,486 s sparse (seed 0 / 1), no stage resumed,
`mass_total[1]` log Z 31,512.9 / 31,503.0 dense and 31,515.5 / 31,534.7 sparse — and seed 0
on both routes has a `mass_total[1]` `truth_delta_sigma` of 5.4σ (dense) / 4.7σ (sparse),
which the re-runs should confirm or retire before anyone reads a mesh comparison off them.

**Why move rather than overwrite.** `seed` is a PyAutoFit identifier field and the config
name sits in the `path_prefix`, so a re-submit at the same config name and seed hashes to the
**same output directory** as the old fit — and PyAutoFit would resume it, report every stage
complete, and re-run nothing. Moving the old `output/` trees aside is what makes the new
jobs fit from scratch. The new runs' trees stay at the normal paths on purpose: a later
variant (lens light fixed to ordinary light profiles from `mass[1]` onwards) resumes from
them.

**Submitted** (RAL, from `hpc/batch_{gpu,cpu}/`, seeds 0–1 each):

_(job ids follow once the submits are accepted)_

**Next.** Pull the rows when they land, commit them to the live tree, and compare the new
per-stage walls against the archived ones — that ratio is the speedup, measured on the
inference rather than on one likelihood call.

