# `scripts/imaging/slam`

Full **SLaM pipeline** runs on imaging data — the science chains as they are actually run, end
to end, stage by stage.

The first leaf landed in phase 3: [`hst.py`](hst.py), the backend-parameterised base-run
driver, exercised by the phase-4 Cortex task `slam_hst_base`. [`hst_delaunay.py`](hst_delaunay.py)
joined it: the same chain and the same runner, with the two pixelized source stages on a
1250-vertex Delaunay mesh — the `delaunay_1250` **run variant**.

## Running a leg

```bash
python3 scripts/imaging/slam/hst.py \
    --backend {jax_cpu,numba_cpu,jax_gpu} --inversion {dense,sparse} \
    --config-name <config> [--seed 0] [--cores N] [--stages source_lp] [--output-dir DIR] \
    [--mesh {rect,delaunay}] [--mesh-pixels N]
```

- The **instrument is a flag** (`--instrument hst`, the default), never a directory. The HST
  cell is the simulated dataset from `scripts/misc/simulators/imaging.py`, auto-simulated on
  first use so RAL needs no data transfer.
- The **backend is a column**: the same leaf runs every leg of a parity row, and the legs
  differ only in the flags above and the resulting `--config-name`. The driver refuses a
  `--config-name` whose backend / inversion / precision fields disagree with the flags — a
  config name that lies about its run makes the whole parity row a lie.
- `--cores` defaults to `SLURM_CPUS_PER_TASK`, else `os.cpu_count()`. `--stages <name>` stops
  the chain after that stage (used to measure a per-backend rate from a `source_lp` leg).
- The **mesh is a run variant**, not a config (see below): `--mesh` / `--mesh-pixels` choose
  the source pixelization, and the variant they resolve to is a directory level and a field
  of the target id. Each leaf carries its own default — `hst.py` is `rect` at 28, the
  workspace default; `hst_delaunay.py` is `delaunay` at 1250 — so neither flag is normally
  typed.
- `--output-dir` overrides the **PyAutoFit run-output root** for this leaf (default
  `output/slam/imaging/hst/<variant>/<config_name>/seed_<n>/`); the result row always lands
  under `results/`.
- `--backend jax_gpu` asserts `jax.default_backend() == "gpu"` and exits 2 otherwise. There is
  no silent CPU fallback: a "GPU" leg that ran on CPU reports a wall clock that is nothing of
  the kind.

## What a leg writes

- One result row per `(variant, config_name, seed)`:
  `results/slam/imaging/hst/<variant>/<config_name>/stages_seed<n>.json`, schema version 2,
  with a `stages` list of five rows (`source_lp[1]`, `source_pix[1]`, `source_pix[2]`, `light[1]`,
  `mass_total[1]`) carrying wall clock, reject-inclusive likelihood evals, log evidence,
  posterior (median + 1σ), `truth_delta_sigma` against `dataset/imaging/hst/tracer.json`,
  `positions_info_present`, `completed` and `resumed`. A wall-per-stage PNG sits beside it.
  Schema v2 added the variant fields — `variant`, `mesh`, `mesh_pixels`, `mesh_areas_factor`,
  `mesh_zeroed_pixels`, `image_mesh`, `image_mesh_weight_power`, `image_mesh_weight_floor`,
  `image_mesh_edge_points` and `regularization` (the resolved scheme name), with `mesh_shape`
  `null` on a family that has no shape. The image-mesh fields are the Delaunay recipe and are
  `null` on the rectangular family: a row saying only "delaunay, 1250" could not be
  reproduced, because what places those 1250 vertices is the image mesh and its weights. v1 rows still read: every reader treats an absent `variant` as `slam_base`,
  which is what those rows are.
- The full PyAutoFit output tree under `output/` (gitignored — see `AGENTS.md`, "Outputs are
  KEPT"). Budget ~150 MB per 5-stage lens per leg.

`build_readme.py` flattens those stage lists into the root README's `slam` table and renders a
**parity view** per (target, seed): the `mass_total[1]` posterior of every config side by side,
each cell the difference from the alphabetically-first config in units of that reference leg's
1σ.

## Run variants: the mesh is not a config

```
results/slam/<dataset_class>/<instrument>/<variant>/<config_name>/stages_seed<n>.json
output/slam/<dataset_class>/<instrument>/<variant>/<config_name>/seed_<n>/
```

The `<variant>` level between the instrument and the config name is **what was fitted**; the
config name is **what ran it**. It has to exist because rows group into a parity row by the
payload's `target`, so without it a second experiment on this cell would share both the
directory and the parity group with the base run and read as another backend of the same
fit. Only the backend may be a column of one table.

| variant | mesh | source stages | regularization | leaf |
|---|---|---|---|---|
| `slam_base` | `rect`, 28x28 = 784 cells | `RectangularBilinearAdaptDensity` then `…AdaptImage` | `al.reg.Adapt` (class) | [`hst.py`](hst.py) |
| `delaunay_1250` | `delaunay`, 1250 interior vertices (+30 zeroed edge) drawn by `al.image_mesh.Hilbert` | `al.mesh.Delaunay` on both | `al.reg.AdaptSplit` (class) | [`hst_delaunay.py`](hst_delaunay.py) |

`--mesh-pixels` means a different thing per family and the two numbers are not comparable:
the side of the square mesh under `rect`, the vertex count outright under `delaunay`. The
workspace default (`rect` at 28) is the one variant not named after its mesh — it is
`slam_base`, the baseline, and the name the rows already on disk were written under. Every
other combination names itself: `delaunay_1250`, `rect_40`. Its target id says so too —
`hst/slam5/seed0` for the base run, `hst/slam5_delaunay_1250/seed0` for the variant.

### Why `hst_delaunay.py` is a separate leaf

Not because the code differs — it is one line, `run_slam(..., default_mesh="delaunay")` —
but because **the wall gate derives a submit's cell from the script path it invokes**, and
`wall/rates.py` keys measured step rates by that cell. Cell id = cost profile = rate key. A
1250-vertex Delaunay chain costs a different amount per likelihood evaluation from a
784-cell rectangular one, so a `--time` justified from `imaging/slam/hst`'s rate would be a
number measured on a different model — the carry that killed 35 of 39 arms of an overnight
A100 block. Its own leaf gives it its own cell, `imaging/slam/hst_delaunay`, and the gate
then demands its own measurement.

### The Delaunay recipe, and why none of it is a free choice

The production group-SLaM recipe, verbatim
(`autolens_workspace/scripts/group/slam.py`), at 1250 interior vertices:

```python
image_mesh = al.image_mesh.Hilbert(pixels=1250, weight_power=3.5, weight_floor=0.01)
grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=source_adapt_image)
grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=grid, centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0, n_points=30,
)
adapt_images = al.AdaptImages(
    galaxy_name_image_dict=…,
    galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": grid},
)
mesh = al.mesh.Delaunay(pixels=1250, zeroed_pixels=30, areas_factor=0.5)
```

rebuilt at **every** pixelized stage from that stage's own S/N-capped source adapt image, as
production does — `light[1]` and `mass_total[1]` inherit the source pixelization and need the
grid too. Four things about it are constraints rather than preferences:

- **Not `al.reg.Adapt`**, which is what the rectangular legs run under. That scheme takes
  its pixel neighbours from a `scipy.spatial.Delaunay` call on the *traced* source grid,
  which raises `TracerArrayConversionError` under `jax.jit`. The mesh family, not the
  regularization alone, decides what can be traced; the Split schemes are the traceable
  pairing for this family.
- **The regularization is the class, not a fixed instance.** The stage helpers build
  `af.Model(al.Pixelization, mesh=..., regularization=...)`, so a class leaves the
  coefficients free exactly as `Adapt` does on the rectangular legs — which is what makes
  the two runs comparable. `_inference_cli.delaunay_regularization()`'s fixed-coefficient
  `AdaptSplit` (built for the profiling-style cells) would silently drop free parameters.
- **The mesh does not place its own vertices.** `al.mesh.Delaunay` alone raises
  `MeshException: the mesh Delaunay was not given an image-plane mesh grid`. The points are
  drawn by an *image mesh* from the capped adapt image and reach the fit only through
  `adapt_images`; `Delaunay.pixels` is documented as a *description* of that grid rather than
  a control over it. The Hilbert weights are production's (3.5 / 0.01), **not** the library
  defaults (0.0 / 0.0), which would draw a mesh that does not adapt to the source at all.
- **The edge ring and `zeroed_pixels` are one number written twice.**
  `Delaunay.total_pixels` is `pixels + zeroed_pixels`, and the grid the mapper receives is
  the 1250 interior vertices plus the appended 30-point circle = 1280. So the ring size is
  not a free knob: change `MESH_EDGE_POINTS` and `zeroed_pixels` follows it, or the
  accounting breaks silently. Those last 30 vertices are fixed to zero, which keeps poorly
  constrained boundary pixels from absorbing flux.
- **The mesh itself is an instance, not an `af.Model`.** It has no free parameters,
  and `af.Model` asks PyAutoFit to prior every constructor argument the caller did not pin —
  including `areas_factor`, which no `config/priors` tree in this stack defines, so
  `source_pix[1]` dies at its first `search.fit` with `ConfigException: No prior config
  found for class: Delaunay … areas_factor`. Production passes the instance
  (`autolens_workspace/scripts/group/slam.py`, the `autolens_profiling` Delaunay cells), and
  so does this. `areas_factor` is passed explicitly at the library default 0.5 and recorded
  in the row as `mesh_areas_factor`, rather than inherited silently.

## The six legs of the parity row

One cell — `imaging/slam/hst` — on the `slam_base` variant, run six ways. The backend is a
column, never a reason to split the table.

| leg | `--backend` | `--inversion` | `--config-name` | where it runs |
|---|---|---|---|---|
| 1 | `jax_gpu` | `dense` | `hpc_a100_jax_gpu_dense_fp64` | RAL `gpu` (A100) |
| 2 | `jax_gpu` | `sparse` | `hpc_a100_jax_gpu_sparse_fp64` | RAL `gpu` (A100) |
| 3 | `jax_cpu` | `dense` | `hpc_a100_jax_cpu_dense_fp64` | RAL `ral` (8 cores) |
| 4 | `jax_cpu` | `sparse` | `hpc_a100_jax_cpu_sparse_fp64` | RAL `ral` (8 cores) |
| 5 | `numba_cpu` | `dense` | `hpc_a100_numba_cpu_dense_fp64` | RAL `ral` (8 cores) |
| 6 | `numba_cpu` | `sparse` | `hpc_a100_numba_cpu_sparse_fp64` | RAL `ral` (8 cores) |

**`hpc_a100_` on a CPU leg is not a typo.** The config-name grammar
(`_inference_cli.CONFIG_NAME_RE`) offers exactly two "where" tokens, `local` and `hpc_a100`,
and the second means *RAL*, not *the A100 partition*. The device is already carried by the
backend token, so legs 3–6 are honestly named; the alternative would be a third token that
says the same thing twice.

Each leg is one submit under `hpc/batch_{gpu,cpu}/submit_slam_hst_<backend>_<inversion>`, each
an `--array=0-1` (one seed per array task), each sized by its own `# WALL-BASIS:` block.
`hpc/sync submit --gpu|--cpu <submit filename>` sends one; `hpc/sync jobs` / `tail` / `pull`
bring it back. The submit names, the seed convention and the memory request are tabulated
in [`../../../hpc/README.md`](../../../hpc/README.md).

**All four HST CPU legs and both GPU legs are RAL jobs, not laptop jobs, and that is a
measurement rather than a preference.** See the memory section below.

## Memory: the HST cell does not fit on a laptop under JAX

The HST cell is **15,361 masked pixels**, and the cost is in the Nautilus batch rather than
the dataset. Measured on 2026-09-11, on the `source_pix[1]` stage with `n_batch=20`:

| leg | RSS |
|---|---|
| `jax_cpu` × dense | **~13 GB** at `source_pix[1]` |
| `jax_cpu` × sparse | **~21.6 GB** at `source_pix[1]` |
| `numba_cpu` × either | not the constraint — there is no vmap at all: with `use_jax=False` Nautilus takes the `number_of_cores` multiprocessing path. The production `source_lp[1]` probe held ~6.3 GB across its nine processes, and the `PYAUTO_TEST_MODE` chain reached `mass_total[1]` on the same 15 GB machine. Its `source_pix[1]` figure at production settings is **not** measured. |

Both JAX-CPU figures OOM a 15 GB laptop, which has three consequences worth stating plainly:

1. The two `jax_cpu` HST legs of the parity row exist **only** as RAL jobs.
2. The **local rate measurements are `--stages source_lp` only** — the parametric opening
   stage builds no inversion and holds no mesh, so it fits in RAM and still measures the
   backend's seconds per likelihood evaluation.
3. The CI witness (`.github/workflows/profile.yml`) runs `--instrument euclid`, not `hst`:
   a GitHub-hosted runner has 7–16 GB and the euclid cell exercises the identical code path
   on a smaller grid. The local `PYAUTO_TEST_MODE` witness for JAX on HST is expected to
   OOM and is not evidence of a bug.

Every submit asks for `--mem=64gb` — the template's figure, kept because it clears the
worst measured leg by 3x and a leg killed by the OOM killer at stage two costs a night.

## Rates, and what they are allowed to justify

Measured 2026-09-11, and narrower than they look — each is `source_lp[1]` **only**, at
production settings, on the HST cell (15,361 masked pixels, 17 free parameters, nautilus
`n_live=200` / `n_batch=50`, seed 0, dense, fp64):

| device key | s / likelihood eval | how it was measured |
|---|---|---|
| `laptop_numba_cpu` | **0.06831** | partial run, 17,950 evals over 1,226 s, 8 cores, `DESKTOP-H143S82` |
| `laptop_jax_cpu` | **0.04925** | partial run, 25,500 evals over 1,256 s, 8 cores, same host |
| `a100` | *not measured* | RAL job **342695** (`submit_slam_hst_rate_jax_gpu`) was still `PENDING(Priority)` when phase 3 shipped |

Both local probes were stopped before `source_lp[1]` finished — nautilus was still in its
exploration phase — so **`results/` carries no production row yet**; these are rates, not
results. The environment is part of the measurement and is recorded in `PROVENANCE`: the
session's ambient `XLA_FLAGS=--xla_disable_hlo_passes=constant_folding` was removed with
`env -u XLA_FLAGS`, and `OMP_NUM_THREADS=1` was kept (correct for the numba leg, which takes
nautilus's `number_of_cores` multiprocessing path — kernel threading under multiprocessing
costs rather than pays; the JAX legs are sized by `NPROC` instead).

The device key names the **host** as well as the backend on purpose. The submits declare
`ral_numba_cpu`, `ral_jax_cpu` and `a100`, so a laptop rate cannot be cited by a RAL submit
even by accident: the gate simply does not match the key.

`wall/rates.py` holds these rows and `wall/check_submits.py` gates every submit's `--time`
against them. Read [`../../misc/wall/README.md`](../../misc/wall/README.md) before citing
one: the rule is that **a `--time` justification never crosses cells**, and the corollary
here is that a `source_lp` rate is the *cheapest* stage of this cell. Four of the five
stages are pixelized and per-eval-inversion bound at many times that cost, so the six
production submits all declare `source: unmeasured  probe-first: yes` — their first run is
itself the probe that will replace the declaration with `measured-wall`. The measured rows
are cited by one submit only: `submit_slam_hst_rate_jax_gpu`, the `--stages source_lp` probe
they were measured on.

## The chain, and where it departs from the workspace

The chain mirrors `autolens_workspace/scripts/guides/modeling/slam_start_here.py`. Four
deliberate decisions, all recorded in the driver's module docstring — the first two are
departures from the workspace script, the last two are choices this repo had to make
because the workspace script never faced them:

1. A **positions likelihood is attached to all four pixelized stages**, not just
   `source_pix[1]` and `mass_total[1]`. A mesh stage with no `positions.info` beside it is not
   citable here.
2. **`log_evidence_err` is always `null`** — nautilus 1.0.5 exposes no `log_z_err`, and a
   derived one would be an invention. The row carries a sibling
   `log_evidence_err_note` saying so, so a reader never has to guess whether the field is
   missing or zero.
3. **The over-sample map before `source_pix[2]` is kept** (it is the workspace default, and
   dropping it would make this chain a different pipeline from the one science runs), and
   the **sparse operator is re-applied immediately after it**. `Imaging.apply_over_sampling`
   rebuilds the `Imaging` object from scratch and silently drops the sparse operator, so a
   sparse leg that did not re-apply would run its last three stages dense while still
   calling itself sparse — a parity row that lies.
4. **`--output-dir` moves the PyAutoFit run-output root, not the results directory.** A
   leg's bulk state is the run tree (~150 MB per 5-stage lens), and that is what a RAL job
   or a scratch disk needs to redirect. The small result row always lands under
   `results/slam/<dataset_class>/<instrument>/<variant>/<config_name>/`, so a pulled run and
   a local one are read from the same place.

Everything else — flags, the config-name grammar and the backend facts that make a leg honest —
is in [`../../../_inference_cli.py`](../../../_inference_cli.py),
[`../../misc/slam/_runner.py`](../../misc/slam/_runner.py) and
[`../../../AGENTS.md`](../../../AGENTS.md).
