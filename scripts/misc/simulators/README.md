# simulators

The dataset simulators — the scripts that produce the mock lensed datasets every run in
this repo fits. Seven dataset classes are covered, one script per dataset type (the script
name is the `dataset_type` that `_inference_cli.auto_simulate_if_missing` invokes):

| Script | What it simulates |
|--------|-------------------|
| [`imaging.py`](./imaging.py) | Single-band imaging (HST / Euclid / JWST / AO presets). Grid + over-sampling setup, tracer + galaxies, `tracer.image_2d_from` (eager + JIT), `simulator.via_tracer_from`, FITS output. |
| [`interferometer.py`](./interferometer.py) | Visibility-space datasets with synthetic uv-wavelengths. Transformer setup, then `tracer.image_2d_from` eager + JIT. |
| [`point_source.py`](./point_source.py) | Point-source datasets: lens + source tracer, eager and JIT `solver.solve`, `tracer.time_delays_from`. |
| [`group.py`](./group.py) | Group-scale imaging at `euclid` (250×250 @ 0.1″/px): one main lens + two dPIE extra galaxies + SersicCore source, over-sampled on every galaxy centre. Writes `extra_galaxies_centres.json` and `simulation.json` beside the FITS. |
| [`multi_galaxy.py`](./multi_galaxy.py) | Two co-dominant lenses + external shear + a light-only extra galaxy at `hst` (200×200 @ 0.05″/px). |
| [`cluster.py`](./cluster.py) | Cluster-scale `simple`: 2 main dPIE lenses + 10 scaling dPIE members + NFW host halo, two sources (z = 1, 2); per-source JIT `solver.solve`, 1000×1000 `via_tracer_from`. Writes FITS, per-source point datasets and `scaling_galaxies.csv`. The heaviest (minutes). |
| [`weak.py`](./weak.py) | Weak-lensing shear catalogue `simple`: 1500 sources around an elliptical isothermal cluster lens. Also holds the weak model (`standard_model`, `truth_vector`). Writes no timing summary (milliseconds). |

Each is dual-purpose: it **writes the dataset** under `dataset/<class>/<instrument>/`, and
**times each phase** of doing so into `results/misc/simulators/<script>_<instrument>_summary_v<version>.{json,png}`.
The timings are a by-product — this repo's subject is inference, not simulation — but they
are cheap and they catch the case where a dataset that used to take seconds starts taking
minutes.

`INSTRUMENTS` lives in [`instruments/`](../../../instruments/README.md), not here; these
modules re-export it so `from simulators.imaging import INSTRUMENTS` keeps working.

The four ported from `autolens_profiling` (`group`, `multi_galaxy`, `cluster`, `weak`) write
the same datasets the profiling likelihood cells fit — pixel-identical FITS and JSON identical
once parsed. `group.py` and `multi_galaxy.py` are thin CLIs over
[`_galaxy_group.py`](./_galaxy_group.py), the vendored dataset half of profiling's
`cell_builders/galaxy_group.py`; its `SPECS` carries what a runner needs to mask the data as
those cells did (mask radius, the multi_galaxy noise-scaled region, over-sampling bins, galaxy
centres). `weak.py` keeps its presets in a module-level `INSTRUMENTS`.

## Running

From the repo root:

```bash
python scripts/misc/simulators/imaging.py --instrument hst
python scripts/misc/simulators/interferometer.py --instrument sma
python scripts/misc/simulators/point_source.py --instrument simple
python scripts/misc/simulators/group.py --instrument euclid
python scripts/misc/simulators/multi_galaxy.py --instrument hst
python scripts/misc/simulators/cluster.py --instrument simple
python scripts/misc/simulators/weak.py --instrument simple
```

Every script takes `--output-root <dir>` to write `dataset/` and `results/` under another root
(the auto-simulate call passes the repo root).

Sandboxed / restricted environments:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/misc/simulators/imaging.py
```

## Datasets are inputs, not results

`dataset/` is gitignored. `_inference_cli.auto_simulate_if_missing` re-runs the matching
simulator when a run finds its dataset absent, so a missing dataset costs a few minutes of
CPU rather than a lost run. If you add an instrument whose dataset does **not** regenerate
byte-identically, track it instead and say in `.gitignore` why.

## Smoke

Every script reads `AUTOLENS_INFERENCE_SMOKE` at module top and exits 0 immediately after
the import + setup section, without simulating anything. The `lint` workflow runs the
simulators that way on every PR, which catches import-graph breakage (a broken `sys.path`
injection, a missing dependency, a renamed module) in seconds.

```bash
AUTOLENS_INFERENCE_SMOKE=1 python scripts/misc/simulators/imaging.py
```
