# simulators

The dataset simulators — the scripts that produce the mock lensed datasets every run in
this repo fits. Three dataset classes are covered, matching `scripts/`:

| Script | What it simulates |
|--------|-------------------|
| [`imaging.py`](./imaging.py) | Single-band imaging (HST / Euclid / JWST / AO presets). Grid + over-sampling setup, tracer + galaxies, `tracer.image_2d_from` (eager + JIT), `simulator.via_tracer_from`, FITS output. |
| [`interferometer.py`](./interferometer.py) | Visibility-space datasets with synthetic uv-wavelengths. Transformer setup, then `tracer.image_2d_from` eager + JIT. |
| [`point_source.py`](./point_source.py) | Point-source datasets: lens + source tracer, eager and JIT `solver.solve`, `tracer.time_delays_from`. |

Each is dual-purpose: it **writes the dataset** under `dataset/<class>/<instrument>/`, and
**times each phase** of doing so into `results/simulators/<script>_<instrument>_summary_v<version>.{json,png}`.
The timings are a by-product — this repo's subject is inference, not simulation — but they
are cheap and they catch the case where a dataset that used to take seconds starts taking
minutes.

`INSTRUMENTS` lives in [`instruments/`](../../../instruments/README.md), not here; these
modules re-export it so `from simulators.imaging import INSTRUMENTS` keeps working.

## Running

From the repo root:

```bash
python scripts/misc/simulators/imaging.py --instrument hst
python scripts/misc/simulators/interferometer.py --instrument sma
python scripts/misc/simulators/point_source.py --instrument simple
```

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
the import + setup section, without simulating anything. The `lint` workflow runs all
three that way on every PR, which catches import-graph breakage (a broken `sys.path`
injection, a missing dependency, a renamed module) in seconds.

```bash
AUTOLENS_INFERENCE_SMOKE=1 python scripts/misc/simulators/imaging.py
```
