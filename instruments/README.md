# `instruments` — per-instrument dataset presets

This subpackage owns the per-instrument configuration dicts that drive
both the simulators and the inference leaves. Two modules:

- `instruments.imaging` — `INSTRUMENTS` for imaging (hst, jwst, ao, euclid).
- `instruments.interferometer` — `INSTRUMENTS` for interferometer
  (sma, alma, alma_high, jvla).

## Why a separate package?

The INSTRUMENTS dicts used to live inside `simulators/imaging.py` and
`simulators/interferometer.py`. As the repo grew, multiple consumers ended
up reading them:

- `scripts/misc/simulators/*.py` — drives dataset simulation.
- `scripts/<dataset_class>/slam/*.py` — reads `pixel_scale`, `mask_radius`,
  `real_space_shape`, `transformer_chunk_size` to set up the dataset the
  pipeline fits.
- `scripts/<dataset_class>/searches/<sampler>/*.py` — same.

Splitting the dicts into a dedicated home means:

- Each consumer imports from one canonical location.
- Adding a new instrument is one row in one file.
- Helpers like `mask_radius_pixels(instrument)` can centralise math that
  was previously inlined across multiple files.

## Schema

### Imaging fields

| Field | Type | Meaning |
|-------|------|---------|
| `pixel_scale` | float | arcsec / pixel |
| `mask_radius` | float | arcsec (circular mask) |
| `psf_shape` | tuple[int, int] | PSF kernel shape (n_y, n_x) |
| `psf_sigma` | float | Gaussian PSF width (arcsec) |
| `seed` | int | RNG seed for noise generation |

### Interferometer fields

| Field | Type | Meaning |
|-------|------|---------|
| `pixel_scale` | float | arcsec / pixel |
| `real_space_shape` | tuple[int, int] | (n_y, n_x) real-space image grid |
| `mask_radius` | float | arcsec (circular mask) |
| `n_visibilities` | int | number of (u, v) baselines |
| `uv_scale` | float | RNG sampling scale for (u, v) |
| `noise_sigma` | float | noise per visibility |
| `seed` | int | RNG seed |
| `transformer` | "dft" or "nufft" | transformer class |
| `transformer_chunk_size` | int or None | NUFFT gather-buffer cap |

## Helpers

- `imaging.mask_radius_pixels(instrument) -> int` — mask radius / pixel_scale, rounded.
- `imaging.shape_native(instrument) -> tuple[int, int]` — data grid shape derived from mask.
- `interferometer.mask_radius_pixels(instrument) -> int` — same math, on interferometer.
- `interferometer.transformer_chunk_size_for(instrument) -> int | None` — convenience accessor.

## Backward compatibility

The legacy import paths still work:

```python
from simulators.imaging import INSTRUMENTS  # still valid
from simulators.interferometer import INSTRUMENTS  # still valid
```

These re-export from `instruments.{imaging,interferometer}` so existing
consumers don't have to migrate. New code should prefer
`from instruments.imaging import INSTRUMENTS`.

## Adding a new instrument

1. Add a row to the appropriate `INSTRUMENTS` dict.
2. Simulate the dataset by running
   `python scripts/misc/simulators/<imaging|interferometer>.py --instrument <name>`.
3. Run one short arm on the new instrument before sizing any SLURM `--time`
   against it — an unmeasured cell has no wall-clock basis
   (see [`../scripts/misc/wall/README.md`](../scripts/misc/wall/README.md)).
