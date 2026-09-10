# `scripts/point_source/searches`

Single-**search** runs on point source data: one sampler, one model, one seed. Where `slam/` asks
"what does the production chain cost and does it agree across backends", this asks "does
*this* search find the right answer, how often, and how fast".

Empty in phase 1. Leaves are added one sampler at a time, each in its own directory:

```
scripts/point_source/searches/<sampler>/<target>.py
```

Nautilus first, because it is what production runs; the gradient-based optimisers and
samplers follow once the base run exists to compare them against.

## Rules that apply to every leaf here

- **The instrument is a flag** (`--instrument`), never a directory.
- **A reliability claim needs seeds.** One run tells you a search *can* find the answer;
  it takes a spread of `--seed` values to say how often it does.
- **Result rows go to `results/searches/`**, one per run, carrying `target`,
  `sampler`, `config_name`, `instrument`, `wall_s` and the evidence — the columns the
  README auto-table renders.
- **No submit without a wall-clock basis.** See
  [`../../misc/wall/README.md`](../../misc/wall/README.md): an unmeasured cell declares
  itself as one and probes first.
