# `scripts/point_source/searches`

Single-**search** runs on point source data: one sampler, one model, one seed. Where `slam/` asks
"what does the production chain cost and does it agree across backends", this asks "does
*this* search find the right answer, how often, and how fast".

The first leaf is `nautilus/simple_source_plane.py`: a five-parameter Isothermal lens with
a solved source centre, fitted with `FitPositionsSourceSolved`. Five RAL CPU fp64 seeds
recover truth within 0.74σ and take 50–59 seconds. The warmed batch timing implies
0.0405–0.0447% of search wall in steady likelihood evaluation; see the
[admission-bar record](../../../wiki/project/state.md#2026-10-02--point-source-nautilus-admission-bar-five-seeds-recovered)
for the timing assumptions and limits. This is a source-plane measurement; it does not
measure PointSolver image-plane performance or gradient-sampler costs.

Leaves are added one sampler at a time, each in its own directory:

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
