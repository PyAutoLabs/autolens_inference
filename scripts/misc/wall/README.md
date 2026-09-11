# `wall` — per-cell step rates + the submit `--time` gate

This subpackage owns one rule:

> **Never carry a `--time` justification across cells.** Derive it from a rate
> measured on *that* cell. When no measurement exists, run one short arm first.

Two responsibilities:

1. **Rates** (`wall.rates`) — the per-cell step-rate table, populated only from
   rates measured in this repo, on the cell each row names.
2. **Gate** (`wall.check_submits`) — the CI check that every cell a submit
   actually runs has its own basis row, that cited rates match the table, and
   that `--time` clears the declared headroom.

## Why it exists

A submit set `--time=0:30:00` and justified it in its own comment as "16 starts
x 3000 steps at the validated pixelized throughput is ~5 min including compile
per task ... `--time` below gives it 6x headroom."

That was a **parametric** (MGE) rate, for an array whose arms were mostly
pixelized. Pixelized cells are per-eval-inversion bound at tens of times the
cost, so the claimed 6x headroom was an order of magnitude short. **35 of 39
arms were killed at ~12% of budget**, losing an overnight A100 block; the 4 that
finished were the parametric controls — the only cells the citation actually
described.

Nothing about a step rate is portable across that boundary, and no amount of
confident prose around the number makes it so. Hence a table that refuses to
answer for a cell it has not measured, and a gate that reads what the job will
really run rather than what its header claims.

## The table started empty, and is still narrower than it looks

`rates.py` shipped with `STEP_RATE = {}`. This repo was born 2026-09-10 and
deliberately does **not** inherit the retired `inference_programme`'s rate table —
those numbers were measured on a different tree by runs this project cannot name.

Its first two rows landed **2026-09-11**, and both describe `source_lp[1]`, the
*parametric* opening stage of `imaging/slam/hst`, measured on a laptop. Neither
may size a five-stage submit (the other four stages are pixelized and
per-eval-inversion bound), and neither may size a RAL job (the device key names
the host). So every submit in `hpc/batch_*` still declares `source: unmeasured`
with `probe-first: yes`. That is the honest state, and the gate accepts it.

## The `# WALL-BASIS:` block

One row per cell the submit runs, in the header above the `#SBATCH` stanza:

```
# WALL-BASIS: — one row per cell this submit runs.
#   cell: imaging/slam/hst  device: a100  precision: fp64
#   lanes: 1  steps: 3000  source: unmeasured  probe-first: yes
```

A row starts at its `cell:` key and runs to the next `cell:` or the end of the
block. **`cell:` is the invoked script's path below `scripts/`, without the
`.py`** — `python3 scripts/imaging/slam/hst.py` is the cell `imaging/slam/hst`,
i.e. `<dataset_class>/<task>/<leaf>`. The task directory is part of the identity
because this repo names a leaf for the *target* (`AGENTS.md`: the instrument is a
flag, never a directory), so `imaging/slam/hst.py` and
`imaging/searches/nautilus/hst.py` would otherwise both be the cell
`imaging/hst` — two pipelines sharing one rate row, which is the carry this gate
exists to prevent. The rate-table key splits the id as `(dataset, task, leaf)`,
and the leaf doubles as the instrument the `--instrument` check reads. Prose
lines inside the block are ignored, so the human "why" can sit next to the
machine-checked "what".

### The three `source:` kinds

| `source:` | needs | headroom floor | means |
|---|---|---|---|
| `rates` | `lanes`, `batch_size`, `steps`, `rate`, `device`, `precision` | 1.5x | a step rate measured on **this** cell, matching `rates.py` within 5% |
| `measured-wall` | `wall` (seconds), `ref` | 1.25x | a directly observed total for this cell — a prior run of this same arm |
| `unmeasured` | `probe-first: yes` | 3x on any `wall` offered | nothing measured. Legal, honest, and the normal case here today |

### What the gate checks

1. **Every cell the submit runs has its own row.** The cells are read from the
   `python3 scripts/<dataset_class>/.../<cell>.py` invocation — resolving
   `${CELLS[$I]}` arrays, `case` arms and variable indirection, and ignoring
   commands merely *mentioned* in comments. A cell with no row of its own is a
   cell whose `--time` was justified by some other cell's rate.
2. **A declared cell is one the submit actually runs** (and its instrument).
3. **A `source: rates` row matches `rates.py`** within 5%.
4. **`--time` >= headroom x estimated wall**, over the *slowest* row. An array
   submit is sized by its slowest cell, never its fastest.

### Which files must carry one

Every `submit_*` under `hpc/batch_gpu/` and `hpc/batch_cpu/`, **and the
`template` beside them**. The template is what every future submit is copied
from, so a template without a block seeds the defect into every child.

This is a **path predicate, not an allowlist**. No submit is individually
exempted, because an exemption list would hide precisely the class of leak this
gate exists to close.

## Running it

```bash
python scripts/misc/wall/check_submits.py           # report on every submit
python scripts/misc/wall/check_submits.py --check   # exit non-zero on any violation
```

`--check` runs in the `lint` workflow on every PR and push to main, alongside
`build_readme.py --check`.

## Adding a measured rate

1. **Measure it on the cell itself.** A full arm, or a truncated one — steps
   completed / wall elapsed, read from the arm's own log. A 30-minute killed arm
   still measures s/step.
2. Add the row to `STEP_RATE` in `rates.py` with an inline comment naming the
   RAL job and the date. The key is
   `(dataset, cell, instrument, device, precision, n_lanes, batch_size)` —
   `batch_size` is in the key because an unbatched lane row and a
   `batch_size=4` row are different configurations of the same cell.
3. Add the matching `PROVENANCE` entry, including what the row must **not** be
   used for.
4. Point the submit's row at it with `source: rates`.

**Do not interpolate.** An interpolated rate is an unearned citation in a more
respectable coat: it sits between two measured rows and was never measured, so a
submit resting on one declares `measured-wall` against its observed runs, or
`unmeasured`.
