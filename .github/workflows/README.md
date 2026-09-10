# `.github/workflows/`

CI for `autolens_inference`. Two workflows, deliberately split.

## `lint.yml` — PR + push-to-main gate

Runs on every PR and every push to `main`. CPU-only, target wall time under 5 minutes.

| Step | Purpose |
|------|---------|
| `ruff check .` | Pyflakes + pycodestyle + isort + pyupgrade + flake8-bugbear (see `ruff.toml`) |
| `ruff format --check .` | Formatting parity with the sibling PyAutoLabs repos (black-compatible defaults) |
| `python scripts/misc/tooling/build_readme.py --check` | Dashboard idempotence — the auto-generated tables in `README.md` must match what the renderer would produce from the current `results/`. Catches "forgot to rerun the dashboard generator after committing a result" |
| `python scripts/misc/wall/check_submits.py --check` | Every submit — and both `hpc/batch_*/template` files — justifies its `--time` from a rate measured on the cell it runs |
| `pytest scripts/misc/test -q` | The renderer's fixed-point tests |
| `lychee` | Markdown link-rot across every `README.md` |
| Smoke — the three simulators | Runs each under `AUTOLENS_INFERENCE_SMOKE=1`. Every script reads that at module top and exits 0 after the import + setup section, so this catches import-graph breakage (a broken `sys.path` injection, a missing dependency, a renamed module) without simulating or fitting anything |

The job checks out the five PyAuto* `main`s and installs them for their dependency chain,
then resolves the imports from source via `PYTHONPATH` — the checkouts stay authoritative.
Nothing here produces a result artifact.

## `profile.yml` — manual, and empty in phase 1

`workflow_dispatch` only, and currently a placeholder that prints why. Phase 3 of the
`autolens-inference` epic wires it to the backend-parameterised driver.

It will never be a PR gate. The real legs of a parity row are an A100 job and a multi-hour
CPU job; neither runs on a GitHub-hosted runner, and a leg that quietly ran on different
hardware would be a row that lies about what it measured. Real runs go to RAL by hand:

```bash
hpc/sync submit --gpu submit_<name>
hpc/sync jobs
hpc/sync tail gpu
hpc/sync pull
```

See [`../../hpc/README.md`](../../hpc/README.md).

## Design decisions

- **CPU-only runner, single Python version (3.12).** Lens modelling is Linux-only in
  practice, and a matrix would buy nothing this gate is for.
- **No coverage reporting.** This repo is a scripts collection; the smoke step and the
  renderer's fixed-point tests are the practical equivalents.
- **The wall-clock gate runs on templates too.** A template is what every future submit is
  copied from, so a template without a `# WALL-BASIS:` block seeds the defect into every
  child.
