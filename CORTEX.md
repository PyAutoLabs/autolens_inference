# Where the rulings of record live

The **rulings of record** for this repository — what is decided, and therefore what a run
is allowed to assume — live in
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex), in `projects.yaml`, row
`autolens_inference`. That row is the machine-readable body map for this repo: the remote,
the RAL root, the sync CLI and its verbs, the ledger, the assistant and the witness every
run is judged by.

Campaign intent and pending inference tasks now live in
[PyAutoInsight](https://github.com/PyAutoLabs/PyAutoInsight), whose `CHECKIN.md`
coordinates the complete inference status sweep. Existing project drivers execute
runs. PyAutoMind retains bounded implementation issues, PR lifecycle and claims.

## The current scientific ledger

**`PyAutoCortex/projects/autolens_inference.md`** holds the authoritative current
run records and append-only log of observations, lessons and human conclusions.
Use the current `scripts/cortex.py` verbs described in Cortex's `REFERENCE.md`;
finished runs become log entries. `wiki/project/state.md` remains the project's
commentary/history ledger linked from the Cortex body map. Insight links both;
it does not duplicate their scientific conclusions as editable campaign facts.

- **<https://pyautolabs.github.io/PyAutoCortex/>** — the current scientific board.
- **<https://pyautolabs.github.io/PyAutoInsight/>** — inference campaigns and evidence.
- **`PyAutoCortex/archive/`** — frozen historical tasks and rulings. The former
  `tasks/` and `rulings/` workflow is retired, not a current scheduling surface.

## Nothing is inherited

This repo was born 2026-09-10 as the **from-scratch restart** of the retired Cortex
project `inference_programme` (retired in PyAutoCortex#22). None of that programme's
baselines, notes, tolerance tables, target code or run JSONs came across, and none of its
evidence may be cited by a ruling here. Every number this project stands on is measured
in this repo, by a run it can name.
