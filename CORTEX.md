# Where the rulings of record live

The **rulings of record** for this repository — what is decided, and therefore what a run
is allowed to assume — live in
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex), in `projects.yaml`, row
`autolens_inference`. That row is the machine-readable body map for this repo: the remote,
the RAL root, the sync CLI and its verbs, the ledger, the assistant and the witness every
run is judged by.

This repo is managed as **science**, not as software development. A question about which
search to run, on what, and whether the answer that came back is believable is a Cortex
task; only changes to the code in this repo are a PyAutoMind development task.

## The ledger

`wiki/project/state.md` is this project's **ledger** — the Cortex row points at it. It
holds the science goal, where the project currently is, and a dated journal entry per
piece of work (template: `wiki/project/_template.md`). It is commentary that explains why
a decision was reached and what it cost; where it and a Cortex ruling disagree, the
**ruling is the one that counts**.

## The rulings themselves

There is no hand-maintained list here — a copy of a ledger is not a ledger. Read the
rulings where they live:

- **`PyAutoCortex/rulings/<YYYY>/<MM>/`** — every ruling file, append-only.
- **<https://pyautolabs.github.io/PyAutoCortex/>** — the Cortex board: what is awaiting a
  ruling, what is running, what has been ruled, by project.
- **`PyAutoCortex/tasks/autolens_inference/`** — this project's tasks; each one's
  `Ruling:` header names its chain head.

## Nothing is inherited

This repo was born 2026-09-10 as the **from-scratch restart** of the retired Cortex
project `inference_programme` (retired in PyAutoCortex#22). None of that programme's
baselines, notes, tolerance tables, target code or run JSONs came across, and none of its
evidence may be cited by a ruling here. Every number this project stands on is measured
in this repo, by a run it can name.
