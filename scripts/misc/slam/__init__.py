"""The SLaM pipeline framework shared by every ``scripts/<dataset_class>/slam/`` leaf.

``scripts/misc/`` is on ``sys.path`` (see ``AGENTS.md``, "Import model"), so a
leaf imports the driver by its top-level name::

    from slam._runner import run_slam

    sys.exit(run_slam("imaging", "hst"))

Everything the run does — flag validation, backend environment, the five-stage
chain, the per-stage result rows — lives in :mod:`slam._runner`. The leaf is a
name and a dataset class, nothing more.
"""
