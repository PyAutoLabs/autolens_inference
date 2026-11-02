"""The validation drivers shared by every ``scripts/<dataset_type>/<variant>/validation/`` leaf.

``scripts/misc/`` is on ``sys.path`` (see ``AGENTS.md``, "Import model"), so a
leaf imports its driver by its top-level name::

    from slam._runner import run_slam

    sys.exit(run_slam("imaging", "hst"))

:mod:`slam._core` holds everything a run does that is not a dataset — flag
validation, backend environment, the per-stage result rows, the ``stages.json``
writer — and :func:`slam._core.run_family`, the driver every non-imaging family
runs through. :mod:`slam._runner` is the imaging family: the five-stage chain
and ``run_slam``. ``README.md`` here says how a family registers its own chain.
The leaf is a name and a dataset class, nothing more.
"""
