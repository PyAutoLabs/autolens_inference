"""The single-search framework shared by every ``scripts/<dataset_class>/searches/`` leaf.

``scripts/misc/`` is on ``sys.path`` (see ``AGENTS.md``, "Import model"), so a
leaf imports its runner by top-level name::

    from searches._point_runner import run_point_search

    sys.exit(run_point_search(sampler="nautilus", default_instrument="simple"))

Flag validation, the backend environment, the posterior / truth helpers and the
device record are imported from :mod:`slam._runner` and :mod:`_inference_cli`
rather than copied; what lives here is only what a single search adds — the
point-source model and the admission-bar timing fields.
"""
