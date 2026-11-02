"""Campaign generators: a manifest of legs, rendered into ``hpc/batch_{cpu,gpu}/`` submits.

``scripts/misc/`` is on ``sys.path`` (``AGENTS.md``, "Import model"), so a test imports a
generator by its top-level name (``from campaigns import validation_p4``).
"""
