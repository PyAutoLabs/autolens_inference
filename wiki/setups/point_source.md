# Simple point source

Stable setup: `point_source/simple/simple`.
Run `python scripts/point_source_source/source_plane/samplers/baseline.py --instrument simple
--backend jax_cpu --inversion dense --config-name local_jax_cpu_dense_fp64 --seed 0`
(as one shell command). This is the existing Nautilus source-plane solved model,
shared in `scripts/misc/searches/_point_runner.py`.
Historical rows retain their original source_plane_solved result identities.
No prepared scientific identity or accepted reference is inferred.

Gradient search: `scripts/point_source_source/source_plane/samplers/multi_start_adam.py` (library-default
300-step ceiling) and `multi_start_adam_n3000.py` (ceiling 3000; auto-convergence ends
the run) fit the same model with `af.MultiStartAdam` in the analysis's declared forward
gradient mode. First rows: issue #25 (2026-10-17, released 2026.10.17.1 stack).
