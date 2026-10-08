# Simple point source

Stable setup: `point_source/simple/simple`.
Run `python scripts/point_source/simple/baseline.py --instrument simple
--backend jax_cpu --inversion dense --config-name local_jax_cpu_dense_fp64 --seed 0`
(as one shell command). This is the existing Nautilus source-plane solved model,
shared in `scripts/misc/searches/_point_runner.py`.
Historical rows retain their original source_plane_solved result identities.
No prepared scientific identity or accepted reference is inferred.
