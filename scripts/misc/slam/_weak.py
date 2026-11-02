"""The weak-lensing family: one Nautilus search, on the shared core.

The pipeline is ``autolens_workspace/scripts/weak/modeling.py``: ONE Nautilus search
(``n_live=100``; the workspace passes no ``n_batch``, so the library default, 100) of a free
elliptical ``Isothermal`` lens and an empty source galaxy, fitted with ``al.AnalysisWeak`` to a
shear catalogue — the model shear at every catalogue position from the deflections' Hessian,
then a Gaussian log-likelihood over the ``2N`` shear components. There is no linear inversion.

The dataset and the model are the profiled cell's (``weak/isothermal``,
``autolens_profiling/scripts/misc/likelihood_breakdown/weak_standard.py``): the 1500-galaxy
``simple`` catalogue of ``scripts/misc/simulators/weak.py`` (ported byte-for-byte), the same
five free parameters (``centre``, ``ell_comps``, ``einstein_radius``) and the simulator truth
(:func:`simulators.weak.truth_vector`) as the row's truth.

Departures from ``modeling.py``:

1. **The priors are the workspace's, not the profiled cell's.** ``standard_model`` puts tight
   Gaussians on the truth because profiling times one call at the prior medians; a search
   takes the workspace priors (``einstein_radius ~ U(0, 60)``, each centre ``N(0, 20)``,
   library-default ``ell_comps``).
2. **The analysis is given** ``use_jax=leg.use_jax``. The workspace runs ``AnalysisWeak`` in
   NumPy (its default, ``use_jax=False``); the profiled setups are JAX
   (``jax_cpu_dense_fp64`` on a RAL CPU node, ``jax_gpu_dense_fp64`` on an A100;
   ``autolens_profiling/catalogue/runtime_estimates.yaml``, role optimal), and a validation
   row must run the setup it is paired with.
3. **The visualizer's fit is NumPy** (:func:`analysis_weak`). Under ``use_jax=True`` the
   library's ``fit_for_visualization`` returns the JAX fit, whose ``model_shear`` is a bare
   array the weak plotter cannot draw (``AttributeError: ... no attribute 'grid'``, which
   killed the ``PYAUTO_TEST_MODE`` witness at the final output). The likelihood the sampler
   evaluates is untouched: only the plotted fit is rebuilt eagerly. A library fix belongs in
   PyAutoLens (``autolens/weak/plot/weak_dataset_plots.py``) and makes this a no-op.

Profiling files the weak likelihood under the ``dense`` setups, although it solves no linear
system, so the leaf accepts only ``--inversion dense``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from simulators import weak as weak_sim  # noqa: E402

from _inference_cli import auto_simulate_if_missing  # noqa: E402
from slam._core import Family, Stage  # noqa: E402
from slam._point import SEARCH_NAME, STAGE_KEY, refuse_unprofiled_setup  # noqa: E402

#: The setups profiling measured for ``weak/isothermal`` (role optimal).
PROFILED_SETUPS = ("jax_cpu_dense_fp64", "jax_gpu_dense_fp64")

#: The free parameters, in ``weak_sim.truth_vector``'s order; each leaf name is unique, so
#: it is the row's posterior key (``slam._core.posterior_keys``).
TRUTH_KEYS = ("centre_0", "centre_1", "ell_comps_0", "ell_comps_1", "einstein_radius")


def load(leg):
    code = refuse_unprofiled_setup(leg, PROFILED_SETUPS)
    if code is not None:
        return code
    if leg.instrument not in weak_sim.INSTRUMENTS:
        print(
            f"ERROR: unknown --instrument {leg.instrument!r}; "
            f"choose from {sorted(weak_sim.INSTRUMENTS)}",
            file=sys.stderr,
        )
        return 2
    auto_simulate_if_missing(
        leg.dataset_path,
        dataset_type=leg.dataset_class,
        instrument=leg.instrument,
        workspace_root=leg.root,
    )
    return SimpleNamespace(
        dataset=leg.al.from_json(file_path=leg.dataset_path / "dataset.json"),
        config=weak_sim.INSTRUMENTS[leg.instrument],
        truths=dict(zip(TRUTH_KEYS, weak_sim.truth_vector(leg.instrument))),
    )


def weak_model(af, al, config: dict):
    """``modeling.py``'s model: a free elliptical ``Isothermal`` under the workspace priors and
    an empty source galaxy, at the preset's redshifts. 5 free parameters."""
    mass = af.Model(al.mp.Isothermal)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=60.0)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=20.0)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=20.0)
    lens = af.Model(al.Galaxy, redshift=config["lens_redshift"], mass=mass)
    source = af.Model(al.Galaxy, redshift=config["source_redshift"])
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def analysis_weak(al, dataset, use_jax: bool):
    """``al.AnalysisWeak`` whose visualizer draws a NumPy fit (departure 3)."""

    class AnalysisWeak(al.AnalysisWeak):
        def fit_for_visualization(self, instance):
            import numpy as np

            tracer = self.tracer_via_instance_from(instance=instance)
            return al.FitWeak(dataset=self.dataset, tracer=tracer, xp=np)

    return AnalysisWeak(dataset=dataset, use_jax=use_jax)


def build_modeling(leg, context, results):
    analysis = analysis_weak(leg.al, context.dataset, leg.use_jax)
    return weak_model(leg.af, leg.al, context.config), analysis, leg.nautilus(MODELING)


#: Nautilus at ``autolens_workspace/scripts/weak/modeling.py``'s settings (``n_batch`` the
#: ``af.Nautilus`` default, which the workspace leaves unset).
MODELING = Stage(STAGE_KEY, SEARCH_NAME, 100, 100, build_modeling)


def model_fields(leg, context) -> dict:
    return {"n_galaxies": int(context.dataset.n_galaxies), "mass_profile": "Isothermal"}


ISOTHERMAL = Family(
    dataset_type="weak",
    likelihood_variant=weak_sim.VARIANT,
    stages=(MODELING,),
    load=load,
    pipeline="modeling",
    inversions=("dense",),
    model_fields=model_fields,
    title="Weak-lensing modeling run",
)
