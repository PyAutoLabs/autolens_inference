"""The point-source families: one Nautilus search, on the shared core.

Two likelihoods, one pipeline. ``autolens_workspace/scripts/point_source/modeling.py`` fits a
point-source lens with ONE Nautilus search (``n_live=100``, ``n_batch=50``), an
``Isothermal`` lens and a ``PointSolved`` source (5 free parameters: the source-plane centre
is solved analytically for every trial mass model), and the fit class picks the likelihood:

- :data:`SOURCE_PLANE` (``point_source_source/source_plane``) —
  ``al.FitPositionsSourceSolved``, the tensor-weighted source-plane chi-squared, the
  workspace's "fast" option;
- :data:`IMAGE_PLANE` (``point_source_image/image_plane``) —
  ``al.FitPositionsImagePairAllSolved``, the all-to-all image-plane chi-squared the
  workspace analysis uses by default, which forward-solves the lens equation with the
  ``PointSolver`` at every evaluation.

Both are the headline cells autolens_profiling times
(``scripts/point_source_<source|image>/<variant>/likelihood_runtime_solved.py``, the runtime
cells of its point-source campaign ``scripts/misc/campaigns/point_source_p2c.py``: human
ruling 2026-10-29, "Switch 2c to Solved"): the same model, fit class and ``PointSolver``
(100 x 100 @ 0.2", ``pixel_scale_precision`` 0.001, ``magnification_threshold`` 0.1 — the
workspace solver, passed to both analyses as the workspace passes it) on the same
``dataset/point_source/simple`` positions. Each family is a :class:`slam._core.Family` of one
stage, ``modeling[1]``, so its row is a one-stage ``stages.json`` that pairs with the
likelihood's profiling record.

What is NOT mirrored: the profiled cell's priors. The profiling scripts time one call at the
prior medians and so put truth-centred Gaussians on every parameter (sigma 0.005" on the
centre); a search's evaluation count depends on where the prior sits, so these families take
the priors of the committed point-source search leaves instead
(``searches._point_runner.point_model``): the workspace's library-default ``Isothermal``
priors with θ_E bounded to ``U(0.5, 4.0)``. Re-using that model keeps the source-plane
validation row comparable with the committed Nautilus sampler rows of the same target, which
it does not replace: ``scripts/point_source_source/source_plane/samplers/`` still owns the
admission-bar search rows.

The likelihood has no linear inversion, so the leaves accept only ``--inversion na`` and the
Optimal setups profiling measured (``jax_cpu_na_fp64`` on a RAL CPU node, ``jax_gpu_na_fp64``
on an A100; ``autolens_profiling/catalogue/runtime_estimates.yaml``, role optimal).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from searches._point_runner import point_model, point_truth_dict  # noqa: E402

from _inference_cli import auto_simulate_if_missing, split_config_name  # noqa: E402
from slam._core import Family, Stage  # noqa: E402

#: The one search of every single-search family here, as ``modeling.py`` names it.
STAGE_KEY = "modeling"
SEARCH_NAME = "modeling[1]"

#: The setups profiling measured for a point-source likelihood (role optimal). A leaf refuses
#: any other: a row of an unprofiled setup would pair with no record.
PROFILED_SETUPS = ("jax_cpu_na_fp64", "jax_gpu_na_fp64")

#: The workspace (and profiled) ``PointSolver``: ``autolens_workspace/scripts/point_source/
#: modeling.py`` and both ``likelihood_runtime_solved.py`` cells.
SOLVER = {"shape_native": (100, 100), "pixel_scales": 0.2, "pixel_scale_precision": 0.001}
MAGNIFICATION_THRESHOLD = 0.1


def refuse_unprofiled_setup(leg, setups: tuple[str, ...]) -> int | None:
    """2 (after saying why) when the leg's setup id is not one profiling measured, else None.

    ``Family.inversions`` already refuses a wrong inversion before anything loads; this is
    the backend / precision half, which the family core leaves to the family (its loader
    may return an exit code). Shared by the cluster and weak families.
    """
    setup_id, _ = split_config_name(leg.config_name)
    if setup_id in setups:
        return None
    print(
        f"ERROR: setup {setup_id!r} has no profiling record for this likelihood; it accepts "
        f"only {', '.join(setups)} (autolens_profiling/catalogue/runtime_estimates.yaml). "
        "Nothing was fitted.",
        file=sys.stderr,
    )
    return 2


def _load(fit_positions_cls: str):
    def load(leg):
        code = refuse_unprofiled_setup(leg, PROFILED_SETUPS)
        if code is not None:
            return code
        al = leg.al
        auto_simulate_if_missing(
            leg.dataset_path,
            dataset_type=leg.dataset_class,
            instrument=leg.instrument,
            workspace_root=leg.root,
        )
        grid = al.Grid2D.uniform(
            shape_native=SOLVER["shape_native"], pixel_scales=SOLVER["pixel_scales"]
        )
        return SimpleNamespace(
            dataset=al.from_json(file_path=leg.dataset_path / "point_dataset_positions_only.json"),
            solver=al.PointSolver.for_grid(
                grid=grid,
                pixel_scale_precision=SOLVER["pixel_scale_precision"],
                magnification_threshold=MAGNIFICATION_THRESHOLD,
            ),
            fit_positions_cls=getattr(al, fit_positions_cls),
            truths=point_truth_dict(json.loads((leg.dataset_path / "truth.json").read_text())),
        )

    return load


def build_modeling(leg, context, results):
    """``modeling.py``'s one search: the point model, ``AnalysisPoint`` with the family's fit
    class, Nautilus at the workspace settings."""
    analysis = leg.al.AnalysisPoint(
        dataset=context.dataset,
        solver=context.solver,
        fit_positions_cls=context.fit_positions_cls,
        use_jax=leg.use_jax,
    )
    return point_model(leg.af, leg.al), analysis, leg.nautilus(MODELING)


#: Nautilus at ``autolens_workspace/scripts/point_source/modeling.py``'s settings.
MODELING = Stage(STAGE_KEY, SEARCH_NAME, 100, 50, build_modeling)


def _model_fields(fit_positions_cls: str):
    def model_fields(leg, context) -> dict:
        return {
            "fit_positions_cls": fit_positions_cls,
            "point_solver": {
                "grid_shape_native": list(SOLVER["shape_native"]),
                "grid_pixel_scales": SOLVER["pixel_scales"],
                "pixel_scale_precision": SOLVER["pixel_scale_precision"],
                "magnification_threshold": MAGNIFICATION_THRESHOLD,
            },
        }

    return model_fields


def _family(dataset_type: str, variant: str, fit_positions_cls: str) -> Family:
    return Family(
        dataset_type=dataset_type,
        likelihood_variant=variant,
        stages=(MODELING,),
        load=_load(fit_positions_cls),
        dataset_class="point_source",
        pipeline="modeling",
        inversions=("na",),
        model_fields=_model_fields(fit_positions_cls),
        title="Point-source modeling run",
    )


SOURCE_PLANE = _family("point_source_source", "source_plane", "FitPositionsSourceSolved")
IMAGE_PLANE = _family("point_source_image", "image_plane", "FitPositionsImagePairAllSolved")
