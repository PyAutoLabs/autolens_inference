"""Likelihood-lifecycle 4c-3 (autolens_inference#54): the point-source, cluster and weak
single-search families — each is one ``modeling[1]`` stage that declares its likelihood, its
leaf refuses every setup profiling did not measure, its row resolves to the likelihood's
Optimal profiling record, and validation_p4 emits its two Optimal legs (seeds 0 and 1) as
submits that pass the wall gate.

Nothing here fits: the modelling stack is imported only by the cluster-model test, which
builds a model and no analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from .test_inference_summary import ROOT, put, summary
from .test_walltime_validation import _walltime

_MISC = str(ROOT / "scripts" / "misc")
for _path in (str(ROOT), _MISC, str(ROOT / "scripts" / "misc" / "tooling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from campaigns import validation_p4 as p4  # noqa: E402
from slam import _cluster, _core, _point, _weak  # noqa: E402
from wall import check_submits  # noqa: E402

from _inference_cli import config_name_for, split_config_name  # noqa: E402

PULSE = json.loads((ROOT / summary.PULSE_REFERENCE).read_text())["records"]

#: family -> (leaf, the Optimal setups of autolens_profiling/catalogue/runtime_estimates.yaml
#: with their record ids, the workspace's n_live / n_batch).
FAMILIES = {
    _point.SOURCE_PLANE: (
        "point_source_source/source_plane",
        {
            "jax_cpu_na_fp64": "lifecycle/61082dc032d7280b3bd1",
            "jax_gpu_na_fp64": "lifecycle/fcb319f0f24b8cc9fd37",
        },
        (100, 50),
    ),
    _point.IMAGE_PLANE: (
        "point_source_image/image_plane",
        {
            "jax_cpu_na_fp64": "lifecycle/6cbf08c848ae6cdb820f",
            "jax_gpu_na_fp64": "lifecycle/6282556b81be530f7bce",
        },
        (100, 50),
    ),
    _cluster.SOURCE_PLANE: (
        "cluster/source_plane",
        {
            "jax_cpu_na_fp64": "lifecycle/004001caf29a727421e0",
            "jax_gpu_na_fp64": "lifecycle/02df871c3e8378b505ca",
        },
        (100, 50),
    ),
    _cluster.IMAGE_PLANE: (
        "cluster/image_plane",
        {
            "jax_cpu_na_fp64": "lifecycle/86d6277f36e3c8ec1d2f",
            "jax_gpu_na_fp64": "lifecycle/d92190013ae10eaa21f6",
        },
        (100, 50),
    ),
    _weak.ISOTHERMAL: (
        "weak/isothermal",
        {
            "jax_cpu_dense_fp64": "lifecycle/03601d3a29aca4f5c389",
            "jax_gpu_dense_fp64": "lifecycle/fbd315b1a0e03f308c76",
        },
        (100, 100),
    ),
}
IDS = [f"{f.dataset_type}/{f.likelihood_variant}" for f in FAMILIES]
LEGS = [leg for leg in p4.planned_legs() if leg.key in p4.EVALS_BASIS]
RENDERED = p4.submits()


def _host(setup_id: str) -> str:
    return "hpc_a100" if "_gpu_" in setup_id else "hpc_ral"


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_family_is_one_modeling_search_of_its_likelihood(family):
    leaf, setups, (n_live, n_batch) = FAMILIES[family]
    assert f"{family.dataset_type}/{family.likelihood_variant}" == leaf
    (stage,) = family.stages
    assert (stage.key, stage.search_name) == ("modeling", "modeling[1]")
    assert stage.evaluates_likelihood and not stage.inversion_applies
    assert (stage.n_live, stage.n_batch) == (n_live, n_batch)
    assert family.inversions == tuple({s.split("_")[2] for s in setups})
    assert family.variant_of(None) == f"modeling_{family.likelihood_variant}"
    assert (ROOT / "scripts" / leaf / "validation" / "modeling.py").is_file()


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_the_leaf_refuses_an_inversion_its_likelihood_has_not(monkeypatch, family):
    wrong = "dense" if family.inversions == ("na",) else "na"
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["leaf", "--inversion", wrong, "--config-name", f"local_jax_cpu_{wrong}_fp64"],
    )
    assert _core.run_family(family, default_instrument="simple") == 2


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_the_loader_refuses_every_setup_profiling_did_not_measure(family):
    _, setups, _ = FAMILIES[family]
    inversion = family.inversions[0]
    for setup_id in setups:
        leg = type("Leg", (), {"config_name": config_name_for(_host(setup_id), setup_id)})
        assert _point.refuse_unprofiled_setup(leg, tuple(setups)) is None
    for setup_id in (f"numba_cpu_{inversion}_fp64", f"jax_cpu_{inversion}_mp"):
        leg = type("Leg", (), {"config_name": f"hpc_ral_{setup_id}"})
        # The loader refuses before it touches the dataset or the modelling stack.
        assert family.load(leg) == 2


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_family_row_pairs_with_its_optimal_profiling_record(tmp_path, family):
    """The row shape the leaf writes (a PYAUTO_TEST_MODE witness of each leaf did the same on
    a scratch copy) resolves to the family's likelihood and setup id and pairs its one stage
    with the Optimal runtime record of runtime_estimates.yaml."""
    _, setups, _ = FAMILIES[family]
    for setup_id, record_id in setups.items():
        config_name = config_name_for(_host(setup_id), setup_id)
        variant = family.variant_of(None)
        row = {
            "schema_version": _core.SCHEMA_VERSION,
            "target": family.target("simple", variant, 0),
            "variant": variant,
            "likelihood": family.likelihood,
            "model_variant": family.model_variant,
            "transformer": None,
            "config_name": config_name,
            "instrument": "simple",
            "dataset_class": family.data_class,
            "seed": 0,
            "sampler": "nautilus",
            "cores": 8,
            "status": "complete",
            "stages": [
                {
                    "name": "modeling[1]",
                    "completed": True,
                    "wall_s": 60.0,
                    "likelihood_evals": 5000,
                    "inversion_applies": False,
                    "likelihood_evaluated": family.likelihood,
                }
            ],
        }
        json_path, _ = _core.validation_row_paths(
            tmp_path,
            family.dataset_type,
            family.likelihood_variant,
            "simple",
            family.model_variant,
            config_name,
            0,
        )
        assert split_config_name(config_name)[0] == setup_id
        put(tmp_path, row, json_path.relative_to(tmp_path / "results").as_posix())
        (record,) = [r for r in _walltime(tmp_path, PULSE)["records"] if r["setup_id"] == setup_id]
        assert record["likelihood"] == family.likelihood
        assert record["stage"] == "modeling[1]"
        assert record["profiling_record_id"] == record_id, record["profiling_unmatched_reason"]


def test_the_cluster_model_is_the_profiled_one_and_every_parameter_has_a_truth():
    import autofit as af
    import autolens as al

    context = type(
        "Context",
        (),
        {
            "main_centres": [c for c, *_ in _cluster.MAIN_LENS_PARAMS],
            "host_centre": (0.0, 0.0),
            "scaling_centres": [(5.5, -6.5), (-7.5, 3.0)],
            "scaling_luminosities": [0.4, 0.32],
            "dataset_list": [
                type("D", (), {"name": "point_0", "redshift": 1.0}),
                type("D", (), {"name": "point_1", "redshift": 2.0}),
            ],
        },
    )
    model, truths = _cluster.cluster_model(af, al, context)
    # 2 main lenses x (centre, ra, rs, b0) + one scaling b0_ref + the host centre.
    assert model.prior_count == 13
    assert len(truths) == 13
    assert truths["galaxies.main_0.mass.rs"] == 20.0 and truths["galaxies.main_1.mass.b0"] == 1.2
    assert _cluster.SCALING_B0_REF in truths.values()


# ---------------------------------------------------------------------------
# validation_p4: the Optimal legs and their submits
# ---------------------------------------------------------------------------


def test_every_family_gets_its_two_optimal_legs_over_seeds_0_and_1():
    got = {(leg.key, leg.setup_id, leg.host_class, leg.seeds, leg.instrument) for leg in LEGS}
    expected = {
        (
            (family.dataset_type, family.likelihood_variant),
            setup_id,
            "hpc_a100" if "_gpu_" in setup_id else "hpc_ral_cpu",
            (0, 1),
            "simple",
        )
        for family, (_, setups, _) in FAMILIES.items()
        for setup_id in setups
    }
    assert got == expected
    assert all(leg.role in ("Optimal CPU", "Optimal GPU") for leg in LEGS)
    assert len({leg.name for leg in LEGS}) == len(LEGS)


def test_the_source_plane_evaluations_are_the_committed_search_rows():
    rows = sorted(
        (ROOT / "results/point_source_source/source_plane/simple/samplers/default/nautilus").rglob(
            "search.json"
        )
    )
    most = max(json.loads(p.read_text())["likelihood_evals"] for p in rows)
    evals, ref = p4.EVALS_BASIS[("point_source_source", "source_plane")]
    assert evals == most
    assert json.loads((ROOT / ref).read_text())["likelihood_evals"] == most


@pytest.mark.parametrize("leg", LEGS, ids=lambda leg: leg.name)
def test_a_family_submit_runs_one_search_and_passes_the_wall_gate(leg):
    text = RENDERED[leg.path]
    assert check_submits.check_text(text, leg.name) == []
    (row,) = check_submits.parse_basis_rows(text)
    assert row["source"] == "unmeasured" and row["probe-first"] == "yes"
    assert row["cell"] == f"{leg.dataset_type}/search/simple_{leg.likelihood}"
    assert leg.family
    cells, instruments = check_submits.cells_run(text)
    assert cells == {row["cell"]} and instruments == {"simple"}
    run = check_submits.strip_comments(text)
    script = f"scripts/{leg.dataset_type}/{leg.likelihood}/validation/modeling.py"
    assert f"python3 {script} " in run and (ROOT / script).is_file()
    assert f"--inversion {leg.inversion} " in run
    for flag in ("--stages-from", "--lens-light", "--export-prepared", "--stages "):
        assert flag not in run
    assert leg.fitted_stages("default") == ("modeling[1]",)
