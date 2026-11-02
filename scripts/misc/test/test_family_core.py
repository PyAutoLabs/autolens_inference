"""Likelihood-lifecycle 4c-0 (autolens_inference#54): the shared foundation the family runners
stand on — every profiled likelihood resolves, the Pulse setup ids are expressible config names,
a stage is paired with the likelihood it evaluates, delaunay_nn files under its own folder, and
the core driver refuses setups a family does not have.

No test here imports autolens: the core's flag checks run before the modelling stack loads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_inference_summary import ROOT, feed, put, summary
from .test_walltime_validation import _walltime

_MISC = str(ROOT / "scripts" / "misc")
for _path in (str(ROOT), _MISC, str(ROOT / "scripts" / "misc" / "tooling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import inference_catalogue  # noqa: E402
from slam import _core, _runner  # noqa: E402

from _inference_cli import (  # noqa: E402
    DELAUNAY_NN_BASE_VARIANT,
    MESH_PIXELS_STANDARD,
    InferenceCLI,
    config_name_for,
    parse_config_name,
    split_config_name,
    validation_layout,
)

PULSE = json.loads((ROOT / summary.PULSE_REFERENCE).read_text())["records"]

#: Every likelihood the cached profiling reference times — the same 20 as autolens_profiling's
#: catalogue/runtime_estimates.yaml (imaging x4, interferometer x3, datacube, group x3,
#: multi_galaxy x3, multi_dataset, point source x2, cluster x2, weak).
LIKELIHOODS = sorted({(r["likelihood"]["dataset_type"], r["likelihood"]["variant"]) for r in PULSE})


def test_the_reference_holds_every_profiled_likelihood():
    assert len(LIKELIHOODS) == 20


# ---------------------------------------------------------------------------
# Config names (1c)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setup_id", sorted({r["setup_id"] for r in PULSE}))
def test_every_pulse_setup_id_is_a_config_name(setup_id):
    """The grammar expresses every setup the profiling records time, na / nufft / wtilde too."""
    for where in ("local", "hpc_a100", "hpc_ral"):
        name = config_name_for(where, setup_id)
        assert split_config_name(name)[0] == setup_id


def test_the_transformer_belongs_to_one_inversion():
    assert parse_config_name("hpc_ral_jax_cpu_dense_fp64_nufft").transformer == "nufft"
    assert parse_config_name("hpc_ral_numba_cpu_sparse_fp64_wtilde").transformer == "wtilde"
    assert parse_config_name("hpc_ral_jax_cpu_na_fp64").inversion == "na"
    assert parse_config_name("local_jax_cpu_dense_fp64").transformer is None
    for bad in (
        "hpc_ral_jax_cpu_sparse_fp64_nufft",
        "hpc_ral_jax_cpu_dense_fp64_wtilde",
        "hpc_ral_jax_cpu_na_fp64_wtilde",
        "hpc_ral_jax_cpu_dense_fp64_",
    ):
        assert parse_config_name(bad) is None, bad
        with pytest.raises(ValueError):
            split_config_name(bad)


def _cli(**overrides):
    fields = dict(
        config_name="local_jax_cpu_dense_fp64",
        output_dir=None,
        use_mixed_precision=False,
        instrument=None,
        backend="jax_cpu",
        inversion="dense",
        seed=0,
        use_sparse_operator=False,
        rect_mesh="bilinear",
        mesh="rect",
        mesh_pixels=28,
        regularization=None,
        memo="off",
    )
    fields.update(overrides)
    return InferenceCLI(**fields)


def test_the_name_must_agree_with_the_transformer_and_na_flags():
    ok = _cli(config_name="local_jax_cpu_dense_fp64_nufft", transformer="nufft")
    assert _core.validate_config_name(ok) is None
    assert "transformer" in _core.validate_config_name(_cli(transformer="nufft"))
    lying = _cli(config_name="local_jax_cpu_dense_fp64_nufft")
    assert "transformer" in _core.validate_config_name(lying)
    na = _cli(config_name="hpc_ral_jax_cpu_na_fp64", inversion="na")
    assert _core.validate_config_name(na) is None
    assert "inversion" in _core.validate_config_name(_cli(config_name="hpc_ral_jax_cpu_na_fp64"))


def test_a_leaf_accepts_only_the_setups_its_likelihood_has():
    assert _core.refuse_setup(_cli(), ("dense", "sparse")) is None
    assert "na" in _core.refuse_setup(_cli(inversion="na"), _runner.IMAGING_INVERSIONS)
    assert "transformer" in _core.refuse_setup(_cli(transformer="wtilde"), ("dense", "sparse"))
    wtilde = _cli(inversion="sparse", transformer="wtilde")
    assert _core.refuse_setup(wtilde, ("dense", "sparse"), ("nufft", "wtilde")) is None


def test_the_imaging_chain_refuses_na_before_anything_loads(monkeypatch):
    monkeypatch.setattr(_runner, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["leaf", "--inversion", "na", "--config-name", "local_jax_cpu_na_fp64"],
    )
    assert _runner.run_slam() == 2


# ---------------------------------------------------------------------------
# delaunay_nn: the mesh, the variant and its folder (1b, runner)
# ---------------------------------------------------------------------------


def test_delaunay_nn_files_under_its_own_likelihood():
    assert validation_layout(DELAUNAY_NN_BASE_VARIANT) == ("delaunay_nn", "default")
    assert validation_layout("slam_delaunay_nn_light_fixed") == ("delaunay_nn", "light_fixed")
    assert validation_layout("delaunay_nn_1500") == ("delaunay_nn", "delaunay_nn_1500")
    # The rows already on disk keep their folders.
    assert validation_layout("slam_delaunay") == ("delaunay", "default")
    assert validation_layout("slam_delaunay_light_fixed") == ("delaunay", "light_fixed")
    assert validation_layout("slam_base") == ("rectangular", "default")
    assert validation_layout("rect_35") == ("rectangular", "rect_35")


def test_the_variant_of_the_delaunay_nn_and_standard_rectangular_meshes():
    from _inference_cli import base_variant_name

    assert base_variant_name(_cli(mesh="delaunay_nn", mesh_pixels=1250)) == "slam_delaunay_nn"
    assert base_variant_name(_cli(mesh="rect", mesh_pixels=35)) == "rect_35"
    assert MESH_PIXELS_STANDARD["rect"] ** 2 == 1225


def test_delaunay_nn_is_the_delaunay_recipe_on_the_natural_neighbour_mesh():
    class Mesh:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DelaunayNN(Mesh):
        pass

    class Delaunay(Mesh):
        pass

    al = SimpleNamespace(
        mesh=SimpleNamespace(Delaunay=Delaunay, DelaunayNN=DelaunayNN),
        reg=SimpleNamespace(AdaptSplit=object, Adapt=None),
    )
    nn = _runner.mesh_models(None, al, _cli(mesh="delaunay_nn", mesh_pixels=1250))
    plain = _runner.mesh_models(None, al, _cli(mesh="delaunay", mesh_pixels=1250))
    assert type(nn.mesh) is DelaunayNN and type(nn.mesh_init) is DelaunayNN
    assert type(plain.mesh) is Delaunay
    assert (
        nn.mesh.kwargs == plain.mesh.kwargs == dict(pixels=1250, zeroed_pixels=30, areas_factor=0.5)
    )
    assert nn._replace(mesh=None, mesh_init=None) == plain._replace(mesh=None, mesh_init=None)


def test_a_delaunay_nn_row_resolves_and_counts_its_vertices(tmp_path):
    row = {
        "schema_version": 3,
        "dataset_class": "imaging",
        "instrument": "hst",
        "variant": "slam_delaunay_nn",
        "mesh": "delaunay_nn",
        "mesh_pixels": 1250,
        "config_name": "hpc_a100_jax_gpu_sparse_fp64",
        "sampler": "nautilus",
        "seed": 0,
        "stages": [
            {"name": "source_pix[1]", "completed": True, "wall_s": 9.0, "likelihood_evals": 9}
        ],
    }
    put(
        tmp_path,
        row,
        "imaging/delaunay_nn/hst/validation/default/jax_gpu_sparse_fp64/seed_0/stages.json",
    )
    (record,) = _walltime(tmp_path, PULSE)["records"]
    assert record["likelihood"] == {"dataset_type": "imaging", "variant": "delaunay_nn"}
    assert record["source_pixels"] == 1250
    assert record["profiling_record_id"] is not None


# ---------------------------------------------------------------------------
# Likelihood resolution and stage pairing (1a, 1d)
# ---------------------------------------------------------------------------


def _family_row(likelihood, setup_id, stages, instrument="simple"):
    dataset_type, variant = likelihood
    where = "hpc_a100" if "_gpu_" in setup_id else "hpc_ral"
    return {
        "schema_version": 3,
        "target": f"{dataset_type}/{variant}/{instrument}/slam_{variant}/seed0",
        "variant": f"slam_{variant}",
        "likelihood": {"dataset_type": dataset_type, "variant": variant},
        "model_variant": "default",
        "transformer": parse_config_name(f"{where}_{setup_id}").transformer,
        "config_name": f"{where}_{setup_id}",
        "instrument": instrument,
        "dataset_class": dataset_type,
        "seed": 0,
        "sampler": "nautilus",
        "cores": 8,
        "stages": stages,
    }


def _stage(name, *, evaluated, inversion_applies=False):
    return {
        "name": name,
        "completed": True,
        "wall_s": 100.0,
        "likelihood_evals": 1000,
        "inversion_applies": inversion_applies,
        "likelihood_evaluated": evaluated,
    }


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_every_profiled_likelihood_resolves_and_pairs(tmp_path, likelihood):
    """A family row of each likelihood, at a setup its profiling records hold, resolves to that
    likelihood and setup id and pairs every stage — no inversion needed."""
    ref = next(
        r
        for r in sorted(PULSE, key=lambda r: r["id"])
        if (r["likelihood"]["dataset_type"], r["likelihood"]["variant"]) == likelihood
        and r["host_class"] == ("hpc_a100" if "_gpu_" in r["setup_id"] else "hpc_ral_cpu")
        and isinstance(r["per_call_s"], (int, float))
    )
    block = {"dataset_type": likelihood[0], "variant": likelihood[1]}
    row = _family_row(
        likelihood, ref["setup_id"], [_stage("fit[1]", evaluated=block)], ref["instrument"]
    )
    path = f"{likelihood[0]}/{likelihood[1]}/{ref['instrument']}/validation/default/{ref['setup_id']}/seed_0/stages.json"
    put(tmp_path, row, path)
    stage = next(r for r in feed(tmp_path)["records"] if r["parent_run_id"])
    assert stage["likelihood"] == block and stage["model_variant"] == "default"
    assert stage["setup_id"] == ref["setup_id"]
    (record,) = _walltime(tmp_path, PULSE)["records"]
    assert record["likelihood"] == block
    assert record["profiling_record_id"] is not None, record["profiling_unmatched_reason"]


def test_a_parametric_stage_of_a_pixelized_chain_pairs_with_nothing(tmp_path):
    likelihood = ("interferometer", "rectangular")
    block = {"dataset_type": "interferometer", "variant": "rectangular"}
    stages = [
        _stage("source_lp[1]", evaluated=None),
        _stage("source_pix[1]", evaluated=block, inversion_applies=True),
    ]
    row = _family_row(likelihood, "jax_gpu_sparse_fp64_wtilde", stages, "sma")
    put(tmp_path, row, "interferometer/rectangular/sma/validation/default/x/seed_0/stages.json")
    records = {r["stage"]: r for r in _walltime(tmp_path, PULSE)["records"]}
    lp = records["source_lp[1]"]
    assert lp["profiling_record_id"] is None
    assert lp["profiling_unmatched_reason"] == summary.NO_LIKELIHOOD_REASON
    # Filed under its chain's likelihood all the same, so the chain stays one group.
    assert lp["likelihood"] == block
    assert records["source_pix[1]"]["profiling_record_id"] is not None


def test_a_no_inversion_stage_of_its_own_likelihood_pairs(tmp_path):
    """MGE / point / cluster / weak solve no inversion yet evaluate their own likelihood."""
    block = {"dataset_type": "point_source_image", "variant": "image_plane"}
    stage = _stage("point[1]", evaluated=block, inversion_applies=False)
    row = _family_row(("point_source_image", "image_plane"), "jax_gpu_na_fp64", [stage])
    put(
        tmp_path,
        row,
        "point_source_image/image_plane/simple/validation/default/x/seed_0/stages.json",
    )
    (record,) = _walltime(tmp_path, PULSE)["records"]
    assert record["profiling_record_id"] is not None
    assert record["setup_id"] == "jax_gpu_na_fp64"


def test_a_row_without_a_declaration_keeps_the_old_rule(tmp_path):
    """Imaging SLaM rows (no likelihood block, no likelihood_evaluated) resolve as before."""
    stages = [
        {"name": "source_lp[1]", "completed": True, "wall_s": 1.0, "inversion_applies": False},
        {"name": "mass_total[1]", "completed": True, "wall_s": 1.0, "inversion_applies": True},
    ]
    row = {
        "schema_version": 3,
        "dataset_class": "imaging",
        "instrument": "hst",
        "variant": "slam_base",
        "mesh": "rect",
        "config_name": "hpc_a100_jax_gpu_dense_fp64",
        "sampler": "nautilus",
        "seed": 0,
        "stages": stages,
    }
    put(tmp_path, row, "imaging/rectangular/hst/validation/default/x/seed_0/stages.json")
    records = {r["stage"]: r for r in _walltime(tmp_path, PULSE)["records"]}
    assert records["source_lp[1]"]["profiling_unmatched_reason"] == summary.NO_INVERSION_REASON
    assert records["mass_total[1]"]["likelihood"] == {
        "dataset_type": "imaging",
        "variant": "rectangular",
    }


def test_the_committed_walltime_records_are_unchanged_by_4c():
    """The 56 adopted records regenerate exactly (only limitation text may move)."""
    committed = json.loads((ROOT / "dashboard/walltime_validation.json").read_text())
    rebuilt = summary.build_walltime(ROOT, feed(ROOT), committed["producer_revision"])
    assert rebuilt["records"] == committed["records"]


# ---------------------------------------------------------------------------
# Catalogue (1e)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_every_likelihood_has_a_catalogue_family(likelihood):
    assert likelihood in inference_catalogue.FAMILIES


def test_a_family_row_gets_its_catalogue_setup_id():
    for dataset_type, variant, instrument, _ in inference_catalogue.REFERENCE_SETUPS:
        row = {
            "likelihood": {"dataset_type": dataset_type, "variant": variant},
            "dataset": {"instrument": instrument},
            "pipeline": "slam",
        }
        assert inference_catalogue.setup_id(row) == f"{dataset_type}/{variant}/{instrument}"
        row["dataset"]["instrument"] = "nowhere"
        assert inference_catalogue.setup_id(row) is None


# ---------------------------------------------------------------------------
# The core driver
# ---------------------------------------------------------------------------


def _toy_family(**extra):
    stage = _core.Stage("fit", "fit[1]", 100, 50, build=lambda *a: pytest.fail("built"))
    return _core.Family(
        dataset_type="point_source_image",
        likelihood_variant="image_plane",
        stages=(stage,),
        load=lambda leg: pytest.fail("loaded"),
        inversions=("na",),
        **extra,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--inversion", "dense", "--config-name", "local_jax_cpu_dense_fp64"],
        ["--inversion", "na", "--config-name", "local_jax_cpu_dense_fp64"],
        ["--inversion", "na", "--config-name", "local_jax_cpu_na_fp64", "--lens-light", "fixed"],
    ],
)
def test_run_family_refuses_before_the_environment(monkeypatch, argv):
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    monkeypatch.setattr(sys, "argv", ["leaf", *argv])
    assert _core.run_family(_toy_family(), default_instrument="simple") == 2


def test_a_family_takes_its_own_stage_vocabulary(monkeypatch):
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    monkeypatch.setattr(sys, "argv", ["leaf", "--stages", "source_lp"])
    with pytest.raises(SystemExit):  # argparse: not this family's stage
        _core.run_family(_toy_family(), default_instrument="simple")


def test_a_family_row_lands_under_its_declared_likelihood(tmp_path):
    json_path, png_path = _core.validation_row_paths(
        tmp_path, "cluster", "source_plane", "simple", "default", "hpc_ral_jax_cpu_na_fp64", 1
    )
    assert json_path == (
        tmp_path
        / "results/cluster/source_plane/simple/validation/default/jax_cpu_na_fp64/seed_1/stages.json"
    )
    family = _toy_family()
    assert family.variant_of(_cli()) == "slam_image_plane"
    assert family.likelihood == {"dataset_type": "point_source_image", "variant": "image_plane"}
    assert family.target("simple", "slam_image_plane", 0) == (
        "point_source_image/image_plane/simple/slam_image_plane/seed0"
    )
    assert _core.output_path_prefix("weak", "simple", "v", "c", 0, "search") == Path(
        "autolens/weak/simple/search/v/c/seed_0"
    )


def test_the_imaging_results_path_is_the_core_path(tmp_path):
    old = _runner.results_paths(
        tmp_path, "imaging", "hst", "slam_delaunay_nn", "hpc_a100_jax_gpu_dense_fp64", 0
    )
    new = _core.validation_row_paths(
        tmp_path, "imaging", "delaunay_nn", "hst", "default", "hpc_a100_jax_gpu_dense_fp64", 0
    )
    assert old == new
