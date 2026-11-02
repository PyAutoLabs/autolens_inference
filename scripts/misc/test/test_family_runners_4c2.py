"""Likelihood-lifecycle 4c-2 (autolens_inference#54): the group, multi_galaxy and multi_dataset
runners — each family declares its likelihood and the chain it mirrors, its stages say which
evaluate the profiled likelihood, a row built from its declaration files and pairs under that
likelihood at its Optimal setups, and validation_p4 renders one honest submit per Optimal setup.

No test here imports autolens: the declarations and the flag checks load without the stack.
"""

from __future__ import annotations

import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from .test_inference_summary import ROOT, put, summary
from .test_walltime_validation import _walltime

for _path in (
    str(ROOT),
    str(ROOT / "scripts" / "misc"),
    str(ROOT / "scripts" / "misc" / "tooling"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from campaigns import validation_p4 as p4  # noqa: E402
from slam import _core, _group, _multi_dataset, _multi_galaxy, _runner  # noqa: E402
from wall import check_submits  # noqa: E402

from _inference_cli import MESH_PIXELS_STANDARD, InferenceCLI, parse_config_name  # noqa: E402

PULSE = json.loads((ROOT / summary.PULSE_REFERENCE).read_text())["records"]

#: (family, instrument, leaf, mesh, the chain's stage search names, stages that evaluate it).
PIXELIZED_GROUP = (
    "source_lp[0]",
    "source_lp[1]",
    "source_pix[1]",
    "source_pix[2]",
    "light[1]",
    "mass_total[1]",
)
FIVE = ("source_lp[1]", "source_pix[1]", "source_pix[2]", "light[1]", "mass_total[1]")
MGE = ("source_lp[1]", "light[1]", "mass_total[1]")
FAMILIES = {
    ("group", "rectangular"): (_group.FAMILY_RECTANGULAR, "euclid", "rect", PIXELIZED_GROUP, 4),
    ("group", "delaunay"): (_group.FAMILY_DELAUNAY, "euclid", "delaunay", PIXELIZED_GROUP, 4),
    ("group", "mge"): (_group.FAMILY_MGE, "euclid", None, MGE, 3),
    ("multi_galaxy", "rectangular"): (_multi_galaxy.FAMILY_RECTANGULAR, "hst", "rect", FIVE, 4),
    ("multi_galaxy", "delaunay"): (_multi_galaxy.FAMILY_DELAUNAY, "hst", "delaunay", FIVE, 4),
    ("multi_galaxy", "mge"): (_multi_galaxy.FAMILY_MGE, "hst", None, MGE, 3),
    ("multi_dataset", "delaunay"): (_multi_dataset.FAMILY_DELAUNAY, "hst", "delaunay", FIVE, 4),
}
#: The Optimal setups of autolens_profiling catalogue/runtime_estimates.yaml (role: optimal).
OPTIMAL = {
    "rect": ("numba_cpu_sparse_fp64", "jax_gpu_dense_fp64"),
    "delaunay": ("numba_cpu_sparse_fp64", "jax_gpu_dense_fp64"),
    None: ("jax_cpu_dense_fp64", "jax_gpu_dense_fp64"),
}


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
        mesh_pixels=MESH_PIXELS_STANDARD["rect"],
        regularization=None,
        memo="off",
    )
    fields.update(overrides)
    return InferenceCLI(**fields)


def _meshes(mesh):
    """The ``MeshModels`` record fields of the profiled mesh (the models need autolens)."""
    if mesh == "rect":
        return _runner.MeshModels(None, None, None, "adapt", (35, 35))
    return _runner.MeshModels(
        None, None, None, "adapt_split", None, 0.5, "hilbert", 1250, 3.5, 0.01, 30, 30
    )


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("likelihood", sorted(FAMILIES))
def test_each_family_declares_its_likelihood_and_chain(likelihood):
    family, _, mesh, names, evaluated = FAMILIES[likelihood]
    assert (family.dataset_type, family.likelihood_variant) == likelihood
    assert tuple(stage.search_name for stage in family.stages) == names
    assert len(set(family.stage_keys)) == len(names)
    # A pixelized chain's parametric stages evaluate none of the profiled likelihoods; every
    # MGE stage is the MGE likelihood.
    flags = [stage.evaluates_likelihood for stage in family.stages]
    assert sum(flags) == evaluated and flags[-evaluated:] == [True] * evaluated
    assert all(
        s.inversion_applies == (mesh is not None) for s in family.stages if s.evaluates_likelihood
    )
    assert family.inversions == (("dense",) if mesh is None else ("dense", "sparse"))
    assert family.variant_of(_cli()) == f"slam_{likelihood[1]}"
    assert family.model_variant == "default" and family.data_class == likelihood[0]


def test_the_meshes_differ_only_where_the_workspace_pipelines_do():
    """group/slam.py (Delaunay) and its pixelization script (rectangular) differ in n_live on
    light[1] / mass_total[1]; the stages, searches and batches are the same."""
    rect, delaunay = _group.FAMILY_RECTANGULAR.stages, _group.FAMILY_DELAUNAY.stages
    lives = {s.key: (r.n_live, s.n_live) for r, s in zip(rect, delaunay)}
    assert lives == {
        "source_lp_0": (190, 190),
        "source_lp_1": (240, 240),
        "source_pix_1": (150, 150),
        "source_pix_2": (75, 75),
        "light": (150, 300),
        "mass_total": (150, 200),
    }
    assert [s.n_batch for s in rect] == [50, 50, 20, 20, 20, 20]
    assert [s.n_live for s in _multi_galaxy.FAMILY_RECTANGULAR.stages] == [250, 200, 75, 250, 250]
    assert [s.n_live for s in _multi_dataset.FAMILY_DELAUNAY.stages] == [200, 150, 75, 250, 150]


@pytest.mark.parametrize("likelihood", sorted(FAMILIES))
def test_the_leaf_runs_its_family_at_the_profiled_mesh(likelihood, monkeypatch):
    family, instrument, mesh, _, _ = FAMILIES[likelihood]
    leaf = ROOT / "scripts" / likelihood[0] / likelihood[1] / "validation" / f"{instrument}_slam.py"
    seen = {}

    def fake(fam, default_instrument, default_mesh=None, default_mesh_pixels=None):
        seen.update(fam=fam, inst=default_instrument, mesh=default_mesh, px=default_mesh_pixels)
        return 0

    monkeypatch.setattr(_core, "run_family", fake)
    with pytest.raises(SystemExit) as exit_:
        runpy.run_path(str(leaf), run_name="__main__")
    assert exit_.value.code == 0
    assert seen["fam"] is family and seen["inst"] == instrument and seen["mesh"] == mesh
    assert seen["px"] == (MESH_PIXELS_STANDARD[mesh] if mesh else None)


def test_an_mge_leaf_refuses_a_sparse_setup_before_the_environment(monkeypatch):
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    argv = ["leaf", "--inversion", "sparse", "--config-name", "local_jax_cpu_sparse_fp64"]
    monkeypatch.setattr(sys, "argv", argv)
    assert _core.run_family(_group.FAMILY_MGE, default_instrument="euclid") == 2


@pytest.mark.parametrize(
    ("variant", "mesh", "pixels", "refused"),
    [
        ("delaunay", "delaunay", 1250, False),
        ("delaunay", "delaunay", 1500, True),
        ("delaunay", "rect", 35, True),
        ("rectangular", "rect", 35, False),
        ("rectangular", "rect", 28, True),
        ("mge", "rect", 28, False),
    ],
)
def test_a_pixelized_leaf_runs_only_the_profiled_mesh(variant, mesh, pixels, refused):
    leg = SimpleNamespace(cli=_cli(mesh=mesh, mesh_pixels=pixels))
    assert (_group.refuse_mesh(leg, variant) == 2) is refused


def test_the_model_fields_carry_the_mesh_the_exporter_sizes():
    rect = _group.model_fields(None, SimpleNamespace(meshes=_meshes("rect"), cli=_cli()))
    assert rect["mesh_shape"] == [35, 35] and rect["regularization"] == "adapt"
    cli = _cli(mesh="delaunay", mesh_pixels=1250)
    delaunay = _group.model_fields(None, SimpleNamespace(meshes=_meshes("delaunay"), cli=cli))
    assert delaunay["mesh"] == "delaunay" and delaunay["mesh_pixels"] == 1250
    assert _group.model_fields(None, SimpleNamespace(meshes=None, cli=_cli()))["mesh"] is None
    md = _multi_dataset.model_fields(None, SimpleNamespace(meshes=_meshes("delaunay"), cli=cli))
    assert md["dataset_scales"] == [1.0, 0.5] and md["mesh_pixels"] == 1250


# ---------------------------------------------------------------------------
# The exporter: a row built from the family's declaration resolves and pairs
# ---------------------------------------------------------------------------


def _row(likelihood, setup_id):
    family, instrument, mesh, _, _ = FAMILIES[likelihood]
    where = "hpc_a100" if "_gpu_" in setup_id else "hpc_ral"
    config = f"{where}_{setup_id}"
    fields = parse_config_name(config)
    cli = _cli(
        mesh=mesh or "rect",
        mesh_pixels=MESH_PIXELS_STANDARD[mesh or "rect"],
        backend=fields.backend,
        inversion=fields.inversion,
    )
    meshes = _meshes(mesh) if mesh else None
    model = family.model_fields(None, SimpleNamespace(meshes=meshes, cli=cli))
    stages = [
        {
            "name": stage.search_name,
            "completed": True,
            "wall_s": 100.0,
            "likelihood_evals": 1000,
            "inversion_applies": stage.inversion_applies,
            "likelihood_evaluated": family.likelihood if stage.evaluates_likelihood else None,
        }
        for stage in family.stages
    ]
    variant = family.variant_of(cli)
    return {
        "schema_version": 3,
        "target": family.target(instrument, variant, 0),
        "variant": variant,
        "likelihood": family.likelihood,
        "model_variant": family.model_variant,
        "transformer": None,
        "config_name": config,
        "instrument": instrument,
        "dataset_class": family.data_class,
        "seed": 0,
        "sampler": "nautilus",
        "cores": 8,
        **model,
        "stages": stages,
    }


@pytest.mark.parametrize(
    ("likelihood", "setup_id"),
    [(lk, setup) for lk, spec in sorted(FAMILIES.items()) for setup in OPTIMAL[spec[2]]],
)
def test_a_family_row_resolves_and_pairs_at_its_optimal_setups(tmp_path, likelihood, setup_id):
    family, instrument, mesh, names, evaluated = FAMILIES[likelihood]
    dataset_type, variant = likelihood
    rel = f"{dataset_type}/{variant}/{instrument}/validation/default/{setup_id}/seed_0/stages.json"
    json_path, _ = _core.validation_row_paths(
        tmp_path, dataset_type, variant, instrument, "default", f"hpc_ral_{setup_id}", 0
    )
    assert json_path == tmp_path / "results" / rel
    put(tmp_path, _row(likelihood, setup_id), rel)
    records = {r["stage"]: r for r in _walltime(tmp_path, PULSE)["records"]}
    assert set(records) == set(names)
    for stage in family.stages:
        record = records[stage.search_name]
        assert record["likelihood"] == family.likelihood
        assert record["setup_id"] == setup_id and record["instrument"] == instrument
        if stage.evaluates_likelihood:
            assert record["profiling_record_id"] is not None, record["profiling_unmatched_reason"]
            assert record["size_match"] is (True if mesh else None)
        else:
            assert record["profiling_unmatched_reason"] == summary.NO_LIKELIHOOD_REASON
    paired = sum(1 for r in records.values() if r["profiling_record_id"])
    assert paired == evaluated


# ---------------------------------------------------------------------------
# validation_p4: Optimal CPU + GPU, default + light_fixed, unmeasured, probe first
# ---------------------------------------------------------------------------

RECORDS = p4.adopted_records()
OURS = [leg for leg in p4.planned_legs(RECORDS) if leg.key in FAMILIES]


def test_every_4c2_likelihood_has_its_two_optimal_legs_with_light_fixed():
    light = ("default", "light_fixed")
    got = {(leg.key, leg.setup_id, leg.seeds, leg.model_variants) for leg in OURS}
    expected = {
        (lk, setup, (0, 1), light) for lk, spec in FAMILIES.items() for setup in OPTIMAL[spec[2]]
    }
    assert got == expected
    for leg in OURS:
        assert leg.family and leg.instrument == FAMILIES[leg.key][1]
        assert leg.role == ("Optimal GPU" if leg.on_gpu else "Optimal CPU")
        assert not leg.share_source_lp and leg.lens_light == "both"
        assert (ROOT / leg.script).is_file()
        # The light_fixed row refits only the last stage.
        assert leg.fitted_stages("light_fixed") == ("mass_total[1]",)
    assert all(family.light_fixed for family, *_ in FAMILIES.values())


def test_the_proxy_basis_is_the_most_any_adopted_imaging_row_took():
    """The pinned EVALS_BASIS counts are the adopted imaging maxima per stage (a refresh of the
    adopted rows that moves one must move the pin)."""
    imaging = [
        r
        for r in RECORDS
        if r["likelihood"]["dataset_type"] == "imaging"
        and r["model_variant"] == "default"
        and r["likelihood_evals"]
        and ":cores_" not in r["sampler"]
    ]
    for stage, (evals, ref) in p4._4C2_IMAGING_EVALS.items():
        most = max((r for r in imaging if r["stage"] == stage), key=lambda r: r["likelihood_evals"])
        assert evals == most["likelihood_evals"]
        assert ref in {
            r["id"] for r in imaging if r["stage"] == stage and r["likelihood_evals"] == evals
        }
    for key in FAMILIES:
        basis = p4.EVALS_BASIS[key]
        assert set(basis) == set(p4.STAGES_OF[key])
        assert all(ref.startswith("proxy: max adopted imaging ") for _, ref in basis.values())


@pytest.mark.parametrize("leg", OURS, ids=lambda leg: leg.name)
def test_a_family_submit_is_unmeasured_probe_first_and_runs_its_leaf(leg):
    text = p4.render(leg, RECORDS, p4.references())
    assert check_submits.check_text(text, leg.name) == []
    (row,) = check_submits.parse_basis_rows(text)
    assert (
        row["cell"]
        == leg.cell
        == f"{leg.dataset_type}/{leg.likelihood}/validation/{leg.instrument}_slam"
    )
    assert row["source"] == "unmeasured" and row["probe-first"] == "yes" and "wall" in row
    assert check_submits.cells_run(text) == ({leg.cell}, {leg.instrument})
    run = check_submits.strip_comments(text)
    assert f"--instrument {leg.instrument} " in run and f"--config-name {leg.config_name} " in run
    assert "--export-prepared" not in run and "--stages-from" not in run
    assert "--lens-light both" in run and "--array=0-1" in text
    # Every stage priced (the parametric ones too), at this likelihood's own profiling record,
    # plus the light_fixed mass_total[1].
    walls = p4.stage_walls(leg, RECORDS, p4.references())
    stages = FAMILIES[leg.key][3]
    assert [(w.model_variant, w.stage) for w in walls] == [
        *(("default", s) for s in stages),
        ("light_fixed", "mass_total[1]"),
    ]
    assert all(w.derived_s and w.evals_ref.startswith("proxy: ") for w in walls)
    assert len({w.per_call_ref for w in walls}) == 1
