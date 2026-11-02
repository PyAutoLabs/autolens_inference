"""Likelihood-lifecycle 4c-1 (autolens_inference#54): the runners of imaging mge / delaunay_nn,
interferometer rectangular / delaunay / mge and datacube delaunay.

Each family declares its likelihood and which stages evaluate it; a row of each, as
``run_family`` writes it, resolves to that likelihood and setup and pairs every evaluating stage
with its profiling record at the profiled size; the lens-light hook fits only the last stage
twice; and validation_p4 emits exactly the Optimal-setup legs, each passing the wall gate.

No test here imports autolens: the family modules and the core's flag checks import none.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_inference_summary import ROOT, put, summary
from .test_walltime_validation import _walltime

_MISC = str(ROOT / "scripts" / "misc")
for _path in (str(ROOT), _MISC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from campaigns import validation_p4 as p4  # noqa: E402
from slam import _core, _datacube, _imaging_mge, _interferometer  # noqa: E402
from wall import check_submits  # noqa: E402

PULSE = json.loads((ROOT / summary.PULSE_REFERENCE).read_text())["records"]
Stage = _core.Stage

#: family, the leaf that runs it, its reference instrument, an Optimal setup and its mesh fields.
_RECT = {"mesh": "rect", "mesh_pixels": 35, "mesh_shape": [35, 35]}
_DELAUNAY = {"mesh": "delaunay", "mesh_pixels": 1250, "mesh_shape": None}
_NO_MESH = {"mesh": None, "mesh_pixels": None, "mesh_shape": None}
FAMILIES = {
    "imaging/mge": (_imaging_mge.FAMILY, "hst", "jax_gpu_dense_fp64", _NO_MESH),
    "interferometer/rectangular": (
        _interferometer.RECTANGULAR,
        "sma",
        "numba_cpu_sparse_fp64_wtilde",
        _RECT,
    ),
    "interferometer/delaunay": (
        _interferometer.DELAUNAY,
        "sma",
        "jax_gpu_sparse_fp64_wtilde",
        _DELAUNAY,
    ),
    "interferometer/mge": (_interferometer.MGE, "sma", "jax_cpu_dense_fp64_nufft", _NO_MESH),
    "datacube/delaunay": (_datacube.FAMILY, "sma", "numba_cpu_sparse_fp64_wtilde", _DELAUNAY),
}
LEAVES = {
    "imaging/mge": "scripts/imaging/mge/validation/baseline_slam.py",
    "imaging/delaunay_nn": "scripts/imaging/delaunay_nn/validation/baseline_slam.py",
    "interferometer/rectangular": "scripts/interferometer/rectangular/validation/baseline_slam.py",
    "interferometer/delaunay": "scripts/interferometer/delaunay/validation/baseline_slam.py",
    "interferometer/mge": "scripts/interferometer/mge/validation/baseline_slam.py",
    "datacube/delaunay": "scripts/datacube/delaunay/validation/baseline.py",
}


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------


def test_each_family_declares_its_chain_and_which_stages_evaluate_it():
    chains = {
        name: [(s.search_name, s.evaluates_likelihood) for s in family.stages]
        for name, (family, *_rest) in FAMILIES.items()
    }
    assert chains == {
        # Every MGE stage solves the profiled MGE.
        "imaging/mge": [("source_lp[1]", True), ("light[1]", True), ("mass_total[1]", True)],
        # The parametric source_lp[1] ahead of a pixelized chain evaluates none of them.
        "interferometer/rectangular": [
            ("source_lp[1]", False),
            ("source_pix[1]", True),
            ("source_pix[2]", True),
            ("mass_total[1]", True),
        ],
        "interferometer/delaunay": [
            ("source_lp[1]", False),
            ("source_pix[1]", True),
            ("source_pix[2]", True),
            ("mass_total[1]", True),
        ],
        "interferometer/mge": [("source_lp[1]", True), ("mass_total[1]", True)],
        # A single search over the factor graph of channels.
        "datacube/delaunay": [("delaunay", True)],
    }
    for name, (family, *_rest) in FAMILIES.items():
        assert "/".join((family.dataset_type, family.likelihood_variant)) == name
    # Only imaging carries lens light; the interferometer data and the cube carry none.
    assert [n for n, (f, *_r) in FAMILIES.items() if f.light_fixed] == ["imaging/mge"]


def test_the_families_accept_only_their_profiled_setups():
    assert _imaging_mge.FAMILY.inversions == ("dense",)
    assert _interferometer.MGE.transformers == ("nufft",)
    for family in (_interferometer.RECTANGULAR, _interferometer.DELAUNAY, _datacube.FAMILY):
        assert family.transformers == ("nufft", "wtilde")
    assert _datacube.FAMILY.data_class == "interferometer"


def test_the_cube_is_the_profiled_cube():
    scales = _datacube.channel_scales()
    assert len(scales) == _datacube.N_CHANNELS == 34
    assert scales[0] == 1.0 and all(a > b for a, b in zip(scales, scales[1:]))
    assert scales[-1] == pytest.approx(math.exp(-0.5 * (33 / 17) ** 2))


def _cli(**overrides):
    fields = dict(mesh="rect", mesh_pixels=35)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_non_standard_mesh_names_its_own_variant():
    assert _interferometer.RECTANGULAR.variant_of(_cli()) == "slam_rectangular"
    assert _interferometer.RECTANGULAR.variant_of(_cli(mesh_pixels=40)) == "rect_40"
    assert _interferometer.DELAUNAY.variant_of(_cli(mesh="delaunay", mesh_pixels=1250)) == (
        "slam_delaunay"
    )
    assert _datacube.FAMILY.variant_of(_cli(mesh="delaunay", mesh_pixels=1250)) == (
        "search_delaunay"
    )
    assert _imaging_mge.FAMILY.variant_of(_cli()) == "slam_mge"


@pytest.mark.parametrize(
    "argv",
    [
        # An interferometer route without its transformer, or with the other inversion's.
        ["--inversion", "sparse", "--config-name", "local_numba_cpu_sparse_fp64"],
        ["--inversion", "dense", "--config-name", "local_jax_cpu_dense_fp64"],
        # No lens light in visibilities.
        [
            "--inversion",
            "sparse",
            "--transformer",
            "wtilde",
            "--config-name",
            "local_numba_cpu_sparse_fp64_wtilde",
            "--lens-light",
            "fixed",
        ],
    ],
)
def test_the_interferometer_leaf_refuses_before_the_environment(monkeypatch, argv):
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    monkeypatch.setattr(sys, "argv", ["leaf", *argv])
    assert _core.run_family(_interferometer.RECTANGULAR, default_instrument="sma") == 2


def test_the_mge_leaf_refuses_a_sparse_or_visibility_setup(monkeypatch):
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: pytest.fail("env before refusal"))
    for argv in (
        ["--inversion", "sparse", "--config-name", "local_jax_cpu_sparse_fp64"],
        ["--transformer", "nufft", "--config-name", "local_jax_cpu_dense_fp64_nufft"],
    ):
        monkeypatch.setattr(sys, "argv", ["leaf", *argv])
        assert _core.run_family(_imaging_mge.FAMILY, default_instrument="hst") == 2


# ---------------------------------------------------------------------------
# The lens-light hook: the last stage once per mode
# ---------------------------------------------------------------------------


class _FakeLeg:
    """Just enough of ``_core.Leg`` for ``run_family``'s loop: it records what it is asked."""

    def __init__(self, root):
        self.root, self.config_name, self.seed = root, "local_numba_cpu_dense_fp64", 0
        self.lens_light_mode = "linear"
        self.stage_settings_search = {}
        self.prefixes, self.built, self.rows = [], [], []

    def share_stages(self, *a):
        return None

    def sample_provenance(self):
        pass

    def settings_search_for(self, variant):
        self.prefixes.append(variant)
        return f"settings[{variant}]", self.root / variant

    def run_stage(self, key, name, built, *, stage_search_root, **kw):
        row = {"name": name, "root": stage_search_root and stage_search_root.name}
        row["settings"] = self.stage_settings_search.get(key)
        return f"result[{name}]", row

    def payload(self, **kw):
        return kw

    def write_row(self, json_path, png_path, payload):
        self.rows.append((json_path.relative_to(self.root).as_posix(), payload))


def _light_family():
    def build(leg, context, results):
        leg.built.append(leg.lens_light_mode)
        return None, None, None

    stages = tuple(Stage(k, f"{k}[1]", 1, 1, build) for k in ("source_lp", "light", "mass_total"))
    return _core.Family(
        dataset_type="imaging",
        likelihood_variant="mge",
        stages=stages,
        load=lambda leg: SimpleNamespace(truths={}),
        inversions=("dense",),
        light_fixed=True,
    )


@pytest.mark.parametrize("light", ["linear", "both"])
def test_light_both_fits_only_the_last_stage_twice(monkeypatch, tmp_path, light):
    leg = _FakeLeg(tmp_path)
    monkeypatch.setattr(_core, "set_backend_env", lambda *a: {})
    monkeypatch.setattr(_core, "open_leg", lambda *a, **k: leg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["leaf", "--backend", "numba_cpu", "--config-name", leg.config_name, "--lens-light", light],
    )
    assert _core.run_family(_light_family(), default_instrument="hst") == 0
    rows = {path.split("/")[5]: payload for path, payload in leg.rows}
    if light == "linear":
        assert leg.built == ["linear"] * 3 and set(rows) == {"default"}
    else:
        assert leg.built == ["linear", "linear", "linear", "fixed"]
        assert set(rows) == {"default", "light_fixed"}
        fixed = rows["light_fixed"]
        assert fixed["variant"] == "slam_mge_light_fixed"
        assert fixed["identity"]["model_variant"] == "light_fixed"
        assert fixed["identity"]["lens_light_mode"] == "fixed"
        # Stages 1-2 are shared; the fixed mass stage is fitted under its own prefix.
        names = [s["name"] for s in fixed["stages"]]
        assert names == ["source_lp[1]", "light[1]", "mass_total[1]"]
        assert fixed["stages"][-1]["root"] == "slam_mge_light_fixed"
        assert fixed["stages"][-1]["settings"] == "settings[slam_mge_light_fixed]"
    default = rows["default"]
    assert default["identity"]["lens_light_mode"] == "linear"
    assert default["stages"][-1]["root"] == "slam_mge"
    assert all(p["status"] == "complete" for p in rows.values())


# ---------------------------------------------------------------------------
# The exporter: each row resolves and pairs
# ---------------------------------------------------------------------------


def _family_row(family, instrument, setup_id, mesh):
    where = "hpc_a100" if "_gpu_" in setup_id else "hpc_ral"
    config = f"{where}_{setup_id}"
    block = family.likelihood
    variant = f"{family.pipeline}_{family.likelihood_variant}"
    return {
        "schema_version": 3,
        "target": family.target(instrument, variant, 0),
        "variant": variant,
        "likelihood": block,
        "model_variant": "default",
        "transformer": _core.parse_config_name(config).transformer,
        "config_name": config,
        "instrument": instrument,
        "dataset_class": family.data_class,
        "seed": 0,
        "sampler": "nautilus",
        "cores": 8,
        **mesh,
        "stages": [
            {
                "name": stage.search_name,
                "completed": True,
                "wall_s": 100.0,
                "likelihood_evals": 1000,
                "inversion_applies": stage.inversion_applies,
                "likelihood_evaluated": block if stage.evaluates_likelihood else None,
            }
            for stage in family.stages
        ],
    }


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_family_row_resolves_and_pairs_every_evaluating_stage(tmp_path, name):
    family, instrument, setup_id, mesh = FAMILIES[name]
    row = _family_row(family, instrument, setup_id, mesh)
    json_path, _ = _core.validation_row_paths(
        tmp_path, family.dataset_type, family.likelihood_variant, instrument, "default",
        row["config_name"], 0,
    )  # fmt: skip
    put(tmp_path, row, json_path.relative_to(tmp_path / "results").as_posix())
    records = {r["stage"]: r for r in _walltime(tmp_path, PULSE)["records"]}
    assert set(records) == {s.search_name for s in family.stages}
    for stage in family.stages:
        record = records[stage.search_name]
        assert record["likelihood"] == family.likelihood
        assert record["setup_id"] == setup_id and record["instrument"] == instrument
        if stage.evaluates_likelihood:
            assert record["profiling_record_id"] is not None, record["profiling_unmatched_reason"]
            assert record["size_match"] in ((None,) if mesh["mesh"] is None else (True,))
        else:
            assert record["profiling_unmatched_reason"] == summary.NO_LIKELIHOOD_REASON


# ---------------------------------------------------------------------------
# validation_p4: the Optimal-setup legs
# ---------------------------------------------------------------------------

KEYS_4C1 = {tuple(k.split("/")) for k in LEAVES}
LEGS = [leg for leg in p4.planned_legs() if leg.key in KEYS_4C1]
RENDERED = p4.submits()


def test_the_4c1_manifest_is_every_optimal_setup_on_its_reference_instrument():
    got = {(leg.key, leg.instrument, leg.setup_id, leg.model_variants, leg.seeds) for leg in LEGS}
    light, default = ("default", "light_fixed"), ("default",)
    assert got == {
        (("imaging", "delaunay_nn"), "hst", "jax_gpu_dense_fp64", light, (0, 1)),
        (("imaging", "delaunay_nn"), "hst", "numba_cpu_sparse_fp64", light, (0, 1)),
        (("imaging", "mge"), "hst", "jax_cpu_dense_fp64", light, (0, 1)),
        (("imaging", "mge"), "hst", "jax_gpu_dense_fp64", light, (0, 1)),
        (("interferometer", "rectangular"), "sma", "numba_cpu_sparse_fp64_wtilde", default, (0, 1)),
        (("interferometer", "rectangular"), "sma", "jax_gpu_sparse_fp64_wtilde", default, (0, 1)),
        (("interferometer", "delaunay"), "sma", "numba_cpu_sparse_fp64_wtilde", default, (0, 1)),
        (("interferometer", "delaunay"), "sma", "jax_gpu_sparse_fp64_wtilde", default, (0, 1)),
        (("interferometer", "mge"), "sma", "jax_cpu_dense_fp64_nufft", default, (0, 1)),
        (("interferometer", "mge"), "sma", "jax_gpu_dense_fp64_nufft", default, (0, 1)),
        # The datacube A100 cells are held provisional: its only Optimal setup is the CPU one.
        (("datacube", "delaunay"), "sma", "numba_cpu_sparse_fp64_wtilde", default, (0, 1)),
    }
    assert len({leg.path for leg in LEGS}) == len(LEGS)


@pytest.mark.parametrize("leg", LEGS, ids=lambda leg: leg.name)
def test_every_4c1_submit_runs_its_leaf_and_passes_the_wall_gate(leg):
    text = RENDERED[leg.path]
    assert check_submits.check_text(text, leg.name) == []
    (row,) = check_submits.parse_basis_rows(text)
    assert row["cell"] == leg.cell and row["source"] == "unmeasured"
    assert row["probe-first"] == "yes"
    cells, instruments = check_submits.cells_run(text)
    assert cells == {leg.cell} and instruments == {leg.instrument}
    run = check_submits.strip_comments(text)
    assert f"python3 {LEAVES['/'.join(leg.key)]} " in run
    assert f"--config-name {leg.config_name} " in run
    transformer = _core.parse_config_name(leg.config_name).transformer
    assert (f"--transformer {transformer}" in run) == (transformer is not None)
    assert ("--lens-light both" in run) == (leg.model_variants == ("default", "light_fixed"))
    # Only the run_slam leaf (delaunay_nn) shares source_lp[1] and exports mass_total[1].
    assert ("--stages-from" in run) == (leg.key == ("imaging", "delaunay_nn"))
    assert ("--export-prepared" in run) == (leg.key == ("imaging", "delaunay_nn"))


def test_delaunay_nn_counts_its_evaluations_on_the_delaunay_chain_never_its_walls():
    records, refs = p4.adopted_records(), p4.references()
    for leg in (leg for leg in LEGS if leg.key == ("imaging", "delaunay_nn")):
        budget, walls = p4.budget(leg, records, refs)
        assert walls and all(w.measured_s is None for w in walls)
        assert all("/delaunay/" in w.evals_ref and w.derived_s for w in walls)
        assert budget.source == "unmeasured" and budget.wall_s == budget.estimate_s


def test_a_likelihood_no_adopted_row_counts_is_contained_at_the_cap():
    records, refs = p4.adopted_records(), p4.references()
    for leg in (leg for leg in LEGS if leg.family):
        budget, walls = p4.budget(leg, records, refs)
        assert walls == [] and budget.wall_s is None and budget.time_s == p4.MAX_TIME_S


@pytest.mark.parametrize("name", sorted(LEAVES))
def test_every_leaf_is_its_own_cell(name):
    cell = check_submits.LEGACY_CELLS[LEAVES[name].removeprefix("scripts/").removesuffix(".py")]
    assert cell == p4.CELL[tuple(name.split("/"))]
    assert Path(ROOT / LEAVES[name]).is_file()
