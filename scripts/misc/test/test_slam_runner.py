"""Unit tests for the SLaM driver's two pure seams.

Neither test runs a fit; both cover a place where getting it wrong produces a
result row that *looks* fine:

- ``_apply_inversion`` — sparse is **two different calls**, and calling the JAX
  one on a numba leg does not error. It produces a leg that is neither, and
  nothing downstream can tell.
- ``_stage_row`` — every disk-read field of a stage row. ``positions_info_present``
  and ``completed`` are the two the phase-4 witness reads, and a row that
  silently reports ``False`` for a stage that did write ``positions.info`` (or
  ``True`` for one that did not) is a ruling made on a lie.

``_runner`` must also be importable **without** importing autolens: the whole
point of parsing argv first is that the backend environment is set before the
modelling stack loads, which is impossible if importing the module loads it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam import _runner  # noqa: E402

from _inference_cli import variant_name  # noqa: E402

# ---------------------------------------------------------------------------
# Import graph
# ---------------------------------------------------------------------------


def test_importing_the_runner_does_not_import_autolens():
    """Run in a subprocess: the point is what a *fresh* interpreter loads.

    ``run_slam`` sets JAX_PLATFORMS / PYAUTO_DISABLE_JAX / NPROC and only then
    imports the stack. If importing ``slam._runner`` pulled autolens in at module
    level, every one of those exports would land after JAX had already read its
    environment, and a ``numba_cpu`` leg would quietly be a JAX leg.
    """
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts' / 'misc')!r})\n"
        "import slam._runner\n"
        "leaked = [m for m in ('autolens', 'autofit', 'jax', 'numba') if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "", (
        f"importing slam._runner pulled in {completed.stdout.strip()} — the backend "
        f"environment must be set before any of these load"
    )


# ---------------------------------------------------------------------------
# _apply_inversion
# ---------------------------------------------------------------------------


class _StubDataset:
    """Records which sparse call was made, and returns a distinct dataset for each."""

    def __init__(self, label="base"):
        self.label = label

    def apply_sparse_operator(self, batch_size: int = 128):
        return _StubDataset("jax")

    def apply_sparse_operator_cpu(self):
        return _StubDataset("numba")


def test_dense_is_a_no_op_on_every_backend():
    for backend in ("jax_cpu", "jax_gpu", "numba_cpu"):
        dataset = _StubDataset()
        assert _runner._apply_inversion(dataset, backend, "dense") is dataset


def test_sparse_dispatches_on_the_backend_family():
    assert _runner._apply_inversion(_StubDataset(), "jax_cpu", "sparse").label == "jax"
    assert _runner._apply_inversion(_StubDataset(), "jax_gpu", "sparse").label == "jax"
    assert _runner._apply_inversion(_StubDataset(), "numba_cpu", "sparse").label == "numba"


def test_unknown_inversion_is_refused():
    try:
        _runner._apply_inversion(_StubDataset(), "jax_cpu", "wtilde")
    except ValueError as exc:
        assert "wtilde" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown inversion must not be silently ignored")


# ---------------------------------------------------------------------------
# _stage_row
# ---------------------------------------------------------------------------


def _fake_search_dir(tmp_path: Path, *, completed=True, positions=True, info=None) -> Path:
    output_path = tmp_path / "mass_total[1]" / "abc123"
    (output_path / "files").mkdir(parents=True)
    (output_path / "files" / "samples_info.json").write_text(
        json.dumps(
            info
            if info is not None
            else {
                "log_evidence": -5342.5,
                "total_samples": 31415,
                "total_accepted_samples": 31415,
                "time": 1234.5,
                "number_live_points": 150,
            }
        )
    )
    if completed:
        (output_path / ".completed").write_text("")
    if positions:
        (output_path / "positions.info").write_text("positions summary\n")
    return output_path


def test_wall_s_is_a_number_even_though_autofit_writes_it_as_a_string(tmp_path):
    """``Timer.update`` writes ``str(time.time() - start)``, so ``samples_info["time"]``
    is a JSON string. A string in ``wall_s`` renders as an em dash in the README and
    raises in the stage plot — coerce it once, here."""
    output_path = _fake_search_dir(
        tmp_path,
        info={
            "log_evidence": "-5342.5",
            "total_samples": 31415,
            "time": "1234.5",
            "number_live_points": 150,
        },
    )
    row = _runner._stage_row(
        "mass_total[1]",
        output_path,
        n_live=150,
        n_batch=20,
        free_parameters=14,
        total_wall_s=1300.0,
        compile_s=None,
        resumed=False,
        inversion_applies=True,
    )
    assert row["wall_s"] == 1234.5
    assert isinstance(row["wall_s"], float)
    assert row["log_evidence"] == -5342.5
    assert isinstance(row["log_evidence"], float)


def test_stage_row_reads_every_disk_field(tmp_path):
    output_path = _fake_search_dir(tmp_path)

    row = _runner._stage_row(
        "mass_total[1]",
        output_path,
        n_live=150,
        n_batch=20,
        free_parameters=14,
        total_wall_s=1300.0,
        compile_s=42.0,
        resumed=False,
        inversion_applies=True,
    )

    assert row["name"] == "mass_total[1]"
    assert row["wall_s"] == 1234.5
    assert row["likelihood_evals"] == 31415
    assert row["log_evidence"] == -5342.5
    assert row["completed"] is True
    assert row["positions_info_present"] is True
    assert row["resumed"] is False
    assert row["total_wall_s"] == 1300.0
    assert row["compile_s"] == 42.0
    assert row["free_parameters"] == 14
    assert row["inversion_applies"] is True


def test_stage_row_never_invents_a_log_evidence_error(tmp_path):
    row = _runner._stage_row(
        "mass_total[1]",
        _fake_search_dir(tmp_path),
        n_live=150,
        n_batch=20,
        free_parameters=14,
        total_wall_s=1.0,
        compile_s=None,
        resumed=False,
        inversion_applies=True,
    )
    assert row["log_evidence_err"] is None
    assert "log_z_err" in row["log_evidence_err_note"]


def test_stage_row_reports_a_missing_positions_info(tmp_path):
    row = _runner._stage_row(
        "source_pix[2]",
        _fake_search_dir(tmp_path, positions=False, completed=False),
        n_live=75,
        n_batch=20,
        free_parameters=8,
        total_wall_s=1.0,
        compile_s=None,
        resumed=True,
        inversion_applies=True,
    )
    assert row["positions_info_present"] is False
    assert row["completed"] is False
    assert row["resumed"] is True


def test_stage_row_survives_a_search_dir_that_never_appeared():
    row = _runner._stage_row(
        "light[1]",
        None,
        n_live=150,
        n_batch=20,
        free_parameters=None,
        total_wall_s=None,
        compile_s=None,
        resumed=False,
        inversion_applies=True,
    )
    assert row["wall_s"] is None
    assert row["likelihood_evals"] is None
    assert row["completed"] is False


# ---------------------------------------------------------------------------
# Resume marker + config-name grammar
# ---------------------------------------------------------------------------


def test_checkpoint_is_detected_before_it_is_deleted(tmp_path):
    assert _runner.checkpoint_exists(tmp_path, "mass_total[1]") is False
    checkpoint = tmp_path / "mass_total[1]" / "abc123" / "files" / "search_internal"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.hdf5").write_text("")
    assert _runner.checkpoint_exists(tmp_path, "mass_total[1]") is True


def _cli(**overrides):
    from _inference_cli import InferenceCLI

    fields = {
        "config_name": "local_jax_cpu_dense_fp64",
        "output_dir": None,
        "use_mixed_precision": False,
        "instrument": "hst",
        "backend": "jax_cpu",
        "inversion": "dense",
        "seed": 0,
        "use_sparse_operator": False,
        "rect_mesh": "bilinear",
        "mesh": "rect",
        "mesh_pixels": 28,
        "regularization": None,
        "memo": "off",
        "cores": 4,
        "stages": None,
    }
    fields.update(overrides)
    return InferenceCLI(**fields)


def test_a_config_name_that_agrees_with_the_flags_is_accepted():
    assert _runner.validate_config_name(_cli()) is None


def test_a_config_name_that_lies_about_the_backend_is_refused():
    error = _runner.validate_config_name(_cli(backend="numba_cpu"))
    assert error is not None and "backend" in error


def test_a_config_name_that_lies_about_the_inversion_is_refused():
    error = _runner.validate_config_name(_cli(inversion="sparse"))
    assert error is not None and "inversion" in error


def test_a_config_name_that_lies_about_the_precision_is_refused():
    error = _runner.validate_config_name(_cli(use_mixed_precision=True))
    assert error is not None and "precision" in error


def test_a_malformed_config_name_is_refused():
    assert "does not match the grammar" in _runner.validate_config_name(_cli(config_name="nope"))


def test_a_missing_config_name_is_refused():
    assert "required" in _runner.validate_config_name(_cli(config_name=None))


# ---------------------------------------------------------------------------
# Run variant: the name, the two paths and the target id
# ---------------------------------------------------------------------------


def test_the_workspace_default_mesh_is_the_base_variant():
    """28x28 rect is `slam_base`, and nothing else is.

    The four A100 rows already on disk were written under that name and carry
    the bare `hst/slam5/seed<n>` target; a rename would orphan them.
    """
    assert variant_name(_cli()) == "slam_base"
    assert variant_name(_cli(mesh_pixels=40)) == "rect_40"
    assert variant_name(_cli(mesh="delaunay", mesh_pixels=1250)) == "delaunay_1250"


def test_the_variant_is_a_directory_level_in_the_results_path(tmp_path):
    json_path, png_path = _runner.results_paths(
        tmp_path, "imaging", "hst", "slam_base", "hpc_a100_jax_gpu_dense_fp64", 0
    )
    assert json_path.relative_to(tmp_path) == Path(
        "results/slam/imaging/hst/slam_base/hpc_a100_jax_gpu_dense_fp64/stages_seed0.json"
    )
    assert png_path.name == "stages_seed0.png"

    delaunay, _ = _runner.results_paths(
        tmp_path, "imaging", "hst", "delaunay_1250", "hpc_a100_jax_gpu_dense_fp64", 1
    )
    assert delaunay.relative_to(tmp_path) == Path(
        "results/slam/imaging/hst/delaunay_1250/hpc_a100_jax_gpu_dense_fp64/stages_seed1.json"
    )


def test_the_variant_is_a_directory_level_in_the_output_tree():
    """Neither `config_name` nor the mesh is an autofit identifier field, so two
    legs differing only in those must be kept apart by the path or they resume
    each other's fit."""
    assert _runner.output_path_prefix(
        "imaging", "hst", "slam_base", "local_jax_cpu_dense_fp64", 0
    ) == Path("slam/imaging/hst/slam_base/local_jax_cpu_dense_fp64/seed_0")
    assert _runner.output_path_prefix(
        "imaging", "hst", "delaunay_1250", "local_jax_cpu_dense_fp64", 0
    ) == Path("slam/imaging/hst/delaunay_1250/local_jax_cpu_dense_fp64/seed_0")


def test_only_the_base_variant_keeps_the_bare_target_id():
    """The dashboard groups legs into a parity row by `target`. A second
    experiment on this cell must not land in the base run's group claiming to be
    the same run under a different backend."""
    assert _runner.target_id("hst", "slam_base", 0) == "hst/slam5/seed0"
    assert _runner.target_id("hst", "delaunay_1250", 0) == "hst/slam5_delaunay_1250/seed0"
    assert _runner.target_id("hst", "rect_40", 1) == "hst/slam5_rect_40/seed1"
    assert _runner.target_id("euclid", "delaunay_1250", 2) == "euclid/slam5_delaunay_1250/seed2"


# ---------------------------------------------------------------------------
# Mesh models: which pixelization and which regularization each stage gets
# ---------------------------------------------------------------------------


class _StubModel:
    """Stands in for ``af.Model`` — records the class and kwargs it was built from."""

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs


class _StubAf:
    Model = _StubModel


class _StubMesh:
    class Delaunay:
        """Constructed directly — the Delaunay mesh is an instance, not an ``af.Model``."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class RectangularBilinearAdaptDensity:
        pass

    class RectangularBilinearAdaptImage:
        pass

    class RectangularRTUAdaptDensity:
        pass

    class RectangularRTUAdaptImage:
        pass


class _StubReg:
    class Adapt:
        pass

    class AdaptSplit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class _StubAl:
    """The modelling namespace as far as ``mesh_models`` is concerned.

    Stubbed rather than imported: ``_runner`` exists to set the backend
    environment before autolens loads, and a unit test that imports the stack
    would be testing a different process from the one the driver runs.
    """

    mesh = _StubMesh
    reg = _StubReg


def test_the_rect_branch_is_the_workspace_default_and_is_unchanged():
    meshes = _runner.mesh_models(_StubAf, _StubAl, _cli())

    assert meshes.mesh_init.cls is _StubMesh.RectangularBilinearAdaptDensity
    assert meshes.mesh_init.kwargs == {"shape": (28, 28)}
    assert meshes.mesh.cls is _StubMesh.RectangularBilinearAdaptImage
    assert meshes.mesh.kwargs == {"shape": (28, 28)}
    assert meshes.regularization is _StubReg.Adapt
    assert meshes.scheme == "adapt"
    assert meshes.shape == (28, 28)
    # A rectangular row has no areas_factor, no image mesh and no edge ring to
    # record, and must not invent any of them.
    assert meshes.areas_factor is None
    assert meshes.image_mesh is None
    assert (meshes.weight_power, meshes.weight_floor) == (None, None)
    assert (meshes.edge_points, meshes.zeroed_pixels, meshes.pixels) == (None, None, None)


def test_the_rect_branch_still_honours_rect_mesh_and_mesh_pixels():
    meshes = _runner.mesh_models(_StubAf, _StubAl, _cli(rect_mesh="rtu", mesh_pixels=40))
    assert meshes.mesh_init.cls is _StubMesh.RectangularRTUAdaptDensity
    assert meshes.mesh.cls is _StubMesh.RectangularRTUAdaptImage
    assert meshes.shape == (40, 40)
    assert meshes.regularization is _StubReg.Adapt


def test_the_delaunay_mesh_is_an_instance_not_a_model():
    """An ``af.Model`` here asks PyAutoFit to prior every unpinned argument.

    ``al.mesh.Delaunay(pixels, zeroed_pixels, areas_factor)`` has no free
    parameters in this chain, and no ``config/priors`` tree in the stack defines
    one for ``areas_factor`` — so ``af.Model(al.mesh.Delaunay, pixels=...,
    zeroed_pixels=0)`` raises ``ConfigException: No prior config found for
    class: Delaunay ... areas_factor`` at the first ``search.fit`` of
    ``source_pix[1]``, which is exactly what the pre-submit smoke caught.
    Production passes the instance, and so does this.
    """
    cli = _cli(mesh="delaunay", mesh_pixels=1250)
    meshes = _runner.mesh_models(_StubAf, _StubAl, cli)

    for mesh in (meshes.mesh_init, meshes.mesh):
        assert isinstance(mesh, _StubMesh.Delaunay), "an af.Model would need a prior it has none of"
        assert mesh.kwargs == {"pixels": 1250, "zeroed_pixels": 30, "areas_factor": 0.5}
    # Two searches, two objects: sharing one would share its state.
    assert meshes.mesh_init is not meshes.mesh
    # The knobs are recorded rather than inherited, and the row carries them.
    assert meshes.areas_factor == _runner.MESH_AREAS_FACTOR == 0.5


def test_the_edge_ring_and_zeroed_pixels_are_one_number():
    """``Delaunay.total_pixels`` is ``pixels + zeroed_pixels``, and the mesh grid
    handed to the mapper is the Hilbert interior vertices plus the appended ring.
    The two counts are therefore the same number written twice; a reader who
    changes one must change the other, and this test is what says so."""
    meshes = _runner.mesh_models(_StubAf, _StubAl, _cli(mesh="delaunay", mesh_pixels=1250))
    assert meshes.edge_points == meshes.zeroed_pixels == _runner.MESH_EDGE_POINTS == 30
    assert meshes.mesh_init.kwargs["zeroed_pixels"] == meshes.edge_points
    assert meshes.mesh_init.kwargs["pixels"] + meshes.edge_points == 1280


def test_the_delaunay_recipe_is_the_production_one():
    """The Hilbert weights are production's (group SLaM), not the library defaults.

    ``weight_power`` / ``weight_floor`` default to 0.0 / 0.0, which draws a mesh
    that does not adapt to the source at all — a uniform mesh wearing the name of
    an adaptive one. The recipe is recorded in the row for the same reason.
    """
    meshes = _runner.mesh_models(_StubAf, _StubAl, _cli(mesh="delaunay", mesh_pixels=1250))
    assert meshes.image_mesh == "hilbert"
    assert meshes.pixels == 1250
    assert (meshes.weight_power, meshes.weight_floor) == (3.5, 0.01)


def test_the_delaunay_branch_pairs_delaunay_with_the_adaptsplit_class():
    """The regularization half of the pairing is load-bearing too.

    ``al.reg.Adapt`` — what the rectangular legs run under — takes its
    neighbours from a ``scipy.spatial.Delaunay`` call on the traced source grid
    and raises ``TracerArrayConversionError`` under jit on this mesh family. And
    it must be the **class**: the stage helpers build
    ``af.Model(al.Pixelization, regularization=...)``, so a class leaves the
    coefficients free exactly as ``Adapt`` does on the rectangular legs, while
    ``delaunay_regularization()``'s fixed instance would silently drop free
    parameters and make the two runs incomparable.
    """
    cli = _cli(mesh="delaunay", mesh_pixels=1250)
    meshes = _runner.mesh_models(_StubAf, _StubAl, cli)

    assert meshes.regularization is _StubReg.AdaptSplit
    assert meshes.regularization is not _StubReg.Adapt
    assert isinstance(meshes.regularization, type), (
        "the regularization must be the class, not a fixed-coefficient instance — "
        "an instance drops the coefficients from the model"
    )
    assert meshes.scheme == "adapt_split"
    # A vertex count is not a shape, and the row must not invent one.
    assert meshes.shape is None


def test_all_five_slam_stage_models_carry_the_expected_flat_field(monkeypatch):
    """Exercise the real model builders without running a likelihood or search."""
    import autofit as af
    import autolens as al

    monkeypatch.setattr(al, "AnalysisImaging", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        al.model_util,
        "mge_model_from",
        lambda **kwargs: af.Model(al.lp.Sersic),
    )
    monkeypatch.setattr(af, "Nautilus", lambda **kwargs: kwargs)
    monkeypatch.setattr(_runner, "_adapt_images", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        al.util.chaining,
        "source_custom_model_from",
        lambda **kwargs: af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge=af.Model(al.lp.Sersic),
        ),
    )
    monkeypatch.setattr(
        al.util.chaining,
        "source_from",
        lambda **kwargs: af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge=af.Model(al.lp.Sersic),
        ),
    )

    def result(model):
        return SimpleNamespace(
            model=model,
            instance=model.instance_from_prior_medians(),
        )

    settings_search = SimpleNamespace(search_dict={})
    dataset = SimpleNamespace(pixel_scales=(0.1,))
    common = dict(use_jax=False, seed=1)

    source_lp_model, _, _ = _runner.source_lp(
        af,
        al,
        settings_search,
        dataset,
        3.0,
        0.5,
        1.0,
        **common,
    )
    source_lp_result = result(source_lp_model)

    source_pix_1_model, _, _ = _runner.source_pix_1(
        af,
        al,
        settings_search,
        dataset,
        source_lp_result,
        af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=(3, 3)),
        al.reg.Adapt,
        meshes=None,
        mask_radius=3.0,
        positions_likelihood=object(),
        **common,
    )
    source_pix_1_result = result(source_pix_1_model)

    source_pix_2_model, _, _ = _runner.source_pix_2(
        af,
        al,
        settings_search,
        dataset,
        source_lp_result,
        source_pix_1_result,
        af.Model(al.mesh.RectangularBilinearAdaptImage, shape=(3, 3)),
        al.reg.Adapt,
        meshes=None,
        mask_radius=3.0,
        positions_likelihood=object(),
        **common,
    )
    source_pix_2_result = result(source_pix_2_model)

    light_model, _, _ = _runner.light_lp(
        af,
        al,
        settings_search,
        dataset,
        3.0,
        source_pix_2_result,
        source_pix_2_result,
        meshes=None,
        positions_likelihood=object(),
        **common,
    )
    light_result = result(light_model)

    mass_total_model, _, _ = _runner.mass_total(
        af,
        al,
        settings_search,
        dataset,
        source_pix_1_result,
        source_pix_2_result,
        light_result,
        meshes=None,
        mask_radius=3.0,
        positions_likelihood=object(),
        **common,
    )

    models = (
        source_lp_model,
        source_pix_1_model,
        source_pix_2_model,
        light_model,
        mass_total_model,
    )
    free_field = (True, True, False, False, True)

    for model, is_free in zip(models, free_field, strict=True):
        assert not hasattr(model.galaxies.lens, "shear")
        assert isinstance(model.instance_from_prior_medians().fields, al.MassField)
        assert (("fields", "shear", "gamma_1") in model.unique_prior_paths) is is_free


# ---------------------------------------------------------------------------
# Backend environment
# ---------------------------------------------------------------------------


def test_backend_env_is_what_each_backend_needs(monkeypatch):
    for key in ("PYAUTO_DISABLE_JAX", "JAX_PLATFORMS", "NPROC", "JAX_ENABLE_X64"):
        monkeypatch.delenv(key, raising=False)

    assert _runner.set_backend_env("numba_cpu", 8) == {
        "JAX_ENABLE_X64": "True",
        "PYAUTO_DISABLE_JAX": "1",
    }
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)

    assert _runner.set_backend_env("jax_cpu", 8) == {
        "JAX_ENABLE_X64": "True",
        "JAX_PLATFORMS": "cpu",
        "NPROC": "8",
    }

    gpu = _runner.set_backend_env("jax_gpu", 4)
    assert gpu["JAX_PLATFORMS"] == "cuda,cpu"
    assert gpu["NPROC"] == "4"
    assert gpu["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


# ---------------------------------------------------------------------------
# Posterior keys + truth deltas
# ---------------------------------------------------------------------------


def test_unique_leaf_names_become_short_keys_and_repeats_stay_dotted():
    names = [
        ("galaxies", "lens", "mass", "einstein_radius"),
        ("galaxies", "lens", "mass", "centre", "centre_0"),
        ("galaxies", "lens", "bulge", "centre", "centre_0"),
        ("galaxies", "lens", "shear", "gamma_1"),
    ]
    assert _runner.posterior_keys(names) == [
        "einstein_radius",
        "galaxies.lens.mass.centre.centre_0",
        "galaxies.lens.bulge.centre.centre_0",
        "gamma_1",
    ]


def test_autofit_alias_tuples_resolve_to_their_leaf():
    """``all_names`` yields a tuple of *alias strings* per prior, not path segments.

    Indexing that tuple gives another full dotted name, so the naive
    ``name[-1]`` made every key a dotted path — which meant no `einstein_radius`
    row in the parity view, no `shear_magnitude` (its gamma_1/gamma_2 columns
    were dotted and never matched), and an empty `truth_delta_sigma`. Caught by
    the first real witness run, 2026-09-11.
    """
    names = [
        ("galaxies.lens.mass.einstein_radius",),
        ("galaxies.lens.mass.centre_0",),
        ("galaxies.lens.bulge.centre_0",),
        ("fields.shear.gamma_1",),
    ]
    assert _runner.posterior_keys(names) == [
        "einstein_radius",
        "galaxies.lens.mass.centre_0",
        "galaxies.lens.bulge.centre_0",
        "gamma_1",
    ]


def test_a_prior_with_several_aliases_takes_the_first():
    assert _runner.posterior_path(("a.b.slope", "z.slope")) == "a.b.slope"


def test_truth_delta_sigma_is_a_key_intersection():
    posterior = {
        "einstein_radius": {"median": 1.62, "sigma": 0.02},
        "slope": {"median": 2.10, "sigma": 0.05},
        "nuisance": {"median": 1.0, "sigma": 0.1},
    }
    truths = {"einstein_radius": 1.60, "slope": 2.00, "unfitted": 3.0}
    deltas = _runner.truth_delta_sigma_from(posterior, truths)
    assert set(deltas) == {"einstein_radius", "slope"}
    assert deltas["einstein_radius"] == pytest.approx(1.0)
    assert deltas["slope"] == pytest.approx(2.0)


def test_a_zero_width_posterior_records_none_rather_than_vanishing():
    """A collapsed posterior is an outcome, not an absence.

    Dropping the key would make "σ was zero" indistinguishable from "the truth
    file had no such parameter" — and a resumed test-mode fit produces exactly
    this, with every sample on one point.
    """
    deltas = _runner.truth_delta_sigma_from(
        {"einstein_radius": {"median": 2.07, "sigma": 0.0}}, {"einstein_radius": 1.60}
    )
    assert deltas == {"einstein_radius": None}
