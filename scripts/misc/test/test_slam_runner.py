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

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slam import _runner  # noqa: E402

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
        ("galaxies.lens.shear.gamma_1",),
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
