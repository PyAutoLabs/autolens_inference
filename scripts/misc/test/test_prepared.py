"""Prepared identity failures and explicit sampler starts; no scientific fits."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]
from experiments import prepared, run  # noqa: E402


class Prior:
    def dict(self):
        return {"type": "Uniform", "lower_limit": 0, "upper_limit": 2}


class Model:
    path_priors_tuples = [(("mass",), Prior())]

    def dict(self):
        return {"mass": self.path_priors_tuples[0][1].dict()}


@pytest.fixture
def problem(tmp_path, monkeypatch):
    monkeypatch.setattr(prepared, "revision", lambda _: "a" * 40)
    return prepared.export_problem(
        tmp_path,
        Path("output/prepared/example"),
        Model(),
        SimpleNamespace(dataset={"data": [1, 2], "adapt": [3]}, positions=[4]),
        setup_id="imaging/rectangular/hst",
        baseline_record_id="baseline#mass",
        controls={"backend": "numba_cpu"},
        preparation_s=2,
        dependency_revisions={"PyAutoFit": "b" * 40},
    )


def test_roundtrip_shared_problem_and_priors(tmp_path, problem):
    doc, model, analysis, _ = prepared.load_problem(tmp_path, problem)
    assert model.dict() == Model().dict()
    assert analysis.dataset["adapt"] == [3]
    assert analysis.positions == [4]
    assert doc["baseline_evidence"] is None
    assert doc["preparation_s"] == 2


def test_tampered_artifact_rejected_before_loading(tmp_path, problem):
    doc = json.loads(problem.read_text())
    (tmp_path / doc["artifacts"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        prepared.load_problem(tmp_path, problem)


def test_identity_and_revision_failures(tmp_path, problem):
    with pytest.raises(ValueError, match="identity mismatch"):
        prepared.verify_manifest(tmp_path, problem, expected_problem="other")
    with pytest.raises(ValueError, match="revisions differ"):
        prepared.verify_manifest(tmp_path, problem, revisions={"PyAutoFit": "c" * 40})
    doc = json.loads(problem.read_text())
    doc["priors_id"] = "sha256:" + "f" * 64
    problem.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="priors_id mismatch"):
        prepared.verify_manifest(tmp_path, problem)


def test_missing_and_escaping_artifacts(tmp_path, problem):
    doc = json.loads(problem.read_text())
    (tmp_path / doc["artifacts"][0]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        prepared.verify_manifest(tmp_path, problem)
    with pytest.raises(ValueError, match="relative"):
        prepared.contained(tmp_path, "../outside")


def args(**kwargs):
    return SimpleNamespace(
        start="cold",
        sampler="emcee",
        run_id="fresh",
        cores=1,
        walkers=10,
        steps=20,
        chains=1,
        seed=0,
        jitter=0.05,
        warmup=10,
        particles=20,
        live=10,
        batch=5,
        **kwargs,
    )


class FakeAF:
    InitializerPrior = staticmethod(lambda: "prior")
    Emcee = staticmethod(lambda **kw: kw)
    Nautilus = staticmethod(lambda **kw: kw)


def test_cold_does_not_load_baseline_samples():
    search = run.make_search(
        FakeAF,
        Model(),
        {"controls": {"backend": "numba_cpu"}},
        {"posterior_samples": {"path": "missing"}},
        args(),
    )
    assert search["initializer"] == "prior"


def test_warm_missing_evidence_refused():
    option = args()
    option.start = "warm"
    with pytest.raises(ValueError, match="completed baseline"):
        run.make_search(FakeAF, Model(), {"controls": {"backend": "numba_cpu"}}, {}, option)


def test_gradient_backend_and_resume_capability():
    option = args()
    option.sampler = "nuts"
    with pytest.raises(ValueError, match="JAX analysis"):
        run.make_search(FakeAF, Model(), {"controls": {"backend": "numba_cpu"}}, {}, option)
    option.start = "resume"
    with pytest.raises(ValueError, match="not resumable"):
        run.make_search(FakeAF, Model(), {"controls": {"backend": "jax_cpu"}}, {}, option)


def test_warm_reuses_pinned_samples_without_replacing_priors(tmp_path, monkeypatch):
    import dill

    monkeypatch.setattr(run, "ROOT", tmp_path)
    sample_path = tmp_path / "baseline.dill"
    sample_path.write_bytes(dill.dumps({"posterior": [1, 2]}))

    class WarmAF(FakeAF):
        class InitializerParamStartPoints:
            @staticmethod
            def from_result(result, **kwargs):
                return dict(samples=result.samples, **kwargs)

    model = Model()
    before = model.dict()
    option = args()
    option.start = "warm"
    search = run.make_search(
        WarmAF,
        model,
        {"controls": {"backend": "numba_cpu"}},
        {"posterior_samples": {"path": "baseline.dill"}},
        option,
    )
    assert search["initializer"]["samples"] == {"posterior": [1, 2]}
    assert search["initializer"]["n_points"] == option.walkers
    assert model.dict() == before


def test_resume_preserves_source_and_copies_checkpoint(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "revision", lambda _: "a" * 40)
    state = tmp_path / "source.hdf"
    with h5py.File(state, "w") as output:
        output.create_dataset("mcmc/chain", shape=(100, 10, 1))
        output["mcmc"].attrs["iteration"] = 5
    before = state.read_bytes()
    source = {
        "problem_id": "p",
        "sampler": "emcee",
        "seed": 0,
        "experiment_options": {"walkers": 10},
        "checkpoint": {"path": "source.hdf", "sha256": prepared.digest(before)},
    }
    record = tmp_path / "source.json"
    record.write_text(json.dumps(source))
    options = args()
    options.resume_record = "source.json"
    options.sampler = "emcee"
    search = SimpleNamespace(
        paths=SimpleNamespace(model=None), backend_filename=tmp_path / "new/state.hdf"
    )
    source_artifact = run.resume_source(
        options, {"id": "p"}, search, SimpleNamespace(prior_count=1)
    )
    assert state.read_bytes() == before
    assert search.backend_filename.read_bytes() == before
    assert source_artifact["record_id"] == "source.json"
    assert (tmp_path / source_artifact["path"]).read_bytes() == before
    options.steps = 5
    with pytest.raises(ValueError, match="larger total step"):
        run.resume_source(options, {"id": "p"}, search, SimpleNamespace(prior_count=1))
    options.steps = 20
    state.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run.resume_source(options, {"id": "p"}, search, SimpleNamespace(prior_count=1))


def test_executable_snapshot_swap_cannot_keep_problem_id(tmp_path, problem):
    doc = json.loads(problem.read_text())
    snapshot = next(a for a in doc["artifacts"] if a["kind"] == "model_snapshot")
    replacement = b"different executable model with the same displayed priors"
    (tmp_path / snapshot["path"]).write_bytes(replacement)
    snapshot["sha256"] = prepared.digest(replacement)
    problem.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="scientific identity mismatch"):
        prepared.verify_manifest(tmp_path, problem)


@pytest.mark.parametrize(
    "key,value",
    [
        ("controls", {"backend": "jax_cpu"}),
        ("dependency_revisions", {"PyAutoFit": "f" * 40}),
        ("setup_id", "imaging/delaunay/hst"),
        ("stage", "light[1]"),
    ],
)
def test_controls_and_library_identity_cannot_change(tmp_path, problem, key, value):
    doc = json.loads(problem.read_text())
    doc[key] = value
    problem.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="scientific identity mismatch"):
        prepared.verify_manifest(tmp_path, problem)


def test_fit_clock_excludes_later_diagnostics(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])

    class Search:
        def fit(self, **kwargs):
            clock[0] += 2
            return "fit-result"

    row = {}
    assert run.timed_fit(Search(), Model(), object(), row) == "fit-result"
    clock[0] += 10  # later posterior/diagnostic extraction must not enter fit clock
    assert row["total_wall_s"] == 2

    class FailedSearch:
        def fit(self, **kwargs):
            clock[0] += 3
            raise RuntimeError("stopped")

    with pytest.raises(RuntimeError, match="stopped"):
        run.timed_fit(FailedSearch(), Model(), object(), row)
    assert row["total_wall_s"] == 3


def test_export_keeps_portable_metadata_without_bulk(tmp_path, problem):
    import shutil

    doc = json.loads(problem.read_text())
    tracked = tmp_path / "prepared" / doc["id"] / "manifest.json"
    assert json.loads(tracked.read_text()) == doc
    shutil.rmtree(tmp_path / "output")
    assert tracked.exists()
    assert all(
        a["path"].startswith("output/") for a in json.loads(tracked.read_text())["artifacts"]
    )


def test_library_revisions_follow_selected_import_sources(tmp_path, monkeypatch):
    import importlib.util

    modules = {
        "autofit": "fit/PyAutoFit",
        "autoarray": "array/PyAutoArray",
        "autogalaxy": "galaxy/PyAutoGalaxy",
        "autolens": "lens/PyAutoLens",
        "autonerves": "organs/PyAutoNerves",
    }
    origins = {name: tmp_path / path / name / "__init__.py" for name, path in modules.items()}
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(origins[name]))
    )
    calls = []

    def command(argv, cwd, **kwargs):
        calls.append((argv, Path(cwd)))
        if "--show-toplevel" in argv:
            return str(Path(cwd).parent) + "\n"
        if "status" in argv:
            return ""
        if "ls-files" in argv:
            return b"tracked"
        return "a" * 40 + "\n"

    monkeypatch.setattr(prepared.subprocess, "check_output", command)
    result = prepared.library_revisions(tmp_path / "lens/autolens_inference")
    assert set(result) == {"PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens", "PyAutoNerves"}
    assert any(cwd == tmp_path / "fit/PyAutoFit" for _, cwd in calls)
    assert not any(cwd == tmp_path / "lens/PyAutoFit" for _, cwd in calls)
