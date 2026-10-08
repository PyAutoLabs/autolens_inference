"""Catalogue migration witnesses without scientific execution."""

from .test_inference_summary import ROOT, feed, put, summary


def test_real_migration_preserves_records_and_archive():
    old = feed(ROOT)
    new = summary.build_catalogue(old, ROOT)
    assert new["version"] == 2
    assert [
        (r["id"], r["parent_run_id"], r["archived"], r["evidence_paths"]) for r in new["records"]
    ] == [(r["id"], r["parent_run_id"], r["archived"], r["evidence_paths"]) for r in old["records"]]
    assert {r["setup_id"] for r in new["records"] if r["setup_id"]} == {
        "imaging/rectangular/hst",
        "imaging/delaunay/hst",
        "point_source/simple/simple",
    }
    assert not new["prepared_problems"]
    assert all(r["initialization"]["mode"] == "unknown" for r in new["records"])
    assert all(s["reference_record_id"] is None for s in new["setups"])


def test_unmapped_history_has_no_fabricated_problem(tmp_path):
    put(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_class": "imaging",
            "instrument": "hst",
            "variant": "unidentified",
        },
    )
    new = summary.build_catalogue(feed(tmp_path), tmp_path)
    row = new["records"][0]
    assert row["setup_id"] is None and row["setup_id_reason"]
    assert row["problem_id"] is None and row["problem_id_reason"]
    assert row["timings"]["preparation_s"] is None
    assert row["work"]["gradient_evaluations"] is None


def test_bad_manifest_visible(tmp_path):
    import json

    p = tmp_path / "prepared/problem/manifest.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {"schema": "prepared-inference-problem", "version": 1, "dataset_id": "fabricated"}
        )
    )
    new = summary.build_catalogue(feed(tmp_path), tmp_path)
    assert not new["prepared_problems"]
    assert new["coverage"]["excluded"][0]["path"] == "prepared/problem/manifest.json"


def test_prepared_baseline_binds_exact_identity(tmp_path):
    import hashlib
    import json

    name = "slam/imaging/hst/slam_base/local/stages_seed0.json"
    put(
        tmp_path,
        {
            "schema_version": 2,
            "dataset_class": "imaging",
            "instrument": "hst",
            "variant": "slam_base",
            "stages": [{"name": "mass_total[1]", "completed": True}],
        },
        name,
    )
    rid = "results/" + name + "#stage-" + hashlib.sha256(b"mass_total[1]").hexdigest()[:16]
    manifest = dict(
        schema="prepared-inference-problem",
        version=1,
        id="prepared-" + "c" * 64,
        setup_id="imaging/rectangular/hst",
        dataset_id="sha256:" + "a" * 64,
        model_id="sha256:" + "b" * 64,
        priors_id="sha256:" + "c" * 64,
        stage="mass_total[1]",
        baseline_record_id=rid,
        artifacts=[],
        baseline_evidence={"max_log_likelihood": 12.0, "max_likelihood_parameters": {"x": 1.0}},
    )
    path = tmp_path / "prepared/example/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    doc = summary.build_catalogue(feed(tmp_path), tmp_path)
    assert len(doc["prepared_problems"]) == 1
    stage = next(r for r in doc["records"] if r["id"] == rid)
    assert stage["dataset"]["id"] == manifest["dataset_id"]
    assert stage["problem_id"] == manifest["id"]
    assert stage["scientific"]["acceptance"] == "not_assessed"
    assert stage["initialization"]["mode"] == "unknown"
    assert stage["diagnostics"]["values"]["max_log_likelihood"] == 12.0


def test_new_clock_labels_do_not_rewrite_historical_records(tmp_path):
    put(tmp_path, {"schema_version": 2, "total_wall_s": 1}, "searches/a.json")
    put(
        tmp_path,
        {
            "schema_version": 2,
            "total_wall_s": 2,
            "timing_definitions": {"total_s": "Experiment segment fit only"},
        },
        "searches/b.json",
    )
    old = feed(tmp_path)
    original = old["records"][0]["timings"]["definitions"]["total_s"]
    new = summary.build_catalogue(old, tmp_path)
    assert new["records"][0]["timings"]["definitions"]["total_s"] == original
    assert new["records"][1]["timings"]["definitions"]["total_s"] == "Experiment segment fit only"
    assert old["records"][1]["timings"]["definitions"]["total_s"] == original
    assert summary.DEFINITIONS["total_s"] == original


def test_portable_export_survives_missing_local_output(tmp_path, monkeypatch):
    import hashlib
    import shutil
    from pathlib import Path
    from types import SimpleNamespace

    from experiments import prepared

    from .test_prepared import Model

    monkeypatch.setattr(prepared, "revision", lambda _: "a" * 40)
    name = "slam/imaging/hst/slam_base/local/stages_seed0.json"
    rid = "results/" + name + "#stage-" + hashlib.sha256(b"mass_total[1]").hexdigest()[:16]
    put(
        tmp_path,
        {
            "schema_version": 2,
            "dataset_class": "imaging",
            "instrument": "hst",
            "variant": "slam_base",
            "stages": [{"name": "mass_total[1]", "completed": True}],
        },
        name,
    )
    local = prepared.export_problem(
        tmp_path,
        Path("output/prepared/test"),
        Model(),
        SimpleNamespace(dataset={"data": [1]}),
        setup_id="imaging/rectangular/hst",
        baseline_record_id=rid,
        controls={"backend": "numba_cpu"},
        preparation_s=1,
        dependency_revisions={"PyAutoFit": "a" * 40},
    )
    identity = prepared.verify_manifest(tmp_path, local)["id"]
    with_local = summary.build_catalogue(feed(tmp_path), tmp_path)
    assert len(with_local["prepared_problems"]) == 1
    assert not with_local["coverage"]["excluded"]
    shutil.rmtree(tmp_path / "output")
    portable = summary.build_catalogue(feed(tmp_path), tmp_path)
    assert portable["prepared_problems"][0]["id"] == identity
    assert next(r for r in portable["records"] if r["id"] == rid)["problem_id"] == identity
    assert not portable["coverage"]["excluded"]
