"""Exchange witnesses: preserve adverse results without executing science."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "summary", ROOT / "scripts/misc/tooling/export_inference_summary.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def put(root, data, name="searches/example.json"):
    path = root / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def feed(root):
    return summary.build(root, "a" * 40, "2026-10-04T00:00:00+00:00")


def test_empty(tmp_path):
    result = feed(tmp_path)
    assert result["records"] == []
    assert result["coverage"]["expected"]["runs"] is None


def test_failed_seed_unknown_provenance_and_archive(tmp_path):
    for seed in range(3):
        put(
            tmp_path,
            {
                "schema_version": 1,
                "target": "lens",
                "seed": seed,
                "status": "failed: timeout" if seed == 2 else "complete",
            },
            f"searches/seed{seed}.json",
        )
    put(
        tmp_path,
        {"schema_version": 1, "status": "stopped_early"},
        "archive/September/searches/old.json",
    )
    rows = feed(tmp_path)["records"]
    assert len(rows) == 4
    assert any(r["execution"]["status"] == "failed: timeout" for r in rows)
    assert sum(r["archived"] for r in rows) == 1
    assert all(r["scientific"]["acceptance"] == "not_assessed" for r in rows)
    assert all(r["samples"]["availability"] == "unknown" for r in rows)
    assert all(r["diagnostics"]["status"] == "missing" for r in rows)
    assert all("dependency_revisions" in r["unknown_reasons"] for r in rows)


def test_partial_pipeline_no_summed_clocks(tmp_path):
    put(
        tmp_path,
        {
            "schema_version": 2,
            "status": "failed: later stage",
            "stages": [
                {"name": "one", "completed": True, "wall_s": 30, "total_wall_s": 40},
                {"name": "two", "completed": False, "wall_s": 10, "total_wall_s": 20},
            ],
        },
        "slam/chain.json",
    )
    rows = feed(tmp_path)["records"]
    assert len(rows) == 3
    assert rows[0]["timings"]["total_s"] is None
    assert rows[1]["parent_run_id"] == rows[0]["id"]
    assert rows[1]["execution"]["status"] == "complete"
    assert rows[2]["execution"]["status"] == "incomplete"
    assert rows[1]["timings"]["sampling_s"] == 30
    assert rows[1]["timings"]["setup_s"] is None


def test_changed_target_no_implied_comparison(tmp_path):
    for i, target in enumerate(("lensA", "lensB")):
        put(tmp_path, {"schema_version": 1, "target": target}, f"searches/{i}.json")
    rows = feed(tmp_path)["records"]
    assert rows[0]["target"] != rows[1]["target"]
    assert all(r["comparison"]["group"] is None for r in rows)


def test_invalid_results_visible(tmp_path):
    for i, extra in enumerate(
        (
            {"schema_version": 99},
            {"wall_s": float("nan")},
            {"wall_s": -2},
            {"stages": [{"name": "a"}, {"name": "a"}]},
        )
    ):
        put(tmp_path, {"schema_version": 1, **extra}, f"searches/{i}.json")
    result = feed(tmp_path)
    assert not result["records"]
    assert len(result["coverage"]["excluded"]) == 4


def test_safe_paths_missing_archives(tmp_path):
    for i, path in enumerate(
        ("/mnt/private/archive", "../escape", "searches/seed_1", "https://example.com")
    ):
        put(tmp_path, {"schema_version": 1, "output_path": path}, f"searches/{i}.json")
    rows = feed(tmp_path)["records"]
    assert [r["samples"]["path"] for r in rows] == [None, None, "output/searches/seed_1", None]
    assert all(r["samples"]["availability"] == "unknown" for r in rows)


def test_reproducible_actual_results():
    first = feed(ROOT)
    assert first == feed(ROOT)
    assert first["records"]
    assert not first["coverage"]["excluded"]


def test_utc_flat_counts_and_unknown_completion(tmp_path):
    put(tmp_path, {"schema_version": 1, "status": "failed: timeout"})
    result = summary.build(tmp_path, "a" * 40, "2026-10-04T02:00:00+02:00")
    assert result["generated_at"] == "2026-10-04T00:00:00Z"
    assert all(type(value) is int for value in result["coverage"]["observed"].values())
    assert result["coverage"]["observed"]["execution_failed"] == 1
    row = result["records"][0]
    assert row["execution"]["completed"] is None
    assert "execution.completed" in row["unknown_reasons"]


def test_pipeline_clock_definitions_are_explicit(tmp_path):
    put(tmp_path, {"schema_version": 2, "stages": []}, "slam/empty.json")
    row = feed(tmp_path)["records"][0]
    assert all(
        "aggregate not measured" in value for value in row["timings"]["definitions"].values()
    )
    assert "pipeline-wide" in row["unknown_reasons"]["execution.completed"]
