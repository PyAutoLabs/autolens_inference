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


def test_installed_release_null_revisions_dropped_with_reason(tmp_path):
    revisions = {"autofit": None, "autolens": None}
    put(
        tmp_path,
        {"schema_version": 3, "target": "lens", "library_revisions": revisions},
        "searches/seed0.json",
    )
    (row,) = feed(tmp_path)["records"]
    assert row["dependency_revisions"] == {}
    assert "installed release" in row["unknown_reasons"]["dependency_revisions"]


def _slam_row(config, variant="slam_base", host=None, seed=0, stages=None, **extra):
    row = {
        "schema_version": 3,
        "target": "hst/slam5/seed0",
        "dataset_class": "imaging",
        "instrument": "hst",
        "variant": variant,
        "mesh": "delaunay" if "delaunay" in variant else "rect",
        "config_name": config,
        "sampler": "nautilus",
        "seed": seed,
        "cores": 8,
        "host": host,
        "device": {"omp_num_threads": "1"},
        "stages": stages or [{"name": "mass_total[1]", "completed": True, "wall_s": 10.0}],
        **extra,
    }
    return row


def test_setup_fields_from_the_config_table(tmp_path):
    put(tmp_path, _slam_row("hpc_a100_jax_gpu_sparse_fp64"), "a/gpu.json")
    put(
        tmp_path,
        _slam_row("hpc_ral_numba_cpu_sparse_fp64", "slam_delaunay_light_fixed"),
        "a/n.json",
    )
    put(tmp_path, _slam_row("local_jax_cpu_dense_mp"), "a/local.json")
    rows = {r["id"]: r for r in feed(tmp_path)["records"] if r["parent_run_id"] is None}
    gpu = rows["results/a/gpu.json"]
    assert gpu["setup_mode"] == "explicit"
    assert gpu["setup_id"] == "jax_gpu_sparse_fp64" and gpu["host_class"] == "hpc_a100"
    assert gpu["setup_resolved"] == {
        "backend": "jax",
        "device": "gpu",
        "operator": "sparse",
        "precision": "fp64",
        "transformer": "na",
        "threads": 1,
    }
    assert gpu["likelihood"] == {"dataset_type": "imaging", "variant": "rectangular"}
    assert gpu["model_variant"] == "default"
    numba = rows["results/a/n.json"]
    assert numba["host_class"] == "hpc_ral_cpu" and numba["model_variant"] == "light_fixed"
    assert numba["likelihood"]["variant"] == "delaunay"
    local = rows["results/a/local.json"]
    assert local["setup_id"] == "jax_cpu_dense_mixed" and local["host_class"] == "local_laptop"


def test_cpu_rows_under_the_a100_prefix_relabel_only_on_host_proof(tmp_path):
    proven = _slam_row("hpc_a100_jax_cpu_dense_fp64", host="euclid-ral-compute-10-2")
    put(tmp_path, proven, "a/proven.json")
    put(tmp_path, _slam_row("hpc_a100_jax_cpu_dense_fp64", host="euclid-ral-gpu-2"), "a/gpu.json")
    put(tmp_path, _slam_row("hpc_a100_jax_cpu_dense_fp64"), "a/none.json")
    rows = {r["id"]: r for r in feed(tmp_path)["records"] if r["parent_run_id"] is None}
    assert rows["results/a/proven.json"]["host_class"] == "hpc_ral_cpu"
    assert "host_class" not in rows["results/a/proven.json"]["unknown_reasons"]
    for name in ("gpu", "none"):
        row = rows[f"results/a/{name}.json"]
        assert row["host_class"] == "hpc_a100" and "host_class" in row["unknown_reasons"]


def test_rows_without_a_config_name_say_why(tmp_path):
    put(tmp_path, {"schema_version": 1, "target": "lens"})
    (row,) = feed(tmp_path)["records"]
    assert row["setup_id"] is None and row["host_class"] is None
    assert row["setup_mode"] == "explicit"
    assert {"setup_id", "host_class", "likelihood"} <= set(row["unknown_reasons"])


def test_every_current_record_carries_the_setup_fields():
    for row in feed(ROOT)["records"]:
        assert row["setup_mode"] == "explicit"
        assert row["model_variant"] in ("default", "light_fixed", "delaunay_1250")
        if not row["archived"]:
            assert row["setup_id"] and row["host_class"] and row["likelihood"]
