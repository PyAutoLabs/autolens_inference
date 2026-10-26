"""walltime-validation@1: full-fit records, producer-computed numbers only."""

import json
import os
import sys
from pathlib import Path

import pytest

from .test_inference_summary import ROOT, feed, put, summary

# The validator lives in PyAutoPulse: grouped (organs/) or flat task-bundle layout,
# or $PYAUTO_PULSE. Absent, the validator test skips and the shape tests still run.
PULSE = next(
    (
        p
        for p in (
            Path(os.environ.get("PYAUTO_PULSE", "/nonexistent")),
            ROOT.parent / "PyAutoPulse",
            ROOT.parent.parent / "organs" / "PyAutoPulse",
        )
        if (p / "pulse" / "walltime.py").exists()
    ),
    ROOT / "absent",
)


def _walltime(root, references=(), backfill=None):
    return summary.build_walltime(
        root, feed(root), "a" * 40, references=list(references), backfill=backfill or {}
    )


def _row(config, variant="slam_base", seed=0, stages=None, job="1_0"):
    return {
        "schema_version": 3,
        "dataset_class": "imaging",
        "instrument": "hst",
        "variant": variant,
        "mesh": "rect",
        "config_name": config,
        "sampler": "nautilus",
        "seed": seed,
        "cores": 4,
        "release_pin": "2026.10.17.1",
        "slurm_job_id": job,
        "stages": stages,
    }


def _stage(name, wall, evals, **extra):
    return {"name": name, "completed": True, "wall_s": wall, "likelihood_evals": evals, **extra}


def _put_validation(root, row, model_variant="default"):
    setup = row["config_name"].split("_", 2)[2] if row["config_name"].startswith("hpc_") else ""
    name = (
        f"imaging/rectangular/hst/validation/{model_variant}/{setup}/seed_{row['seed']}/stages.json"
    )
    put(root, row, name)


def test_one_record_per_stage_with_declared_numbers(tmp_path):
    stages = [_stage("source_lp[1]", 100.0, 1000), _stage("mass_total[1]", 50.0, 500)]
    _put_validation(tmp_path, _row("hpc_a100_jax_gpu_dense_fp64", stages=stages))
    doc = _walltime(tmp_path)
    assert doc["schema"] == "walltime-validation" and doc["version"] == 1
    records = {r["stage"]: r for r in doc["records"]}
    assert set(records) == {"source_lp[1]", "mass_total[1]"}
    mass = records["mass_total[1]"]
    assert mass["sampler"] == "nautilus:mass_total[1]"
    assert mass["implied_ms_per_eval"] == pytest.approx(100.0)
    assert mass["setup_id"] == "jax_gpu_dense_fp64" and mass["host_class"] == "hpc_a100"
    assert mass["reference_runtime_record_id"] is None and mass["predicted_ms_per_eval"] is None
    assert mass["parallel_efficiency"] is None and mass["parity_status"] == "not_assessed"
    assert mass["cores"] == 4 and mass["slurm_job_id"] == "1_0"
    assert mass["profiling_record_id"] is None and mass["size_match"] is None
    assert mass["implied_s_per_eval"] == pytest.approx(0.1)
    assert any("0 of 2 records pair with a profiling" in x for x in doc["limitations"])


def test_shared_and_duplicate_stages_are_not_re_counted(tmp_path):
    shared = _stage("source_lp[1]", 100.0, 1000, shared_from_config="hpc_a100_jax_cpu_dense_fp64")
    common = _stage("light[1]", 20.0, 200)
    default = _row(
        "hpc_ral_numba_cpu_sparse_fp64", stages=[shared, common, _stage("mass_total[1]", 9.0, 90)]
    )
    fixed = _row(
        "hpc_ral_numba_cpu_sparse_fp64",
        variant="slam_base_light_fixed",
        stages=[shared, common, _stage("mass_total[1]", 7.0, 70)],
    )
    _put_validation(tmp_path, default)
    _put_validation(tmp_path, fixed, "light_fixed")
    doc = _walltime(tmp_path)
    got = sorted((r["model_variant"], r["stage"]) for r in doc["records"])
    assert got == [
        ("default", "light[1]"),
        ("default", "mass_total[1]"),
        ("light_fixed", "mass_total[1]"),
    ]
    assert any("--stages-from" in x for x in doc["limitations"])
    assert any("listed once" in x for x in doc["limitations"])


def test_declared_divergence_is_the_only_non_default_parity(tmp_path):
    stages = [_stage("light[1]", 20.0, 200), _stage("mass_total[1]", 9.0, 90)]
    _put_validation(
        tmp_path,
        _row("hpc_a100_jax_gpu_sparse_fp64", variant="slam_base_light_fixed", stages=stages),
        "light_fixed",
    )
    statuses = {r["stage"]: r["parity_status"] for r in _walltime(tmp_path)["records"]}
    assert statuses == {"light[1]": "not_assessed", "mass_total[1]": "diverged"}


def _reference(id, **extra):
    return {
        "id": id,
        "likelihood": {"dataset_type": "imaging", "variant": "rectangular"},
        "instrument": "hst",
        "setup_id": "jax_gpu_dense_fp64",
        "host_class": "hpc_a100",
        "model_variant": "default",
        "status": "current",
        "measured_at": "2026-10-25T00:00:00Z",
        "regime": "standard",
        "source_pixels": 1225,
        "per_call_s": 0.05,
        "per_call_source": "vmap.per_call",
        **extra,
    }


def test_a_profiling_record_matches_only_on_the_exact_key(tmp_path):
    row = _row("hpc_a100_jax_gpu_dense_fp64", stages=[_stage("mass_total[1]", 50.0, 500)])
    row["mesh_shape"] = [35, 35]
    _put_validation(tmp_path, row)
    other_host = _reference("lifecycle/host", host_class="local_gpu")
    other_setup = _reference("lifecycle/setup", setup_id="jax_gpu_sparse_fp64")
    (record,) = _walltime(tmp_path, [other_host, other_setup])["records"]
    assert record["profiling_record_id"] is None
    assert record["profiling_unmatched_reason"] == (
        "no profiling runtime record with host_class hpc_a100"
    )
    exact = _reference("lifecycle/exact")
    (record,) = _walltime(tmp_path, [other_host, other_setup, exact])["records"]
    assert record["profiling_record_id"] == "lifecycle/exact"
    assert record["reference_runtime_record_id"] == "lifecycle/exact"
    assert record["profiling_per_call_s"] == pytest.approx(0.05)
    assert record["predicted_ms_per_eval"] == pytest.approx(50.0)
    assert record["ratio_measured_over_predicted"] == pytest.approx(2.0)
    assert record["profiling_status"] == "current" and record["profiling_unmatched_reason"] is None
    assert record["source_pixels"] == 1225 and record["size_match"] is True


def test_current_is_preferred_else_the_newest_and_size_is_compared(tmp_path):
    _put_validation(
        tmp_path, _row("hpc_a100_jax_gpu_dense_fp64", stages=[_stage("mass_total[1]", 50.0, 500)])
    )
    old = _reference("lifecycle/old", status="superseded", measured_at="2026-09-01T00:00:00Z")
    new = _reference("lifecycle/new", status="superseded", measured_at="2026-09-10T00:00:00Z")
    (record,) = _walltime(tmp_path, [old, new])["records"]
    assert record["profiling_record_id"] == "lifecycle/new"
    assert record["profiling_status"] == "superseded"
    # The row records no mesh_shape: the run's size is unknown, so no size verdict.
    assert record["source_pixels"] is None and record["size_match"] is None
    current = _reference("lifecycle/current", measured_at="2026-08-01T00:00:00Z")
    (record,) = _walltime(tmp_path, [old, new, current])["records"]
    assert record["profiling_record_id"] == "lifecycle/current"


def test_search_summary_timings_come_from_the_row_else_the_backfill(tmp_path):
    own = {"log_likelihood_eval_s": 0.4, "time_per_sample_s": 1.9, "path": "x"}
    stages = [
        _stage("source_lp[1]", 100.0, 1000, search_summary=own),
        _stage("mass_total[1]", 50.0, 500),
        _stage("light[1]", 20.0, 200),
    ]
    row = _row("hpc_a100_jax_gpu_dense_fp64", stages=stages)
    row["mesh_shape"] = [28, 28]
    _put_validation(tmp_path, row)
    path = (
        "results/imaging/rectangular/hst/validation/default/jax_gpu_dense_fp64/seed_0/stages.json"
    )
    backfill = {
        path: {
            "source_lp[1]": {"log_likelihood_eval_s": 9.0, "time_per_sample_s": 9.0},
            "mass_total[1]": {"log_likelihood_eval_s": 0.3, "time_per_sample_s": 0.1},
        }
    }
    doc = _walltime(tmp_path, [_reference("lifecycle/exact")], backfill)
    records = {r["stage"]: r for r in doc["records"]}
    assert records["source_lp[1]"]["likelihood_eval_s_search_summary"] == 0.4
    assert records["source_lp[1]"]["search_summary_source"] == "row"
    assert records["mass_total[1]"]["time_per_sample_s_search_summary"] == 0.1
    assert records["mass_total[1]"]["search_summary_source"] == "backfill"
    assert records["light[1]"]["likelihood_eval_s_search_summary"] is None
    assert records["light[1]"]["search_summary_source"] is None
    assert records["mass_total[1]"]["size_match"] is False
    assert any("2 of 3 records carry search.summary" in x for x in doc["limitations"])
    assert any("0 match, 3 differ" in x for x in doc["limitations"])


def test_the_reference_cache_is_profiling_summary_v3_runtime_records():
    doc = json.loads((ROOT / summary.PULSE_REFERENCE).read_text())
    assert doc["schema"] == "profiling-summary@3" and doc["records"]
    for ref in doc["records"]:
        assert set(summary.PROFILING_MATCH_KEYS) <= set(ref)
        assert ref["status"] in {"current", "superseded", "archived"}


def test_committed_artefact_is_current_and_keyed_uniquely():
    doc = json.loads((ROOT / "dashboard/walltime_validation.json").read_text())
    assert doc["schema"] == "walltime-validation" and doc["records"]
    keys = {
        json.dumps(
            {
                k: r[k]
                for k in (
                    "likelihood",
                    "instrument",
                    "setup_id",
                    "host_class",
                    "model_variant",
                    "sampler",
                    "seed",
                )
            },
            sort_keys=True,
        )
        for r in doc["records"]
    }
    assert len(keys) == len(doc["records"])
    assert all(not r["host_class"].startswith("local") for r in doc["records"])


@pytest.mark.skipif(not (PULSE / "pulse" / "walltime.py").exists(), reason="PyAutoPulse not beside")
def test_pulse_validator_accepts_the_committed_artefact():
    sys.path.insert(0, str(PULSE))
    try:
        from pulse import walltime
    finally:
        sys.path.remove(str(PULSE))
    doc = json.loads((ROOT / "dashboard/walltime_validation.json").read_text())
    assert walltime.validate(doc) == []
