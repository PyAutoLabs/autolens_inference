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


def _walltime(root, references=()):
    return summary.build_walltime(root, feed(root), "a" * 40, references=list(references))


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
    assert any("no Pulse runtime reference" in x for x in doc["limitations"])


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


def test_a_reference_matches_only_a_fully_determined_setup(tmp_path):
    _put_validation(
        tmp_path, _row("hpc_a100_jax_gpu_dense_fp64", stages=[_stage("mass_total[1]", 50.0, 500)])
    )
    legacy = {
        "id": "measurement/legacy",
        "dataset": "imaging",
        "model": "rectangular",
        "instrument": "hst",
        "configuration": {},
        "identity": {"backend": "gpu", "device": "a100", "precision": "float64"},
        "metric": "vmap.per_call",
        "measurement": {"vmap.per_call": 0.05},
    }
    (record,) = _walltime(tmp_path, [legacy])["records"]
    assert record["reference_runtime_record_id"] is None
    exact = dict(legacy, id="measurement/exact", configuration={"operator": "dense"})
    (record,) = _walltime(tmp_path, [legacy, exact])["records"]
    assert record["reference_runtime_record_id"] == "measurement/exact"
    assert record["predicted_ms_per_eval"] == pytest.approx(50.0)
    assert record["ratio_measured_over_predicted"] == pytest.approx(2.0)


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
