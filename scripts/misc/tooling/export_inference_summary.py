"""Publish the project-owned inference-summary@1 exchange; no scientific imports.

Reads all current and archived inference result JSONs, never simulator summaries.
Git dates describe publication provenance only, never measurement freshness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import UTC, datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
DEFINITIONS = {
    "setup_s": "Not measured separately; unknown, not total minus sampling",
    "sampling_s": "wall_s: sampler clock from samples_info.json; includes sampler overhead",
    "total_s": "total_wall_s: elapsed search.fit call; excludes prior compile probe and dataset/model setup",
    "compile_s": "Separate likelihood warm-up probe; not added to total_s",
}
DIAGNOSTICS = (
    "log_evidence",
    "log_evidence_err",
    "log_evidence_err_note",
    "max_log_likelihood",
    "posterior",
    "truth_delta_sigma",
    "likelihood_evals",
    "positions_info_present",
    "inversion_applies",
    "per_call_s",
    "likelihood_share",
    "timing",
    "truths",
    "log_likelihood_at_truth",
    "ess",
    "r_hat",
    "acceptance_fraction",
)
CONFIG = (
    "config_name",
    "variant",
    "inversion",
    "priors",
    "fit_positions_cls",
    "free_parameters",
    "n_live",
    "n_batch",
    "cores",
    "use_jax",
    "test_mode",
    "mesh",
    "mesh_pixels",
    "rect_mesh",
    "memo",
    "mesh_areas_factor",
    "mesh_zeroed_pixels",
    "image_mesh",
    "image_mesh_weight_power",
    "image_mesh_weight_floor",
    "image_mesh_edge_points",
    "regularization",
    "mesh_shape",
    "stages_requested",
    "resumed",
)


def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    p = PurePosixPath(value)
    return not p.is_absolute() and all(s not in ("", ".", "..") for s in value.split("/"))


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite metric")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def record(payload, path, *, stage=None, parent=None, archived=False):
    row = dict(payload)
    row.pop("stages", None)
    if stage is not None:
        row.update(stage)
    stage_name = stage.get("name") if stage is not None else None
    identifier = (
        path
        if stage is None
        else f"{path}#stage-{hashlib.sha256(str(stage_name).encode()).hexdigest()[:16]}"
    )
    status = row.get("status")
    if stage is not None:
        status = "complete" if stage.get("completed") is True else "incomplete"
    elif not status:
        status = "complete" if row.get("completed") is True else "unknown"
    is_pipeline = "stages" in payload
    is_parent = is_pipeline and stage is None
    raw_path = row.get("output_path")
    output_path = None
    if safe_path(raw_path):
        output_path = raw_path if raw_path.startswith("output/") else f"output/{raw_path}"
    values = {k: row[k] for k in DIAGNOSTICS if k in row and row[k] is not None}
    if is_parent:
        values = {}
    result = {
        "id": identifier,
        "parent_run_id": parent,
        "stage": stage_name,
        "target": row.get("target"),
        "dataset": {
            "class": row.get("dataset_class"),
            "instrument": row.get("instrument"),
            "id": row.get("dataset_id"),
        },
        "model": row.get("model"),
        "pipeline": "slam" if is_pipeline else row.get("pipeline"),
        "sampler": row.get("sampler"),
        "configuration": {k: row[k] for k in CONFIG if k in row},
        "backend": row.get("backend"),
        "hardware": {
            k: row[k]
            for k in (
                "device",
                "host",
                "cores",
                "slurm_job_id",
                "slurm_array_job_id",
                "slurm_array_task_id",
            )
            if k in row
        },
        "precision": row.get("precision"),
        "seed": row.get("seed"),
        "dependency_revisions": row.get("library_revisions") or {},
        "library_version": row.get("version"),
        "execution": {
            "status": status,
            "completed": row.get("completed") if not is_parent else None,
            "reason": status if str(status).startswith("failed") else None,
        },
        "scientific": {
            "convergence": "not_assessed",
            "acceptance": "not_assessed",
            "reason": "Exporter does not make scientific rulings; consult Cortex",
        },
        "timings": {
            "setup_s": None,
            "sampling_s": None if is_parent else row.get("wall_s"),
            "total_s": None if is_parent else row.get("total_wall_s"),
            "compile_s": None if is_parent else row.get("compile_s"),
            "definitions": (
                {
                    k: "Pipeline aggregate not measured; stage clocks may overlap and are never summed"
                    for k in DEFINITIONS
                }
                if is_parent
                else DEFINITIONS
            ),
        },
        "diagnostics": {
            "status": "available"
            if any(v not in ({}, []) for k, v in values.items() if not k.endswith("note"))
            else "missing",
            "values": values,
        },
        "comparison": {
            "group": None,
            "declared_target": row.get("target"),
            "note": "Declared target is not a verified comparison group; no ranking",
        },
        "samples": {
            "availability": "unknown",
            "path": output_path,
            "access": "project-storage-not-published",
            "note": "Committed results do not verify current archive/sample availability; unsafe absolute paths omitted",
        },
        "evidence_paths": [path],
        "measured_at": row.get("measured_at"),
        "archived": archived,
        "unknown_reasons": {},
    }
    for key in ("target", "model", "sampler", "backend", "precision", "seed", "measured_at"):
        if result[key] is None:
            result["unknown_reasons"][key] = "Not recorded in source result"
    if result["dataset"]["id"] is None:
        result["unknown_reasons"]["dataset.id"] = "Dataset content identity not recorded"
    if not result["dependency_revisions"]:
        result["unknown_reasons"]["dependency_revisions"] = "Dependency revisions not recorded"
    if result["execution"]["completed"] is None:
        result["unknown_reasons"]["execution.completed"] = (
            "No pipeline-wide completion marker is recorded"
            if is_parent
            else "Completion marker not recorded"
        )
    if is_parent:
        result["unknown_reasons"]["timings"] = (
            "Pipeline aggregate not measured; stage clocks may overlap and are never summed"
        )
    return result


def build(root, revision, generated_at):
    timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Generation timestamp must include timezone")
    generated_at = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    records, excluded = [], []
    runs = 0
    for path in sorted((root / "results").rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(root / "results").parts
        if not (
            parts[0] in ("slam", "searches")
            or (parts[0] == "archive" and ("slam" in parts or "searches" in parts))
        ):
            continue
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("unsafe result path")
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("result must be an object")
            finite(payload)
            if payload.get("schema_version") not in (1, 2):
                raise ValueError("unsupported result schema_version")
            stages = payload.get("stages", [])
            if not isinstance(stages, list) or any(not isinstance(s, dict) for s in stages):
                raise ValueError("stages must be objects")
            names = [s.get("name") for s in stages]
            if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError("duplicate or missing stage identity")
            kwargs = {"archived": parts[0] == "archive"}
            rows = [record(payload, rel, **kwargs)]
            rows.extend(record(payload, rel, stage=s, parent=rel, **kwargs) for s in stages)
            for row in rows:
                for key in ("setup_s", "sampling_s", "total_s", "compile_s"):
                    value = row["timings"][key]
                    if value is not None and (
                        isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0
                    ):
                        raise ValueError("invalid timing")
            records.extend(rows)
            runs += 1
        except (ValueError, TypeError, OSError) as exc:
            excluded.append({"path": rel, "reason": str(exc), "status": "invalid"})
    counts = Counter(str(r["execution"]["status"]).split(":", 1)[0] for r in records)
    return {
        "schema": "inference-summary",
        "version": 1,
        "project": "autolens_inference",
        "scope": "inference",
        "source_commit": revision,
        "producer_revision": revision,
        "generated_at": generated_at,
        "generation_basis": "Source commit timestamp; not a measurement or check-in time",
        "evidence_updated_at": None,
        "evidence_updated_at_reason": "Legacy result rows do not record measurement timestamps",
        "valid_until": None,
        "records": records,
        "coverage": {
            "expected": {
                "runs": None,
                "reason": "No complete expected-run manifest; unrecorded jobs cannot be enumerated",
            },
            "observed": {
                "runs": runs,
                "records": len(records),
                "archived_records": sum(r["archived"] for r in records),
                **{f"execution_{key}": value for key, value in sorted(counts.items())},
            },
            "excluded": excluded,
        },
        "comparisons": [],
        "comparison_policy": {
            "id": "declared-target-seed-stage-v1",
            "ranking": "none",
            "note": "Targets/seeds/stages retained; no automatic parity certification or best-seed selection",
        },
        "limitations": [
            "No scientific acceptance or convergence inferred from process completion",
            "Sample archives remain in project storage; current availability unknown",
            "No evidence from the retired inference programme is imported",
            "Archived rows belong to this project's September 2026 runs and are historical, not current baselines",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="dashboard/summary.json")
    args = parser.parse_args()
    revision = subprocess.check_output(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            "results",
            "scripts/misc/tooling/export_inference_summary.py",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    date = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", revision], cwd=ROOT, text=True
    ).strip()
    target = ROOT / args.output
    content = (
        json.dumps(build(ROOT, revision, date), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit("inference summary is stale; regenerate")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


if __name__ == "__main__":
    main()
