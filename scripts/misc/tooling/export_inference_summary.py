"""Publish the project-owned inference-summary@1 exchange; no scientific imports.

Reads all current and archived inference result JSONs, never simulator summaries.
Git dates describe publication provenance only, never measurement freshness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from inference_catalogue import build_catalogue  # noqa: E402

from _inference_cli import (  # noqa: E402  (stdlib-only module)
    BASE_VARIANT,
    CONFIG_NAME_RE,
    SEARCH_LAYOUT,
    split_config_name,
    validation_layout,
)

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
    "max_likelihood_parameters",
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
    "lens_light_mode",
    "results_tag",
    "release_pin",
    # --stages-from (autolens_inference#31): a stage loaded from another
    # config's completed fit names that config, and the backend that fitted it.
    "stages_from",
    "shared_from_config",
    "fitted_backend",
)

#: The timing definition a shared stage's record carries instead of the fitted
#: one: its clocks are another config's fit, kept for the record.
SHARED_STAGE_TIMING_NOTE = (
    "Stage loaded from shared_from_config's completed fit (--stages-from), not "
    "fitted by this run; sampling_s is that fit's clock, total_s is null"
)

#: Result schema versions this exporter reads. v3 adds ``lens_light_mode``,
#: ``results_tag``, ``library_revisions``, ``release_pin`` and ``slurm_job_id``;
#: a v1/v2 pipeline row predates the light mode and is ``linear`` by definition.
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)

#: What a pipeline row's ``lens_light_mode`` is when the field is absent.
DEFAULT_LENS_LIGHT_MODE = "linear"


#: Hosts whose name proves a RAL CPU node. An ``hpc_a100_*`` CPU-backend row is
#: relabelled ``hpc_ral_cpu`` only when its own ``host`` field matches.
RAL_CPU_HOST = re.compile(r"^euclid-ral-compute-")

#: Instruments a SLaM imaging row may bind to (the catalogue's supported presets).
IMAGING_INSTRUMENTS = ("hst", "euclid", "jwst", "jwst_lw", "ao")

PRECISIONS = {"fp64": "fp64", "mp": "mixed"}


def _family(row, is_pipeline):
    """``rectangular`` / ``delaunay`` for a SLaM row, from its mesh or run variant."""
    mesh = row.get("mesh")
    if mesh == "delaunay":
        return "delaunay"
    if mesh == "rect":
        return "rectangular"
    variant = row.get("variant")
    if variant is None:
        # Rows written before the variant field were the base run.
        return "rectangular" if is_pipeline else None
    if not isinstance(variant, str):
        return None
    if variant.startswith((BASE_VARIANT, "rect_")):
        return "rectangular"
    if variant.startswith(("slam_delaunay", "delaunay_")):
        return "delaunay"
    return None


def setup_fields(row, is_pipeline):
    """The additive setup fields of one record (autolens_inference#41).

    ``setup_id`` follows the Pulse grammar ``<backend>_<device>_<operator>_<precision>``
    and ``host_class`` is provenance, both resolved from the config-name table
    (``_inference_cli.split_config_name``). Every existing row is ``explicit``: its
    setup was named by flags, never chosen automatically.
    """
    reasons = {}
    config = row.get("config_name")
    setup_id = host_class = None
    resolved = dict(
        backend=None, device=None, operator=None, precision=None, transformer="na", threads=None
    )
    if isinstance(config, str) and CONFIG_NAME_RE.match(config):
        _, backend_device, operator, precision = CONFIG_NAME_RE.match(config).groups()
        backend, device = backend_device.split("_")
        resolved.update(
            backend=backend, device=device, operator=operator, precision=PRECISIONS[precision]
        )
        setup_id = f"{backend}_{device}_{operator}_{resolved['precision']}"
        _, host_class = split_config_name(config)
        host = row.get("host")
        if host_class == "hpc_a100" and device == "cpu":
            if isinstance(host, str) and RAL_CPU_HOST.match(host):
                host_class = "hpc_ral_cpu"
            else:
                reasons["host_class"] = (
                    "hpc_a100 config prefix on a CPU backend; no recorded host proves a "
                    "RAL CPU node, so the prefix is kept"
                )
    else:
        reasons["setup_id"] = "No config name of the setup grammar recorded"
        reasons["host_class"] = "No config name of the setup grammar recorded"
    threads = (row.get("device") or {}).get("omp_num_threads")
    if isinstance(threads, str) and threads.isdigit() and int(threads) > 0:
        resolved["threads"] = int(threads)
    dataset_class = row.get("dataset_class")
    variant = row.get("variant")
    likelihood = None
    model_variant = "default"
    if is_pipeline or row.get("pipeline") == "slam":
        family = _family(row, is_pipeline)
        if dataset_class and family:
            likelihood = {"dataset_type": dataset_class, "variant": family}
            if isinstance(variant, str):
                model_variant = validation_layout(variant)[1]
    elif (dataset_class, variant) in SEARCH_LAYOUT:
        dataset_type, likelihood_variant = SEARCH_LAYOUT[(dataset_class, variant)]
        likelihood = {"dataset_type": dataset_type, "variant": likelihood_variant}
    if likelihood is None:
        reasons["likelihood"] = "Dataset class and variant do not resolve to a likelihood"
    return {
        "setup_mode": "explicit",
        "setup_resolved": resolved,
        "setup_id": setup_id,
        "host_class": host_class,
        "model_variant": model_variant,
        "likelihood": likelihood,
    }, reasons


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
    stage_name = stage.get("name") if stage is not None else row.get("stage")
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
        # A run from an installed release has no checkouts: its revisions read None.
        "dependency_revisions": {
            k: v for k, v in (row.get("library_revisions") or {}).items() if v is not None
        },
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
                else dict(DEFINITIONS)
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
    fields, reasons = setup_fields(row, is_pipeline)
    result.update(fields)
    result["unknown_reasons"].update(reasons)
    if is_pipeline:
        result["configuration"].setdefault("lens_light_mode", DEFAULT_LENS_LIGHT_MODE)
    if not is_parent and row.get("shared_from_config"):
        result["timings"]["definitions"] = {
            k: SHARED_STAGE_TIMING_NOTE for k in result["timings"]["definitions"]
        }
    for key in ("target", "model", "sampler", "backend", "precision", "seed", "measured_at"):
        if result[key] is None:
            result["unknown_reasons"][key] = "Not recorded in source result"
    if result["dataset"]["id"] is None:
        result["unknown_reasons"]["dataset.id"] = "Dataset content identity not recorded"
    if not result["dependency_revisions"]:
        result["unknown_reasons"]["dependency_revisions"] = (
            "Run from an installed release (no git checkouts); see library_version"
            if row.get("library_revisions")
            else "Dependency revisions not recorded"
        )
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


def _result_files(root):
    """Every result row: ``results/**`` except ``results/misc/`` (simulator
    summaries, logs) but including ``results/misc/experiments/``, plus the dated
    ``archive/YYYY/MM/<slug>/`` rows (autolens_inference#41 layout). Rows under a
    legacy ``results/archive/`` are still read as archived."""
    for path in sorted((root / "results").rglob("*.json")):
        parts = path.relative_to(root / "results").parts
        if parts[0] == "misc" and parts[1:2] != ("experiments",):
            continue
        if parts[0] == "archive" and not ("slam" in parts or "searches" in parts):
            continue
        yield path
    for path in sorted((root / "archive").rglob("*.json")):
        parts = path.relative_to(root / "archive").parts
        if "slam" in parts or "searches" in parts:
            yield path


def build(root, revision, generated_at):
    timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Generation timestamp must include timezone")
    generated_at = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    records, excluded = [], []
    runs = 0
    for path in _result_files(root):
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts[1:]
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("unsafe result path")
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("result must be an object")
            finite(payload)
            if payload.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError("unsupported result schema_version")
            stages = payload.get("stages", [])
            if not isinstance(stages, list) or any(not isinstance(s, dict) for s in stages):
                raise ValueError("stages must be objects")
            names = [s.get("name") for s in stages]
            if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError("duplicate or missing stage identity")
            kwargs = {"archived": rel.startswith("archive/") or parts[0] == "archive"}
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


# ---------------------------------------------------------------------------
# walltime-validation@1 (PyAutoPulse REFERENCE.md): full-fit evidence
# ---------------------------------------------------------------------------

#: The committed extract of the Pulse ``lens`` snapshot this exporter matches
#: against. Refreshed only by ``--refresh-pulse-reference``, so ``--check`` never
#: depends on a sibling checkout.
PULSE_REFERENCE = "catalogue/pulse_lens_runtime_reference.json"

#: Per-likelihood runtime metrics, in the order a prediction prefers them.
REFERENCE_METRICS = ("vmap.per_call", "single_jit_block")

#: The parity stage of the backend-parity view (build_readme.PARITY_STAGE).
PARITY_STAGE = "mass_total[1]"

#: Declared parity rulings. The exporter computes no parity verdict: a status
#: other than ``not_assessed`` exists only where a ruling of record names it.
#: Key: (dataset_type, variant, instrument, model_variant, setup_id, stage).
PARITY_RULINGS = {
    ("imaging", "rectangular", "hst", "light_fixed", "jax_gpu_sparse_fp64", PARITY_STAGE): (
        "diverged",
        "2026-10-18 backend-parity view: the jax_gpu sparse light_fixed mass_total[1] "
        "posterior sits +274 sigma (einstein_radius) from the jax_cpu dense reference and "
        "its max log likelihood is about -43,000 against about 31,650 on every other leg; "
        "the numba_cpu sparse light_fixed leg agrees (Cortex autolens_inference ledger)",
    ),
}


def refresh_pulse_reference(snapshot_path, root=ROOT):
    """Write the runtime records of a Pulse ``lens`` snapshot to PULSE_REFERENCE."""
    snapshot = json.loads(Path(snapshot_path).read_text())
    doc = snapshot.get("summary", snapshot)
    setups = {s["id"]: s for s in doc.get("setups", [])}
    records = []
    for record in doc.get("records", []):
        if record.get("axis") != "runtime" or record.get("setup_id") not in setups:
            continue
        setup = setups[record["setup_id"]]
        config = {
            k: (v.get("value") if isinstance(v, dict) else v)
            for k, v in (setup.get("configuration") or {}).items()
            if k in ("operator", "inversion", "backend", "source_pixels", "delaunay_vertices")
        }
        records.append(
            dict(
                id=record["id"],
                setup_id=record["setup_id"],
                dataset=setup.get("dataset"),
                model=setup.get("model"),
                instrument=setup.get("instrument"),
                configuration=config,
                identity={
                    k: record.get("identity", {}).get(k) for k in ("backend", "device", "precision")
                },
                metric=record.get("metric"),
                measurement=record.get("measurement"),
            )
        )
    out = {
        "source": "PyAutoPulse snapshots/lens.json",
        "instance": snapshot.get("instance"),
        "schema": f"{doc.get('schema')}@{doc.get('version')}",
        "commit": snapshot.get("commit"),
        "fetched_at": snapshot.get("fetched_at"),
        "records": sorted(records, key=lambda r: r["id"]),
    }
    target = root / PULSE_REFERENCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def _reference_setup(ref):
    """The R1 compute tuple a Pulse runtime record determines, or None.

    A legacy (v2) record names its device and precision but not the operator, so
    it determines no setup_id; matching never fills a field the record omits.
    """
    identity = ref.get("identity") or {}
    config = ref.get("configuration") or {}
    device = {"gpu": "gpu", "cpu": "cpu"}.get(identity.get("backend"))
    precision = {"float64": "fp64", "fp64": "fp64", "mixed": "mixed"}.get(identity.get("precision"))
    operator = config.get("operator") or config.get("inversion")
    backend = config.get("backend") or ("jax" if device == "gpu" else None)
    if None in (device, precision, operator, backend):
        return None
    return f"{backend}_{device}_{operator}_{precision}"


def match_reference(walltime, references):
    """``(record id, predicted ms per eval)`` of the Pulse runtime record whose
    (dataset, model, instrument, setup) equals the walltime record's, else
    ``(None, None)``."""
    likelihood = walltime["likelihood"]
    candidates = [
        ref
        for ref in references
        if ref.get("dataset") == likelihood["dataset_type"]
        and ref.get("model") == likelihood["variant"]
        and ref.get("instrument") == walltime["instrument"]
        and _reference_setup(ref) == walltime["setup_id"]
    ]
    for metric in REFERENCE_METRICS:
        for ref in sorted(candidates, key=lambda r: r["id"]):
            value = (ref.get("measurement") or {}).get(metric)
            if ref.get("metric") == metric and isinstance(value, (int, float)):
                return ref["id"], value * 1000.0
    return None, None


def build_walltime(root, summary_doc, revision, references=None):
    """``walltime-validation@1`` from the current SLaM validation rows.

    One record per (likelihood, instrument, setup_id, host_class, model_variant,
    sampler, seed) and stage. The Pulse key has no stage field, so the stage is
    folded into ``sampler`` as ``<sampler>:<stage>`` (``nautilus:source_pix[1]``) and
    repeated in a ``stage`` field, which the validator tolerates. Every derived
    number is computed here; Pulse computes nothing.
    """
    if references is None:
        path = root / PULSE_REFERENCE
        references = json.loads(path.read_text())["records"] if path.exists() else []
    stages = [
        r
        for r in summary_doc["records"]
        if r["pipeline"] == "slam"
        and r["parent_run_id"] is not None
        and not r["archived"]
        and "/validation/" in r["evidence_paths"][0]
    ]
    default_walls = {
        (r["parent_run_id"].replace("/light_fixed/", "/default/"), r["stage"]): r["timings"][
            "sampling_s"
        ]
        for r in stages
        if r["model_variant"] == "default"
    }
    skipped = Counter()
    records = []
    for r in sorted(stages, key=lambda r: r["id"]):
        values = r["diagnostics"]["values"]
        if r["execution"]["status"] != "complete":
            skipped["incomplete"] += 1
            continue
        if r["configuration"].get("shared_from_config"):
            skipped["shared"] += 1
            continue
        wall = r["timings"]["sampling_s"]
        if (
            r["model_variant"] != "default"
            and default_walls.get(
                (r["parent_run_id"].replace(f"/{r['model_variant']}/", "/default/"), r["stage"])
            )
            == wall
        ):
            skipped["duplicate"] += 1
            continue
        if None in (r["likelihood"], r["setup_id"], r["host_class"], r["seed"]):
            skipped["unresolved"] += 1
            continue
        evals = values.get("likelihood_evals")
        evals = evals if isinstance(evals, int) and not isinstance(evals, bool) else None
        implied = wall / evals * 1000.0 if wall is not None and evals else None
        likelihood = r["likelihood"]
        item = {
            "likelihood": likelihood,
            "instrument": r["dataset"]["instrument"],
            "setup_id": r["setup_id"],
            "host_class": r["host_class"],
            "model_variant": r["model_variant"],
            "sampler": f"{r['sampler']}:{r['stage']}",
            "stage": r["stage"],
            "seed": r["seed"],
        }
        ref_id, predicted = match_reference(item, references)
        status, basis = PARITY_RULINGS.get(
            (
                likelihood["dataset_type"],
                likelihood["variant"],
                item["instrument"],
                item["model_variant"],
                item["setup_id"],
                r["stage"],
            ),
            ("not_assessed", None),
        )
        item.update(
            id="/".join(
                [
                    likelihood["dataset_type"],
                    likelihood["variant"],
                    item["instrument"],
                    item["model_variant"],
                    item["setup_id"],
                    item["host_class"],
                    item["sampler"],
                    f"seed_{item['seed']}",
                ]
            ),
            wall_s=wall,
            likelihood_evals=evals,
            implied_ms_per_eval=implied,
            reference_runtime_record_id=ref_id,
            predicted_ms_per_eval=predicted,
            ratio_measured_over_predicted=implied / predicted if implied and predicted else None,
            cores=r["hardware"].get("cores"),
            parallel_efficiency=None,
            release_pin=r["configuration"].get("release_pin"),
            slurm_job_id=r["hardware"].get("slurm_job_id"),
            max_log_likelihood=values.get("max_log_likelihood"),
            parity_status=status,
            parity_basis=basis,
            evidence_path=r["evidence_paths"][0],
        )
        records.append(item)
    unmatched = sum(1 for item in records if item["reference_runtime_record_id"] is None)
    limitations = [
        "Full-fit measurements: one Nautilus stage of one SLaM chain per record, never a "
        "per-likelihood cost; the sampler is <sampler>:<stage> because the key has no stage",
        "wall_s is the sampler clock (samples_info.json) and includes sampler overhead; "
        "implied_ms_per_eval = wall_s / likelihood_evals * 1000 is therefore an upper bound "
        "on the per-evaluation cost",
        "parallel_efficiency is null: no 1-core row of any key exists yet",
        "parity_status is not_assessed unless a declared ruling of record names the run "
        "(parity_basis); the exporter computes no parity verdict",
    ]
    if unmatched:
        limitations.append(
            f"{unmatched} of {len(records)} records have no Pulse runtime reference: the lens "
            "snapshot's runtime records do not determine the operator (dense/sparse) or match "
            "the instrument and mesh, so reference_runtime_record_id and predicted_ms_per_eval "
            "are null until the lens pin moves to profiling-summary@3"
        )
    kept_prefix = sum(
        1 for item in records if item["host_class"] == "hpc_a100" and "_cpu_" in item["setup_id"]
    )
    if kept_prefix:
        limitations.append(
            f"{kept_prefix} CPU-backend record(s) keep host_class hpc_a100 from their config "
            "prefix: the row records no host that proves a RAL CPU node"
        )
    for reason, label in (
        ("shared", "stages loaded from another config (--stages-from) are not this run's fit"),
        (
            "duplicate",
            "non-default model-variant stages identical to the default run's "
            "(one --lens-light both job, one fit) are listed once, under default",
        ),
        ("incomplete", "incomplete stages are not full-fit measurements"),
        ("unresolved", "stages without a resolved likelihood, setup or host"),
    ):
        if skipped[reason]:
            limitations.append(f"{skipped[reason]} stage(s) omitted: {label}")
    return {
        "schema": "walltime-validation",
        "version": 1,
        "project": "autolens_inference",
        "generated_at": summary_doc["generated_at"],
        "producer_revision": revision,
        "limitations": limitations,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="dashboard/summary.json")
    parser.add_argument("--catalogue-output", default="dashboard/catalogue.json")
    parser.add_argument("--walltime-output", default="dashboard/walltime_validation.json")
    parser.add_argument(
        "--refresh-pulse-reference",
        metavar="SNAPSHOT",
        help=f"rewrite {PULSE_REFERENCE} from a PyAutoPulse snapshots/lens.json, then exit",
    )
    args = parser.parse_args()
    if args.refresh_pulse_reference:
        out = refresh_pulse_reference(args.refresh_pulse_reference)
        print(f"wrote {PULSE_REFERENCE}: {len(out['records'])} runtime record(s)")
        return
    revision = subprocess.check_output(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            "results",
            "archive",
            "catalogue",
            "scripts/misc/tooling/export_inference_summary.py",
            "scripts/misc/tooling/inference_catalogue.py",
            "_inference_cli.py",
            "prepared",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    date = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", revision], cwd=ROOT, text=True
    ).strip()
    target = ROOT / args.output
    summary_doc = build(ROOT, revision, date)
    content = json.dumps(summary_doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    walltime = ROOT / args.walltime_output
    walltime_content = (
        json.dumps(
            build_walltime(ROOT, summary_doc, revision), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    if args.check and (not walltime.exists() or walltime.read_text() != walltime_content):
        raise SystemExit("walltime validation is stale; regenerate")
    if not args.check:
        walltime.parent.mkdir(parents=True, exist_ok=True)
        walltime.write_text(walltime_content)
    catalogue = ROOT / args.catalogue_output
    catalogue_content = (
        json.dumps(
            build_catalogue(build(ROOT, revision, date), ROOT),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if args.check and (not catalogue.exists() or catalogue.read_text() != catalogue_content):
        raise SystemExit("inference catalogue is stale; regenerate")
    if not args.check:
        catalogue.parent.mkdir(parents=True, exist_ok=True)
        catalogue.write_text(catalogue_content)
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit("inference summary is stale; regenerate")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


if __name__ == "__main__":
    main()
