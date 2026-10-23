"""Project-owned v2 setup bindings; historical unknowns remain explicit."""

import copy
import json
import re
from pathlib import PurePosixPath

SETUPS = (
    ("imaging/rectangular/hst", "HST rectangular SLaM", "imaging", "rectangular", "hst"),
    ("imaging/delaunay/hst", "HST Delaunay SLaM", "imaging", "delaunay", "hst"),
    ("point_source/simple/simple", "Simple point source", "point_source", "simple", "simple"),
)
WORK = (
    "likelihood_evaluations",
    "gradient_evaluations",
    "iterations",
    "retained_samples",
    "effective_sample_size",
)


#: ``(dataset_type, likelihood variant)`` -> the stable family id prefix. The
#: point-source family keeps the id it was published under.
FAMILIES = {
    ("imaging", "rectangular"): "imaging/rectangular",
    ("imaging", "delaunay"): "imaging/delaunay",
    ("point_source_source", "source_plane"): "point_source/simple",
}
IMAGING_INSTRUMENTS = ("hst", "euclid", "jwst", "jwst_lw", "ao")


def setup_id(row):
    """The stable family id, from the record's resolved ``likelihood`` and
    instrument (the exporter derives both from the config table and the run
    variant; no path or id heuristic)."""
    likelihood = row.get("likelihood") or {}
    instrument = row["dataset"]["instrument"]
    family = FAMILIES.get((likelihood.get("dataset_type"), likelihood.get("variant")))
    if family is None:
        return None
    if family.startswith("imaging/"):
        if row["pipeline"] != "slam" or instrument not in IMAGING_INSTRUMENTS:
            return None
        return f"{family}/{instrument}"
    return f"{family}/{instrument}" if instrument == "simple" else None


MIGRATION_MAP = "catalogue/migration_2026_10.json"


def _migrated_ids(root, records):
    """``{old record id: new record id}`` through the committed path migration map."""
    path = root / MIGRATION_MAP
    if not path.exists():
        return {}
    old_of = {new: old for old, new in json.loads(path.read_text())["moves"].items()}
    result = {}
    for record_id in records:
        new_path, sep, fragment = record_id.partition("#")
        if new_path in old_of:
            result[old_of[new_path] + sep + fragment] = record_id
    return result


def build_catalogue(summary, root):
    doc = copy.deepcopy(summary)
    doc["version"] = 2
    doc["setups"] = [
        dict(
            id=i,
            label=label,
            dataset_family=d,
            model_family=m,
            instrument=inst,
            reference_record_id=None,
            reference_record_id_reason="No explicitly accepted current baseline with frozen model/prior identities",
        )
        for i, label, d, m, inst in SETUPS
    ]
    # Instruments remain columns under stable family IDs, including empty
    # supported presets; no evidence or baseline is inferred from a preset.
    for instrument in ("euclid", "jwst", "jwst_lw", "ao"):
        for family in ("rectangular", "delaunay"):
            doc["setups"].append(
                dict(
                    id=f"imaging/{family}/{instrument}",
                    label=f"{instrument.upper()} {family} SLaM",
                    dataset_family="imaging",
                    model_family=family,
                    instrument=instrument,
                    reference_record_id=None,
                    reference_record_id_reason="No explicitly accepted current baseline with frozen model/prior identities",
                )
            )
    doc["prepared_problems"] = []
    for row in doc["records"]:
        row["setup_id"] = setup_id(row)
        if row["setup_id"] is None:
            row["setup_id_reason"] = "Historical result does not identify a supported setup"
        for key in ("problem_id", "experiment_protocol"):
            row[key] = None
            row[key + "_reason"] = "Exact prepared scientific identity/protocol not recorded"
        row["initialization"] = dict(
            mode="unknown",
            recipe="Historical sampler initialization was not recorded",
            sources=[],
            reason="No pinned sampler start or checkpoint provenance",
        )
        row["environment"] = dict(
            hardware_id=None,
            hardware_id_reason="Resource/concurrency identity not recorded",
            compilation="unknown",
            compilation_reason="Separate compile probe does not establish fit compilation state",
            cache="unknown",
            cache_reason="Cache state not recorded",
        )
        for key in ("preparation_s", "initialization_s"):
            row["timings"][key] = None
            row["timings"][key + "_reason"] = "Not measured separately"
            row["timings"]["definitions"][key] = (
                "Not measured separately; inclusion in fit/sampling clocks and resumed segment scope unknown; never inferred or summed"
            )
        row["work"] = {
            "definitions": {
                k: "Not recorded with warmup/burn-in and aggregation scope; no conversion from sampler iterations"
                for k in WORK
            }
        }
        for key in WORK:
            row["work"][key] = None
            row["work"][key + "_reason"] = "Work count/estimator scope not recorded"
    # New measurements declare conditions explicitly. Never reconstruct them
    # from legacy flags, the presence of compile timings, or current defaults.
    for row in doc["records"]:
        source = json.loads((root / row["evidence_paths"][0]).read_text())
        if row["parent_run_id"]:
            source = {**source, **next(s for s in source["stages"] if s["name"] == row["stage"])}
        for key in (
            "setup_id",
            "problem_id",
            "experiment_protocol",
            "initialization",
            "environment",
            "work",
        ):
            if key in source:
                row[key] = copy.deepcopy(source[key])
                if source[key] is not None:
                    row.pop(key + "_reason", None)
        for key in ("preparation_s", "initialization_s"):
            if key in source:
                row["timings"][key] = source[key]
                if source[key] is not None:
                    row["timings"].pop(key + "_reason", None)
                    if key == "preparation_s":
                        row["timings"]["definitions"][key] = (
                            "Elapsed prerequisite SLaM stage fits and mass model construction after dataset setup; excludes mass fit and prepared export; shared prerequisite cost excluded from total_s; never sum across experiment rows"
                        )
                elif source.get(key + "_reason"):
                    row["timings"][key + "_reason"] = source[key + "_reason"]
        if source.get("timing_definitions"):
            row["timings"]["definitions"].update(source["timing_definitions"])
    records = {r["id"]: r for r in doc["records"]}
    # Manifests written before the 2026-10 layout move name the old row paths;
    # the committed migration map resolves them without editing the manifests.
    for old_id, new_id in _migrated_ids(root, records).items():
        records.setdefault(old_id, records[new_id])
    known = {s["id"] for s in doc["setups"]}
    seen = {}
    for directory in (root / "prepared", root / "output/autolens/prepared"):
        for path in sorted(directory.rglob("manifest.json")):
            try:
                manifest = json.loads(path.read_text())
                if (
                    manifest.get("schema") != "prepared-inference-problem"
                    or manifest.get("version") != 1
                ):
                    raise ValueError("unsupported prepared manifest")
                for key in ("dataset_id", "model_id", "priors_id"):
                    if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest.get(key, "")):
                        raise ValueError("prepared exact identity missing or invalid")
                for artifact in manifest.get("artifacts", []):
                    artifact_path = artifact.get("path", "")
                    if (
                        not artifact_path
                        or PurePosixPath(artifact_path).is_absolute()
                        or any(p in ("", ".", "..") for p in artifact_path.split("/"))
                        or "\\" in artifact_path
                        or ":" in artifact_path
                    ):
                        raise ValueError("unsafe prepared artifact path")
                    if not re.fullmatch(
                        r"[0-9a-f]{64}", artifact.get("sha256", "")
                    ) or not re.fullmatch(r"[0-9a-f]{40}", artifact.get("revision", "")):
                        raise ValueError("unpinned prepared artifact")
                    source = records.get(artifact.get("record_id"))
                    if (
                        not artifact.get("kind")
                        or source is None
                        or source["setup_id"] != manifest.get("setup_id")
                    ):
                        raise ValueError("prepared artifact source does not resolve in setup")
                baseline = records.get(manifest.get("baseline_record_id"))
                if (
                    baseline is None
                    or manifest.get("setup_id") not in known
                    or baseline["setup_id"] != manifest["setup_id"]
                    or baseline["stage"] != manifest.get("stage")
                ):
                    raise ValueError("prepared baseline/setup/stage does not resolve")
                if manifest["id"] in seen:
                    if manifest != seen[manifest["id"]]:
                        raise ValueError("conflicting prepared problem manifests")
                    continue
                problem = {
                    k: manifest[k]
                    for k in (
                        "id",
                        "setup_id",
                        "dataset_id",
                        "model_id",
                        "priors_id",
                        "stage",
                        "baseline_record_id",
                        "artifacts",
                    )
                }
                # Publish the current record ids: a manifest written before the
                # 2026-10 layout move names the old paths, which consumers cannot
                # resolve. The manifest itself stays as committed evidence.
                problem["baseline_record_id"] = baseline["id"]
                problem["artifacts"] = [
                    {**artifact, "record_id": records[artifact["record_id"]]["id"]}
                    for artifact in problem["artifacts"]
                ]
                if not problem["artifacts"]:
                    problem["artifacts_reason"] = "Manifest declares no retained artifacts"
                seen[problem["id"]] = manifest
                doc["prepared_problems"].append(problem)
                baseline["problem_id"] = problem["id"]
                baseline.pop("problem_id_reason", None)
                baseline["dataset"]["id"] = problem["dataset_id"]
                baseline["unknown_reasons"].pop("dataset.id", None)
                baseline["diagnostics"]["values"].update(manifest.get("baseline_evidence") or {})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                doc["coverage"]["excluded"].append(
                    dict(path=path.relative_to(root).as_posix(), reason=str(exc), status="invalid")
                )
    # Every published reference resolves: a row whose source names a prepared
    # problem without a valid published manifest keeps the name in its reason,
    # never as a dangling ``problem_id``.
    emitted = {problem["id"] for problem in doc["prepared_problems"]}
    for row in doc["records"]:
        named = row.get("problem_id")
        if named is not None and named not in emitted:
            row["problem_id"] = None
            row["problem_id_reason"] = (
                f"Source names prepared problem {named}, whose manifest is not "
                "published (or not valid) under prepared/; reference withheld"
            )
    return doc
