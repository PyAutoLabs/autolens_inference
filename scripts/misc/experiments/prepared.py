"""Project-owned frozen scientific problems, verified before local reuse.

Dill snapshots are trusted local PyAutoFit storage, not an interchange format.
Hash validation precedes deserialization; hashes establish identity, not trust.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def revision(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def library_revisions(root):
    """Pin the actual import-selected source checkouts, never guessed siblings."""
    import importlib.util

    result = {}
    for name, module in (
        ("PyAutoFit", "autofit"),
        ("PyAutoArray", "autoarray"),
        ("PyAutoGalaxy", "autogalaxy"),
        ("PyAutoLens", "autolens"),
        ("PyAutoNerves", "autonerves"),
    ):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Installed {module} source is unavailable")
        source = Path(spec.origin).resolve()
        try:
            checkout = Path(
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=source.parent,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            subprocess.check_output(
                ["git", "ls-files", "--error-unmatch", source.relative_to(checkout).as_posix()],
                cwd=checkout,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise ValueError(f"Installed {module} is not a tracked source checkout") from exc
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout, text=True
        )
        if dirty.strip():
            raise ValueError(f"Installed {module} checkout has uncommitted changes")
        result[name] = revision(checkout)
    return result


def publish_manifest(root, doc):
    """Keep portable metadata tracked while bulk scientific snapshots stay local."""
    target = contained(root, Path("prepared") / doc["id"] / "manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json_bytes(doc))
    return target


def contained(root, path):
    value = Path(path)
    if value.is_absolute() or any(part in (".", "..") for part in value.parts):
        raise ValueError(f"Artifact must be relative to project storage: {path}")
    target = root / value
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"Artifact escapes project storage: {path}")
    return target


def artifact(root, target, content, kind, record_id, source_revision):
    contained(root, target.relative_to(root))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite prepared artifact: {target}")
    target.write_bytes(content)
    return dict(
        kind=kind,
        record_id=record_id,
        path=target.relative_to(root).as_posix(),
        revision=source_revision,
        sha256=digest(content),
    )


def export_problem(
    root,
    directory,
    model,
    analysis,
    *,
    setup_id,
    baseline_record_id,
    controls,
    preparation_s,
    dependency_revisions=None,
):
    """Freeze the exact model, priors, dataset, adapt products and positions LH.

    Called before the mass-stage compile probe and fit. The dataset/analysis
    snapshots retain masks, both over-sampling maps, adapt images and settings;
    no scientific defaults are reconstructed on reuse.
    """
    import dill

    root = root.resolve()
    directory = contained(root, directory)
    source_revision = revision(root)
    priors = [
        {"path": list(path), "prior": prior.dict()} for path, prior in model.path_priors_tuples
    ]
    contents = {
        "model_definition": ("model.json", json_bytes(model.dict())),
        "priors_definition": ("priors.json", json_bytes(priors)),
        "model_snapshot": ("model.dill", dill.dumps(model)),
        "dataset_snapshot": ("dataset.dill", dill.dumps(analysis.dataset)),
        "analysis_snapshot": ("analysis.dill", dill.dumps(analysis)),
    }
    artifacts = [
        artifact(root, directory / filename, data, kind, baseline_record_id, source_revision)
        for kind, (filename, data) in contents.items()
    ]
    hashes = {a["kind"]: a["sha256"] for a in artifacts}
    identity = {
        key: hashes[key]
        for key in (
            "model_definition",
            "model_snapshot",
            "priors_definition",
            "dataset_snapshot",
            "analysis_snapshot",
        )
    }
    resolved_revisions = dependency_revisions or library_revisions(root)
    identity.update(
        controls={k: v for k, v in controls.items() if k != "seed"},
        dependency_revisions=resolved_revisions,
        setup_id=setup_id,
        stage="mass_total[1]",
    )
    manifest = {
        "schema": "prepared-inference-problem",
        "version": 1,
        "id": "prepared-" + digest(json_bytes(identity)),
        "setup_id": setup_id,
        "dataset_id": "sha256:" + hashes["dataset_snapshot"],
        "model_id": "sha256:" + hashes["model_definition"],
        "priors_id": "sha256:" + hashes["priors_definition"],
        "stage": "mass_total[1]",
        "baseline_record_id": baseline_record_id,
        "artifacts": artifacts,
        "controls": controls,
        "dependency_revisions": resolved_revisions,
        "preparation_s": preparation_s,
        "baseline_evidence": None,
        "baseline_evidence_reason": "Baseline fit not completed",
    }
    target = directory / "manifest.json"
    if target.exists() and json.loads(target.read_text())["id"] != manifest["id"]:
        raise ValueError("Prepared directory already identifies another scientific problem")
    target.write_bytes(json_bytes(manifest))
    publish_manifest(root, manifest)
    return target


def complete_baseline(root, manifest_path, result, posterior):
    import dill

    manifest = verify_manifest(root, manifest_path)
    samples = result.samples
    parameters = dict(zip(samples.names, samples.max_log_likelihood(as_instance=False)))
    manifest["baseline_evidence"] = {
        "max_log_likelihood": float(samples.max_log_likelihood_sample.log_likelihood),
        "max_likelihood_parameters": {str(k): float(v) for k, v in parameters.items()},
        "posterior": posterior,
        "scientific_acceptance": "not_assessed",
    }
    manifest.pop("baseline_evidence_reason", None)
    manifest["artifacts"].append(
        artifact(
            root,
            manifest_path.parent / "baseline_samples.dill",
            dill.dumps(samples),
            "posterior_samples",
            manifest["baseline_record_id"],
            revision(root),
        )
    )
    manifest_path.write_bytes(json_bytes(manifest))
    publish_manifest(root, manifest)


def verify_manifest(root, manifest_path, *, expected_problem=None, revisions=None):
    """Fail closed on missing/tampered artifacts, identity or library drift."""
    manifest_path = contained(
        root,
        Path(manifest_path).relative_to(root)
        if Path(manifest_path).is_absolute()
        else manifest_path,
    )
    doc = json.loads(manifest_path.read_text())
    if doc.get("schema") != "prepared-inference-problem" or doc.get("version") != 1:
        raise ValueError("Unsupported prepared manifest")
    if expected_problem is not None and doc["id"] != expected_problem:
        raise ValueError("Prepared problem identity mismatch")
    if revisions is not None and revisions != doc["dependency_revisions"]:
        raise ValueError("Installed library revisions differ from frozen problem")
    hashes = {}
    for item in doc["artifacts"]:
        data = contained(root, item["path"]).read_bytes()
        if digest(data) != item["sha256"]:
            raise ValueError(f"Artifact digest mismatch: {item['path']}")
        if item["kind"] in hashes:
            raise ValueError("Duplicate prepared artifact kind")
        hashes[item["kind"]] = item["sha256"]
    identity = {
        key: hashes[key]
        for key in (
            "model_definition",
            "model_snapshot",
            "priors_definition",
            "dataset_snapshot",
            "analysis_snapshot",
        )
    }
    identity.update(
        controls={k: v for k, v in doc["controls"].items() if k != "seed"},
        dependency_revisions=doc["dependency_revisions"],
        setup_id=doc["setup_id"],
        stage=doc["stage"],
    )
    if doc["id"] != "prepared-" + digest(json_bytes(identity)):
        raise ValueError("Prepared scientific identity mismatch")
    for key, kind in [
        ("dataset_id", "dataset_snapshot"),
        ("model_id", "model_definition"),
        ("priors_id", "priors_definition"),
    ]:
        if doc[key] != "sha256:" + hashes[kind]:
            raise ValueError(f"Prepared {key} mismatch")
    return doc


def load_problem(root, manifest_path, **kwargs):
    import dill

    manifest = verify_manifest(root, manifest_path, **kwargs)
    items = {a["kind"]: a for a in manifest["artifacts"]}

    def load(kind):
        return dill.loads(contained(root, items[kind]["path"]).read_bytes())

    model, analysis = load("model_snapshot"), load("analysis_snapshot")
    if digest(json_bytes(model.dict())) != items["model_definition"]["sha256"]:
        raise ValueError("Deserialized model differs from frozen definition")
    return manifest, model, analysis, items
