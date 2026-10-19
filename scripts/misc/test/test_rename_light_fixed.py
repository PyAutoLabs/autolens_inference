"""The one-off ``_lightfixed`` -> ``_light_fixed`` rename (autolens_inference#38), on a
throwaway tree only."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]

import rename_light_fixed as ren  # noqa: E402
from experiments import prepared  # noqa: E402

CFG = "hpc_a100_jax_cpu_dense_fp64"
NUMBA = "hpc_a100_numba_cpu_sparse_fp64"
RAL = "/mnt/ral/jnightin/inference/autolens_inference"
STAGE = "source_lp[1]/97c9fde39db31280a495381d06812ef3"
OLD_VARIANT, NEW_VARIANT = "slam_base_lightfixed", "slam_base_light_fixed"


class Prior:
    def dict(self):
        return {"type": "Uniform", "lower_limit": 0, "upper_limit": 2}


class Model:
    path_priors_tuples = [(("mass",), Prior())]

    def dict(self):
        return {"mass": self.path_priors_tuples[0][1].dict()}


def _record(variant: str, config: str = CFG) -> str:
    return f"results/slam/imaging/hst/{variant}/{config}/stages_seed0.json#stage-03dd"


def _export(root: Path, variant: str, config: str = CFG) -> Path:
    """A prepared problem written by the real producer (stub model + analysis)."""
    return prepared.export_problem(
        root,
        Path("output/autolens/prepared") / config / "seed_0" / variant,
        Model(),
        SimpleNamespace(dataset={"data": [1, 2]}, positions=[4]),
        setup_id="imaging/rectangular/hst",
        baseline_record_id=_record(variant, config),
        controls={
            "variant": variant,
            "lens_light_mode": "fixed" if variant.endswith("fixed") else "linear",
            "config_name": config,
            "backend": config.split("_")[2] + "_cpu",
            "seed": 0,
        },
        preparation_s=2,
        dependency_revisions={"PyAutoFit": "b" * 40},
    )


def _row(root: Path, variant: str, problem_id: str, config: str = CFG) -> Path:
    path = root / "results" / "slam" / "imaging" / "hst" / variant / config / "stages_seed0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = "hst/slam5/seed0" if variant == "slam_base" else f"hst/slam5_{variant}/seed0"
    doc = {
        "schema_version": 3,
        "target": target,
        "variant": variant,
        "config_name": config,
        "stages": [
            {
                "name": "mass_total[1]",
                "output_path": f"{RAL}/output/autolens/imaging/hst/slam/{variant}/{config}/seed_0/{STAGE}",
                "problem_id": problem_id,
                "wall_s": 1.5,
            }
        ],
    }
    path.write_text(json.dumps(doc, indent=2))  # the runner's own format: no trailing newline
    (path.parent / "stages_seed0.png").write_bytes(b"\x89PNG lightfixed")
    return path


def _tree(root: Path, monkeypatch) -> dict:
    monkeypatch.setattr(prepared, "revision", lambda _: "a" * 40)
    out = root / "output"
    for variant in ("slam_base", OLD_VARIANT):
        seed = out / "autolens" / "imaging" / "hst" / "slam" / variant / CFG / "seed_0"
        (seed / STAGE).mkdir(parents=True)
        (seed / STAGE / ".completed").write_text("")
    linear = json.loads(_export(root, "slam_base").read_text())
    fixed = json.loads(_export(root, OLD_VARIANT).read_text())
    _row(root, "slam_base", linear["id"])
    _row(root, OLD_VARIANT, fixed["id"])
    return {"linear": linear, "fixed": fixed}


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def test_rewrite_is_structural_and_idempotent():
    text = json.dumps(
        {"variant": OLD_VARIANT, "n": 1.5, "k_lightfixed": ["x_lightfixed"]}, indent=2
    )
    new = ren.rewrite_json_text(text, {})
    assert json.loads(new) == {"variant": NEW_VARIANT, "n": 1.5, "k_light_fixed": ["x_light_fixed"]}
    assert ren.rewrite_json_text(new, {}) == new
    sorted_text = prepared.json_bytes({"z": "a_lightfixed", "a": 1}).decode()
    assert (
        ren.rewrite_json_text(sorted_text, {})
        == prepared.json_bytes({"z": "a_light_fixed", "a": 1}).decode()
    )


def test_a_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)
    assert ren.main(["--root", str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "3 move(s), 1 manifest(s), 1 rewrite(s), 1 stale published copy(ies) planned" in out


def test_apply_renames_reidentifies_and_verifies_then_is_a_no_op(tmp_path, monkeypatch, capsys):
    docs = _tree(tmp_path, monkeypatch)
    assert ren.main(["--root", str(tmp_path), "--apply"]) == 0
    assert "Verified 1 rewritten manifest(s)." in capsys.readouterr().out

    slam = tmp_path / "output" / "autolens" / "imaging" / "hst" / "slam"
    assert sorted(p.name for p in slam.iterdir()) == ["slam_base", NEW_VARIANT]
    assert (slam / NEW_VARIANT / CFG / "seed_0" / STAGE / ".completed").is_file()

    manifest_path = (
        tmp_path / "output/autolens/prepared" / CFG / "seed_0" / NEW_VARIANT / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["controls"]["variant"] == NEW_VARIANT
    assert manifest["baseline_record_id"] == _record(NEW_VARIANT)
    assert all(
        NEW_VARIANT in a["path"] and NEW_VARIANT in a["record_id"] for a in manifest["artifacts"]
    )
    # Same bytes, new paths: the digests are unchanged, the id is not (controls are identity).
    old = docs["fixed"]
    assert [a["sha256"] for a in manifest["artifacts"]] == [a["sha256"] for a in old["artifacts"]]
    assert manifest["id"] != old["id"]
    assert manifest["id"] == prepared.manifest_problem_id(manifest)
    assert manifest_path.read_bytes() == prepared.json_bytes(manifest)
    prepared.verify_manifest(tmp_path, manifest_path.relative_to(tmp_path))

    published = tmp_path / "prepared"
    assert sorted(p.name for p in published.iterdir()) == sorted(
        [docs["linear"]["id"], manifest["id"]]
    )
    assert json.loads((published / manifest["id"] / "manifest.json").read_text()) == manifest

    rows = tmp_path / "results" / "slam" / "imaging" / "hst"
    assert sorted(p.name for p in rows.iterdir()) == ["slam_base", NEW_VARIANT]
    row_path = rows / NEW_VARIANT / CFG / "stages_seed0.json"
    row = json.loads(row_path.read_text())
    assert row["variant"] == NEW_VARIANT
    assert row["target"] == f"hst/slam5_{NEW_VARIANT}/seed0"
    assert f"/slam/{NEW_VARIANT}/{CFG}/seed_0/{STAGE}" in row["stages"][0]["output_path"]
    assert row["stages"][0]["problem_id"] == manifest["id"]
    assert row["stages"][0]["wall_s"] == 1.5
    assert not row_path.read_text().endswith("\n")  # the producer's own format kept
    assert (row_path.parent / "stages_seed0.png").read_bytes() == b"\x89PNG lightfixed"
    linear_row = json.loads((rows / "slam_base" / CFG / "stages_seed0.json").read_text())
    assert linear_row["stages"][0]["problem_id"] == docs["linear"]["id"]

    after = _snapshot(tmp_path)
    assert ren.main(["--root", str(tmp_path), "--apply"]) == 0
    assert "Nothing to do" in capsys.readouterr().out
    assert _snapshot(tmp_path) == after


def test_an_existing_target_refuses_before_anything_moves(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch)
    clash = tmp_path / "results" / "slam" / "imaging" / "hst" / NEW_VARIANT
    clash.mkdir(parents=True)
    before = _snapshot(tmp_path)
    assert ren.main(["--root", str(tmp_path), "--apply"]) == 2
    assert _snapshot(tmp_path) == before
    assert f"target already exists: {clash}" in capsys.readouterr().err


def test_a_published_copy_that_is_not_an_exact_stale_copy_is_kept(tmp_path, monkeypatch, capsys):
    docs = _tree(tmp_path, monkeypatch)
    extra = tmp_path / "prepared" / docs["fixed"]["id"] / "notes.txt"
    extra.write_text("keep me")
    assert ren.main(["--root", str(tmp_path), "--apply"]) == 0
    assert "KEEP" in capsys.readouterr().out
    assert extra.read_text() == "keep me"


def test_a_symlinked_output_root_is_verified_through_the_mapping(tmp_path, monkeypatch):
    checkout, mirror = tmp_path / "checkout", tmp_path / "mirror"
    _tree(checkout, monkeypatch)
    (checkout / "output").rename(mirror)
    (checkout / "output").symlink_to(mirror)
    assert ren.main(["--root", str(checkout), "--apply"]) == 0
    manifest_path = mirror / "autolens/prepared" / CFG / "seed_0" / NEW_VARIANT / "manifest.json"
    ren.verify(checkout, mirror.resolve(), manifest_path)
    # Tampering is still caught on the mapped path.
    (manifest_path.parent / "model.json").write_text("{}")
    try:
        ren.verify(checkout, mirror.resolve(), manifest_path)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a tampered artifact verified")


def test_aside_moves_the_superseded_numba_legs_and_nothing_else(tmp_path, monkeypatch, capsys):
    _tree(tmp_path, monkeypatch)
    out = tmp_path / "output" / "autolens"
    seed = out / "imaging" / "hst" / "slam" / "slam_base" / NUMBA / "seed_0"
    (seed / STAGE).mkdir(parents=True)
    numba = json.loads(_export(tmp_path, "slam_base", NUMBA).read_text())
    _row(tmp_path, "slam_base", numba["id"], NUMBA)
    dest = tmp_path / "aside"
    before = _snapshot(tmp_path)

    assert ren.main(["--root", str(tmp_path), "--aside"]) == 2  # --aside-dest is required
    assert ren.main(["--root", str(tmp_path), "--aside", "--aside-dest", str(dest)]) == 0
    assert _snapshot(tmp_path) == before
    assert "4 move(s) planned" in capsys.readouterr().out

    assert ren.main(["--root", str(tmp_path), "--aside", "--aside-dest", str(dest), "--apply"]) == 0
    assert (dest / "autolens/imaging/hst/slam/slam_base" / NUMBA / "seed_0" / STAGE).is_dir()
    assert (dest / "autolens/prepared" / NUMBA / "seed_0/slam_base/manifest.json").is_file()
    archive = tmp_path / "results" / "archive" / ren.ASIDE_LABEL
    assert (archive / "slam/imaging/hst/slam_base" / NUMBA / "stages_seed0.json").is_file()
    assert (archive / "prepared" / numba["id"] / "manifest.json").is_file()
    assert not (out / "imaging/hst/slam/slam_base" / NUMBA).exists()
    assert not (out / "prepared" / NUMBA).exists()
    # The jax legs stay.
    assert (out / "imaging/hst/slam/slam_base" / CFG).is_dir()
    assert (tmp_path / "results/slam/imaging/hst/slam_base" / CFG).is_dir()

    after = _snapshot(tmp_path)
    assert ren.main(["--root", str(tmp_path), "--aside", "--aside-dest", str(dest), "--apply"]) == 0
    assert "Nothing to do" in capsys.readouterr().out
    assert _snapshot(tmp_path) == after
