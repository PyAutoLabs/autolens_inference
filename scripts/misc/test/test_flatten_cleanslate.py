"""The one-off clean-slate flattening (autolens_inference#35), on a throwaway tree only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]

import flatten_cleanslate as flat  # noqa: E402

TAG = flat.TAG
CFG = "hpc_a100_jax_cpu_dense_fp64"
RAL = "/mnt/ral/jnightin/inference/autolens_inference"
STAGE = "source_lp[1]/97c9fde39db31280a495381d06812ef3"
RECORD = f"results/slam/imaging/hst/slam_base/{CFG}__{TAG}/stages_seed0.json#stage-03dd"
RUN = f"cs1017_{CFG}_slam_base_emcee_s0"


def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _tree(root: Path) -> None:
    out = root / "output"
    seed = out / TAG / "slam" / "imaging" / "hst" / "slam_base" / CFG / "seed_0"
    (seed / STAGE).mkdir(parents=True)
    (seed / STAGE / ".completed").write_text("")
    prepared = out / "prepared" / TAG / CFG / "seed_0" / "slam_base"
    manifest = {
        "id": "prepared-abc",
        "baseline_record_id": RECORD,
        "artifacts": [
            {
                "path": f"output/prepared/{TAG}/{CFG}/seed_0/slam_base/model.json",
                "record_id": RECORD,
            }
        ],
    }
    _write(prepared / "manifest.json", manifest)
    _write(root / "prepared" / "prepared-abc" / "manifest.json", manifest)
    (out / "experiments" / RUN / "mass_total[1]").mkdir(parents=True)
    (out / "experiment_sources" / RUN).mkdir(parents=True)
    (out / "experiment_sources" / RUN / "final.hdf").write_text("x")
    _write(
        root
        / "results"
        / "slam"
        / "imaging"
        / "hst"
        / "slam_base"
        / f"{CFG}__{TAG}"
        / "stages_seed0.json",
        {
            "config_name": CFG,
            "results_tag": TAG,
            "stages": [
                {
                    "output_path": f"{RAL}/output/{TAG}/slam/imaging/hst/slam_base/{CFG}/seed_0/{STAGE}"
                }
            ],
        },
    )
    _write(
        root / "results" / "searches" / "experiments" / f"{RUN}.json",
        {
            "output_path": f"output/experiments/{RUN}",
            "checkpoint": {"path": f"output/experiment_sources/{RUN}/final.hdf"},
            "wave": f"{TAG}_wave1",
        },
    )
    _write(
        root / ".cortex" / "pull.json",
        {
            "checkpoints": {
                f"output/{TAG}/slam/imaging/hst/slam_base/{CFG}/seed_0/{STAGE}": {"bytes": 1}
            }
        },
    )


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): (p.read_text() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def test_the_rewrites_reorder_the_slam_levels_and_drop_the_dated_ones():
    text = (
        f'"{RAL}/output/{TAG}/slam/imaging/hst/slam_base/{CFG}/seed_0" '
        f'"output/prepared/{TAG}/{CFG}/" "output/experiments/r" '
        f'"output/experiment_sources/r" "results/slam/x/{CFG}__{TAG}/s.json" '
        f'"results_tag": "{TAG}" "{TAG}_wave1"'
    )
    assert flat.rewrite_text(text, row=True) == (
        f'"{RAL}/output/autolens/imaging/hst/slam/slam_base/{CFG}/seed_0" '
        f'"output/autolens/prepared/{CFG}/" "output/autolens/experiments/r" '
        f'"output/autolens/experiment_sources/r" "results/slam/x/{CFG}/s.json" '
        f'"results_tag": null "{TAG}_wave1"'
    )
    once = flat.rewrite_text(text, row=True)
    assert flat.rewrite_text(once, row=True) == once


def test_a_dry_run_changes_nothing(tmp_path, capsys):
    _tree(tmp_path)
    before = _snapshot(tmp_path)
    assert flat.main(["--root", str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "5 move(s), 5 rewrite(s) planned" in out


def test_apply_moves_and_rewrites_then_a_second_apply_is_a_no_op(tmp_path, capsys):
    _tree(tmp_path)
    assert flat.main(["--root", str(tmp_path), "--apply"]) == 0
    out = tmp_path / "output"
    new = out / "autolens"
    seed = new / "imaging" / "hst" / "slam" / "slam_base" / CFG / "seed_0"
    assert (seed / STAGE / ".completed").is_file()
    assert (new / "prepared" / CFG / "seed_0" / "slam_base" / "manifest.json").is_file()
    assert (new / "experiments" / RUN / "mass_total[1]").is_dir()
    assert (new / "experiment_sources" / RUN / "final.hdf").read_text() == "x"
    # The dated and old top-level folders are gone; nothing else was.
    assert sorted(p.name for p in out.iterdir()) == ["autolens"]

    row_dir = tmp_path / "results" / "slam" / "imaging" / "hst" / "slam_base"
    assert [p.name for p in row_dir.iterdir()] == [CFG]
    row = json.loads((row_dir / CFG / "stages_seed0.json").read_text())
    assert row["results_tag"] is None
    assert row["stages"][0]["output_path"] == (
        f"{RAL}/output/autolens/imaging/hst/slam/slam_base/{CFG}/seed_0/{STAGE}"
    )
    record = f"results/slam/imaging/hst/slam_base/{CFG}/stages_seed0.json#stage-03dd"
    for manifest_path in (
        new / "prepared" / CFG / "seed_0" / "slam_base" / "manifest.json",
        tmp_path / "prepared" / "prepared-abc" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text())
        assert manifest["baseline_record_id"] == record
        assert manifest["artifacts"][0] == {
            "path": f"output/autolens/prepared/{CFG}/seed_0/slam_base/model.json",
            "record_id": record,
        }
    experiment = json.loads(
        (tmp_path / "results" / "searches" / "experiments" / f"{RUN}.json").read_text()
    )
    assert experiment["output_path"] == f"output/autolens/experiments/{RUN}"
    assert experiment["checkpoint"]["path"] == f"output/autolens/experiment_sources/{RUN}/final.hdf"
    assert experiment["wave"] == f"{TAG}_wave1"
    pull = json.loads((tmp_path / ".cortex" / "pull.json").read_text())
    assert list(pull["checkpoints"]) == [
        f"output/autolens/imaging/hst/slam/slam_base/{CFG}/seed_0/{STAGE}"
    ]
    # Formatting is the file's own (indent 2, trailing newline).
    assert (tmp_path / ".cortex" / "pull.json").read_text().endswith("}\n")

    after = _snapshot(tmp_path)
    capsys.readouterr()
    assert flat.main(["--root", str(tmp_path), "--apply"]) == 0
    assert "Nothing to do" in capsys.readouterr().out
    assert _snapshot(tmp_path) == after


def test_an_existing_target_refuses_before_anything_moves(tmp_path, capsys):
    _tree(tmp_path)
    clash = tmp_path / "results" / "slam" / "imaging" / "hst" / "slam_base" / CFG
    clash.mkdir(parents=True)
    before = _snapshot(tmp_path)
    assert flat.main(["--root", str(tmp_path), "--apply"]) == 2
    assert _snapshot(tmp_path) == before
    assert f"target already exists: {clash}" in capsys.readouterr().err


def test_output_root_can_live_outside_the_checkout(tmp_path):
    checkout, mirror = tmp_path / "checkout", tmp_path / "mirror"
    _tree(checkout)
    (checkout / "output").rename(mirror)
    assert flat.main(["--root", str(checkout), "--output-root", str(mirror), "--apply"]) == 0
    assert (mirror / "autolens" / "imaging" / "hst" / "slam" / "slam_base" / CFG).is_dir()
    assert not (mirror / TAG).exists()
