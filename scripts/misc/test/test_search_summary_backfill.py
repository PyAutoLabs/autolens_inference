"""search_summary_backfill: committed rows -> search.summary timings, never guessed."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "backfill", ROOT / "scripts/misc/tooling/search_summary_backfill.py"
)
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "search.summary").read_text()
STAGE = "autolens/imaging/hst/slam/slam_base/hpc_a100_jax_cpu_dense_fp64/seed_0/source_pix[1]"


def _summary(output_root, ident, text=FIXTURE):
    path = output_root / STAGE / ident
    path.mkdir(parents=True)
    (path / "search.summary").write_text(text)


def _stage(ident="recorded"):
    return {"name": "source_pix[1]", "output_path": f"/mnt/ral/x/output/{STAGE}/{ident}"}


def test_the_recorded_identifier_directory_wins(tmp_path):
    _summary(tmp_path, "recorded")
    _summary(tmp_path, "other", FIXTURE.replace("8.641005992889404", "1.0"))
    entry = backfill.resolve_stage(_stage(), tmp_path)
    assert entry["resolution"] == "recorded_identifier"
    assert entry["log_likelihood_eval_s"] == 8.641005992889404
    assert entry["time_per_sample_s"] == 1.1063321578481946
    assert entry["path"] == f"output/{STAGE}/recorded/search.summary"


def test_a_sole_candidate_is_used_and_agreeing_candidates_are_accepted(tmp_path):
    _summary(tmp_path, "a")
    entry = backfill.resolve_stage(_stage("gone"), tmp_path)
    assert entry["resolution"] == "sole_candidate" and entry["log_likelihood_eval_s"]
    _summary(tmp_path, "b")
    entry = backfill.resolve_stage(_stage("gone"), tmp_path)
    assert entry["resolution"] == "agreeing_candidates:2"


def test_disagreeing_candidates_are_refused_as_ambiguous(tmp_path):
    _summary(tmp_path, "a")
    _summary(tmp_path, "b", FIXTURE.replace("8.641005992889404", "1.0"))
    entry = backfill.resolve_stage(_stage("gone"), tmp_path)
    assert entry["log_likelihood_eval_s"] is None and entry["resolution"] is None
    assert entry["reason"].startswith("Ambiguous") and len(entry["candidates"]) == 2


def test_a_stage_without_a_summary_is_listed_with_a_reason(tmp_path):
    entry = backfill.resolve_stage(_stage(), tmp_path)
    assert entry["time_per_sample_s"] is None and "absent" in entry["reason"]
    entry = backfill.resolve_stage({"name": "light[1]", "output_path": None}, tmp_path)
    assert "no output_path" in entry["reason"]


def test_the_committed_sidecar_covers_every_committed_validation_stage():
    doc = json.loads((ROOT / backfill.SIDECAR).read_text())
    for path in sorted(ROOT.glob(backfill.ROW_GLOB)):
        rel = path.relative_to(ROOT).as_posix()
        names = {s["name"] for s in json.loads(path.read_text()).get("stages") or []}
        stages = doc["rows"].get(rel, {})
        has_own = all(
            "search_summary" in s for s in json.loads(path.read_text()).get("stages") or []
        )
        assert has_own or names <= set(stages), rel
