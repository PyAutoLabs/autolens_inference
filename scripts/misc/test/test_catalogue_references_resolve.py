"""Every reference in the committed catalogue resolves (stdlib only).

PyAutoInsight refuses a catalogue whose ``problem_id``, prepared
``baseline_record_id`` or artifact ``record_id`` does not resolve; this witness
keeps the published ``dashboard/catalogue.json`` inside that rule.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _catalogue():
    return json.loads((ROOT / "dashboard/catalogue.json").read_text())


def test_record_problem_ids_resolve():
    doc = _catalogue()
    problems = {p["id"] for p in doc["prepared_problems"]}
    dangling = [
        r["id"]
        for r in doc["records"]
        if r.get("problem_id") is not None and r["problem_id"] not in problems
    ]
    assert not dangling
    for r in doc["records"]:
        if r.get("problem_id") is None:
            assert r.get("problem_id_reason")


def test_prepared_problem_record_ids_resolve():
    doc = _catalogue()
    records = {r["id"]: r for r in doc["records"]}
    for problem in doc["prepared_problems"]:
        baseline = records.get(problem["baseline_record_id"])
        assert baseline is not None, problem["id"]
        assert baseline["problem_id"] == problem["id"]
        assert baseline["setup_id"] == problem["setup_id"]
        assert baseline["stage"] == problem["stage"]
        for artifact in problem["artifacts"]:
            source = records.get(artifact["record_id"])
            assert source is not None, (problem["id"], artifact["record_id"])
            assert source["setup_id"] == problem["setup_id"]
