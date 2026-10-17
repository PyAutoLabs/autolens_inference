"""The experiment row writer: tuple parameter names never cost a completed fit's row.

RAL array 412543 (wave 1 attempt 1) completed every Nautilus/Emcee fit and then lost the
row: ``samples.names`` is PyAutoFit's ``model.all_names``, a list of *tuples* of dotted
aliases, and ``max_likelihood_parameters`` was keyed by them (autolens_inference#33).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]
from experiments import run  # noqa: E402
from experiments.prepared import json_bytes  # noqa: E402

#: What the installed PyAutoFit hands over: one tuple of aliases per prior.
NAMES = [
    ("fields.shear.gamma_1",),
    ("fields.shear.gamma_2",),
    ("galaxies.lens.mass.einstein_radius",),
    ("galaxies.lens.mass.centre.centre_0", "galaxies.source.light.centre.centre_0"),
]


class StubAf:
    @staticmethod
    def marginalize(parameter_list, sigma, weight_list):
        return sum(parameter_list) / len(parameter_list), min(parameter_list), max(parameter_list)


def result(names=NAMES, info=None):
    vectors = [[0.01, 0.02, 1.6, 0.1], [0.03, -0.01, 1.7, 0.2]]
    samples = SimpleNamespace(
        names=names,
        parameter_lists=vectors,
        weight_list=[0.5, 0.5],
        samples_info={"time": 12.5, "total_samples": 900} if info is None else info,
        sample_list=[object(), object()],
        max_log_likelihood_sample=SimpleNamespace(log_likelihood=-123.0),
        max_log_likelihood=lambda as_instance=False: vectors[1],
    )
    return SimpleNamespace(samples=samples)


def test_tuple_names_become_dotted_string_keys():
    assert run.parameter_key(("galaxies.lens.mass.einstein_radius",)) == (
        "galaxies.lens.mass.einstein_radius"
    )
    assert run.parameter_key(("galaxies", "lens", "mass", "slope")) == "galaxies.lens.mass.slope"
    assert run.parameter_key("already.dotted") == "already.dotted"


def test_row_with_tuple_names_serialises():
    fields = run.row_diagnostics(StubAf, result(), "nautilus")
    assert fields["max_likelihood_parameters"] == {
        "fields.shear.gamma_1": 0.03,
        "fields.shear.gamma_2": -0.01,
        "galaxies.lens.mass.einstein_radius": 1.7,
        "galaxies.lens.mass.centre.centre_0": 0.2,
    }
    assert set(fields["posterior"]) == {"gamma_1", "gamma_2", "einstein_radius", "centre_0"} | {
        "shear_magnitude"
    }
    assert fields["work"]["likelihood_evaluations"] == 900 and fields["wall_s"] == 12.5
    doc = json.loads(json_bytes(dict(status="complete", **fields)))
    assert all(isinstance(k, str) for k in doc["max_likelihood_parameters"])


def test_write_row_coerces_rather_than_losing_a_completed_row(tmp_path):
    destination = tmp_path / "results" / "row.json"
    row = dict(
        status="complete",
        completed=True,
        output_path="output/experiments/x",
        max_likelihood_parameters={("a.b",): 1.0, ("c", "d"): float("nan")},
    )
    run.write_row(destination, row)
    doc = json.loads(destination.read_text())
    assert doc["max_likelihood_parameters"] == {"a.b": 1.0, "c.d": None}
    assert doc["status"].startswith("complete; diagnostics failed: row serialisation coerced")
    assert doc["completed"] is True and doc["output_path"] == "output/experiments/x"


def test_write_row_keeps_a_failed_status_and_writes_a_clean_row_unchanged(tmp_path):
    destination = tmp_path / "row.json"
    run.write_row(destination, {"status": "failed: X: y", "bad": {(1, 2): 3}})
    assert json.loads(destination.read_text())["status"].startswith(
        "failed: X: y; row serialisation coerced"
    )
    clean = {"status": "complete", "completed": True, "max_log_likelihood": -1.0}
    run.write_row(destination, clean)
    assert destination.read_bytes() == json_bytes(clean)
