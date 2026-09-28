"""Unit tests for the point-source search runner's pure seams and its dashboard row.

No test runs a fit. They cover the places where a mistake produces a row that
*looks* fine:

- ``likelihood_share`` — the admission bar itself. A share silently clipped to
  1, or computed from a missing evaluation count, is a verdict made on a lie.
- ``time_calls`` — the warm-up calls must really be discarded, or the first
  post-compile call (2.4x steady state on a GPU, 2026-09-28) leaks into the
  median.
- ``point_truth_dict`` / ``truth_vector`` — keyed to agree with the model's
  posterior keys, so ``truth_delta_sigma`` is a key intersection, not a guess.
- the results path / target grammar, and the README renderer's search table.

``searches._point_runner`` must also be importable **without** importing
autolens, for the same reason as the SLaM runner: the backend environment is set
before the modelling stack loads.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (
    str(REPO_ROOT),
    str(REPO_ROOT / "scripts" / "misc"),
    str(REPO_ROOT / "scripts" / "misc" / "tooling"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_readme  # noqa: E402
from searches import _point_runner as runner  # noqa: E402


def test_importing_the_point_runner_does_not_import_autolens():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts' / 'misc')!r})\n"
        "import searches._point_runner\n"
        "leaked = [m for m in ('autolens', 'autofit', 'jax', 'numba') if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""


# ---------------------------------------------------------------------------
# likelihood_share
# ---------------------------------------------------------------------------


def test_likelihood_share_is_per_call_times_evals_over_wall():
    assert runner.likelihood_share(1e-3, 5000, 50.0) == pytest.approx(0.1)


def test_likelihood_share_is_not_clipped_above_one():
    # A share above 1 means the per-call cost was measured on a slower host state
    # than the fit ran in; clipping it would hide that.
    assert runner.likelihood_share(1e-2, 5000, 10.0) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "per_call, evals, wall",
    [(None, 100, 1.0), (1e-3, None, 1.0), (1e-3, 100, None), (1e-3, 100, 0.0), (1e-3, 100, -1)],
)
def test_likelihood_share_is_none_when_it_cannot_be_computed(per_call, evals, wall):
    assert runner.likelihood_share(per_call, evals, wall) is None


def test_likelihood_share_accepts_the_string_wall_clock_samples_info_writes():
    # samples_info["time"] is a string of seconds (autofit Timer.update).
    assert runner.likelihood_share(1e-3, 1000, "10.0") == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_time_calls_discards_the_warmup_and_times_each_call():
    calls = []

    def fn(arg):
        calls.append(arg)
        return arg

    summary = runner.time_calls(fn, 7, warmup=5, n=20)
    assert len(calls) == 25
    assert summary["n"] == 20
    assert summary["warmup"] == 5
    assert summary["min_s"] <= summary["median_s"] <= summary["p84_s"]


def test_time_calls_blocks_on_every_result():
    blocked = []
    runner.time_calls(lambda a: a, 1, warmup=2, n=3, block=blocked.append)
    assert len(blocked) == 5


def test_timing_summary_of_nothing_is_empty_not_an_error():
    assert runner.timing_summary([]) == {
        "median_s": None,
        "p16_s": None,
        "p84_s": None,
        "min_s": None,
        "n": 0,
    }


def test_timing_summary_median_and_quantiles():
    summary = runner.timing_summary([5.0, 1.0, 3.0, 2.0, 4.0])
    assert summary["median_s"] == 3.0
    assert summary["min_s"] == 1.0
    assert summary["p16_s"] == 2.0 or summary["p16_s"] == 1.0
    assert summary["p84_s"] in (4.0, 5.0)


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

TRUTH_RECORD = {
    "lens_centre": [0.0, 0.0],
    "lens_ell_comps": [0.05263157894736841, 3.2e-18],
    "lens_einstein_radius": 1.6,
    "source_centre": [0.07, 0.07],
}

MODEL_KEYS = ["centre_0", "centre_1", "ell_comps_0", "ell_comps_1", "einstein_radius"]


def test_point_truth_dict_keys_match_the_model_posterior_keys():
    truths = runner.point_truth_dict(TRUTH_RECORD)
    assert sorted(truths) == sorted(MODEL_KEYS)
    assert truths["einstein_radius"] == 1.6
    # The source centre is solved (PointSolved), never a free parameter.
    assert "source_centre" not in truths


def test_truth_vector_follows_the_model_order():
    truths = runner.point_truth_dict(TRUTH_RECORD)
    vector = runner.truth_vector(MODEL_KEYS, truths)
    assert vector == [0.0, 0.0, 0.05263157894736841, 3.2e-18, 1.6]


def test_truth_vector_is_none_when_a_parameter_has_no_truth():
    assert (
        runner.truth_vector(MODEL_KEYS + ["slope"], runner.point_truth_dict(TRUTH_RECORD)) is None
    )


# ---------------------------------------------------------------------------
# Paths + target
# ---------------------------------------------------------------------------


def test_results_paths_put_the_seed_in_the_filename(tmp_path):
    json_path, png_path = runner.results_paths(
        tmp_path, "point_source", "nautilus", "simple", "source_plane_solved", "cfg", 3
    )
    assert json_path == (
        tmp_path
        / "results/searches/point_source/nautilus/simple/source_plane_solved/cfg/search_seed3.json"
    )
    assert png_path.name == "search_seed3.png"
    assert json_path.parent.is_dir()


def test_output_path_prefix_carries_config_and_seed():
    prefix = runner.output_path_prefix(
        "point_source", "nautilus", "simple", "source_plane_solved", "cfg", 2
    )
    assert prefix.parts[-2:] == ("cfg", "seed_2")
    assert prefix.parts[0] == "searches"


def test_target_id():
    assert (
        runner.target_id("simple", "source_plane_solved", 4) == "simple/source_plane_solved/seed4"
    )


# ---------------------------------------------------------------------------
# README search table
# ---------------------------------------------------------------------------


def _row(seed: int, status: str = "complete") -> dict:
    return {
        "target": f"simple/source_plane_solved/seed{seed}",
        "sampler": "nautilus",
        "instrument": "simple",
        "config_name": "hpc_a100_jax_cpu_dense_fp64",
        "seed": seed,
        "status": status,
        "wall_s": 50.0,
        "likelihood_evals": 5000,
        "per_call_s": 6e-6,
        "likelihood_share": 6e-4,
        "log_evidence": -7.1,
        "version": "x",
    }


def test_render_searches_hides_incomplete_rows_and_shows_the_admission_bar(tmp_path, monkeypatch):
    directory = tmp_path / "point_source"
    directory.mkdir()
    (directory / "search_seed0.json").write_text(json.dumps(_row(0)))
    (directory / "search_seed1.json").write_text(json.dumps(_row(1, "failed: boom")))
    monkeypatch.setattr(build_readme, "SEARCHES_ROOT", tmp_path)
    monkeypatch.setattr(build_readme, "RESULTS_ROOT", tmp_path)

    rendered = build_readme.render_searches()
    assert "seed0" in rendered
    assert "seed1" not in rendered
    assert "6.0 µs" in rendered
    assert "0.060%" in rendered
    assert "5,000" in rendered


def test_format_share_and_per_call():
    assert build_readme._format_share(0.25) == "25.0%"
    assert build_readme._format_share(None) == "—"
    assert build_readme._format_per_call(2.5e-4) == "250.0 µs"
    assert build_readme._format_per_call(0.0125) == "12.50 ms"
