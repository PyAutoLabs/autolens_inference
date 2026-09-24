"""A multi-stage SLaM payload flattens into per-stage rows, and two legs render a
parity view.

The pipeline result JSON is one file per (config_name, seed) carrying a
``stages`` list, so the dashboard's one-JSON-is-one-row assumption had to be
widened. Two things must hold, and both are things `--check` in CI depends on:

1. Five stages in a file means five rows in the table, each carrying the
   payload's header (target, config, seed, version) — not one row of dashes.
2. Two legs of one parity row render a Δ/σ comparison of the `mass_total[1]`
   posterior, with the alphabetically-first config as the reference, and a
   single leg renders a stated absence rather than a fake column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
_TOOLING = str(REPO_ROOT / "scripts" / "misc" / "tooling")
if _TOOLING not in sys.path:
    sys.path.insert(0, _TOOLING)

import build_readme  # noqa: E402

STAGE_NAMES = build_readme.SLAM_STAGE_ORDER

#: Two legs of one parity row. `local_jax_cpu_dense_fp64` sorts first, so it is
#: the reference; every Δ/σ below is hand-computed against it.
REFERENCE_CONFIG = "local_jax_cpu_dense_fp64"
OTHER_CONFIG = "local_numba_cpu_sparse_fp64"

REFERENCE_POSTERIOR = {
    "einstein_radius": {"median": 1.600, "sigma": 0.020},
    "slope": {"median": 2.000, "sigma": 0.050},
    "shear_magnitude": {"median": 0.0707, "sigma": 0.0040},
    "gamma_1": {"median": 0.050, "sigma": 0.003},
}
OTHER_POSTERIOR = {
    # +0.010 / 0.020 = +0.50 sigma
    "einstein_radius": {"median": 1.610, "sigma": 0.021},
    # -0.075 / 0.050 = -1.50 sigma
    "slope": {"median": 1.925, "sigma": 0.052},
    # +0.0008 / 0.0040 = +0.20 sigma
    "shear_magnitude": {"median": 0.0715, "sigma": 0.0041},
    # 0.0 / 0.003 = +0.00 sigma
    "gamma_1": {"median": 0.050, "sigma": 0.003},
}


def _payload(
    config_name: str,
    backend: str,
    inversion: str,
    posterior: dict,
    *,
    variant: str | None = None,
    target: str = "hst/slam5/seed0",
    status: str | None = None,
    stage_names=STAGE_NAMES,
) -> dict:
    """One result payload.

    ``variant=None`` is a **schema v1** row: the field did not exist, and the
    row sits one directory shallower. Those are the four A100 base-run rows
    already committed, so every renderer has to keep reading them.
    """
    payload = {
        "schema_version": 1 if variant is None else 2,
        "target": target,
        "config_name": config_name,
        "backend": backend,
        "inversion": inversion,
        "precision": "fp64",
        "instrument": "hst",
        "seed": 0,
        "version": "2026.8.17.1",
        "stages": [
            {
                "name": name,
                "free_parameters": 12,
                "n_live": 150,
                "n_batch": 20,
                "wall_s": 100.0 + index,
                "likelihood_evals": 1000 + index,
                "log_evidence": -5000.0 - index,
                "log_evidence_err": None,
                "completed": True,
                "resumed": False,
                "positions_info_present": name != "source_lp[1]",
                "posterior": posterior if name == build_readme.PARITY_STAGE else {},
            }
            for index, name in enumerate(stage_names)
        ],
    }
    if variant is not None:
        payload["variant"] = variant
    if status is not None:
        payload["status"] = status
    return payload


def _write_results(tmp_path: Path, *payloads: dict) -> Path:
    slam_root = tmp_path / "results" / "slam" / "imaging" / "hst"
    for payload in payloads:
        # A v1 payload also lands at the v1 path — one level shallower, with no
        # variant segment — because that is what is on disk for those rows.
        directory = slam_root
        if "variant" in payload:
            directory = directory / payload["variant"]
        directory = directory / payload["config_name"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"stages_seed{payload['seed']}.json").write_text(json.dumps(payload))
    return slam_root


def _point_renderer_at(monkeypatch, tmp_path: Path, slam_root: Path) -> None:
    monkeypatch.setattr(build_readme, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_readme, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(build_readme, "SLAM_ROOT", slam_root)


def test_two_stage_payloads_flatten_into_ten_rows(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
        _payload(OTHER_CONFIG, "numba_cpu", "sparse", OTHER_POSTERIOR),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rows = build_readme._scan_rows(slam_root)

    assert len(rows) == 10, "two 5-stage payloads must flatten to ten rows"
    assert {row["stage"] for row in rows} == set(STAGE_NAMES)
    assert {row["config_name"] for row in rows} == {REFERENCE_CONFIG, OTHER_CONFIG}
    # Header fields ride along on every stage row.
    assert all(row["target"] == "hst/slam5/seed0" for row in rows)
    assert all(row["version"] == "2026.8.17.1" for row in rows)
    # Stage fields are lifted, not left nested.
    assert sorted(row["wall_s"] for row in rows) == sorted([100.0 + i for i in range(5)] * 2)


def test_flat_table_lists_every_stage_in_chain_order(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
        _payload(OTHER_CONFIG, "numba_cpu", "sparse", OTHER_POSTERIOR),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rendered = build_readme.render_slam()

    assert (
        "| Target | Variant | Stage | Config | Seed | Wall | Evals | log Z | Version |"
    ) in rendered
    body = [line for line in rendered.splitlines() if line.startswith("| `hst/slam5")]
    assert len(body) == 10
    # Chain order, not alphabetical: source_lp[1] leads, mass_total[1] closes.
    stages_in_order = [line.split("|")[3].strip() for line in body]
    assert stages_in_order[0] == "source_lp[1]"
    assert stages_in_order[-1] == "mass_total[1]"


def test_parity_view_compares_mass_total_posteriors(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
        _payload(OTHER_CONFIG, "numba_cpu", "sparse", OTHER_POSTERIOR),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rendered = build_readme.render_slam()

    assert "**Parity — `hst/slam5/seed0` (seed 0), `mass_total[1]`**" in rendered
    assert f"| Parameter | `{REFERENCE_CONFIG}` (ref) | `{OTHER_CONFIG}` |" in rendered

    lines = {
        line.split("|")[1].strip(): line for line in rendered.splitlines() if line.startswith("| `")
    }
    # Leading keys come first, in the fixed order, before the alphabetical rest.
    parity_keys = [
        line.split("|")[1].strip().strip("`")
        for line in rendered.splitlines()
        if line.startswith("| `einstein_radius")
        or line.startswith("| `slope")
        or line.startswith("| `shear_magnitude")
        or line.startswith("| `gamma_1")
    ]
    assert parity_keys == ["einstein_radius", "slope", "shear_magnitude", "gamma_1"]

    # Hand-computed Δ / reference sigma.
    assert "+0.50σ" in lines["`einstein_radius`"]
    assert "-1.50σ" in lines["`slope`"]
    assert "+0.20σ" in lines["`shear_magnitude`"]
    assert "+0.00σ" in lines["`gamma_1`"]
    # The reference column carries the value, not a comparison with itself.
    assert "1.6 ± 0.02" in lines["`einstein_radius`"]


def test_parity_view_states_the_absence_for_a_single_leg(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rendered = build_readme.render_slam()

    assert "a parity view needs at least two legs" in rendered
    assert "σ" not in rendered.split("**Parity")[1]


def test_rendering_a_stage_payload_is_a_fixed_point(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
        _payload(OTHER_CONFIG, "numba_cpu", "sparse", OTHER_POSTERIOR),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    assert build_readme.render_slam() == build_readme.render_slam()


# ---------------------------------------------------------------------------
# Run variants: the column, the v1 fallback, and the incomplete-row skip
# ---------------------------------------------------------------------------


def test_a_v1_row_with_no_variant_field_renders_as_the_base_variant(monkeypatch, tmp_path):
    """The four committed A100 rows are schema v1 at the v1 path. They predate
    the field entirely, and a row with no variant segment in its path is the
    base variant, which is what those rows are."""
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rendered = build_readme.render_slam()

    body = [line for line in rendered.splitlines() if line.startswith("| `hst/slam5")]
    assert len(body) == 5
    assert {line.split("|")[2].strip() for line in body} == {"`slam_base`"}


def test_a_v2_row_renders_its_variant_and_never_joins_the_base_parity_row(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR),
        _payload(OTHER_CONFIG, "numba_cpu", "sparse", OTHER_POSTERIOR),
        _payload(
            REFERENCE_CONFIG,
            "jax_cpu",
            "dense",
            OTHER_POSTERIOR,
            variant="delaunay_1250",
            target="hst/slam5_delaunay_1250/seed0",
        ),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rendered = build_readme.render_slam()

    variants = {
        line.split("|")[2].strip()
        for line in rendered.splitlines()
        if line.startswith("| `hst/slam5")
    }
    assert variants == {"`slam_base`", "`delaunay_1250`"}

    # Two targets, two parity groups. The Delaunay leg is a different
    # experiment, so it is a group of one and says so rather than sitting in the
    # base run's group as if it were a third backend.
    assert "**Parity — `hst/slam5/seed0` (seed 0)" in rendered
    assert "**Parity — `hst/slam5_delaunay_1250/seed0` (seed 0)" in rendered
    delaunay_block = rendered.split("**Parity — `hst/slam5_delaunay_1250/seed0`")[1]
    assert "a parity view needs at least two legs" in delaunay_block


def test_a_row_that_did_not_complete_is_not_rendered_as_a_result(monkeypatch, tmp_path):
    """A `--stages source_lp` probe writes one stage of five under the same
    target as the finished run beside it. Rendered, it reads as a result and
    joins that run's parity group."""
    slam_root = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR, status="complete"),
        _payload(
            OTHER_CONFIG,
            "numba_cpu",
            "sparse",
            OTHER_POSTERIOR,
            status="stopped_early",
            stage_names=STAGE_NAMES[:1],
        ),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    rendered = build_readme.render_slam()

    body = [line for line in rendered.splitlines() if line.startswith("| `hst/slam5")]
    assert len(body) == 5, "only the completed leg's five stages render"
    assert OTHER_CONFIG not in rendered
    assert "a parity view needs at least two legs" in rendered


def test_a_failed_row_is_skipped_too(monkeypatch, tmp_path):
    slam_root = _write_results(
        tmp_path,
        _payload(
            REFERENCE_CONFIG,
            "jax_cpu",
            "dense",
            REFERENCE_POSTERIOR,
            status="failed: RuntimeError: boom",
            stage_names=STAGE_NAMES[:2],
        ),
    )
    _point_renderer_at(monkeypatch, tmp_path, slam_root)

    assert "No pipeline runs yet" in build_readme.render_slam()


def test_archived_rows_are_never_live_rows(monkeypatch, tmp_path):
    """``results/archive/`` holds superseded rows kept for the record (the
    pre-likelihood-speedup A100 legs, 2026-09-24). The renderer scans
    ``results/slam/`` and ``results/searches/`` only, so an archived row — even
    a complete one at a full variant/config path — must never reach the
    dashboard or a parity view."""
    live = _write_results(
        tmp_path,
        _payload(REFERENCE_CONFIG, "jax_cpu", "dense", REFERENCE_POSTERIOR, variant="slam_base"),
    )
    archived = tmp_path / "results" / "archive" / "2026-09-24_pre_likelihood_speedup"
    archived_row = archived / "slam" / "imaging" / "hst" / "slam_base" / OTHER_CONFIG
    archived_row.mkdir(parents=True)
    (archived_row / "stages_seed0.json").write_text(
        json.dumps(
            _payload(
                OTHER_CONFIG,
                "numba_cpu",
                "sparse",
                OTHER_POSTERIOR,
                variant="slam_base",
                status="complete",
            )
        )
    )
    _point_renderer_at(monkeypatch, tmp_path, live)
    monkeypatch.setattr(build_readme, "SEARCHES_ROOT", tmp_path / "results" / "searches")

    assert {row["config_name"] for row in build_readme._scan_rows(live)} == {REFERENCE_CONFIG}
    assert OTHER_CONFIG not in build_readme.render_slam()
    assert build_readme._scan_rows(tmp_path / "results" / "searches") == []
