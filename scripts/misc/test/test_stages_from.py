"""``--stages-from``: a numba_cpu chain that loads the jax_cpu ``source_lp[1]``.

autolens_inference#31. The clean-slate numba_cpu base legs share the jax_cpu
dense chain's parametric ``source_lp[1]`` instead of fitting it. These tests
hold the seams where getting that wrong would produce a row that *looks* fine:

- the flag accepts only ``source_lp`` and only a config that is not this run's,
  at this run's precision;
- the shared stage's path_prefix names the OTHER config, at the same output
  root, variant and seed;
- an incomplete shared stage is a refusal (exit 3) before any fit or row;
- the row marks the stage ``shared_from_config`` and ``chain_wall_s`` excludes
  it, and the README and summary say so;
- the two submits pass the wall gate, wait in-job for the stage, and say what
  they claim.

The end-to-end check (a test-mode euclid jax_cpu ``--stages source_lp`` then a
numba_cpu chain sharing it) is the witness on the PR, not a unit test.
"""

from __future__ import annotations

import importlib.util
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
from slam import _runner  # noqa: E402
from wall import check_submits  # noqa: E402

import _inference_cli  # noqa: E402
from _inference_cli import InferenceCLI, parse_stages_from  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "summary_stages_from", REPO_ROOT / "scripts/misc/tooling/export_inference_summary.py"
)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)

JAX = "hpc_a100_jax_cpu_dense_fp64"


def _cli(**overrides) -> InferenceCLI:
    fields = {
        "config_name": "hpc_a100_numba_cpu_dense_fp64",
        "output_dir": None,
        "use_mixed_precision": False,
        "instrument": "hst",
        "backend": "numba_cpu",
        "inversion": "dense",
        "seed": 1,
        "use_sparse_operator": False,
        "rect_mesh": "bilinear",
        "mesh": "rect",
        "mesh_pixels": 28,
        "regularization": None,
        "memo": "off",
        "cores": 8,
        "stages": None,
        "stages_from": (JAX, ("source_lp",)),
    }
    fields.update(overrides)
    return InferenceCLI(**fields)


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def test_parse_accepts_source_lp_only():
    assert parse_stages_from(f"{JAX}:source_lp") == (JAX, ("source_lp",))
    assert parse_stages_from(f"{JAX}:source_lp,source_lp") == (JAX, ("source_lp",))
    for stage in ("source_pix_1", "source_pix_2", "light", "mass_total"):
        with pytest.raises(ValueError, match="cannot be shared"):
            parse_stages_from(f"{JAX}:{stage}")
    with pytest.raises(ValueError, match="cannot be shared"):
        parse_stages_from(f"{JAX}:source_lp,source_pix_1")
    with pytest.raises(ValueError, match="is not one of"):
        parse_stages_from(f"{JAX}:stage_one")


@pytest.mark.parametrize(
    "bad", [JAX, f"{JAX}:", ":source_lp", "jax_cpu_dense:source_lp", "hpc_a100_jax_cpu_dense:x"]
)
def test_parse_rejects_malformed_values(bad):
    with pytest.raises(ValueError):
        parse_stages_from(bad)


def test_cli_parses_and_rejects(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["leaf"])
    assert _inference_cli.parse_inference_cli().stages_from is None

    monkeypatch.setattr(sys, "argv", ["leaf", "--stages-from", f"{JAX}:source_lp"])
    assert _inference_cli.parse_inference_cli().stages_from == (JAX, ("source_lp",))

    monkeypatch.setattr(sys, "argv", ["leaf", "--stages-from", f"{JAX}:light"])
    with pytest.raises(SystemExit) as exc:
        _inference_cli.parse_inference_cli()
    assert exc.value.code == 2


def test_validate_refuses_self_and_precision_mismatch():
    assert _runner.validate_stages_from(_cli()) is None
    assert _runner.validate_stages_from(_cli(stages_from=None)) is None
    assert "own config" in _runner.validate_stages_from(
        _cli(stages_from=("hpc_a100_numba_cpu_dense_fp64", ("source_lp",)))
    )
    assert "precision" in _runner.validate_stages_from(
        _cli(stages_from=("hpc_a100_jax_cpu_dense_mp", ("source_lp",)))
    )


# ---------------------------------------------------------------------------
# Path substitution and the completion test
# ---------------------------------------------------------------------------


def test_the_shared_prefix_names_the_other_config_at_this_variant_and_seed():
    prefixes = _runner.shared_stage_prefixes(_cli(), "imaging", "hst", "slam_base", 1)
    assert prefixes == {"source_lp": Path("autolens/imaging/hst/slam/slam_base") / JAX / "seed_1"}
    own = _runner.output_path_prefix(
        "imaging", "hst", "slam_base", "hpc_a100_numba_cpu_dense_fp64", 1
    )
    assert prefixes["source_lp"].parent.parent == own.parent.parent
    assert _runner.shared_stage_prefixes(_cli(stages_from=None), "imaging", "hst", "x", 1) == {}


def test_a_delaunay_run_shares_the_base_runs_parametric_stage():
    """``source_lp[1]`` has no mesh, so it is one fit for every variant: a
    ``slam_delaunay`` leg (or ``rect_40``) loads the copy fitted under ``slam_base``,
    never a ``slam_delaunay/<config>/`` tree nobody fitted it in (autolens_inference#38)."""
    expected = {"source_lp": Path("autolens/imaging/hst/slam/slam_base") / JAX / "seed_0"}
    delaunay = _cli(mesh="delaunay", mesh_pixels=1250)
    assert _inference_cli.base_variant_name(delaunay) == "slam_delaunay"
    for variant in ("slam_delaunay", "rect_40", "slam_base"):
        assert _runner.shared_stage_prefixes(delaunay, "imaging", "hst", variant, 0) == expected


def _stage_dir(seed_dir: Path, *, completed: bool) -> Path:
    stage = seed_dir / "source_lp[1]" / "0123abcd"
    (stage / "files").mkdir(parents=True)
    if completed:
        (stage / ".completed").write_text("")
    return stage


def test_stage_completed_reads_the_completed_marker(tmp_path):
    seed_dir = tmp_path / "seed_0"
    assert not _runner.stage_completed(seed_dir, "source_lp[1]")
    stage = _stage_dir(seed_dir, completed=False)
    assert not _runner.stage_completed(seed_dir, "source_lp[1]")
    (stage / ".completed").write_text("")
    assert _runner.stage_completed(seed_dir, "source_lp[1]")
    assert _runner.main_stage_complete(["--stage-complete", str(seed_dir)]) == 0
    assert _runner.main_stage_complete(["--stage-complete", str(tmp_path / "nope")]) == 1


def test_wait_helper_returns_on_complete_and_exits_4_on_timeout(tmp_path):
    helper = REPO_ROOT / "hpc" / "wait_for_stage.sh"
    seed_dir = tmp_path / "seed_0"
    _stage_dir(seed_dir, completed=True)
    done = subprocess.run(
        ["bash", str(helper), str(seed_dir), "source_lp", "1", "0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "starting the chain" in done.stdout

    late = subprocess.run(
        ["bash", str(helper), str(tmp_path / "seed_1"), "source_lp", "1", "0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert late.returncode == 4
    assert "never" in late.stderr and "without fitting" in late.stderr


def test_an_incomplete_shared_stage_refuses_with_exit_3_before_any_fit(monkeypatch, tmp_path):
    """The refusal runs before the dataset loads: no fit, no row, exit 3."""
    pytest.importorskip("autolens")
    monkeypatch.setattr(_runner, "set_backend_env", lambda backend, cores: {})
    monkeypatch.setattr(
        _runner,
        "auto_simulate_if_missing",
        lambda *a, **k: pytest.fail("the refusal must come before the dataset"),
    )
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leaf",
            "--instrument",
            "euclid",
            "--backend",
            "jax_cpu",
            "--inversion",
            "dense",
            "--config-name",
            "local_jax_cpu_dense_fp64",
            "--stages-from",
            "local_jax_cpu_sparse_fp64:source_lp",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    # An identifier directory WITHOUT `.completed` (a running fit) is still a refusal.
    _stage_dir(
        tmp_path / "out/slam/imaging/euclid/slam_base/local_jax_cpu_sparse_fp64/seed_0",
        completed=False,
    )
    assert _runner.run_slam(default_instrument="euclid") == 3


# ---------------------------------------------------------------------------
# Row fields, README and summary
# ---------------------------------------------------------------------------


def _stage(name, wall):
    return {"name": name, "completed": True, "wall_s": wall, "total_wall_s": wall}


def test_shared_stage_is_marked_and_excluded_from_the_chain_wall():
    shared = _runner.mark_shared_stage(_stage("source_lp[1]", 1000.0), JAX, 2.5)
    assert shared["shared_from_config"] == JAX
    assert shared["fitted_backend"] == "jax_cpu" and shared["use_jax"] is True
    assert shared["wall_s"] == 1000.0, "the loaded fit's wall is kept for the record"
    assert shared["total_wall_s"] is None and shared["shared_load_s"] == 2.5

    fitted = [dict(_stage(n, 10.0), use_jax=False) for n in ("source_pix[1]", "mass_total[1]")]
    wall = _runner.chain_wall_from([shared, *fitted])
    assert wall["chain_wall_s"] == 20.0
    assert wall["chain_wall_excludes"] == ["source_lp[1]"]
    assert "excluded" in wall["chain_wall_note"]

    assert _runner.chain_wall_from(fitted)["chain_wall_excludes"] == []
    missing = [*fitted, _stage("light[1]", None)]
    assert _runner.chain_wall_from(missing)["chain_wall_s"] is None, "no partial sums"


def _payload():
    shared = _runner.mark_shared_stage(_stage("source_lp[1]", 1000.0), JAX, 2.5)
    stages = [shared] + [
        dict(
            _stage(name, 10.0),
            use_jax=False,
            posterior=(
                {"einstein_radius": {"median": 1.6, "sigma": 0.02}}
                if name == "mass_total[1]"
                else {}
            ),
        )
        for name in build_readme.SLAM_STAGE_ORDER[1:]
    ]
    return {
        "schema_version": 3,
        "target": _runner.target_id("hst", "slam_base", 0),
        "variant": "slam_base",
        "lens_light_mode": "linear",
        "results_tag": "cleanslate_2026_10_17",
        "config_name": "hpc_a100_numba_cpu_dense_fp64",
        "backend": "numba_cpu",
        "inversion": "dense",
        "precision": "fp64",
        "instrument": "hst",
        "dataset_class": "imaging",
        "seed": 0,
        "mesh": "rect",
        "use_jax": False,
        "version": "2026.10.17.1",
        "status": "complete",
        "stages_from": {"config_name": JAX, "stages": ["source_lp"]},
        **_runner.chain_wall_from(stages),
        "stages": stages,
    }


def _put(root, payload):
    path = (
        root
        / "results/slam/imaging/hst/slam_base"
        / f"{payload['config_name']}__{payload['results_tag']}"
        / "stages_seed0.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_readme_marks_the_shared_stage_and_renders_the_chain_wall(monkeypatch, tmp_path):
    _put(tmp_path, _payload())
    monkeypatch.setattr(build_readme, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_readme, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(build_readme, "SLAM_ROOT", tmp_path / "results" / "slam")
    rendered = build_readme.render_slam()
    lp = [line for line in rendered.splitlines() if "| source_lp[1] |" in line]
    assert len(lp) == 1 and f"shared from `{JAX}`" in lp[0]
    shared_wall = build_readme._format_time(1000.0)
    assert shared_wall not in lp[0], "the shared wall is never shown as this config's"
    assert "**Chain walls (fitted stages only)**" in rendered
    chain = rendered.split("**Chain walls (fitted stages only)**")[1]
    assert (
        build_readme._format_time(40.0) in chain
        and f"excludes source_lp[1] (shared from `{JAX}`)" in chain
    )


def test_summary_carries_the_shared_stage(tmp_path):
    _put(tmp_path, _payload())
    doc = summary.build(tmp_path, "a" * 40, "2026-10-17T00:00:00+00:00")
    assert doc["coverage"]["excluded"] == []
    stages = {r["stage"]: r for r in doc["records"] if r["stage"] is not None}
    lp = stages["source_lp[1]"]
    assert lp["configuration"]["shared_from_config"] == JAX
    assert lp["configuration"]["fitted_backend"] == "jax_cpu"
    assert lp["configuration"]["use_jax"] is True
    assert all("shared_from_config" in v for v in lp["timings"]["definitions"].values())
    pix = stages["source_pix[1]"]
    assert "shared_from_config" not in pix["configuration"]
    assert pix["configuration"]["use_jax"] is False


# ---------------------------------------------------------------------------
# The two submits
# ---------------------------------------------------------------------------

#: name -> (inversion, mem, config-name prefix, leaf, thread caps exported). The sparse
#: legs are the #38 reruns on `hpc_ral` with the BLAS pools pinned; the dense submit is
#: the superseded 10-17 leg, kept as written.
NUMBA = {
    "batch_cpu/submit_hst_cleanslate_numba_cpu_dense": (
        "dense",
        "48gb",
        "hpc_a100",
        "scripts/imaging/rectangular/validation/baseline_slam.py",
        False,
    ),
    "batch_cpu/submit_hst_cleanslate_numba_cpu_sparse": (
        "sparse",
        "64gb",
        "hpc_ral",
        "scripts/imaging/rectangular/validation/baseline_slam.py",
        True,
    ),
    "batch_cpu/submit_hst_cleanslate_numba_cpu_sparse_delaunay": (
        "sparse",
        "96gb",
        "hpc_ral",
        "scripts/imaging/delaunay/validation/baseline_slam.py",
        True,
    ),
}

THREAD_CAPS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@pytest.mark.parametrize("name", sorted(NUMBA))
def test_the_numba_submits_pass_the_gate_wait_and_share(name):
    inversion, mem, where, leaf, capped = NUMBA[name]
    config = f"{where}_numba_cpu_{inversion}_fp64"
    text = (REPO_ROOT / "hpc" / name).read_text()
    assert check_submits.check_text(text, name) == []
    for line in (
        "#SBATCH --partition=ral\n",
        "#SBATCH --cpus-per-task=8\n",
        f"#SBATCH --mem={mem}\n",
        "#SBATCH --time=5-00:00:00\n",
        "#SBATCH --array=0-1\n",
        "export PYAUTO_LIB_BASE=/mnt/ral/jnightin/inference/pins/2026.10.17.1",
        "export PYAUTO_RELEASE_PIN=2026.10.17.1",
        "export PYAUTO_DISABLE_JAX=1",
    ):
        assert line in text, line
    sbatch = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert [line for line in sbatch if "--partition" in line] == ["#SBATCH --partition=ral"]
    assert not any("--gres" in line for line in sbatch)
    assert "probe-first: yes" in text and "source: unmeasured" in text
    assert text.index("PYAUTO_LIB_BASE=") < text.index("source $AP_ROOT/activate.sh")
    for flag in (
        "--backend numba_cpu",
        f"--inversion {inversion}",
        f"--config-name {config}",
        "--lens-light both",
        f"--stages-from {JAX}:source_lp",
        f"--export-prepared output/autolens/prepared/{config}/seed_$SLURM_ARRAY_TASK_ID",
        "--seed $SLURM_ARRAY_TASK_ID",
    ):
        assert flag in text, flag
    command = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "--output-dir" not in command and "--results-tag" not in command
    shared_dir = f"output/autolens/imaging/hst/slam/slam_base/{JAX}/seed_$SLURM_ARRAY_TASK_ID"
    assert shared_dir in text
    wait = 'bash hpc/wait_for_stage.sh "$SHARED_SEED_DIR" source_lp 300 28800'
    assert wait in text
    assert text.index(wait) < text.index(f"python3 {leaf}")
    # #38: the reruns pin every BLAS/OpenMP pool to one thread BEFORE python3.
    for var in THREAD_CAPS:
        line = f"\nexport {var}=1\n"
        assert (line in text) == capped, var
        if capped:
            assert text.index(line) < text.index(f"python3 {leaf}")
