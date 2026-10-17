"""Wave-1 arm table: the index mapping, the leaf command and the row/failed/missing accounting."""

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]
from experiments import wave  # noqa: E402
from wall.check_submits import cells_run  # noqa: E402

GPU = "hpc_a100_jax_gpu_dense_fp64"
CPU = "hpc_a100_jax_cpu_dense_fp64"


def test_table_is_24_seed_major_arms():
    arms = wave.arms()
    assert len(arms) == 24
    assert [a.index for a in arms] == list(range(24))
    assert len({(a.variant, a.sampler, a.seed) for a in arms}) == 24
    for a in arms:
        assert a.index == a.seed * 8 + wave.VARIANTS.index(a.variant) * 4 + wave.SAMPLERS.index(
            a.sampler
        )
    # %4 starts one arm of every sampler, on slam_base, sampler seed 0.
    assert [(a.variant, a.sampler, a.seed) for a in arms[:4]] == [
        ("slam_base", s, 0) for s in ("nautilus", "emcee", "nuts", "smc")
    ]
    with pytest.raises(ValueError):
        wave.arm(24)
    with pytest.raises(ValueError):
        wave.arm(-1)


def test_run_ids_follow_the_grammar_and_are_unique():
    ids = [a.run_id(GPU) for a in wave.arms()]
    assert len(set(ids)) == 24
    assert ids[0] == "cs1017_hpc_a100_jax_gpu_dense_fp64_slam_base_nautilus_s0"
    assert ids[23] == "cs1017_hpc_a100_jax_gpu_dense_fp64_slam_base_lightfixed_smc_s2"
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", i) for i in ids)
    assert not set(ids) & {a.run_id(CPU) for a in wave.arms()}


def test_settings_are_the_wave_1_settings():
    assert wave.SETTINGS == {
        "nautilus": ("--live", "150", "--batch", "20"),
        "emcee": ("--walkers", "50", "--steps", "2000"),
        "nuts": ("--chains", "1", "--warmup", "500", "--steps", "1000"),
        "smc": ("--particles", "256", "--steps", "1000"),
    }


def manifest(root, config, variant, pid):
    path = root / wave.manifest_path(config, 0, variant)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "prepared-inference-problem", "id": pid}))


def test_leaf_args_read_the_problem_id_from_the_manifest(tmp_path):
    manifest(tmp_path, GPU, "slam_base", "prepared-aaa")
    manifest(tmp_path, GPU, "slam_base_lightfixed", "prepared-bbb")
    args = wave.leaf_args(tmp_path, 14, GPU, 0, "ral-a100-1gpu-4cpu", 4)
    a = wave.arm(14)
    assert (a.variant, a.sampler, a.seed) == ("slam_base_lightfixed", "nuts", 1)
    opts = dict(zip(args[::2], args[1::2]))
    assert opts["--prepared"] == (
        "output/prepared/cleanslate_2026_10_17/hpc_a100_jax_gpu_dense_fp64/seed_0/"
        "slam_base_lightfixed/manifest.json"
    )
    assert opts["--problem-id"] == "prepared-bbb"
    assert opts["--run-id"] == a.run_id(GPU)
    assert opts["--seed"] == "1" and opts["--cores"] == "4"
    assert opts["--start"] == "cold" and opts["--compilation"] == "cold"
    assert opts["--hardware-id"] == "ral-a100-1gpu-4cpu"
    assert (opts["--warmup"], opts["--steps"], opts["--chains"]) == ("500", "1000", "1")


def test_leaf_args_refuse_a_missing_manifest_or_bad_config(tmp_path):
    with pytest.raises(ValueError, match="manifest missing"):
        wave.leaf_args(tmp_path, 0, GPU, 0, "h", 1)
    with pytest.raises(ValueError, match="grammar"):
        wave.manifest_path("gpu_dense", 0, "slam_base")


def test_every_arm_resolves_to_an_existing_leaf(tmp_path, monkeypatch, capsys):
    """Every arm resolves to a full command naming a leaf that exists (the witness runs them)."""
    for variant in wave.VARIANTS:
        manifest(tmp_path, GPU, variant, f"prepared-{variant}")
    monkeypatch.setattr(wave, "ROOT", tmp_path)
    for a in wave.arms():
        rc = wave.main(
            ["--arm", str(a.index), "--config", GPU, "--hardware-id", "h", "--print-command"]
        )
        words = shlex.split(capsys.readouterr().out)
        assert rc == 0 and words[:2] == ["python3", a.leaf]
        assert (ROOT / a.leaf).is_file()


def test_submit_sampler_order_is_checked(capsys):
    ok = ["--arm", "5", "--config", GPU, "--field", "sampler_index"]
    assert wave.main([*ok, "--samplers", "nautilus emcee nuts smc"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert wave.main([*ok, "--samplers", "emcee nautilus nuts smc"]) == 2


def row(results, a, config, **kw):
    results.mkdir(parents=True, exist_ok=True)
    doc = dict(
        sampler=a.sampler, seed=a.seed, config_name=config, variant=a.variant, problem_id="p"
    )
    doc.update(kw)
    (results / f"{a.run_id(config)}.json").write_text(json.dumps(doc))


def test_account_reports_row_failed_and_missing_never_success(tmp_path):
    results = tmp_path / "results"
    arms = wave.arms()
    row(results, arms[0], GPU, status="complete", completed=True)
    row(results, arms[1], GPU, status="failed: RuntimeError: boom", completed=False)
    row(results, arms[2], GPU, status="incomplete", completed=False)
    row(results, arms[3], GPU, status="complete", completed=True, seed=7)
    row(results, arms[4], GPU, status="complete")  # no completion marker
    (results / f"{arms[5].run_id(GPU)}.json").write_text("{not json")
    row(results, arms[6], CPU, status="complete", completed=True)  # another config's row
    states = {e["index"]: e for e in wave.account(results, GPU)}
    assert len(states) == 24
    assert states[0]["state"] == "row"
    assert states[1]["state"] == "failed" and "boom" in states[1]["detail"]
    assert states[2]["state"] == "failed" and states[2]["detail"] == "incomplete"
    assert states[3]["state"] == "failed" and "identity mismatch" in states[3]["detail"]
    assert states[4]["state"] == "failed"
    assert states[5]["state"] == "failed" and "unreadable" in states[5]["detail"]
    assert all(states[i]["state"] == "missing" for i in range(6, 24))


@pytest.mark.parametrize(
    "path, config",
    [
        ("hpc/batch_gpu/submit_hst_cleanslate_mass_samplers_jax_gpu_dense", GPU),
        ("hpc/batch_cpu/submit_hst_cleanslate_mass_samplers_jax_cpu_dense", CPU),
    ],
)
def test_submits_agree_with_the_table(path, config):
    text = (ROOT / path).read_text()
    assert f"CONFIG={config}\n" in text
    assert f"SAMPLERS=({' '.join(wave.SAMPLERS)})" in text
    assert f"--array=0-{len(wave.arms()) - 1}%" in text
    cells, _ = cells_run(text)
    assert cells == {f"imaging/rectangular/{s}" for s in wave.SAMPLERS}
    if "batch_cpu" in path:
        assert "--partition=ral\n" in text and "gpu" not in re.findall(r"--partition=(\S+)", text)
