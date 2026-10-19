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
        "output/autolens/prepared/hpc_a100_jax_gpu_dense_fp64/seed_0/"
        "slam_base_lightfixed/manifest.json"
    )
    assert opts["--problem-id"] == "prepared-bbb"
    assert opts["--run-id"] == a.run_id(GPU)
    assert opts["--seed"] == "1" and opts["--cores"] == "4"
    assert opts["--start"] == "cold" and opts["--compilation"] == "cold"
    assert opts["--hardware-id"] == "ral-a100-1gpu-4cpu"
    assert (opts["--warmup"], opts["--steps"], opts["--chains"]) == ("500", "1000", "1")


def test_prepared_problems_live_under_the_flat_autolens_root():
    """#35 dropped the dated level from the folder; the wave and run-id names keep it."""
    assert wave.PREPARED_ROOT == "output/autolens/prepared"
    assert wave.WAVE_ID == "cleanslate_2026_10_17_wave1" and wave.RUN_PREFIX == "cs1017"


def test_leaf_args_refuse_a_missing_manifest_or_bad_config(tmp_path):
    with pytest.raises(ValueError, match="manifest missing"):
        wave.leaf_args(tmp_path, 0, GPU, 0, "h", 1)
    with pytest.raises(ValueError, match="grammar"):
        wave.manifest_path("gpu_dense", 0, "slam_base")


def test_every_arm_resolves_to_an_existing_leaf(tmp_path, monkeypatch, capsys):
    """Every runnable arm resolves to a full command naming a leaf that exists."""
    for variant in wave.VARIANTS:
        manifest(tmp_path, GPU, variant, f"prepared-{variant}")
    monkeypatch.setattr(wave, "ROOT", tmp_path)
    for a in wave.arms():
        rc = wave.main(
            ["--arm", str(a.index), "--config", GPU, "--hardware-id", "h", "--print-command"]
        )
        out = capsys.readouterr()
        if a.deferred:
            assert rc == 2 and "deferred" in out.err and "OOM" in out.err
            continue
        words = shlex.split(out.out)
        assert rc == 0 and words[:2] == ["python3", a.leaf]
        assert (ROOT / a.leaf).is_file()


def test_nuts_and_smc_are_deferred_with_the_oom_evidence():
    assert set(wave.DEFERRED) == {"nuts", "smc"}
    for reason in wave.DEFERRED.values():
        assert "A100 OOM 2026-10-17 (385 GiB NUTS, 48 GiB SMC)" in reason
    assert [a.sampler for a in wave.arms() if not a.deferred] == ["nautilus", "emcee"] * 6


def test_attempt_suffix_gives_a_new_run_id():
    a = wave.arm(0)
    assert (
        a.run_id(GPU)
        == a.run_id(GPU, 1)
        == ("cs1017_hpc_a100_jax_gpu_dense_fp64_slam_base_nautilus_s0")
    )
    assert a.run_id(GPU, 2) == a.run_id(GPU) + "_a2"
    assert re.fullmatch(r"[A-Za-z0-9_]+", a.run_id(GPU, 12))
    for bad in (0, -1, "2"):
        with pytest.raises(ValueError, match="attempt"):
            a.run_id(GPU, bad)


def test_sampler_filter_selects_the_twelve_nautilus_and_emcee_arms(tmp_path, monkeypatch, capsys):
    assert wave.sampler_filter(None) == wave.SAMPLERS
    assert wave.sampler_filter("") == wave.SAMPLERS
    assert wave.sampler_filter("nautilus,emcee") == ("nautilus", "emcee")
    assert wave.sampler_filter("nautilus emcee") == ("nautilus", "emcee")
    with pytest.raises(ValueError, match="unknown sampler"):
        wave.sampler_filter("nautilus,dynesty")
    chosen = wave.selection(("nautilus", "emcee"))
    assert [a.index for a in chosen] == [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]
    with pytest.raises(ValueError, match="outside the 12 selected"):
        wave.select(12, ("nautilus", "emcee"))
    with pytest.raises(ValueError, match="deferred"):
        wave.select(0, ("nuts",))
    for variant in wave.VARIANTS:
        manifest(tmp_path, GPU, variant, f"prepared-{variant}")
    monkeypatch.setattr(wave, "ROOT", tmp_path)
    flags = ["--config", GPU, "--attempt", "2", "--sampler-filter", "nautilus,emcee"]
    for position, a in enumerate(chosen):
        assert wave.main(["--arm", str(position), *flags, "--field", "index"]) == 0
        assert capsys.readouterr().out.strip() == str(a.index)
        rc = wave.main(["--arm", str(position), *flags, "--hardware-id", "h", "--print-args"])
        opts = shlex.split(capsys.readouterr().out)
        assert rc == 0 and opts[opts.index("--run-id") + 1] == a.run_id(GPU, 2)
    assert wave.main(["--arm", "12", *flags, "--field", "index"]) == 2


def test_list_shows_the_filtered_attempt(capsys):
    flags = ["--config", GPU, "--attempt", "2", "--sampler-filter", "nautilus,emcee"]
    assert wave.main(["--list", *flags]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    assert len(lines) == 12 and all(ln.split()[3].endswith("_a2") for ln in lines)
    assert wave.main(["--list", "--config", GPU]) == 0
    assert capsys.readouterr().out.count("DEFERRED") == 12


def test_submit_sampler_order_is_checked(capsys):
    ok = ["--arm", "5", "--config", GPU, "--field", "sampler_index"]
    assert wave.main([*ok, "--samplers", "nautilus emcee nuts smc"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert wave.main([*ok, "--samplers", "emcee nautilus nuts smc"]) == 2


def row(results, a, config, attempt=1, **kw):
    results.mkdir(parents=True, exist_ok=True)
    doc = dict(
        sampler=a.sampler, seed=a.seed, config_name=config, variant=a.variant, problem_id="p"
    )
    doc.update(kw)
    (results / f"{a.run_id(config, attempt)}.json").write_text(json.dumps(doc))


def test_account_reports_row_failed_missing_and_deferred_never_success(tmp_path):
    results = tmp_path / "results"
    arms = wave.arms()  # 0 nautilus, 1 emcee, 2 nuts, 3 smc; 4-7 lightfixed; 8-15 seed 1
    row(results, arms[0], GPU, status="complete", completed=True)
    row(results, arms[1], GPU, status="failed: RuntimeError: boom", completed=False)
    row(results, arms[2], GPU, status="failed: JaxRuntimeError: OOM", completed=False)
    row(results, arms[4], GPU, status="incomplete", completed=False)
    row(results, arms[5], GPU, status="complete", completed=True, seed=7)
    row(results, arms[8], GPU, status="complete")  # no completion marker
    (results / f"{arms[9].run_id(GPU)}.json").write_text("{not json")
    row(results, arms[12], CPU, status="complete", completed=True)  # another config's row
    row(results, arms[13], GPU, status="complete; diagnostics failed: KeyError", completed=True)
    states = {e["index"]: e for e in wave.account(results, GPU)}
    assert len(states) == 24
    assert states[0]["state"] == "row"
    assert states[1]["state"] == "failed" and "boom" in states[1]["detail"]
    assert states[4]["state"] == "failed" and states[4]["detail"] == "incomplete"
    assert states[5]["state"] == "failed" and "identity mismatch" in states[5]["detail"]
    assert states[8]["state"] == "failed"
    assert states[9]["state"] == "failed" and "unreadable" in states[9]["detail"]
    assert states[12]["state"] == "missing"
    assert states[13]["state"] == "failed" and "diagnostics failed" in states[13]["detail"]
    deferred = [i for i, e in states.items() if e["state"] == "deferred"]
    assert deferred == [a.index for a in arms if a.sampler in ("nuts", "smc")]
    assert "OOM" in states[2]["detail"]
    # A deferred arm still lists the attempt it made.
    assert states[2]["attempts"][0]["state"] == "failed"
    runnable = {a.index for a in arms if not a.deferred}
    missing = runnable - {0, 1, 4, 5, 8, 9, 13}
    assert all(states[i]["state"] == "missing" for i in missing)


def test_account_lists_every_attempt_and_the_live_row(tmp_path, capsys):
    results = tmp_path / "results"
    a, b = wave.arm(0), wave.arm(1)
    row(results, a, GPU, status="failed: TypeError: keys must be str", completed=False)
    row(results, a, GPU, attempt=2, status="complete", completed=True)
    row(results, b, GPU, status="failed: TypeError: keys must be str", completed=False)
    states = {e["index"]: e for e in wave.account(results, GPU)}
    assert [h["state"] for h in states[0]["attempts"]] == ["failed", "row"]
    assert states[0]["state"] == "row" and states[0]["run_id"] == a.run_id(GPU, 2)
    assert states[0]["attempt"] == 2
    # Attempt 2 is expected: the arm without an attempt-2 row is live-missing, attempt 1 kept.
    states = {e["index"]: e for e in wave.account(results, GPU, attempt=2)}
    assert [h["state"] for h in states[1]["attempts"]] == ["failed", "missing"]
    assert states[1]["state"] == "missing" and states[1]["run_id"] == b.run_id(GPU, 2)
    # A later failed attempt never hides an earlier complete row.
    row(results, a, GPU, attempt=3, status="failed: boom", completed=False)
    states = {e["index"]: e for e in wave.account(results, GPU)}
    assert states[0]["state"] == "row" and states[0]["attempt"] == 2
    flags = ["--config", GPU, "--results-dir", str(results), "--attempt", "2"]
    assert wave.main(["--account", *flags, "--sampler-filter", "nautilus,emcee"]) == 0
    out = capsys.readouterr().out
    assert "attempt 1  failed" in out and "12 expected" in out and "0 deferred" in out
    assert wave.main(["--account", *flags]) == 0
    assert "12 deferred" in capsys.readouterr().out


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
    runnable = tuple(s for s in wave.SAMPLERS if s not in wave.DEFERRED)
    assert "WAVE_SAMPLERS=${PYAUTO_WAVE_SAMPLERS:-" + ",".join(runnable) + "}" in text
    assert f"--array=0-{len(wave.selection(runnable)) - 1}%" in text
    attempt = "2" if "batch_gpu" in path else "1"
    assert "ATTEMPT=${PYAUTO_WAVE_ATTEMPT:-" + attempt + "}" in text
    assert "--attempt $ATTEMPT --sampler-filter $WAVE_SAMPLERS" in text
    cells, _ = cells_run(text)
    assert cells == {f"imaging/rectangular/{s}" for s in wave.SAMPLERS}
    if "batch_cpu" in path:
        assert "--partition=ral\n" in text and "gpu" not in re.findall(r"--partition=(\S+)", text)
