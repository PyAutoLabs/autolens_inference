"""`hpc/sync submit` carries sbatch overrides and environment variables to RAL, logged.

Every likelihood-lifecycle phase-4 timing resubmit needed a 24 h `--time`, `--no-requeue`
and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, and `hpc/sync submit` could pass none of
them, so each went through a raw `sbatch` under a scheduler-guard exception
(autolens_inference#71). These tests run the real script against a stub `ssh`, which runs
the remote command in a local bash, and a stub `sbatch`, which records the arguments and
the environment it was given. So they check what reaches sbatch on the far side of the
quoting, not what the script meant to send.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
SYNC = REPO_ROOT / "hpc" / "sync"

SCRIPT = "submit_p4_validation_mge_hst_jax_cpu_dense"
EIGEN = "--xla_cpu_multi_thread_eigen=false"

_SSH = """#!/bin/bash
# Stub ssh: drop the host, run the remote command line in a fresh local shell, as sshd does.
shift
printf '%s\\n' "$*" >> "$STUB_LOG_DIR/ssh_calls"
exec bash -c "$*"
"""

_SBATCH = """#!/bin/bash
# Stub sbatch: record where it ran, its argv and the variables under test. The variables
# are read here, in bash: a Python environment may set XLA_FLAGS itself at startup.
STUB_XLA_FLAGS="${XLA_FLAGS-<unset>}" STUB_SPACED="${SPACED-<unset>}" python3 - "$@" <<'PY'
import json, os, sys
def seen(name):
    value = os.environ[name]
    return None if value == "<unset>" else value
with open(os.path.join(os.environ["STUB_LOG_DIR"], "sbatch_calls"), "a") as f:
    f.write(json.dumps({
        "cwd": os.getcwd(),
        "argv": sys.argv[1:],
        "XLA_FLAGS": seen("STUB_XLA_FLAGS"),
        "SPACED": seen("STUB_SPACED"),
    }) + "\\n")
PY
echo "Submitted batch job 414999"
exit "${STUB_SBATCH_EXIT:-0}"
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def rig(tmp_path):
    """A laptop project with no sync.conf, a fake RAL checkout, and stubs on PATH."""
    laptop = tmp_path / "laptop" / "autolens_inference"
    (laptop / "hpc").mkdir(parents=True)
    shutil.copy(SYNC, laptop / "hpc" / "sync")

    hpc_base = tmp_path / "ral"
    remote_root = hpc_base / "autolens_inference"
    for kind in ("cpu", "gpu"):
        (remote_root / "hpc" / f"batch_{kind}").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "ssh", _SSH)
    _executable(bin_dir / "sbatch", _SBATCH)
    _executable(
        bin_dir / "scancel", '#!/bin/bash\necho "scancel $*" >> "$STUB_LOG_DIR/scancel_calls"\n'
    )

    logs = tmp_path / "stub_logs"
    logs.mkdir()

    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "HPC_HOST": "ral_stub",
        "HPC_BASE": str(hpc_base),
        "STUB_LOG_DIR": str(logs),
    }

    def run(*args, extra_env=None):
        return subprocess.run(
            ["bash", str(laptop / "hpc" / "sync"), *args],
            env={**env, **(extra_env or {})},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def sbatch_calls():
        f = logs / "sbatch_calls"
        return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []

    def ssh_calls():
        f = logs / "ssh_calls"
        return f.read_text().splitlines() if f.exists() else []

    def actions(campaign="adhoc"):
        f = remote_root / "campaign" / campaign / "actions.log"
        return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []

    run.sbatch_calls = sbatch_calls
    run.ssh_calls = ssh_calls
    run.actions = actions
    run.remote_root = remote_root
    run.logs = logs
    return run


def test_phase4_resubmit_reaches_sbatch_with_time_requeue_and_eigen_flag(rig):
    result = rig(
        "submit",
        "--cpu",
        "--campaign",
        "p4_validation",
        "--time",
        "24:00:00",
        "--no-requeue",
        "--env",
        f"XLA_FLAGS={EIGEN}",
        SCRIPT,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    (call,) = rig.sbatch_calls()
    assert call["argv"] == ["--time=24:00:00", "--no-requeue", SCRIPT]
    assert call["XLA_FLAGS"] == EIGEN
    # Relative `#SBATCH -o/-e` paths need sbatch to run beside the submit scripts.
    assert Path(call["cwd"]) == rig.remote_root / "hpc" / "batch_cpu"

    (line,) = rig.actions("p4_validation")
    assert line["campaign"] == "p4_validation"
    assert line["action"] == "submit"
    assert line["args"] == [
        "--cpu",
        "--time=24:00:00",
        "--no-requeue",
        f"--env=XLA_FLAGS={EIGEN}",
        SCRIPT,
    ]
    assert line["exit"] == 0
    assert line["utc"].endswith("Z") and line["user"]
    assert '"action": "submit"' in result.stdout  # the line is printed, too


def test_equals_forms_mem_array_dependency_and_gpu_default(rig):
    result = rig(
        "submit",
        "--mem=64gb",
        "--array=0-11%6",
        "--dependency=afterok:414392",
        "--time=1-00:00:00",
        "submit_hst_cleanslate_mass_samplers_jax_gpu_dense",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (call,) = rig.sbatch_calls()
    assert call["argv"] == [
        "--mem=64gb",
        "--array=0-11%6",
        "--dependency=afterok:414392",
        "--time=1-00:00:00",
        "submit_hst_cleanslate_mass_samplers_jax_gpu_dense",
    ]
    assert Path(call["cwd"]) == rig.remote_root / "hpc" / "batch_gpu"
    assert call["XLA_FLAGS"] is None
    (line,) = rig.actions()  # no --campaign, no $CAMPAIGN: adhoc
    assert line["args"][0] == "--gpu"


def test_env_value_with_spaces_and_quotes_arrives_intact(rig):
    value = """--a=1 --b="two words" it's $HOME"""
    result = rig("submit", "--cpu", "--env", f"SPACED={value}", SCRIPT)
    assert result.returncode == 0, result.stdout + result.stderr
    (call,) = rig.sbatch_calls()
    assert call["SPACED"] == value
    (line,) = rig.actions()
    assert line["args"][1] == f"--env=SPACED={value}"


def test_campaign_from_environment(rig):
    result = rig("submit", "--cpu", SCRIPT, extra_env={"CAMPAIGN": "lifecycle_p4"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(rig.actions("lifecycle_p4")) == 1


def test_a_failed_sbatch_is_logged_with_its_exit_code(rig):
    result = rig("submit", "--cpu", SCRIPT, extra_env={"STUB_SBATCH_EXIT": "1"})
    assert result.returncode != 0
    (line,) = rig.actions()
    assert line["exit"] == 1


def test_unwritable_log_submits_nothing(rig):
    log_dir = rig.remote_root / "campaign" / "adhoc"
    log_dir.mkdir(parents=True)
    (log_dir / "actions.log").mkdir()  # a directory where the log should be: not writable as a file
    result = rig("submit", "--cpu", SCRIPT)
    assert result.returncode != 0
    assert "nothing was done" in result.stderr
    assert rig.sbatch_calls() == []


def test_dry_run_touches_nothing(rig):
    result = rig(
        "submit",
        "--cpu",
        "--dry-run",
        "--time",
        "24:00:00",
        "--no-requeue",
        "--env",
        f"XLA_FLAGS={EIGEN}",
        SCRIPT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert rig.ssh_calls() == []
    assert rig.sbatch_calls() == []
    assert "sbatch --time=24:00:00 --no-requeue" in result.stdout
    assert "export XLA_FLAGS=" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--partition", "gpu"],  # not on the allow-list
        ["--export=ALL,X=1"],  # not on the allow-list
        ["--time", "24h"],
        ["--time", "24:00:00;id"],
        ["--mem", "64gb $(id)"],
        ["--array", "a-b"],
        ["--dependency", "afterok:1;id"],
        ["--env", "1BAD=x"],
        ["--env", "NOEQUALS"],
        ["--env", "X=line1\nline2"],
        ["--campaign", "../escape"],
        ["--time"],  # missing value
        ["other_script"],  # a second script
    ],
)
def test_refused_before_anything_runs(rig, args):
    result = rig("submit", "--cpu", SCRIPT, *args)
    assert result.returncode != 0
    assert "submit:" in result.stderr
    assert rig.ssh_calls() == []


def test_script_name_must_be_a_plain_name(rig):
    result = rig("submit", "--cpu", "../../etc/passwd")
    assert result.returncode != 0
    assert rig.ssh_calls() == []


def test_cancel_is_logged(rig):
    result = rig("cancel", "414392")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (rig.logs / "scancel_calls").read_text().strip() == "scancel 414392"
    (line,) = rig.actions()
    assert line["action"] == "cancel" and line["args"] == ["414392"] and line["exit"] == 0


def test_cancel_refuses_a_non_job_id(rig):
    result = rig("cancel", "414392;id")
    assert result.returncode != 0
    assert rig.ssh_calls() == []
