#!/usr/bin/env bash
# Activate the shared PyAuto venv and point PYTHONPATH at the canonical
# PyAuto* source checkouts on the HPC.
#
# Mirrors the pattern used by z_projects/euclid/activate.sh and
# autolens_assistant/activate.sh: pip handles only third-party deps in
# the venv, while the five PyAuto* libraries are resolved via PYTHONPATH
# directly from /mnt/ral/jnightin/PyAuto/<repo>. ``HPCPullPyAuto`` then
# becomes the only mechanism needed to keep PyAuto* current — no pip
# install of PyAuto* into the venv ever.
#
# Usage (inside a SLURM submit or interactive shell):
#
#     source /mnt/ral/jnightin/autolens_inference/activate.sh
#     python3 scripts/imaging/slam/hst_base.py ...

# --- SLURM exit-code guard ------------------------------------------------
# Inside a batch job, abort on the first failing command so the job's exit
# status is the failure's, not the last `date`'s.
#
# WHY (RAL job 341988, 2026-08-29). Every hpc/batch_* submit ends
#
#     python3 scripts/... ; echo "Finished." ; date
#
# so the script's exit status is `date`'s and is ALWAYS 0. The W6 delaunay
# n_batch tail died on a cuFFT batched plan wanting 25.31 GiB of scratch
# (`JaxRuntimeError`), ran straight on through those two lines, and SLURM
# recorded both array tasks as COMPLETED 0:0 in 1:02 and 1:07. `sacct` showed a
# clean fast run; the traceback existed only in the .err file, and the only
# other tell was the missing results JSON. That is a failure that reports
# success, and it must not be discoverable only by noticing an absence.
#
# `set -e` is scoped to a SLURM job on purpose: this file is also sourced in
# interactive login-node shells, where errexit would close the terminal on the
# first typo. `pipefail` is deliberately NOT set — the submits pipe nothing
# into python, so pipefail would add risk without covering the failure this
# guard is for. Commands whose non-zero exit is expected already guard
# themselves with `|| echo ...` (run_probe, run_cell), which errexit leaves
# alone.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    set -eE
    trap 'rc=$?; echo "FATAL: command exited ${rc} (line ${LINENO}) — failing the SLURM job rather than reporting COMPLETED 0:0." >&2; exit ${rc}' ERR
    # A SLURM `.out` is a file, not a tty, so Python block-buffers stdout at
    # 8 KiB. RAL 341908_5 (`slam_source_pix_nn`) ran for six hours making
    # 90,000 likelihood calls and its `.out` never advanced past
    # `Calls | 0`: every progress line the driver printed was sitting in an
    # unflushed buffer that the wall-clock kill then discarded. The job was
    # therefore read as "0 calls in 6 h / it thrashes" for two days, and the
    # real failure (a likelihood-overflow flood — DECISIONS.md 2026-08-29)
    # was only found in `checkpoint.hdf5`. Line buffering costs nothing at
    # these cadences and is the difference between a live job you can watch
    # and one you can only autopsy. Scoped to SLURM alongside `set -eE` for
    # the same reason: interactive login shells are not the failure mode.
    export PYTHONUNBUFFERED=1
fi

BASE=/mnt/ral/jnightin/PyAuto

source "$BASE/PyAuto/bin/activate"

export PYTHONPATH=$BASE:\
$BASE/PyAutoNerves:\
$BASE/PyAutoFit:\
$BASE/PyAutoArray:\
$BASE/PyAutoGalaxy:\
$BASE/PyAutoLens
