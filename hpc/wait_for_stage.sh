#!/bin/bash
#
# In-job wait for another chain's completed SLaM stage (autolens_inference#31).
#
#     bash $AP_ROOT/hpc/wait_for_stage.sh <seed_dir> [stage] [poll_s] [max_s]
#
# Polls `python3 scripts/misc/slam/_runner.py --stage-complete <seed_dir> --stage <stage>`
# (the same `.completed` test the runner applies to a `--stages-from` stage) every
# `poll_s` seconds (default 300), up to `max_s` (default 28800 = 8 h). Exit 0 once the stage
# is complete; exit 4 with a message on timeout, so the job fails loudly rather than fitting a
# stage it was meant to share. Run from $AP_ROOT with the venv active.
#
# A poll, not a cell: it fits nothing, which is why it lives here and not in a submit's
# `python3 scripts/...` lines (check_submits reads those as the cells the job runs).
#
# When it had to wait, it sleeps a further `settle_s` (120 s) after the marker appears: the
# owning job writes `.completed` after its archive, then tidies the folder in post_fit_output.

set -u

SEED_DIR="${1:?usage: wait_for_stage.sh <seed_dir> [stage] [poll_s] [max_s]}"
STAGE="${2:-source_lp}"
POLL_S="${3:-300}"
MAX_S="${4:-28800}"
SETTLE_S=120

waited=0
while ! python3 scripts/misc/slam/_runner.py --stage-complete "$SEED_DIR" --stage "$STAGE"; do
    if [ "$waited" -ge "$MAX_S" ]; then
        echo "ERROR: waited ${waited}s (cap ${MAX_S}s) for ${STAGE} under ${SEED_DIR}; it never" \
            "completed. Exiting 4 without fitting: this chain shares that stage (--stages-from)" \
            "and must not fit it itself. Check the owning job, then resubmit." >&2
        exit 4
    fi
    echo "$(date -u +%FT%TZ) waiting for ${STAGE} (${waited}s of ${MAX_S}s); next poll in ${POLL_S}s"
    sleep "$POLL_S"
    waited=$((waited + POLL_S))
done

if [ "$waited" -gt 0 ]; then
    echo "$(date -u +%FT%TZ) ${STAGE} complete after ~${waited}s; settling ${SETTLE_S}s"
    sleep "$SETTLE_S"
fi
echo "$(date -u +%FT%TZ) ${STAGE} complete under ${SEED_DIR}; starting the chain"
