"""Run eight complete conditional ring fits with fresh caches and counterbalanced order."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--profiling", type=Path, required=True)
    a = p.parse_args()
    raw = a.directory.resolve()
    raw.mkdir(parents=True, exist_ok=False)
    cores = os.sched_getaffinity(0)
    if (
        len(cores) != 4
        or len(
            {
                Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text()
                for c in cores
            }
        )
        != 4
    ):
        p.error("Use four CPU IDs on distinct cores")
    rows = []
    arms = ("gpu", "feed", "split", "cpu")
    for seed in (1, 2):
        for arm in arms if seed == 1 else arms[::-1]:
            tag = f"s{seed}-{arm}"
            env = {
                **os.environ,
                "JAX_PLATFORMS": "cpu" if arm == "cpu" else "cuda",
                "JAX_ENABLE_X64": "true",
                "JAX_ENABLE_COMPILATION_CACHE": "false",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            with (raw / (tag + ".log")).open("w") as log:
                tic = time.perf_counter()
                subprocess.run(
                    [
                        sys.executable,
                        Path(__file__).with_name("run_fit.py"),
                        raw / tag,
                        "--source",
                        a.source.resolve(),
                        "--profiling",
                        a.profiling.resolve(),
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=True,
                )
                elapsed = time.perf_counter() - tic
            row = json.loads((raw / tag / "receipt.json").read_text())
            row["process_wall_s"] = elapsed
            rows.append(row)
            (raw / "runs.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            print(tag, round(row["wall_s"], 2), row["calls"], round(row["eval_s"], 2), flush=True)


if __name__ == "__main__":
    main()
