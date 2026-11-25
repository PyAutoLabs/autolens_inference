"""Check native outputs and compute the registered per-seed whole-fit comparisons."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def compare(raw):
    rows = json.loads((raw / "runs.json").read_text())
    arms = ("cpu", "gpu", "split", "feed")
    if len(rows) != 8:
        raise ValueError("Eight completed fits required")
    table = {(r["seed"], r["arm"]): r for r in rows}
    pairs = []
    for seed in (1, 2):
        base = table[seed, "gpu"]
        cases = {}

        def draws(arm, seed=seed):
            return np.load(raw / f"s{seed}-{arm}" / "draws.npz")

        with draws("gpu") as ref:
            for arm in arms:
                row = table[seed, arm]
                case = raw / f"s{seed}-{arm}"
                log = (raw / f"s{seed}-{arm}.log").read_text()
                latent = re.findall(
                    r"Time to compute latent variables: ([0-9.]+) seconds for (\d+) samples", log
                )
                if len(latent) != 1 or int(latent[0][1]) != 100:
                    raise ValueError(f"Missing posterior-derived output timing in {case}")
                row["latent_100_s"] = float(latent[0][0])
                status = json.loads(next(case.rglob("*.status.json")).read_text())
                if status["state"] != "completed" or status["progress"]["n_like"] != row["calls"]:
                    raise ValueError(f"Native run status disagrees in {case}")
                row["resources"] = status["resources"]
                info = json.loads(next(case.rglob("samples_info.json")).read_text())
                if (
                    len(list(case.rglob(".completed"))) != 1
                    or info["total_samples"] != row["calls"]
                    or info["log_evidence"] != row["logz"]
                ):
                    raise ValueError(f"Incomplete output in {case}")
                if (
                    row["native"]
                    != {"n_live": 200, "n_update": 200, "n_batch": 100, "n_networks": 4}
                    or row["ess"] < 500
                    or row["cache_sizes"] != [1]
                ):
                    raise ValueError("Unexpected sampler configuration, stopping or compilation")
                if row["hashes"] != base["hashes"] or row["parameters"] != base["parameters"]:
                    raise ValueError("Fixture changed")
                with draws(arm) as data:
                    if len(data["values"]) != row["calls"]:
                        raise ValueError("Draw count mismatch")
                    equal = np.array_equal(data["points"], ref["points"])
                    violations = (
                        int(
                            np.count_nonzero(
                                ~np.isclose(data["values"], ref["values"], atol=1e-6, rtol=1e-10)
                            )
                        )
                        if equal
                        else None
                    )
                    error = float(np.max(np.abs(data["values"] - ref["values"]))) if equal else None
                dz = row["logz"] - base["logz"]
                dm = float(np.max(np.abs(np.array(row["mean"]) - base["mean"]) / base["std"]))
                cases[arm] = dict(
                    proposals_match=equal,
                    max_log_l_error=error,
                    values_outside_tolerance=violations,
                    logz_delta=dz,
                    max_mean_shift_sd=dm,
                    passes=bool(violations == 0 and abs(dz) <= 0.1 and dm <= 0.1),
                )
        feed = table[seed, "feed"]
        if feed["membership_jobs"] <= 0:
            raise ValueError("Feeder did not execute geometry tasks")
        gain = {
            arm: 100 * (1 - feed["wall_s"] / table[seed, arm]["wall_s"])
            for arm in ("cpu", "gpu", "split")
        }
        pairs.append(
            dict(
                seed=seed,
                correctness=cases,
                feeder_saving_pct=gain,
                passes=bool(
                    all(c["passes"] for c in cases.values())
                    and gain["gpu"] >= 5
                    and gain["split"] > 0
                ),
            )
        )
    return dict(runs=rows, pairs=pairs, advance=all(p["passes"] for p in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    print(json.dumps(compare(parser.parse_args().raw), indent=2, allow_nan=False))
