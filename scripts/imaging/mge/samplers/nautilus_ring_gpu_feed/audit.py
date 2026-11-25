"""Re-evaluate retained draws through the scalar NumPy imaging likelihood."""

import argparse
import json
from pathlib import Path

import autolens as al
import numpy as np
from run_fit import fixture


def audit(source, raw):
    model, analysis, hashes = fixture(source)
    scalar = al.AnalysisImaging(dataset=analysis.dataset, use_jax=False)
    tracer = al.from_json(
        file_path=source / "scripts/cosmos_web_ring/results/sersic_source/tracer.json"
    )
    reference = float(al.FitImaging(dataset=analysis.dataset, tracer=tracer).log_likelihood)
    central = float(scalar.log_likelihood_function(model.instance_from_unit_vector([0.5] * 3)))
    np.testing.assert_allclose(central, reference, atol=1e-6, rtol=1e-10)
    checks = []
    for path in sorted(raw.glob("*/draws.npz")):
        with np.load(path) as data:
            points = data["points"]
            vals = data["values"]
        seed = path.parent.name.split("-")[0]
        with (
            np.load(raw / f"{seed}-gpu" / "draws.npz") as gpu,
            np.load(raw / f"{seed}-cpu" / "draws.npz") as cpu,
        ):
            if not np.array_equal(cpu["points"], gpu["points"]):
                raise ValueError("Worst-point audit requires matching proposal coordinates")
            worst = int(np.argmax(np.abs(cpu["values"] - gpu["values"])))
        indices = np.unique(
            np.r_[np.linspace(0, len(vals) - 1, 8, dtype=int), np.argmax(vals), worst]
        )
        expected = np.array(
            [
                scalar.log_likelihood_function(model.instance_from_unit_vector(points[i]))
                for i in indices
            ]
        )
        close = np.isclose(vals[indices], expected, atol=1e-6, rtol=1e-10)
        checks.append(
            dict(
                case=path.parent.name,
                points=len(indices),
                values_outside_tolerance=int(np.count_nonzero(~close)),
                worst_pair_index=worst,
                worst_pair_numpy=float(expected[np.where(indices == worst)[0][0]]),
                worst_pair_backend=float(vals[worst]),
                max_error=float(np.max(np.abs(vals[indices] - expected))),
            )
        )
    if len(checks) != 8:
        raise ValueError("Expected eight completed fits")
    return dict(
        reference_log_l=reference, conditional_centre_log_l=central, inputs=hashes, checks=checks
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.source, a.raw), indent=2, allow_nan=False) + "\n")
