"""Local conditional COSMOS-Web Ring fit using the CPU/GPU feeder from profiling PR 43."""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.non_linear.objective import evaluate_in_chunks
from autolens import jax_wrapper  # noqa: F401
from nautilus import Sampler
from nautilus.bounds import NeuralBound


def fixture(source):
    band = source / "dataset/imaging/cosmos_web_ring/wavebands/F444W"
    tracer_path = source / "scripts/cosmos_web_ring/results/sersic_source/tracer.json"
    files = [
        band / name
        for name in (
            "data.fits",
            "noise_map.fits",
            "psf.fits",
            "mask_extra_galaxies.fits",
            "info.json",
        )
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*files, tracer_path]}
    scale = json.loads((band / "info.json").read_text())["pixel_scale"]
    dataset = al.Imaging.from_fits(
        data_path=files[0], noise_map_path=files[1], psf_path=files[2], pixel_scales=scale
    )
    extra = al.Mask2D.from_fits(file_path=files[3], pixel_scales=scale, invert=True)
    dataset = dataset.apply_noise_scaling(mask=extra)
    mask = al.Mask2D.circular(shape_native=dataset.shape_native, pixel_scales=scale, radius=2.1)
    dataset = dataset.apply_mask(mask=mask)
    oversampling = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid, sub_size_list=[4, 2, 2], radial_list=[0.3, 0.6], centre_list=[(0.0, 0.0)]
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=oversampling)
    tracer = al.from_json(file_path=tracer_path)
    lens, source_galaxy = tracer.galaxies
    old = lens.mass

    def around(x):
        return af.UniformPrior(lower_limit=float(x) - 0.05, upper_limit=float(x) + 0.05)

    mass = af.Model(
        type(old),
        centre=old.centre,
        ell_comps=old.ell_comps,
        einstein_radius=around(old.einstein_radius),
    )
    light = source_galaxy.bulge
    bulge = af.Model(
        type(light),
        ell_comps=light.ell_comps,
        intensity=float(light.intensity),
        effective_radius=light.effective_radius,
        sersic_index=light.sersic_index,
    )
    bulge.centre.centre_0 = around(light.centre[0])
    bulge.centre.centre_1 = around(light.centre[1])
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=lens.redshift, bulge=lens.bulge, mass=mass),
            source=af.Model(al.Galaxy, redshift=source_galaxy.redshift, bulge=bulge),
        ),
        fields=af.Collection(field=tracer.fields[0]),
    )
    assert model.prior_count == 3 and mask.pixels_in_mask == 3841, (
        model.prior_count,
        model.parameter_names,
        mask.pixels_in_mask,
    )
    return model, al.AnalysisImaging(dataset=dataset, use_jax=True), hashes


def common(model, analysis):
    priors = model.priors_ordered_by_id
    unit = np.vstack([np.full(3, 0.5), np.random.default_rng(43).uniform(0.05, 0.95, (12, 3))])
    points = np.array([[p.value_for(u) for p, u in zip(priors, r, strict=True)] for r in unit])

    def scalar(theta):
        return analysis.log_likelihood_function(model.instance_from_vector(theta, xp=jnp))

    fun = jax.jit(jax.vmap(scalar))
    values = {}
    for width in (50, 100):
        tic = time.perf_counter()
        result = np.asarray(evaluate_in_chunks(fun, points, width))
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite common-point likelihood")
        values[str(width)] = dict(values=result.tolist(), first_s=time.perf_counter() - tic)
    np.testing.assert_allclose(
        values["50"]["values"], values["100"]["values"], atol=1e-6, rtol=1e-10
    )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profiling", type=Path, required=True)
    parser.add_argument("--arm", choices=("cpu", "gpu", "split", "feed"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    a = parser.parse_args()
    sys.path.insert(0, str(a.profiling.resolve() / "experiments/nautilus_gpu_feed"))
    from run import Feed

    directory = a.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    af.conf.instance.output_path = directory / "output"
    device = jax.devices()[0]
    assert jax.config.jax_enable_x64 and device.platform == ("cpu" if a.arm == "cpu" else "gpu")
    model, analysis, hashes = fixture(a.source)
    row = dict(
        arm=a.arm,
        seed=a.seed,
        device=device.device_kind,
        hashes=hashes,
        parameters=list(model.parameter_names),
    )
    if a.probe:
        row["common"] = common(model, analysis)
    else:
        search = af.Nautilus(name="ring", n_live=200, number_of_cores=1, seed=a.seed, verbose=False)
        tic = time.perf_counter()
        with ExitStack() as stack:
            pool = (
                stack.enter_context(ThreadPoolExecutor(max_workers=2)) if a.arm == "feed" else None
            )
            feeder = Feed(a.arm, pool)

            def evaluate(sampler, points):
                return feeder.evaluate(sampler, points)

            native_contains = NeuralBound.contains

            def contains(bound, points):
                return feeder.membership(native_contains, bound, points)

            stack.enter_context(patch.object(Sampler, "evaluate_likelihood", evaluate))
            stack.enter_context(patch.object(NeuralBound, "contains", contains))
            result = search.fit(model=model, analysis=analysis)
        wall = time.perf_counter() - tic
        samples = result.samples
        w = np.array(samples.weight_list)
        w /= w.sum()
        params = np.array(samples.parameter_lists)
        mean = w @ params
        points = np.concatenate(feeder.points)
        values = np.concatenate(feeder.values)
        if samples.samples_info["total_samples"] != len(points) or len(values) != len(points):
            raise ValueError("Missing proposals")
        np.savez(directory / "draws.npz", points=points, values=values)
        row.update(
            wall_s=wall,
            logz=float(samples.log_evidence),
            ess=float(1 / (w @ w)),
            mean=mean.tolist(),
            std=np.sqrt(w @ (params - mean) ** 2).tolist(),
            calls=len(points),
            native=feeder.native,
            batches=len(feeder.times),
            eval_s=sum(feeder.times),
            first_batch_s=feeder.times[0],
            warm_batch_ms=float(np.median(feeder.times[1:]) * 1000),
            cache_sizes=sorted(feeder.cache_sizes),
            width=feeder.width,
            membership_jobs=feeder.geometry_jobs,
            wait_s=feeder.wait_s,
            collect_s=feeder.collect_s,
            digest=hashlib.sha256(points.tobytes()).hexdigest(),
        )
    (directory / "receipt.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
