"""Investigate a frozen mass_total[1] problem without rerunning prerequisites."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/misc")]

from experiments.prepared import (  # noqa: E402
    artifact,
    contained,
    digest,
    json_bytes,
    library_revisions,
    load_problem,
    revision,
)
from slam._runner import set_backend_env  # noqa: E402

SUPPORTED = ("nautilus", "emcee", "nuts", "smc")


def make_search(af, model, manifest, items, args):
    """Grounded installed PyAutoFit APIs; never change model or prior limits."""
    initializer = af.InitializerPrior()
    samples = None
    if args.start == "warm":
        import dill

        source = items.get("posterior_samples")
        if source is None:
            raise ValueError("Warm initialization requires completed baseline samples")
        samples = dill.loads(contained(ROOT, source["path"]).read_bytes())
        initializer = af.InitializerParamStartPoints.from_result(
            SimpleNamespace(samples=samples),
            model=model,
            n_points=args.walkers if args.sampler == "emcee" else args.chains,
            jitter=args.jitter,
            seed=args.seed,
        )
    common = dict(
        name="mass_total[1]", path_prefix=f"experiments/{args.run_id}", number_of_cores=args.cores
    )
    if args.sampler == "nautilus":
        if args.start == "warm":
            raise ValueError("Nautilus has no supported baseline warm initializer; use cold")
        return af.Nautilus(**common, n_live=args.live, n_batch=args.batch, seed=args.seed)
    if args.sampler == "emcee":
        return af.Emcee(**common, nwalkers=args.walkers, nsteps=args.steps, initializer=initializer)
    if not getattr(manifest["controls"], "use_jax", False) and not manifest["controls"][
        "backend"
    ].startswith("jax"):
        raise ValueError("NUTS/SMC require a frozen JAX analysis; export a JAX baseline")
    if args.start == "resume":
        raise ValueError("Installed NUTS/SMC persist diagnostics, not resumable kernel state")
    if args.sampler == "nuts":
        return af.BlackJAXNUTS(
            **common,
            num_warmup=args.warmup,
            num_samples=args.steps,
            num_chains=args.chains,
            seed=args.seed,
            initializer=initializer,
        )
    return af.SMC(
        **common,
        num_particles=args.particles,
        max_smc_steps=args.steps,
        seed=args.seed,
        initializer=initializer,
        inverse_mass_matrix=SimpleNamespace(samples=samples) if samples else None,
    )


def resume_source(args, manifest, search, model):
    """Copy a pinned Emcee backend to a new run; retain the original measurement."""
    if args.sampler != "emcee":
        raise ValueError("Resume currently supported only for Emcee HDF backends")
    source_path = contained(ROOT, args.resume_record)
    source = json.loads(source_path.read_text())
    if source.get("problem_id") != manifest["id"] or source.get("sampler") != "emcee":
        raise ValueError("Resume source must be Emcee on the same prepared problem")
    options = source.get("experiment_options", {})
    if options.get("walkers") != args.walkers or source.get("seed") != args.seed:
        raise ValueError("Resume must preserve walkers and original random seed")
    backend = contained(ROOT, source["checkpoint"]["path"])
    if digest(backend.read_bytes()) != source["checkpoint"]["sha256"]:
        raise ValueError("Resume checkpoint changed since measurement; snapshot it first")
    import h5py

    with h5py.File(backend, "r") as state:
        shape = state["mcmc"]["chain"].shape
        iteration = int(state["mcmc"].attrs["iteration"])
        if (
            shape[1:] != (args.walkers, model.prior_count)
            or iteration <= 0
            or iteration >= args.steps
        ):
            raise ValueError("Resume requires matching dimensions and a larger total step budget")
    search.paths.model = model
    target = Path(search.backend_filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backend, target)
    # Pin a distinct immutable snapshot of the source before the continuation.
    snapshot = ROOT / "output" / "experiment_sources" / args.run_id / "checkpoint.hdf"
    return artifact(
        ROOT,
        snapshot,
        backend.read_bytes(),
        "checkpoint",
        source_path.relative_to(ROOT).as_posix(),
        revision(ROOT),
    )


def timed_fit(search, model, analysis, row):
    """Measure the fit call, including failures, before diagnostic extraction."""
    start = time.perf_counter()
    try:
        return search.fit(model=model, analysis=analysis)
    finally:
        row["total_wall_s"] = time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sampler", choices=SUPPORTED, required=True)
    parser.add_argument("--start", choices=("cold", "warm", "resume"), required=True)
    parser.add_argument("--resume-record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--walkers", type=int, default=50)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--live", type=int, default=150)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument(
        "--hardware-id",
        required=True,
        help="Host/device, resources and concurrency; not a scheduler job ID",
    )
    parser.add_argument(
        "--compilation", choices=("cold", "warm", "not_applicable", "unknown"), default="unknown"
    )
    parser.add_argument(
        "--cache", choices=("cold", "warm", "not_applicable", "unknown"), default="unknown"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_id
    ):
        parser.error("run-id must contain only letters, digits, hyphens and underscores")
    if args.start == "resume" and not args.resume_record:
        parser.error("resume requires --resume-record")
    if args.start != "resume" and args.resume_record:
        parser.error("resume-record is only valid for resume")
    if (
        min(
            args.steps, args.walkers, args.chains, args.particles, args.live, args.batch, args.cores
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("Sampler counts must be positive; warmup nonnegative")
    from experiments.prepared import verify_manifest

    manifest = verify_manifest(
        ROOT, args.prepared, expected_problem=args.problem_id, revisions=library_revisions(ROOT)
    )
    set_backend_env(manifest["controls"]["backend"], args.cores)
    import autofit as af
    import numpy as np
    from autonerves.test_mode import with_test_mode_segment
    from slam._runner import posterior_from_result

    af.conf.instance.push(new_path=ROOT / "config", output_path=ROOT / "output")
    manifest, model, analysis, items = load_problem(
        ROOT, args.prepared, expected_problem=args.problem_id
    )
    if model.dict() != analysis.modify_model(model).dict():
        raise ValueError("Analysis.modify_model changes the frozen problem")
    search = make_search(af, model, manifest, items, args)
    if args.validate_only or os.environ.get("AUTOLENS_INFERENCE_SMOKE") == "1":
        print(f"Validated {manifest['id']} / {args.sampler} / {args.start}; no fit executed")
        return 0
    destination = ROOT / "results" / "searches" / "experiments" / f"{args.run_id}.json"
    output = with_test_mode_segment(ROOT / "output") / "experiments" / args.run_id
    if destination.exists() or output.exists():
        raise ValueError("Run identity already exists; choose a new run-id")
    sources = []
    if args.start == "warm":
        sources = [items["posterior_samples"]]
    if args.start == "resume":
        sources = [resume_source(args, manifest, search, model)]
    np.random.seed(args.seed)
    controls = manifest["controls"]
    row = {
        "schema_version": 2,
        "target": manifest["id"],
        "dataset_class": controls["dataset_class"],
        "instrument": controls["instrument"],
        "variant": controls["variant"],
        "dataset_id": manifest["dataset_id"],
        "model": manifest["model_id"],
        "stage": manifest["stage"],
        "pipeline": "slam",
        "sampler": args.sampler,
        "config_name": controls["config_name"],
        "backend": controls["backend"],
        "precision": controls["precision"],
        "seed": args.seed,
        "setup_id": manifest["setup_id"],
        "problem_id": manifest["id"],
        "experiment_protocol": "frozen-mass-total-v1",
        "library_revisions": manifest["dependency_revisions"],
        "initialization": {
            "mode": args.start,
            "recipe": {
                "cold": "Draw from frozen prepared priors; no baseline sampler information",
                "warm": (
                    "Baseline maximum-likelihood start points with physical-space covariance-scaled jitter (absolute physical jitter fallback); priors unchanged"
                    + (
                        "; baseline covariance seeds SMC Gaussian reference whitening"
                        if args.sampler == "smc"
                        else ""
                    )
                ),
                "resume": "Continue pinned Emcee HDF backend; cumulative samples, segment fit clock",
            }[args.start],
            "sources": sources,
        },
        "environment": {
            "hardware_id": args.hardware_id,
            "compilation": args.compilation,
            "cache": args.cache,
        },
        "preparation_s": manifest["preparation_s"],
        "initialization_s": None,
        "initialization_s_reason": "Sampler initialization/warmup not timed separately from fit",
        "timing_definitions": {
            "total_s": "Elapsed experiment search.fit call only; segment-only for resume; excludes shared prerequisite preparation and artifact validation",
            "sampling_s": "Sampler-native clock; cumulative on resume where reported; scope depends on sampler",
            "preparation_s": "Elapsed prerequisite SLaM stage fits and mass model construction after dataset setup; excludes mass fit and prepared export; shared prerequisite cost excluded from total_s; never sum across experiment rows",
            "initialization_s": "Sampler initialization and warmup included in total_s; not separately instrumented",
        },
        "experiment_options": {
            k: getattr(args, k)
            for k in (
                "steps",
                "warmup",
                "walkers",
                "chains",
                "particles",
                "live",
                "batch",
                "jitter",
            )
        },
        "measured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "completed": False,
        "status": "incomplete",
        "output_path": output.relative_to(ROOT).as_posix(),
    }
    for key in ("compilation", "cache"):
        if row["environment"][key] == "unknown":
            row["environment"][key + "_reason"] = (
                "Not independently measured; operator did not declare state"
            )
    try:
        result = timed_fit(search, model, analysis, row)
        row.update(
            completed=True,
            status="complete",
            posterior=posterior_from_result(af, result),
            max_log_likelihood=float(result.samples.max_log_likelihood_sample.log_likelihood),
        )
        info = result.samples.samples_info or {}
        native_time = info.get("time")
        row["wall_s"] = float(native_time) if native_time is not None else None
        definitions = {
            "likelihood_evaluations": "Nautilus total_samples counts rejected and accepted likelihood calls, including exploration; other samplers not instrumented",
            "gradient_evaluations": "Not instrumented; neither iterations nor particle counts are gradient counts",
            "iterations": "Sampler-native Emcee steps aggregated over synchronized walkers or SMC tempering steps; includes native warmup where reported; cumulative on resume",
            "retained_samples": "Length of PyAutoFit returned sample list after sampler-native burn-in/filtering; cumulative on resume",
            "effective_sample_size": "No common posterior ESS estimator instrumented; terminal SMC weight ESS is not substituted",
        }
        row["work"] = {
            "likelihood_evaluations": info.get("total_samples")
            if args.sampler == "nautilus"
            else None,
            "gradient_evaluations": None,
            "iterations": info.get("total_steps", info.get("n_smc_steps")),
            "retained_samples": len(result.samples.sample_list),
            "effective_sample_size": None,
            "definitions": definitions,
        }
        for key in definitions:
            if row["work"][key] is None:
                row["work"][key + "_reason"] = (
                    "Not measured by this producer/sampler; no estimate substituted"
                )
        row["max_likelihood_parameters"] = dict(
            zip(
                result.samples.names,
                map(float, result.samples.max_log_likelihood(as_instance=False)),
            )
        )
    except BaseException as exc:
        row["status"] = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        if args.sampler == "emcee" and Path(search.backend_filename).exists():
            row["checkpoint"] = artifact(
                ROOT,
                ROOT / "output" / "experiment_sources" / args.run_id / "final.hdf",
                Path(search.backend_filename).read_bytes(),
                "checkpoint",
                destination.relative_to(ROOT).as_posix(),
                revision(ROOT),
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(json_bytes(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
