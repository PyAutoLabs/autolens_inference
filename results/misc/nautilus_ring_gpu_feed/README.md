# CPU workers feeding GPU batches on the COSMOS-Web Ring

Retain native serial training and dispatch on this conditional JWST imaging workload. Two CPU feeder threads made the complete GPU fit 12.8% longer for seed 1 and 13.6% longer for seed 2 compared with the serial-host full-batch GPU. CPU-only finished fastest on both seeds. The GPU was faster for the imaging likelihood itself. [Registered decision rule](https://github.com/PyAutoLabs/autolens_inference/issues/74).

| Seed | JAX CPU | GPU full batch | GPU serial half batches | GPU with CPU feeders |
|---:|---:|---:|---:|---:|
| 1 | 149.8 s | 197.5 s | 207.3 s | 222.7 s |
| 2 | 136.3 s | 190.0 s | 230.7 s | 215.9 s |

These clocks cover native PyAutoFit `search.fit`, including first compilation, serial neural training, checkpointing, normal visualization, posterior output and feeder setup/shutdown. The receipts also give the fresh-process time. Each worker count ran on the same four physical CPU cores available to WSL on an i9-10885H laptop with an RTX 2060 Max-Q. We ran each seed in reverse arm order from the other. The first GPU fit built the shared font cache. Two seeds on a shared laptop establish the observed paired comparisons.

## Where CPU and GPU time differed

| Seed | CPU warm batch | GPU warm batch | Worker-fed GPU warm batch | CPU 100-sample derived output | GPU 100-sample derived output |
|---:|---:|---:|---:|---:|---:|
| 1 | 85.4 ms | 44.5 ms | 67.2 ms | 10.4 s | 42.6 s |
| 2 | 76.9 ms | 45.4 ms | 56.9 ms | 9.4 s | 39.8 s |

Warm times are medians over complete logical likelihood batches after the first. The serial-host GPU spent 11.68 and 12.25 s evaluating all likelihood batches, or 5.9% and 6.4% of whole-fit wall. PyAutoFit separately timed the calculation of derived quantities for 100 posterior samples, including its own compilation and materialization. The CPU took about 30–32 s less on that step in the CPU-only fits. This suggests a hardware-placement experiment: replay **identical** posterior samples on each device, then measure the full cost of moving the output calculation to CPU after GPU sampling. The timings here come from separate fits and cannot establish the saving from a hybrid pipeline. Native output reported an unavailable latent name and omitted uncalibrated flux quantities because the analysis had no magnitude zero point. The timing includes that native output path. The inference checks use the sampled parameters and evidence.

## Observational fixture

This is a three-parameter conditional fit on real JWST/NIRCam F444W observations. We used the committed COSMOS-Web Ring single-Sérsic-source tracer, including its MGE lens light, isothermal mass, external shear and source. We loaded the observed data, noise and PSF; applied the reference contaminant noise scaling, 2.1 arcsec circular mask (3,841 image pixels) and 4/2/2 radial oversampling; and evaluated the native PyAutoLens `AnalysisImaging` likelihood. Only Einstein radius and the two source-centre coordinates varied, over uniform priors of ±0.05 arcsec about the saved values. A direct native NumPy `FitImaging` evaluation at the centre reproduced the committed tracer's likelihood within 2.3e-11. No lens parameters were estimated for publication. [Archived model and data preparation](https://github.com/PyAutoLabs/autolens_assistant/blob/a752b429e8442228e7030499e82905ec724e429b/scripts/cosmos_web_ring/explore_committed_models.py).

Each search used native cadence, 200 live points, four serial neural networks, 100 proposals per logical batch and the native ESS target of 500. The two half-batch arms used width 50. The worker arm reused the existing two-thread CPU prior-transform and read-only bound-membership implementation. One controller owned the RNG, JAX device, ordered admission and adaptation. Each complete fit used one cached batched likelihood shape. CPU and GPU passed the common-point preflight with 13 real rows padded independently to widths 50 and 100. [Feeder implementation](https://github.com/PyAutoLabs/autofit_profiling/blob/7db263c56cddd3c177aab815cce467b0393ed537/experiments/nautilus_gpu_feed/run.py) · [Input hashes, preflight and library revisions](provenance.json).

## Numerical audit

All four arms followed identical proposal coordinates within each seed, with 14,300 and 15,000 admitted draws respectively. GPU-only and worker-fed log likelihoods differed by at most 3.64e-12. Across JAX CPU and full-width GPU, **8 of 14,300** pairs on seed 1 and **11 of 15,000** on seed 2 exceeded the preflight tolerance of 1e-6 absolute plus 1e-10 relative when applied to every draw. The largest difference was 4.51e-6 log-likelihood units. We retain these failures. Paired log evidence differed by at most 4.87e-9 and marginal means by less than 1e-8 posterior standard deviations. ESS was approximately 1,769 and 1,827. The two GPU feeder comparisons failed the whole-fit speed threshold independently of numerical parity. [All timings, sample counts and per-arm numerical differences](result.json).

A separate scalar NumPy audit covered evenly spaced draws, the maximum-likelihood draw and the point with the largest JAX CPU/GPU discrepancy in each seed. Some near-peak values also exceeded the same tolerance against NumPy, with a largest difference of 1.01e-4. The sampled points are intentionally selected to include outliers. They cannot estimate an error frequency for the NumPy path. [Independent scalar audit](audit.json).

## Reproduction

Use the checkout revisions in `provenance.json`. Put the five PyAuto libraries on `PYTHONPATH` in a CUDA-capable JAX environment. Point `RING_SOURCE` to the pinned autolens_assistant checkout and `FEED_SOURCE` to the pinned autofit_profiling checkout. From the autolens_inference repository root, choose four CPU IDs on separate physical cores and new output directories.

```sh
script=scripts/imaging/mge/samplers/nautilus_ring_gpu_feed
export JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cpu python "$script/run_fit.py" cpu-probe --source "$RING_SOURCE" --profiling "$FEED_SOURCE" --arm cpu --probe
JAX_PLATFORMS=cuda python "$script/run_fit.py" gpu-probe --source "$RING_SOURCE" --profiling "$FEED_SOURCE" --arm gpu --probe
taskset -c 0,2,4,6 python "$script/pilot.py" fresh-output --source "$RING_SOURCE" --profiling "$FEED_SOURCE"
python "$script/compare.py" fresh-output
JAX_PLATFORMS=cpu python "$script/audit.py" --raw fresh-output --source "$RING_SOURCE" --output scalar-audit.json
```

The eight runs contain 117,200 likelihood evaluations. All completed in fresh directories. An initial preparation check caught two source-centre coordinates that were missing from the free-parameter list before any timed fit. The experiment verifies the three parameters, input hashes, CPU/GPU placement, complete status receipts, native settings and counts. These local results are part of the Nautilus implementation epic. [Insight imaging task](https://github.com/PyAutoLabs/PyAutoInsight/blob/main/tasks/hst_mass_samplers_wave1.md) · [Pulse producer task](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/tasks/autofit_profiling_bootstrap.md).
