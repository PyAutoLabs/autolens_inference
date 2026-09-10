# autolens_inference

Non-linear search and inference benchmarking for strong lensing with
[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) — the proving ground for *which search
finds the right lens model, how reliably, and at what cost*, across gradient-free and
gradient-based samplers, on CPU and on A100 GPUs. It is the sibling of
[autolens_profiling](https://github.com/PyAutoLabs/autolens_profiling), which owns likelihood
*timing*; this repo owns everything downstream of one likelihood call. Science runs, tasks and
rulings of record are managed by [PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex).
