"""Shared CLI / JSON / auto-simulate helpers for the inference scripts.

Used by every leaf under ``scripts/<dataset_class>/{slam,searches}/`` so the
per-script boilerplate stays minimal and the run-defining flags
(``--config-name``, ``--backend``, ``--inversion``, ``--seed``,
``--use-mixed-precision``, ``--instrument``) and the dataset auto-simulate hook
are defined in one place.

Imported after the standard root-finding preamble, since the leaves live under
several sibling directories::

    import sys
    from pathlib import Path

    _ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
    sys.path.insert(0, str(_ROOT))
    from _inference_cli import (
        parse_inference_cli, device_info_dict, resolve_output_paths,
        auto_simulate_if_missing,
    )

Adapted from ``autolens_inference/_inference_cli.py``. The two repos share the
flag vocabulary on purpose: a config name means the same thing in both.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The ``--backend`` vocabulary — the middle field of the config-name grammar
#: ``{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp}``.
BACKENDS = ("jax_cpu", "numba_cpu", "jax_gpu")

#: The ``--inversion`` vocabulary — the third field of the same grammar.
INVERSIONS = ("dense", "sparse")

#: The ``--config-name`` grammar, as a regex with one group per field:
#: ``(where, backend, inversion, precision)``. Drivers match against this and
#: refuse a name whose fields disagree with the flags actually passed — a config
#: name that lies about its run turns a parity row into a lie, and nothing
#: downstream (the result JSON, the README table, a Cortex ruling) can detect it.
CONFIG_NAME_RE = re.compile(
    r"^(local|hpc_a100)_(jax_cpu|numba_cpu|jax_gpu)_(dense|sparse)_(fp64|mp)$"
)

#: The ``--mesh`` vocabulary — which source pixelization family the pixelized
#: stages are built from. A folder and a payload field, never part of the
#: config-name grammar: the config name says which *backend* ran, the mesh says
#: which *model* ran, and conflating them would make one label carry two
#: independent axes.
MESHES = ("rect", "delaunay")

#: The mesh family a run uses when neither the leaf nor ``--mesh`` names one.
MESH_DEFAULT = "rect"

#: What ``--mesh-pixels`` resolves to per family. 28 for ``rect`` is the 28x28
#: shape of the workspace default (``slam_start_here.py``); 1250 for
#: ``delaunay`` is a vertex count, not a shape — the two numbers are not
#: comparable and are not meant to be.
MESH_PIXELS_DEFAULT = {"rect": 28, "delaunay": 1250}

#: The variant name of the workspace-default run: a 28x28 rectangular mesh. It
#: is the one variant whose name is not derived from the mesh, because it is the
#: baseline every other variant is measured against — and because the rows
#: already on disk were written under it.
BASE_VARIANT = "slam_base"

#: The ``--stages`` vocabulary — the SLaM chain in order. ``--stages <name>``
#: stops the driver after that stage, which is how a per-backend rate is
#: measured from a ``source_lp``-only leg without paying for the whole chain.
STAGES = ("source_lp", "source_pix_1", "source_pix_2", "light", "mass_total")


def default_cores() -> int:
    """Cores to parallelise over when ``--cores`` is not given.

    ``SLURM_CPUS_PER_TASK`` first (inside a job, that is the allocation and
    taking more is taking someone else's), then the machine's CPU count.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return int(slurm)
        except ValueError:
            pass
    return os.cpu_count() or 1


@dataclass(frozen=True)
class InferenceCLI:
    config_name: str | None
    output_dir: Path | None
    use_mixed_precision: bool
    instrument: str | None
    backend: str
    inversion: str
    seed: int
    use_sparse_operator: bool
    rect_mesh: str
    mesh: str
    mesh_pixels: int
    regularization: str | None
    memo: str
    cores: int = 1
    stages: str | None = None


def parse_inference_cli(
    default_config_name: str | None = None,
    default_mesh: str | None = None,
    default_mesh_pixels: int | None = None,
) -> InferenceCLI:
    """Parse the run flags accepted by every leaf script.

    When ``--config-name`` is omitted, falls back to ``default_config_name``.

    ``default_mesh`` / ``default_mesh_pixels`` are the leaf's own mesh defaults,
    in the style of ``default_instrument``: a leaf that exists to run one mesh
    (``scripts/imaging/slam/hst_delaunay.py``) names it once here rather than
    making every caller pass ``--mesh`` and get it right. ``--mesh-pixels``
    still overrides, and the leaf's pixel count applies only while the leaf's
    own mesh is the one in force — asking a Delaunay leaf for ``--mesh rect``
    must not hand the rectangular mesh a vertex count meant for Delaunay.

    ``--instrument`` is optional; when omitted (None) leaf scripts keep their
    module-level default (typically ``"hst"``).

    ``--backend``, ``--inversion`` and ``--seed`` are the **phase-3 contract**:
    they are parsed and carried on the returned dataclass now so that every
    leaf, submit script and result row written from phase 1 onward already
    speaks the vocabulary the backend-parameterised driver will consume. Phase-1
    scripts read them; nothing in phase 1 branches on them yet.
    """
    parser = argparse.ArgumentParser(
        description="Inference run driver flags.",
        # Keep unknown args; per-script argparse is not exhaustive.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help=(
            "Output-filename label for the run, from the grammar "
            "{local,hpc_a100}_{jax_cpu,numba_cpu,jax_gpu}_{dense,sparse}_{fp64,mp} "
            "(e.g. local_jax_cpu_dense_fp64, hpc_a100_jax_gpu_sparse_fp64). "
            "Runs sharing a target and differing only in this label are one "
            "parity row. When omitted, the script keeps its single-config "
            "filename pattern."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the results dir. Each leaf defaults to its task's section "
            "under <autolens_inference>/results/."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="jax_cpu",
        help=(
            "PHASE-3 CONTRACT. Which likelihood backend the run uses — the "
            "middle field of the config name. 'jax_cpu' (the default) and "
            "'jax_gpu' pass ``use_jax=True`` to every Analysis and differ only "
            "in JAX_PLATFORMS; 'numba_cpu' passes ``use_jax=False`` to EVERY "
            "Analysis a pipeline builds (one missed stage silently makes that "
            "stage a JAX run) and is belt-and-braced with PYAUTO_DISABLE_JAX=1. "
            "Parsed and recorded in phase 1; the driver that acts on it lands "
            "in phase 3."
        ),
    )
    parser.add_argument(
        "--inversion",
        choices=INVERSIONS,
        default="dense",
        help=(
            "PHASE-3 CONTRACT. Which inversion path the run takes — the third "
            "field of the config name. 'dense' (the default) leaves the "
            "mapping-matrix path in place; 'sparse' applies the w-tilde "
            "operator, which is TWO different calls: "
            "``dataset.apply_sparse_operator()`` under JAX on any device, and "
            "``dataset.apply_sparse_operator_cpu()`` under numba. Calling the "
            "JAX one on a numba leg does not error — it produces a leg that is "
            "neither. Parsed and recorded in phase 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "PHASE-3 CONTRACT. Seed for the search. A parity row holds the seed "
            "fixed across its backends; a reliability measurement varies only "
            "the seed. Default 0."
        ),
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help=(
            "Pass use_mixed_precision=True to al.Settings — the targeted fp32 "
            "paths in the JAX inversion. This is the 'mp' tail of the config name."
        ),
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help=(
            "Instrument preset to run. The instrument is a FLAG, never a "
            "directory. When omitted, the leaf script's module-level default "
            "applies (typically 'hst')."
        ),
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help=(
            "Legacy spelling of ``--inversion sparse``, kept because the "
            "sibling autolens_inference scripts use it. Sets the same "
            "``use_sparse_operator`` field; ``--inversion sparse`` is preferred "
            "in new scripts."
        ),
    )
    parser.add_argument(
        "--rect-mesh",
        choices=("bilinear", "rtu"),
        default="bilinear",
        help=(
            "Which adaptive rectangular mesh family the run uses "
            "(PyAutoArray#462 split): 'bilinear' — "
            "RectangularBilinearAdaptDensity/AdaptImage, the empirical rank-CDF "
            "workspace default — or 'rtu' — RectangularRTUAdaptDensity/"
            "AdaptImage, the kernel-CDF advanced/GPU/gradient meshes (the "
            "pre-split behaviour). Scripts that read this flag resolve the "
            "classes via ``rect_mesh_classes`` and embed ``rect_mesh`` in the "
            "result row; '_rtu' is appended to the output basename so the two "
            "families' results never clobber each other."
        ),
    )
    parser.add_argument(
        "--regularization",
        choices=("adapt_split", "constant_split"),
        default=None,
        help=(
            "Regularization scheme for the Delaunay-family models. "
            "'adapt_split' — ``al.reg.AdaptSplit(inner_coefficient=0.1, "
            "outer_coefficient=10.0, signal_scale=0.1)``, what production "
            "(SLaM, the Euclid pipeline) pairs Delaunay with — or "
            "'constant_split' — ``al.reg.ConstantSplit(coefficient=1.0)``. "
            "Omitted (None) leaves each script on its own default; models "
            "outside the Delaunay family ignore the flag. The resolved scheme "
            "is embedded in the result row as ``regularization``."
        ),
    )
    parser.add_argument(
        "--mesh",
        choices=MESHES,
        default=None,
        help=(
            "Which source pixelization family the two pixelized source stages "
            "are built from. 'rect' — the adaptive rectangular mesh selected by "
            "--rect-mesh, the workspace default, at a --mesh-pixels x "
            "--mesh-pixels shape — or 'delaunay' — al.mesh.Delaunay at "
            "--mesh-pixels VERTICES, paired with a Split regularization because "
            "al.reg.Adapt cannot be traced on this mesh family under jit. The "
            "mesh is NOT part of the config-name grammar: it names the model "
            "under test, while the config name names the backend that ran it. "
            "It is instead a run-VARIANT folder under "
            "results/slam/<dataset_class>/<instrument>/ and a field of the "
            "target id, so a mesh experiment never joins the base run's parity "
            "row. Omitted = the leaf script's own default (usually 'rect')."
        ),
    )
    parser.add_argument(
        "--mesh-pixels",
        type=int,
        default=None,
        help=(
            "How many pixels the source mesh gets. The number means a different "
            "thing per family and the two are not comparable: under '--mesh "
            "rect' it is the side of the square mesh (28 -> 28x28 = 784 cells, "
            "the workspace default), under '--mesh delaunay' it is the vertex "
            "count outright (default 1250). Non-default values name their own "
            "variant folder (rect_40, delaunay_1250), so a mesh-resolution "
            "sweep never overwrites the base run."
        ),
    )
    parser.add_argument(
        "--memo",
        choices=("off", "on"),
        default="off",
        help=(
            "The NNLS cross-evaluation warm-start memo "
            "(``aa.Settings(nnls_warm_start_memo=...)``, PyAutoArray#498). Off "
            "by default: its measured gains come from a random-walk stream a "
            "Nautilus pool never hands one worker. The library default is "
            "``true`` and production leaves it unset, so 'on' is what "
            "production pays; the resolved flag is recorded in every result row."
        ),
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Number of CPU cores the run may use. Consumed two different ways "
            "depending on the backend: a numba leg passes it to "
            "``SettingsSearch(number_of_cores=...)`` (nautilus forks that many "
            "likelihood workers), while a JAX leg exports it as ``NPROC``, which "
            "sizes XLA's CPU thread pool — never both, since nautilus under "
            "``use_jax=True`` ignores the pool and vmaps with ``n_batch``. "
            "Defaults to SLURM_CPUS_PER_TASK inside a job, else os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--stages",
        choices=STAGES,
        default=None,
        help=(
            "Stop the pipeline after this stage instead of running the whole "
            "chain. Used to measure a per-backend likelihood rate from a "
            "``--stages source_lp`` leg (which no inversion flag changes) "
            "without paying for the four pixelized stages. Omitted = run all."
        ),
    )

    args, _unknown = parser.parse_known_args()
    config_name = args.config_name or default_config_name
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    mesh = args.mesh or default_mesh or MESH_DEFAULT
    # The leaf's pixel count is only its own mesh's; asking a Delaunay leaf for
    # `--mesh rect` falls back to the rectangular family's default rather than
    # building a 1250x1250 rectangular mesh.
    if args.mesh_pixels is not None:
        mesh_pixels = int(args.mesh_pixels)
    elif default_mesh_pixels is not None and mesh == (default_mesh or MESH_DEFAULT):
        mesh_pixels = int(default_mesh_pixels)
    else:
        mesh_pixels = MESH_PIXELS_DEFAULT[mesh]
    return InferenceCLI(
        config_name=config_name,
        output_dir=output_dir,
        use_mixed_precision=bool(args.use_mixed_precision),
        instrument=args.instrument,
        backend=args.backend,
        inversion=args.inversion,
        seed=int(args.seed),
        use_sparse_operator=bool(args.sparse) or args.inversion == "sparse",
        rect_mesh=args.rect_mesh,
        mesh=mesh,
        mesh_pixels=mesh_pixels,
        regularization=args.regularization,
        memo=args.memo,
        cores=int(args.cores) if args.cores else default_cores(),
        stages=args.stages,
    )


def variant_name(cli: InferenceCLI) -> str:
    """The run-variant name for this leg — one directory level and one target field.

    A variant is *what was fitted*, as distinct from the config name, which is
    *what ran it*. Every leg of one variant is a parity row; two variants are
    two experiments and must not be compared as if a backend were the only
    thing between them.

    ``rect`` at the workspace-default 28x28 shape is :data:`BASE_VARIANT`
    (``"slam_base"``) — the baseline, and the name the result rows already on
    disk were written under. Anything else names its mesh and its pixel count:
    ``rect_40``, ``delaunay_1250``.
    """
    if cli.mesh == "rect" and cli.mesh_pixels == MESH_PIXELS_DEFAULT["rect"]:
        return BASE_VARIANT
    return f"{cli.mesh}_{cli.mesh_pixels}"


#: What ``--regularization`` resolves to for the Delaunay-family models when
#: the flag is omitted. Production (SLaM, the Euclid pipeline) pairs Delaunay
#: with ``AdaptSplit``, so that is what a run measures unless it says otherwise.
DELAUNAY_REGULARIZATION_DEFAULT = "adapt_split"


def delaunay_regularization(cli: InferenceCLI):
    """Resolve ``--regularization`` into the Delaunay-family regularization object.

    Returns ``(scheme, regularization, provenance)``: the resolved scheme name,
    the ``al.reg`` object to hand ``al.Pixelization``, and the dict every
    Delaunay-family result JSON records under ``regularization``.

    ``AdaptSplit`` is given the in-repo production-shaped coefficients
    (``inner=0.1``, ``outer=10.0``, ``signal_scale=0.1``, as used by
    the values production runs Delaunay with) rather than its
    ``inner == outer == 1.0`` defaults, which make the per-pixel weights uniform
    and the scheme numerically indistinguishable from ``ConstantSplit``.

    ``AdaptSplit`` reads the mapper's ``adapt_data``, so a cell using it must
    also pass ``galaxy_image_dict`` / ``galaxy_name_image_dict`` to its
    ``al.AdaptImages``; scripts do so unconditionally, since ``ConstantSplit``
    ignores them and the two legs then differ only in the scheme.

    Imports autolens lazily so ``_inference_cli`` stays importable without the
    modelling stack.
    """
    import autolens as al

    scheme = cli.regularization or DELAUNAY_REGULARIZATION_DEFAULT

    if scheme == "constant_split":
        return (
            scheme,
            al.reg.ConstantSplit(coefficient=1.0),
            {"scheme": scheme, "coefficient": 1.0},
        )

    return (
        scheme,
        al.reg.AdaptSplit(inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1),
        {
            "scheme": scheme,
            "inner_coefficient": 0.1,
            "outer_coefficient": 10.0,
            "signal_scale": 0.1,
        },
    )


def rect_mesh_classes(cli: InferenceCLI, al=None):
    """Resolve the explicit adaptive rectangular mesh classes for this run.

    Returns ``(density_cls, image_cls)`` for ``cli.rect_mesh``:
    ``RectangularBilinearAdaptDensity/AdaptImage`` (rank-CDF, the workspace
    default) or ``RectangularRTUAdaptDensity/AdaptImage`` (kernel-CDF, the
    pre-split behaviour).

    ``al`` is the already-imported modelling namespace, which callers that have
    one pass in — a driver that has imported autolens after setting the backend
    environment must not re-enter the import, and a unit test can hand this a
    namespace of its own. When it is ``None`` autolens is imported lazily, so
    ``_inference_cli`` stays importable without the modelling stack.
    """
    if al is None:
        import autolens as al

    if cli.rect_mesh == "rtu":
        return al.mesh.RectangularRTUAdaptDensity, al.mesh.RectangularRTUAdaptImage
    return (
        al.mesh.RectangularBilinearAdaptDensity,
        al.mesh.RectangularBilinearAdaptImage,
    )


def device_info_dict() -> dict:
    """Capture backend / device / nvidia-smi summary for the current JAX process.

    Imports jax lazily so callers can collect this near the JSON write without
    re-importing.
    """
    import jax

    info = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        # Environment provenance: a stray XLA_FLAGS (e.g. disabling
        # constant_folding) or thread pinning silently rescales every timing
        # in a result by integer factors — record them so drift between runs
        # is attributable (found the hard way: autolens_inference#59).
        "xla_flags": os.environ.get("XLA_FLAGS") or None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS") or None,
        "cpu_count": os.cpu_count(),
    }
    if info["backend"] == "gpu":
        try:
            out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.used,memory.total",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                .decode()
                .strip()
            )
            info["nvidia_smi"] = out.replace("\n", "; ")
        except Exception:
            pass
    return info


def resolve_output_paths(
    cli: InferenceCLI,
    default_dir: Path,
    default_basename: str,
    cell: str | None = None,
) -> tuple[Path, Path]:
    """Resolve (json_path, png_path) for the per-cell write.

    - When ``cli.config_name`` is unset: use
      ``<output_dir>/<default_basename>.{json,png}`` (the single-config
      filename pattern).
    - When ``cli.config_name`` is set: use ``<output_dir>/<cell>_<config_name>.{json,png}``,
      where ``<cell>`` defaults to the first ``_``-separated token of
      ``default_basename`` (the leaf scripts use ``<cell>_<purpose>_<inst>_...``
      so the cell name is usually the leading token).
      This keeps per-cell JSONs disjoint even when the same config name is
      shared across cells in a sweep — without it, every cell writes to the
      same ``<config_name>.json`` and a sweep loses all but one of its results
      to clobbering (autolens_profiling#44).
    - ``cell`` overrides that first-token derivation. **Required for any cell
      whose name itself contains an underscore**: ``delaunay_nn`` derives to
      ``delaunay`` under the default rule and would silently clobber the
      Delaunay cell's ``delaunay_<config_name>.json`` (autolens_profiling#219).
      Callers that pass nothing keep the default behaviour exactly.
    - ``cli.output_dir`` overrides ``default_dir`` when set.
    - When ``cli.use_sparse_operator`` is set, ``_sparse`` is appended to the
      resolved basename so dense and sparse JSONs from the same config don't
      clobber each other.
    """
    results_dir = cli.output_dir if cli.output_dir is not None else default_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    if cli.config_name is None:
        basename = default_basename
    else:
        # First underscore-separated token of default_basename is the cell,
        # unless the caller named it explicitly. Leaf scripts follow the
        # ``<cell>_<purpose>_<inst>_v<version>`` convention.
        cell_name = cell if cell is not None else default_basename.split("_", 1)[0]
        basename = f"{cell_name}_{cli.config_name}"
    if cli.use_sparse_operator:
        basename = f"{basename}_sparse"
    if cli.rect_mesh == "rtu":
        # Keep the two rectangular-mesh families' results disjoint; the
        # bilinear default keeps the unsuffixed (pre-split) filenames.
        basename = f"{basename}_rtu"
    return results_dir / f"{basename}.json", results_dir / f"{basename}.png"


def auto_simulate_if_missing(
    dataset_path: Path,
    *,
    dataset_type: str,
    instrument: str,
    workspace_root: Path,
) -> None:
    """If the dataset is missing, invoke the matching simulator script.

    ``dataset_type`` maps to ``scripts/misc/simulators/<dataset_type>.py``
    (imaging / interferometer / point_source / cluster / …). The simulator is
    invoked via subprocess with ``--instrument <instrument>``, so both the
    likelihood-fit dataset and a versioned simulator-timing JSON+PNG
    land at the right path in one shot.

    The dataset gate uses ``al.util.dataset.should_simulate`` (which also
    handles the ``PYAUTO_SMALL_DATASETS=1`` cleanup case). ``autolens`` is
    imported lazily so this helper can sit in any module without forcing
    the heavy import chain on every caller.
    """
    import sys

    import autolens as al  # noqa: F401 — imported lazily to defer side effects

    if not al.util.dataset.should_simulate(str(dataset_path)):
        return

    simulator_script = workspace_root / "scripts" / "misc" / "simulators" / f"{dataset_type}.py"
    if not simulator_script.exists():
        raise FileNotFoundError(
            f"Auto-simulate could not find simulator script at {simulator_script}. "
            f"Expected <dataset_type>.py (imaging / interferometer / point_source / cluster / …) "
            f"under scripts/misc/simulators/."
        )

    print(
        f"  [auto-simulate] {dataset_path} missing; invoking "
        f"scripts/misc/simulators/{dataset_type}.py --instrument {instrument}"
    )
    subprocess.run(
        [
            sys.executable,
            str(simulator_script),
            "--instrument",
            instrument,
            "--output-root",
            str(workspace_root),
        ],
        check=True,
    )


def check_pinned(got, expected, *, label: str, rtol: float = 1e-4):
    """Compare a computed likelihood/evidence against its pinned baseline.

    Runs here **record and flag** drift; they never adjudicate library
    correctness (that is autolens_workspace_test's remit). Returns ``None`` when ``got`` is
    within ``rtol`` of ``expected``; otherwise prints a loud warning and
    returns a drift record for the result JSON, which PyAutoHeart's vitals
    scan picks up. Never raises — a changed computation must not kill a
    inference run (the timing numbers are still data; the flag marks them
    non-comparable to the pinned baseline).
    """
    import numpy as _np

    arr = _np.asarray(got, dtype=float).ravel()
    rel = float(_np.max(_np.abs(arr - expected)) / max(abs(expected), 1e-300))
    got_f = float(arr[0]) if arr.size == 1 else float(arr[int(_np.argmax(_np.abs(arr - expected)))])
    if rel <= rtol:
        return None
    print(
        f"  WARNING: PINNED-VALUE DRIFT [{label}] — got {got_f!r}, "
        f"pinned {expected!r} (rel diff {rel:.3e} > rtol {rtol:g}). "
        f"Timings from this run are NOT comparable to the pinned baseline; "
        f"file a bug / check autolens_workspace_test before trusting trends."
    )
    return {
        "label": label,
        "expected": expected,
        "got": got_f,
        "rel_diff": rel,
        "rtol": rtol,
    }


def check_pinned_vector(got, expected, *, label: str, rtol: float = 1e-6):
    """Vector sibling of :func:`check_pinned` — compare an array of pinned values.

    For a pin that is a vector — a deflection sample, a per-stage log evidence
    chain — rather than a single scalar. The
    comparison reduces to the **maximum relative deviation** over the flattened
    pair, and keeps ``check_pinned``'s soft-failure discipline exactly: returns
    ``None`` when every element is within ``rtol``; otherwise prints a loud
    warning and returns a drift record for the result JSON. Never raises — a
    changed computation must not kill a inference run.
    """
    import numpy as _np

    got_arr = _np.asarray(got, dtype=float).ravel()
    exp_arr = _np.asarray(expected, dtype=float).ravel()

    if got_arr.shape != exp_arr.shape:
        print(
            f"  WARNING: PINNED-VECTOR SHAPE CHANGE [{label}] — got {got_arr.size} value(s), "
            f"pinned {exp_arr.size}. The pin and the computation no longer describe the same "
            f"quantity; re-pin deliberately with --repin --repin-reason."
        )
        return {
            "label": label,
            "expected_size": int(exp_arr.size),
            "got_size": int(got_arr.size),
            "rel_diff": float("inf"),
            "rtol": rtol,
        }

    # NaN is a legitimate pinned value here (a mass profile whose deflection
    # field is non-finite somewhere on the grid pins that fact). Two NaNs at the
    # same index match; a NaN facing a number — in either direction — is the
    # loudest possible drift, so it reduces to inf rather than to NaN.
    with _np.errstate(invalid="ignore"):
        rel_each = _np.abs(got_arr - exp_arr) / _np.maximum(_np.abs(exp_arr), 1e-300)
    rel_each = _np.where(_np.isnan(got_arr) & _np.isnan(exp_arr), 0.0, rel_each)
    rel_each = _np.where(_np.isnan(rel_each), _np.inf, rel_each)
    idx = int(_np.argmax(rel_each))
    rel = float(rel_each[idx])
    if rel <= rtol:
        return None
    print(
        f"  WARNING: PINNED-VALUE DRIFT [{label}] — element {idx} got {float(got_arr[idx])!r}, "
        f"pinned {float(exp_arr[idx])!r} (max rel diff {rel:.3e} > rtol {rtol:g}). "
        f"Timings from this run are NOT comparable to the pinned baseline; "
        f"file a bug / check autolens_workspace_test before trusting trends."
    )
    return {
        "label": label,
        "index": idx,
        "expected": float(exp_arr[idx]),
        "got": float(got_arr[idx]),
        "rel_diff": rel,
        "rtol": rtol,
    }


def record_pinned_check(json_path, expected, drift_records) -> None:
    """Merge the pinned-value check outcome into an already-written result JSON.

    Adds ``pinned_expected`` (the baseline value, or ``None`` when the
    instrument has no pin) and ``pinned_drift`` (list of drift records —
    empty means every compared value matched). The guard blocks run after
    the summary JSON is written, so this rewrites the file in place.
    """
    import json

    data = json.loads(json_path.read_text())
    data["pinned_expected"] = expected
    data["pinned_drift"] = drift_records
    json_path.write_text(json.dumps(data, indent=2))
