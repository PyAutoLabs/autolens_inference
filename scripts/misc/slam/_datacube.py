"""The datacube family: one Nautilus search over a factor graph of channels, on the shared core.

Likelihood-lifecycle 4c-1 (autolens_inference#54). The likelihood is ``datacube/delaunay`` as
the profiling catalogue names it, on the reference instrument ``sma``. The search mirrors
``autolens_workspace/scripts/interferometer/features/datacube/delaunay.py``: ONE Nautilus search
(n_live 100, n_batch 20) of an ``Isothermal`` lens with external shear and a Delaunay source,
over an ``af.FactorGraphModel`` of one ``AnalysisInterferometer`` per channel, every channel
fitting the same lens model, with the simulator's positions at threshold 0.3. It is a
single-stage family; that stage evaluates the likelihood and pairs with its profiling record.

The cube and the source are the profiled cell's (autolens_profiling
``scripts/misc/likelihood_breakdown/datacube_standard.py``, decision D2): :data:`N_CHANNELS`
channels of the ``sma`` visibilities, channel ``c`` scaled by a Gaussian emission-line amplitude
(:func:`channel_scales`; channel 0 is the interferometer dataset itself), sharing the uv
coverage, the noise map and the real-space mask, built in memory; each channel's analysis has
``shared_preloads=True``, so the lead channel builds the mapper and the curvature matrix once
per evaluation. The mesh is the profiled interferometer Delaunay preset (ruling R3): 1250
Hilbert vertices (weight power 3.5, floor 0.01) drawn from the truth lensed source
(``lensed_source.fits``, uncapped: a single search has no earlier result to adapt to) plus the
30 zeroed edge points, under a free ``AdaptSplit``, where the workspace example uses a 26 x 26
``Overlay`` grid under ``ConstantSplit``.

Setups: ``--transformer wtilde`` (``--inversion sparse``) builds the NUFFT W~ operator once —
it depends only on the shared uv coverage and noise map — and hands it to every channel, each
of which computes its own dirty image; ``--transformer nufft`` (``--inversion dense``) fits the
plain ``TransformerNUFFT`` mapping matrix. Departures from the workspace script, as the other
families: ``use_jax=leg.use_jax`` on every analysis and on the factor graph, and the library's
positive-only solver in ``al.Settings`` (the workspace turns it off, which would time a
different solve from the profiled cell's).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _inference_cli import MESH_PIXELS_STANDARD  # noqa: E402
from slam._core import Family, Stage  # noqa: E402
from slam._interferometer import (  # noqa: E402
    apply_wtilde,
    hilbert_mesh_grid,
    load_interferometer,
    pixelized_fields,
)
from slam._runner import REDSHIFT_LENS, REDSHIFT_SOURCE, mesh_models  # noqa: E402

#: Channels per cube, and the emission line's width as a fraction of them (decision D2 of
#: the profiled cell: ``a_0 = 1`` at the line peak, ``a_{N-1} ~ exp(-2)``).
N_CHANNELS = 34
LINE_SIGMA_FRACTION = 0.5

#: The workspace datacube search's positions threshold.
POSITIONS_THRESHOLD = 0.3


def channel_scales(n_channels: int = N_CHANNELS) -> list[float]:
    """The per-channel visibility scale ``a_c``: a Gaussian emission line peaking at channel 0
    (``datacube_standard.channel_scales``, the profiled cube's)."""
    n = max(1, int(n_channels))
    sigma = LINE_SIGMA_FRACTION * n
    return [math.exp(-0.5 * (c / sigma) ** 2) for c in range(n)]


def channel_datasets(leg, context, scales: list[float]) -> list:
    """The cube's channels: channel 0 the dataset itself, channel ``c`` its visibilities x
    ``a_c``, on this leg's route (W~ built once and shared, or the plain NUFFT)."""
    import numpy as np

    from slam._interferometer import transformer_class

    al = leg.al
    base = context.dataset_nufft
    raw = [base] + [
        al.Interferometer(
            data=al.Visibilities(np.asarray(base.data) * a),
            noise_map=base.noise_map,
            uv_wavelengths=base.uv_wavelengths,
            real_space_mask=base.real_space_mask,
            transformer_class=transformer_class(al, context.chunk_size),
        )
        for a in scales[1:]
    ]
    if leg.cli.transformer != "wtilde":
        return raw
    lead = context.dataset  # load_interferometer already built W~ on channel 0
    precision = np.asarray(lead.sparse_operator.nufft_precision_operator)
    return [lead] + [
        apply_wtilde(dataset, leg, context.chunk_size, nufft_precision_operator=precision)
        for dataset in raw[1:]
    ]


def load(leg):
    """The ``sma`` dataset, its cube of channels, the truth-drawn Delaunay vertices and the
    positions likelihood."""
    if leg.cli.mesh != "delaunay":
        print(f"ERROR: --mesh {leg.cli.mesh!r}: the datacube runs only delaunay", file=sys.stderr)
        return 2
    context = load_interferometer(leg)
    if isinstance(context, int):
        return context
    al = leg.al
    context.meshes = mesh_models(leg.af, al, leg.cli)
    context.scales = channel_scales()
    context.channels = channel_datasets(leg, context, context.scales)
    mask = context.dataset.real_space_mask
    adapt_image = al.Array2D.from_fits(
        file_path=leg.dataset_path / "lensed_source.fits", pixel_scales=mask.pixel_scales
    ).apply_mask(mask=mask)
    name = "('galaxies', 'source')"
    context.adapt_images = al.AdaptImages(
        galaxy_name_image_dict={name: adapt_image},
        galaxy_name_image_plane_mesh_grid_dict={
            name: hilbert_mesh_grid(al, context.meshes, mask, context.mask_radius, adapt_image)
        },
    )
    context.positions_likelihood = al.PositionsLH(
        positions=context.fallback_positions, threshold=POSITIONS_THRESHOLD
    )
    return context


def build_cube(leg, context, results):
    """The factor graph of every channel, fitting one shared lens model."""
    af, al = leg.af, leg.al
    meshes = context.meshes
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=REDSHIFT_LENS, mass=af.Model(al.mp.Isothermal)),
            source=af.Model(
                al.Galaxy,
                redshift=REDSHIFT_SOURCE,
                pixelization=af.Model(
                    al.Pixelization, mesh=meshes.mesh, regularization=meshes.regularization
                ),
            ),
        ),
        fields=af.Model(al.MassField, redshift=REDSHIFT_LENS, shear=af.Model(al.mp.ExternalShear)),
    )
    analyses = [
        al.AnalysisInterferometer(
            dataset=dataset,
            adapt_images=context.adapt_images,
            positions_likelihood_list=[context.positions_likelihood],
            settings=context.settings,
            use_jax=leg.use_jax,
            shared_preloads=True,
        )
        for dataset in context.channels
    ]
    # One prior model for every factor (the profiled cube's; the workspace's per-factor
    # `model.copy()` keeps the same prior ids, so it is the same global model).
    graph = af.FactorGraphModel(
        *[af.AnalysisFactor(prior_model=model, analysis=analysis) for analysis in analyses],
        use_jax=leg.use_jax,
    )
    print(
        f"  channels: {len(analyses)}; global free parameters: "
        f"{graph.global_prior_model.total_free_parameters}"
    )
    return graph.global_prior_model, graph, leg.nautilus(STAGES[0])


STAGES = (Stage("delaunay", "delaunay", 100, 20, build_cube, inversion_applies=True),)


def model_fields(leg, context) -> dict:
    """The Delaunay fields of the interferometer rows, and the cube the row fitted."""
    return {
        **pixelized_fields(leg, context),
        "n_channels": len(context.scales),
        "channel_scales": [float(a) for a in context.scales],
        "adapt_image": "truth lensed source (lensed_source.fits), uncapped",
    }


FAMILY = Family(
    dataset_type="datacube",
    likelihood_variant="delaunay",
    stages=STAGES,
    load=load,
    dataset_class="interferometer",
    pipeline="search",
    variant=lambda cli: (
        "search_delaunay"
        if cli.mesh_pixels == MESH_PIXELS_STANDARD["delaunay"]
        else f"delaunay_{cli.mesh_pixels}"
    ),
    inversions=("dense", "sparse"),
    transformers=("nufft", "wtilde"),
    model_fields=model_fields,
    title="Datacube search",
)
