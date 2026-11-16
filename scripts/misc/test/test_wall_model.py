"""The calibrated wall model (autolens_inference#69) and its witness.

Phase-4 submit estimates missed the measured walls by 0.55-89.5x. The wall model replaces
that derivation; this file pins what it must do: every completed phase-4 leg's actual wall
within 0.5-2x of its estimate, from inputs available before the run.
"""

from __future__ import annotations

import pytest
from campaigns import validation_p4 as p4
from wall import model


@pytest.fixture(scope="module")
def validated():
    return p4.validate()


def test_every_completed_phase4_leg_is_within_the_band(validated):
    low, high = p4.VALIDATE_BAND
    assert len(validated) >= 40
    misses = [
        (leg.name, round(actual / estimate, 2))
        for leg, estimate, actual, _ in validated
        if not low <= actual / estimate <= high
    ]
    assert misses == []


def test_the_legs_the_old_derivation_missed_worst_are_covered(validated):
    """The finding's worst misses (89.5x, 81x, 49.6x, 4.5x+) are all re-priced."""
    names = {leg.name for leg, *_ in validated}
    for name in (
        "submit_p4_validation_multi_galaxy_mge_hst_jax_gpu_dense",
        "submit_p4_validation_point_source_source_source_plane_simple_jax_cpu_na",
        "submit_p4_validation_cluster_source_plane_simple_jax_gpu_na",
        "submit_p4_validation_multi_galaxy_mge_hst_jax_cpu_dense",
    ):
        assert name in names


def test_numba_is_priced_per_worker_at_the_measured_efficiency():
    assert model.effective_per_call(0.55, "numba_cpu", 8) == pytest.approx(0.55 / (8 * 0.55))
    assert model.effective_per_call(0.1, "jax_gpu", 8) == 0.1
    assert model.parallel_efficiency(12) == model.PARALLEL_EFFICIENCY[8]


def test_a_microsecond_likelihood_still_pays_the_sampler_floor():
    evals = model.stage_evals(5, 100, "weak")
    assert model.stage_seconds(evals, 1e-5) >= evals * model.FLOOR_S


def test_the_multi_galaxy_source_lp_count_is_not_the_imaging_proxy():
    """414160: 250,550-271,650 evaluations at d=24, n_live=250, against a proxy of 92,550."""
    evals = model.stage_evals(24, 250, "multi_galaxy")
    assert 0.5 <= 271_650 / evals <= 2.0
