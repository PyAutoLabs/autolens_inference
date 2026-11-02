"""Likelihood-lifecycle phase 4 legs (autolens_inference#51): the manifest is the gaps around
the adopted 2026.10.17.1 rows, the submits are current and pass the wall gate, and each
cores-sweep leg has its own core count, results tag and output root."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
_MISC = str(REPO_ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

from campaigns import validation_p4 as p4  # noqa: E402
from wall import check_submits  # noqa: E402

RECORDS = p4.adopted_records()
LEGS = p4.planned_legs(RECORDS)
RENDERED = p4.submits()
SWEEP = [leg for leg in LEGS if leg.results_tag]


def _sbatch(text, key):
    return re.search(rf"^#SBATCH --{key}=(\S+)", text, re.M).group(1)


def test_the_manifest_is_the_ruled_gap_set():
    got = {
        (
            leg.path,
            leg.likelihood,
            leg.setup_id,
            leg.host_class,
            leg.model_variants,
            leg.seeds,
            leg.cores,
            leg.results_tag,
        )
        for leg in LEGS
    }
    gpu, cpu = "hpc/batch_gpu/submit_p4_", "hpc/batch_cpu/submit_p4_"
    expected = {
        (
            gpu + "validation_delaunay_hst_jax_gpu_dense",
            "delaunay",
            "jax_gpu_dense_fp64",
            "hpc_a100",
            ("default", "light_fixed"),
            (0, 1),
            8,
            None,
        ),
        # Optimal setups only (human, 2026-11-02). The adopted Delaunay numba rows hold seed 1
        # (default + light_fixed), so only seed 0 is generated.
        (
            cpu + "validation_delaunay_hst_numba_cpu_sparse_seed_0",
            "delaunay",
            "numba_cpu_sparse_fp64",
            "hpc_ral_cpu",
            ("default", "light_fixed"),
            (0,),
            8,
            None,
        ),
    } | {
        (
            cpu + f"cores_rectangular_hst_numba_cpu_sparse_cores_{n}",
            "rectangular",
            "numba_cpu_sparse_fp64",
            "hpc_ral_cpu",
            ("default",),
            (0,),
            n,
            f"cores_{n}",
        )
        for n in (1, 2, 4, 8, 16, 32)
    }
    assert got == expected
    assert set(RENDERED) == {leg.path for leg in LEGS}


def test_light_fixed_is_run_only_under_an_optimal_setup():
    for leg in LEGS:
        if "light_fixed" in leg.model_variants:
            assert leg.role.startswith("Optimal") and leg.lens_light == "both"
        else:
            assert leg.lens_light == "linear"
    with pytest.raises(ValueError, match="Optimal"):
        p4.Target("delaunay", "jax_cpu_sparse_fp64", "Secondary CPU", ("light_fixed",), True)


def test_no_leg_duplicates_an_adopted_row():
    literal = {
        (
            r["likelihood"]["variant"],
            r["instrument"],
            r["setup_id"],
            r["host_class"],
            r["model_variant"],
            r["seed"],
        )
        for r in RECORDS
    }
    # The kept hpc_a100 prefix of a CPU row is read as the RAL CPU node it ran on.
    normalised = {p4.chain_key(r) for r in RECORDS}
    with_sampler = {(*p4.chain_key(r), r["sampler"]) for r in RECORDS}
    for leg in LEGS:
        for key in leg.keys():
            assert key not in with_sampler, key
            if leg.results_tag is None:
                assert key[:6] not in literal and key[:6] not in normalised, key


def test_an_adopted_seed_drops_out_of_the_manifest():
    """The gaps follow the adopted rows: one more default row removes that seed's leg."""
    extra = dict(
        next(r for r in RECORDS if r["setup_id"] == "numba_cpu_sparse_fp64"),
        likelihood={"dataset_type": "imaging", "variant": "delaunay"},
        setup_id="jax_gpu_dense_fp64",
        host_class="hpc_a100",
        model_variant="default",
        seed=0,
    )
    legs = {leg.name: leg for leg in p4.planned_legs([*RECORDS, extra])}
    assert legs["submit_p4_validation_delaunay_hst_jax_gpu_dense_seed_1"].seeds == (1,)
    assert "submit_p4_validation_delaunay_hst_jax_gpu_dense" not in legs
    numba = next(
        r
        for r in RECORDS
        if r["likelihood"]["variant"] == "rectangular"
        and r["setup_id"] == "numba_cpu_sparse_fp64"
        and r["model_variant"] == "default"
        and r["seed"] == 0
    )
    swept = dict(numba, sampler="nautilus:source_pix[1]:cores_4")
    assert not any(leg.results_tag == "cores_4" for leg in p4.planned_legs([*RECORDS, swept]))


def test_the_committed_submits_are_current():
    assert p4.write_or_check(RENDERED, REPO_ROOT, write=False) == 0
    assert p4.generated_paths(REPO_ROOT) == set(RENDERED)


def test_a_hand_edit_or_an_orphan_fails_the_check(tmp_path):
    for sub in ("batch_gpu", "batch_cpu"):
        (tmp_path / "hpc" / sub).mkdir(parents=True)
    for rel, text in RENDERED.items():
        (tmp_path / rel).write_text(text)
    assert p4.write_or_check(RENDERED, tmp_path, write=False) == 0
    rel = next(iter(RENDERED))
    (tmp_path / rel).write_text(RENDERED[rel].replace("--time=", "--time=1", 1))
    assert p4.write_or_check(RENDERED, tmp_path, write=False) == 1
    (tmp_path / rel).write_text(RENDERED[rel])
    orphan = tmp_path / "hpc" / "batch_cpu" / "submit_p4_retired"
    orphan.write_text(RENDERED[rel])
    assert p4.write_or_check(RENDERED, tmp_path, write=False) == 1
    p4.write_or_check(RENDERED, tmp_path, write=True)
    assert not orphan.exists()


@pytest.mark.parametrize("leg", LEGS, ids=lambda leg: leg.name)
def test_every_submit_passes_the_wall_gate_and_says_what_it_runs(leg):
    text = RENDERED[leg.path]
    assert check_submits.check_text(text, leg.name) == []
    (row,) = check_submits.parse_basis_rows(text)
    assert row["cell"] == leg.cell and row["device"] == leg.device
    # measured-wall only where this cell and device have an adopted sibling (no Delaunay A100).
    assert row["source"] == ("unmeasured" if leg.on_gpu else "measured-wall")
    if row["source"] == "unmeasured":
        assert row["probe-first"] == "yes"
    assert _sbatch(text, "partition") == ("gpu" if leg.on_gpu else "ral")
    assert f"export PYAUTO_LIB_BASE={p4.PIN_BASE}\n" in text
    assert f"export PYAUTO_RELEASE_PIN={p4.PIN}\n" in text
    assert f"--config-name {leg.config_name} " in text
    assert (
        "--output-dir $OUTPUT_ROOT" in text and f"OUTPUT_ROOT=$AP_ROOT/{leg.output_dir}\n" in text
    )
    assert ("--stages-from hpc_a100_jax_cpu_dense_fp64:source_lp" in text) == leg.share_source_lp
    assert "JAX_PLATFORMS=cuda,cpu" in text if leg.on_gpu else "JAX_PLATFORMS=cpu" in text
    assert "NPROC=$SLURM_CPUS_PER_TASK" in text and "JAX_ENABLE_X64=True" in text


def test_the_cores_sweep_legs_are_one_core_count_each():
    assert [leg.cores for leg in SWEEP] == list(p4.SWEEP_CORES)
    assert len({leg.output_dir for leg in LEGS}) == len(LEGS)
    assert len({leg.results_tag for leg in SWEEP}) == len(SWEEP)
    for leg in SWEEP:
        text = RENDERED[leg.path]
        assert _sbatch(text, "cpus-per-task") == str(leg.cores)
        assert "--cores $SLURM_CPUS_PER_TASK" in text
        assert f"--results-tag cores_{leg.cores}" in text
        assert leg.output_dir.endswith(f"_cores_{leg.cores}")
        run = check_submits.strip_comments(text)
        assert "--stages source_pix_1" in run and "--export-prepared" not in run
        assert "--array" not in text and "--seed $SEED" in text and "SEED=0\n" in text
        # numba legs pin the BLAS / OpenMP pools.
        assert "PYAUTO_DISABLE_JAX=1" in text and "OMP_NUM_THREADS=1" in text
