"""The lens-light mode as a run variant, schema v3 provenance, and the pinned submits.

``--lens-light {linear,fixed,both}`` changes only ``mass_total[1]``: ``fixed``
replaces ``light[1]``'s linear light profiles by ordinary ones carrying the
intensities its maximum-likelihood fit solved for. These tests hold the seams
where getting that wrong would produce a row that *looks* fine:

- the variant names (``slam_base`` stays the linear name, so no existing row or
  output path moves; ``fixed`` is ``<variant>_lightfixed``);
- the fixed-light mass model really holds no linear light profile, and differs
  from the linear one in the lens light only;
- schema v3 rows carry the light mode and the provenance, and v1/v2 rows still
  read as linear with null provenance;
- a ``both`` run exports two prepared problems that both verify;
- ``activate.sh`` honours a library-root override without moving the venv;
- the four clean-slate submits pass the wall gate and say what they claim.

The end-to-end check — one ``PYAUTO_TEST_MODE`` euclid-cell process writing
both rows — is the witness recorded on autolens_inference#24's PR, not a unit
test: a five-stage chain is minutes of compute, not a CI test.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _path in (
    str(REPO_ROOT),
    str(REPO_ROOT / "scripts" / "misc"),
    str(REPO_ROOT / "scripts" / "misc" / "tooling"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_readme  # noqa: E402
from experiments import prepared, run  # noqa: E402
from slam import _runner  # noqa: E402
from wall import check_submits  # noqa: E402

import _inference_cli  # noqa: E402
from _inference_cli import (  # noqa: E402
    InferenceCLI,
    base_variant_name,
    lens_light_modes,
    variant_name,
)

_spec = importlib.util.spec_from_file_location(
    "summary_v3", REPO_ROOT / "scripts/misc/tooling/export_inference_summary.py"
)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def _cli(**overrides) -> InferenceCLI:
    fields = {
        "config_name": "local_jax_cpu_dense_fp64",
        "output_dir": None,
        "use_mixed_precision": False,
        "instrument": "euclid",
        "backend": "jax_cpu",
        "inversion": "dense",
        "seed": 0,
        "use_sparse_operator": False,
        "rect_mesh": "bilinear",
        "mesh": "rect",
        "mesh_pixels": 28,
        "regularization": None,
        "memo": "off",
        "cores": 4,
        "stages": None,
    }
    fields.update(overrides)
    return InferenceCLI(**fields)


# ---------------------------------------------------------------------------
# Variant naming and the CLI
# ---------------------------------------------------------------------------


def test_linear_keeps_todays_names_and_fixed_appends_lightfixed():
    assert variant_name(_cli()) == "slam_base"
    assert variant_name(_cli(lens_light="linear")) == "slam_base"
    assert variant_name(_cli(lens_light="fixed")) == "slam_base_lightfixed"
    assert variant_name(_cli(mesh="delaunay", mesh_pixels=1250)) == "delaunay_1250"
    assert (
        variant_name(_cli(mesh="delaunay", mesh_pixels=1250, lens_light="fixed"))
        == "delaunay_1250_lightfixed"
    )
    assert variant_name(_cli(mesh_pixels=40, lens_light="fixed")) == "rect_40_lightfixed"


def test_both_has_one_variant_per_mode_and_no_single_variant():
    cli = _cli(lens_light="both")
    assert lens_light_modes(cli) == ("linear", "fixed")
    assert [variant_name(cli, mode) for mode in lens_light_modes(cli)] == [
        "slam_base",
        "slam_base_lightfixed",
    ]
    with pytest.raises(ValueError, match="pass lens_light_mode explicitly"):
        variant_name(cli)
    # Stages 1-4 never see the light mode: they keep the mesh-only variant.
    for mode in ("linear", "fixed", "both"):
        assert base_variant_name(_cli(lens_light=mode)) == "slam_base"
    assert lens_light_modes(_cli()) == ("linear",)
    assert lens_light_modes(_cli(lens_light="fixed")) == ("fixed",)


def test_only_the_mass_stage_prefix_carries_the_light_mode():
    stages_1_to_4 = _runner.output_path_prefix(
        "imaging", "euclid", base_variant_name(_cli(lens_light="fixed")), "c", 0
    )
    linear_mass = _runner.output_path_prefix("imaging", "euclid", "slam_base", "c", 0)
    fixed_mass = _runner.output_path_prefix("imaging", "euclid", "slam_base_lightfixed", "c", 0)
    assert stages_1_to_4 == linear_mass, "fixed must resume the linear chain's stages 1-4"
    assert fixed_mass != linear_mass, "the two mass fits must never resume each other"


def test_the_target_id_carries_the_light_variant():
    assert _runner.target_id("hst", "slam_base", 0) == "hst/slam5/seed0"
    assert (
        _runner.target_id("hst", "slam_base_lightfixed", 0)
        == "hst/slam5_slam_base_lightfixed/seed0"
    )


def test_cli_parses_lens_light_and_results_tag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["leaf"])
    cli = _inference_cli.parse_inference_cli()
    assert cli.lens_light == "linear" and cli.results_tag is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["leaf", "--lens-light", "both", "--results-tag", "cleanslate_2026_10_17"],
    )
    cli = _inference_cli.parse_inference_cli()
    assert cli.lens_light == "both" and cli.results_tag == "cleanslate_2026_10_17"

    for bad in ("Clean-Slate", "a/b", "_x"):
        monkeypatch.setattr(sys, "argv", ["leaf", "--results-tag", bad])
        with pytest.raises(SystemExit):
            _inference_cli.parse_inference_cli()


def test_a_results_tag_suffixes_the_config_directory_only(tmp_path):
    untagged, _ = _runner.results_paths(
        tmp_path, "imaging", "hst", "slam_base", "hpc_a100_jax_gpu_dense_fp64", 0
    )
    tagged, _ = _runner.results_paths(
        tmp_path,
        "imaging",
        "hst",
        "slam_base",
        "hpc_a100_jax_gpu_dense_fp64",
        0,
        "cleanslate_2026_10_17",
    )
    assert untagged.relative_to(tmp_path) == Path(
        "results/slam/imaging/hst/slam_base/hpc_a100_jax_gpu_dense_fp64/stages_seed0.json"
    )
    assert tagged.relative_to(tmp_path) == Path(
        "results/slam/imaging/hst/slam_base/"
        "hpc_a100_jax_gpu_dense_fp64__cleanslate_2026_10_17/stages_seed0.json"
    )


# ---------------------------------------------------------------------------
# Schema v3 provenance
# ---------------------------------------------------------------------------


def test_schema_v3_provenance_helpers():
    assert _runner.SCHEMA_VERSION == 3
    assert _runner.slurm_job_id_from({}) is None
    assert _runner.slurm_job_id_from({"SLURM_JOB_ID": "412011"}) == "412011"
    assert (
        _runner.slurm_job_id_from(
            {"SLURM_JOB_ID": "412013", "SLURM_ARRAY_JOB_ID": "412011", "SLURM_ARRAY_TASK_ID": "1"}
        )
        == "412011_1"
    )
    assert _runner.release_pin_from({}) is None
    assert _runner.release_pin_from({"PYAUTO_RELEASE_PIN": "2026.10.17.1"}) == "2026.10.17.1"
    # v1/v2 rows predate the field: they are linear by definition.
    assert _runner.lens_light_mode_of({"schema_version": 2}) == "linear"
    assert _runner.lens_light_mode_of({"lens_light_mode": "fixed"}) == "fixed"


def test_library_revisions_never_kill_a_leg(monkeypatch, tmp_path):
    def refuse(_root):
        raise ValueError("Installed autofit checkout has uncommitted changes")

    monkeypatch.setattr(prepared, "library_revisions", refuse)
    revisions, note = _runner.library_revisions_or_note(tmp_path)
    assert revisions is None and "uncommitted" in note

    monkeypatch.setattr(prepared, "library_revisions", lambda _root: {"PyAutoFit": "a" * 40})
    assert _runner.library_revisions_or_note(tmp_path) == ({"PyAutoFit": "a" * 40}, None)


def _row(variant, mode=None, *, schema=3, tag=None, config="local_jax_cpu_dense_fp64"):
    payload = {
        "schema_version": schema,
        "target": _runner.target_id("euclid", variant, 0),
        "variant": variant,
        "config_name": config,
        "backend": "jax_cpu",
        "inversion": "dense",
        "precision": "fp64",
        "instrument": "euclid",
        "dataset_class": "imaging",
        "seed": 0,
        "mesh": "rect",
        "version": "2026.10.17.1",
        "status": "complete",
        "stages": [
            {
                "name": name,
                "completed": True,
                "wall_s": 1.0,
                "total_wall_s": 1.0,
                "posterior": (
                    {"einstein_radius": {"median": 1.6, "sigma": 0.02}}
                    if name == "mass_total[1]"
                    else {}
                ),
            }
            for name in build_readme.SLAM_STAGE_ORDER
        ],
    }
    if schema >= 3:
        payload.update(
            lens_light_mode=mode,
            results_tag=tag,
            library_revisions={"PyAutoFit": "f" * 40},
            release_pin="2026.10.17.1",
            slurm_job_id="412011_0",
        )
    return payload


def _put(root, payload, tag=None):
    config_dir = payload["config_name"] + (f"__{tag}" if tag else "")
    path = (
        root
        / "results/slam/imaging/euclid"
        / payload["variant"]
        / config_dir
        / f"stages_seed{payload['seed']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_summary_reads_v3_and_v2_rows(tmp_path):
    _put(tmp_path, _row("slam_base", schema=2))
    _put(tmp_path, _row("slam_base_lightfixed", "fixed"))
    doc = summary.build(tmp_path, "a" * 40, "2026-10-17T00:00:00+00:00")
    assert doc["coverage"]["excluded"] == []
    parents = {r["configuration"]["variant"]: r for r in doc["records"] if r["stage"] is None}
    assert parents["slam_base"]["configuration"]["lens_light_mode"] == "linear"
    assert "release_pin" not in parents["slam_base"]["configuration"]
    assert "dependency_revisions" in parents["slam_base"]["unknown_reasons"]

    fixed = parents["slam_base_lightfixed"]
    assert fixed["configuration"]["lens_light_mode"] == "fixed"
    assert fixed["configuration"]["release_pin"] == "2026.10.17.1"
    assert fixed["dependency_revisions"] == {"PyAutoFit": "f" * 40}
    assert fixed["hardware"]["slurm_job_id"] == "412011_0"
    assert fixed["target"] != parents["slam_base"]["target"]


def test_readme_keeps_light_modes_and_tags_in_separate_parity_groups(monkeypatch, tmp_path):
    _put(tmp_path, _row("slam_base", "linear"))
    _put(tmp_path, _row("slam_base", "linear", config="local_numba_cpu_dense_fp64"))
    _put(tmp_path, _row("slam_base_lightfixed", "fixed"))
    tagged = _row("slam_base", "linear", tag="cleanslate_2026_10_17")
    _put(tmp_path, tagged, tag="cleanslate_2026_10_17")
    _put(tmp_path, _row("slam_base", schema=2, config="local_jax_cpu_sparse_fp64"))
    monkeypatch.setattr(build_readme, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_readme, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(build_readme, "SLAM_ROOT", tmp_path / "results" / "slam")

    rendered = build_readme.render_slam()

    assert "| Target | Variant | Light |" in rendered
    assert "**Parity — `euclid/slam5/seed0` (seed 0), `mass_total[1]`**" in rendered
    assert "**Parity — `euclid/slam5_slam_base_lightfixed/seed0` (seed 0)" in rendered
    assert "**Parity — `euclid/slam5/seed0` (seed 0, tag `cleanslate_2026_10_17`)" in rendered
    assert "`local_jax_cpu_dense_fp64__cleanslate_2026_10_17`" in rendered
    lights = {line.split("|")[3].strip() for line in rendered.splitlines() if "| `euclid/" in line}
    assert lights == {"linear", "fixed"}, "a v2 row must read as linear"
    untagged = rendered.split("**Parity — `euclid/slam5/seed0` (seed 0), `mass_total[1]`**")[1]
    untagged = untagged.split("**Parity")[0]
    assert "`local_jax_cpu_dense_fp64` (ref)" in untagged
    assert "cleanslate" not in untagged, "a tagged re-measurement never joins the untagged group"


# ---------------------------------------------------------------------------
# Prepared export under both modes
# ---------------------------------------------------------------------------


class _Prior:
    def __init__(self, upper):
        self.upper = upper

    def dict(self):
        return {"type": "Uniform", "lower_limit": 0, "upper_limit": self.upper}


class _Model:
    def __init__(self, upper):
        self.path_priors_tuples = [(("mass",), _Prior(upper))]

    def dict(self):
        return {"mass": self.path_priors_tuples[0][1].dict()}


def test_both_modes_export_two_verifiable_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(prepared, "revision", lambda _: "a" * 40)
    modes = ("linear", "fixed")
    export_root = Path("output/prepared/witness")
    manifests = {}
    for index, mode in enumerate(modes):
        variant = variant_name(_cli(lens_light="both"), mode)
        directory = _runner.prepared_export_dir(export_root, modes, variant)
        assert directory == export_root / variant
        manifests[mode] = prepared.export_problem(
            tmp_path,
            directory,
            _Model(2 + index),
            SimpleNamespace(dataset={"data": [1, 2]}, positions=[4], lens=mode),
            setup_id="imaging/rectangular/euclid",
            baseline_record_id=f"results/{variant}#stage",
            controls={"backend": "jax_cpu", "variant": variant, "lens_light_mode": mode},
            preparation_s=1,
            dependency_revisions={"PyAutoFit": "b" * 40},
        )
    docs = {mode: prepared.verify_manifest(tmp_path, path) for mode, path in manifests.items()}
    assert docs["linear"]["id"] != docs["fixed"]["id"]
    assert docs["fixed"]["controls"]["lens_light_mode"] == "fixed"
    for doc in docs.values():
        run.validate_setup_binding(doc, ("imaging", "rectangular"))
    # A single-mode run keeps exporting to the directory it was given.
    assert _runner.prepared_export_dir(export_root, ("fixed",), "slam_base_lightfixed") == (
        export_root
    )


# ---------------------------------------------------------------------------
# The fixed-light mass model
# ---------------------------------------------------------------------------


def _mass_models(monkeypatch):
    """Build both mass_total[1] models from one real light[1] fit.

    The light stage's lens bulge is the real linear MGE; ``max_log_likelihood_fit``
    is a real ``al.FitImaging`` on a tiny masked dataset, so the conversion under
    test is the library's own ``model_obj_linear_light_profiles_to_light_profiles``.
    """
    import autofit as af
    import autolens as al
    import numpy as np

    monkeypatch.setattr(al, "AnalysisImaging", lambda **kwargs: kwargs)
    monkeypatch.setattr(af, "Nautilus", lambda **kwargs: kwargs)
    monkeypatch.setattr(_runner, "_adapt_images", lambda *args, **kwargs: object())
    source = af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.Sersic))
    monkeypatch.setattr(al.util.chaining, "source_custom_model_from", lambda **kwargs: source)
    monkeypatch.setattr(al.util.chaining, "source_from", lambda **kwargs: source)

    settings_search = SimpleNamespace(search_dict={})
    pixel_scale = 0.2
    dataset = SimpleNamespace(pixel_scales=(pixel_scale,))
    common = dict(use_jax=False, seed=1)

    def result(model, fit=None):
        return SimpleNamespace(
            model=model, instance=model.instance_from_prior_medians(), max_log_likelihood_fit=fit
        )

    lp_model, _, _ = _runner.source_lp(af, al, settings_search, dataset, 1.5, 0.5, 1.0, **common)
    source_pix_result = result(lp_model)

    light_model, _, _ = _runner.light_lp(
        af,
        al,
        settings_search,
        dataset,
        1.5,
        source_pix_result,
        source_pix_result,
        meshes=None,
        positions_likelihood=object(),
        **common,
    )
    light_instance = light_model.instance_from_prior_medians()
    assert _runner.linear_light_profiles_in(al, light_instance.galaxies.lens), (
        "the light stage must hold linear profiles, or this test proves nothing"
    )

    shape = (25, 25)
    imaging = al.Imaging(
        data=al.Array2D.no_mask(
            values=np.random.default_rng(0).normal(1.0, 0.1, shape), pixel_scales=pixel_scale
        ),
        noise_map=al.Array2D.full(fill_value=0.1, shape_native=shape, pixel_scales=pixel_scale),
        psf=al.Convolver.from_gaussian(shape_native=(3, 3), sigma=0.1, pixel_scales=pixel_scale),
        over_sample_size_lp=1,
    )
    mask = al.Mask2D.circular(shape_native=shape, pixel_scales=pixel_scale, radius=1.5)
    tracer = al.Tracer(galaxies=light_instance.galaxies, fields=light_instance.fields)
    fit = al.FitImaging(dataset=imaging.apply_mask(mask=mask), tracer=tracer)
    light_result = SimpleNamespace(
        model=light_model, instance=light_instance, max_log_likelihood_fit=fit
    )

    models = {}
    for mode in ("linear", "fixed"):
        models[mode], _, _ = _runner.mass_total(
            af,
            al,
            settings_search,
            dataset,
            source_pix_result,
            source_pix_result,
            light_result,
            meshes=None,
            mask_radius=1.5,
            positions_likelihood=object(),
            lens_light_mode=mode,
            **common,
        )
    return al, fit, models


def test_the_fixed_light_mass_model_holds_no_linear_profiles(monkeypatch):
    al, fit, models = _mass_models(monkeypatch)

    linear_lens = models["linear"].instance_from_prior_medians().galaxies.lens
    fixed_lens = models["fixed"].instance_from_prior_medians().galaxies.lens
    assert _runner.linear_light_profiles_in(al, linear_lens), "linear keeps today's lens light"
    assert _runner.linear_light_profiles_in(al, fixed_lens) == []

    # The fixed profiles are the linear ones with their solved intensities.
    linear_profiles = list(linear_lens.bulge.profile_list)
    fixed_profiles = list(fixed_lens.bulge.profile_list)
    assert len(fixed_profiles) == len(linear_profiles)
    solved = fit.linear_light_profile_intensity_dict
    for linear, fixed in zip(linear_profiles, fixed_profiles, strict=True):
        assert fixed.centre == linear.centre and fixed.sigma == linear.sigma
        assert fixed.intensity == pytest.approx(float(solved[linear]))


def test_the_two_mass_models_differ_only_in_the_lens_light(monkeypatch):
    _al, _fit, models = _mass_models(monkeypatch)
    linear, fixed = models["linear"], models["fixed"]

    assert linear.prior_count == fixed.prior_count
    assert [path for path, _ in linear.path_priors_tuples] == [
        path for path, _ in fixed.path_priors_tuples
    ]

    def without_id(prior):
        # Each build makes fresh prior objects (the PowerLaw slope), so ids differ.
        return {k: v for k, v in prior.dict().items() if k != "id"}

    assert [without_id(prior) for _, prior in linear.path_priors_tuples] == [
        without_id(prior) for _, prior in fixed.path_priors_tuples
    ]

    def state(obj):
        # Instances made by two builds carry different object ids; the values must match.
        return re.sub(r"id: \d+", "", str(obj))

    linear_instance = linear.instance_from_prior_medians()
    fixed_instance = fixed.instance_from_prior_medians()
    for attr in ("mass", "redshift"):
        assert state(getattr(linear_instance.galaxies.lens, attr)) == state(
            getattr(fixed_instance.galaxies.lens, attr)
        )
    assert state(linear_instance.galaxies.source) == state(fixed_instance.galaxies.source)
    assert state(linear_instance.fields) == state(fixed_instance.fields)
    assert state(linear_instance.galaxies.lens.bulge) != state(fixed_instance.galaxies.lens.bulge)


# ---------------------------------------------------------------------------
# activate.sh and the clean-slate submits
# ---------------------------------------------------------------------------


def _source_activate(tmp_path, lib_base: str | None) -> str:
    shared = tmp_path / "shared"
    (shared / "PyAuto" / "bin").mkdir(parents=True)
    (shared / "PyAuto" / "bin" / "activate").write_text("export VENV_FROM=shared\n")
    script = (REPO_ROOT / "activate.sh").read_text()
    script = script.replace("SHARED_BASE=/mnt/ral/jnightin/PyAuto", f"SHARED_BASE={shared}")
    activate = tmp_path / "activate.sh"
    activate.write_text(script)
    env = {"PATH": "/usr/bin:/bin", "PYAUTO_HPC_CACHE": str(tmp_path / "cache")}
    if lib_base is not None:
        env["PYAUTO_LIB_BASE"] = lib_base
    completed = subprocess.run(
        ["bash", "-c", f'source "{activate}" && echo "$VENV_FROM|$PYTHONPATH"'],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return completed.stdout.strip()


def test_activate_honours_a_library_root_override_and_keeps_the_shared_venv(tmp_path):
    default = _source_activate(tmp_path / "a", None)
    venv, pythonpath = default.split("|")
    assert venv == "shared"
    assert f"{tmp_path / 'a' / 'shared'}/PyAutoFit" in pythonpath.split(":")

    pinned = _source_activate(tmp_path / "b", "/pins/2026.10.17.1")
    venv, pythonpath = pinned.split("|")
    assert venv == "shared", "a pin directory must not need a venv of its own"
    entries = pythonpath.split(":")
    for repo in ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens"):
        assert f"/pins/2026.10.17.1/{repo}" in entries
    assert not any("shared" in entry for entry in entries)


CLEANSLATE = {
    "batch_gpu/submit_hst_cleanslate_jax_gpu_dense": ("gpu", "jax_gpu", "dense", "64gb"),
    "batch_gpu/submit_hst_cleanslate_jax_gpu_sparse": ("gpu", "jax_gpu", "sparse", "64gb"),
    "batch_cpu/submit_hst_cleanslate_jax_cpu_dense": ("ral", "jax_cpu", "dense", "48gb"),
    "batch_cpu/submit_hst_cleanslate_jax_cpu_sparse": ("ral", "jax_cpu", "sparse", "64gb"),
}


@pytest.mark.parametrize("name", sorted(CLEANSLATE))
def test_the_cleanslate_submits_pass_the_gate_and_say_what_they_claim(name):
    partition, backend, inversion, mem = CLEANSLATE[name]
    text = (REPO_ROOT / "hpc" / name).read_text()
    assert check_submits.check_text(text, name) == []
    assert f"#SBATCH --partition={partition}\n" in text
    assert f"#SBATCH --mem={mem}\n" in text
    assert "#SBATCH --array=0-1\n" in text
    assert "probe-first: yes" in text and "source: unmeasured" in text
    assert "export PYAUTO_LIB_BASE=/mnt/ral/jnightin/inference/pins/2026.10.17.1" in text
    assert "export PYAUTO_RELEASE_PIN=2026.10.17.1" in text
    assert text.index("PYAUTO_LIB_BASE=") < text.index("source $AP_ROOT/activate.sh")
    for flag in (
        f"--backend {backend}",
        f"--inversion {inversion}",
        f"--config-name hpc_a100_{backend}_{inversion}_fp64",
        "--lens-light both",
        f"--export-prepared output/autolens/prepared/hpc_a100_{backend}_{inversion}_fp64/"
        "seed_$SLURM_ARRAY_TASK_ID",
        "--seed $SLURM_ARRAY_TASK_ID",
    ):
        assert flag in text, flag
    # #35: the default output root and plain rows; the dated root and tag are gone.
    command = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "--output-dir" not in command and "--results-tag" not in command
    assert "export JAX_ENABLE_X64=True" in text and "export NPROC=$SLURM_CPUS_PER_TASK" in text
    if partition == "gpu":
        assert "#SBATCH --gres=gpu:1\n" in text and "export JAX_PLATFORMS=cuda,cpu" in text
        assert "#SBATCH --time=12:00:00\n" in text and "nvidia-smi" in text
    else:
        assert "--gres" not in text and "export JAX_PLATFORMS=cpu" in text
        assert "#SBATCH --time=5-00:00:00\n" in text
