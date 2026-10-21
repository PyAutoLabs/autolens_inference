"""``results/`` mirrors ``scripts/`` (autolens_inference#41).

Every ``results/<dataset_type>/<variant>/`` has a ``scripts/<dataset_type>/<variant>/``
that runs it, and every scripts variant with runnable leaves (a ``validation/`` or
``samplers/`` folder) has results. ``results/misc/`` and the root ``archive/`` are
outside the mirror, as are the README-only folders (``scripts/imaging/slam/`` and
the like) that document a family rather than run one.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PURPOSES = ("validation", "samplers")


def _subdirs(path):
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


def results_variants(root=ROOT):
    return {
        f"{d.name}/{v.name}"
        for d in _subdirs(root / "results")
        if d.name != "misc"
        for v in _subdirs(d)
    }


def script_variants(root=ROOT):
    return {
        f"{d.name}/{v.name}"
        for d in _subdirs(root / "scripts")
        if d.name != "misc"
        for v in _subdirs(d)
        if any((v / purpose).is_dir() for purpose in PURPOSES)
    }


def test_every_results_variant_has_its_scripts():
    assert results_variants() <= script_variants()


def test_every_script_variant_has_results():
    assert script_variants() <= results_variants()


def test_every_row_sits_under_a_purpose_folder():
    """``results/<dataset_type>/<variant>/<instrument>/{validation,samplers}/...``."""
    stray = [
        path.relative_to(ROOT).as_posix()
        for d in _subdirs(ROOT / "results")
        if d.name != "misc"
        for path in d.rglob("*.json")
        if path.relative_to(d).parts[2] not in PURPOSES
    ]
    assert stray == []


def test_the_mirror_check_sees_a_missing_side(tmp_path):
    (tmp_path / "results/imaging/rectangular/hst/validation").mkdir(parents=True)
    (tmp_path / "scripts/imaging/delaunay/samplers").mkdir(parents=True)
    (tmp_path / "scripts/imaging/slam").mkdir(parents=True)
    (tmp_path / "results/misc/logs").mkdir(parents=True)
    assert results_variants(tmp_path) == {"imaging/rectangular"}
    assert script_variants(tmp_path) == {"imaging/delaunay"}
