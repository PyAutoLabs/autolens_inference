"""The README auto-table renderer is idempotent on this checkout.

Two properties, both of which the `lint` workflow depends on:

1. **`--check` exits 0 on a committed tree.** If it does not, the committed
   README disagrees with what the current results would render, and every PR
   from here on inherits a red gate that has nothing to do with its own change.

2. **Rendering twice changes nothing the second time.** A renderer that is not a
   fixed point turns the CI gate into a coin flip: `--check` passes or fails
   depending on how many times it happened to run. The empty-state path is
   included on purpose — this repo was born with no results, so the empty render
   is the one that has to be stable first.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
_TOOLING = str(REPO_ROOT / "scripts" / "misc" / "tooling")
if _TOOLING not in sys.path:
    sys.path.insert(0, _TOOLING)

import build_readme  # noqa: E402


def test_check_is_clean_on_the_committed_tree():
    assert build_readme.main(["--check"]) == 0


def test_rendering_is_a_fixed_point():
    readme = REPO_ROOT / "README.md"
    before = readme.read_text()

    _, once, unknown = build_readme._rewrite_file(readme, build_readme.RENDERERS)
    assert unknown == [], f"unknown sentinels left un-rendered: {unknown}"

    readme.write_text(once)
    try:
        _, twice, _ = build_readme._rewrite_file(readme, build_readme.RENDERERS)
        assert twice == once, "second render differs from the first — renderer is not idempotent"
        assert once == before, "committed README differs from what the renderer produces"
    finally:
        readme.write_text(before)


def test_every_sentinel_in_the_readme_has_a_renderer():
    readme = (REPO_ROOT / "README.md").read_text()
    names = {m.group("name") for m in build_readme.SENTINEL_RE.finditer(readme)}
    assert names, "README.md has no auto-table sentinels — did a rewrite drop them?"
    missing = names - set(build_readme.RENDERERS)
    assert not missing, f"README sentinels with no renderer: {sorted(missing)}"


def test_empty_state_renders_a_single_line():
    for renderer in build_readme.RENDERERS.values():
        rendered = renderer()
        assert rendered.startswith("\n") and rendered.endswith("\n")
        assert rendered.strip(), "an empty section must still say something"
