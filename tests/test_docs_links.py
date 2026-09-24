"""Docs smoke tests: relative markdown links in docs/ must resolve.

Checks the pages reachable from mkdocs nav (plus ``docs/index.md``) for
relative ``.md`` links pointing at files that do not exist. External
http(s) links and anchors are out of scope — no network requests here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

# Inline [text](target) links; images follow the same shape. Skips code
# spans fenced in backticks naively — good enough for this repo's prose.
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _nav_pages() -> list[Path]:
    """Markdown pages listed in mkdocs.yml nav (falls back to index.md)."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a hard dependency
        return [DOCS_DIR / "index.md"]

    try:
        data = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [DOCS_DIR / "index.md"]

    pages: list[Path] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str) and node.endswith(".md"):
            pages.append(DOCS_DIR / node)

    _walk((data or {}).get("nav"))
    if not pages:
        pages.append(DOCS_DIR / "index.md")
    return pages


def _relative_md_links(page: Path) -> list[str]:
    """Relative .md link targets in one page (anchors and URLs stripped)."""
    text = page.read_text(encoding="utf-8")
    targets: list[str] = []
    for raw in _LINK_RE.findall(text):
        if raw.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = raw.split("#", 1)[0]
        if target.endswith(".md"):
            targets.append(target)
    return targets


def _pages_under_test() -> list[Path]:
    pages = {DOCS_DIR / "index.md", *_nav_pages()}
    return sorted(p for p in pages if p.is_file())


def test_nav_pages_exist() -> None:
    """Every md file listed in mkdocs nav exists in docs/."""
    missing = [str(p.relative_to(REPO_ROOT)) for p in _nav_pages() if not p.is_file()]
    assert not missing, f"mkdocs nav references missing pages: {missing}"


@pytest.mark.parametrize("page", _pages_under_test(), ids=lambda p: p.name)
def test_relative_md_links_resolve(page: Path) -> None:
    """Relative .md links in each page point at existing files."""
    missing = [
        target
        for target in _relative_md_links(page)
        if not (page.parent / target).resolve().is_file()
    ]
    assert not missing, f"{page.name} has dead relative links: {missing}"
