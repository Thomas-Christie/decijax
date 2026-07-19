"""Generate the API reference pages from the ``decijax`` package source.

At docs-build time (via the ``gen-files`` plugin) this script walks the
installed package tree under ``src/decijax`` and, for every importable module,
emits a *virtual* Markdown page containing a single ``mkdocstrings`` directive
(``::: decijax.<module>``). ``mkdocstrings`` then renders that module's
signatures and docstrings in place.

It also writes ``api/SUMMARY.md``, which the ``literate-nav`` plugin turns into
the nested "Reference" navigation.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

# --- Configuration -----------------------------------------------------------

PACKAGE = "decijax"
# Source root for the src-layout package (repo_root/src/decijax/...).
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC_ROOT / PACKAGE

# Casing overrides for nav titles that ``str.title()`` gets wrong. Keyed by the
# raw path part (module or package directory name).
TITLE_OVERRIDES = {
    "gps": "GPs",
    "decijax": "decijax",
}

# Module basenames to skip entirely (private/dunder handled separately).
EXCLUDE_STEMS = {"__main__", "__about__"}


def _title(part: str) -> str:
    """Human-readable nav title for a path part."""
    if part in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[part]
    return part.replace("_", " ").title()


nav = mkdocs_gen_files.Nav()

for path in sorted(PACKAGE_ROOT.rglob("*.py")):
    module_path = path.relative_to(SRC_ROOT).with_suffix("")
    doc_path = path.relative_to(SRC_ROOT).with_suffix(".md")
    parts = tuple(module_path.parts)

    is_package_index = parts[-1] == "__init__"
    if is_package_index:
        # Package landing page: a docstring-only overview.
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        if not parts:
            continue  # top-level src has no __init__ to document
    elif parts[-1] in EXCLUDE_STEMS or parts[-1].startswith("_"):
        continue

    full_doc_path = Path("api", doc_path)
    nav_key = tuple(_title(p) for p in parts[1:]) or (_title(parts[0]),)
    nav[nav_key] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"::: {identifier}", file=fd)
        if is_package_index:
            print("    options:", file=fd)
            print("      members: false", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(SRC_ROOT.parent))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
