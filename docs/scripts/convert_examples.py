"""Render the executable examples in ``examples/`` into Markdown for the docs.

Each ``examples/*.py`` file is a `jupytext <https://jupytext.readthedocs.io>`_
"percent" formatted notebook. At docs-build time this script:

1. converts the script to a notebook with ``jupytext``,
2. executes it and renders it to Markdown with ``nbconvert``, and
3. writes the Markdown (plus any figures it produced) into the virtual docs
   tree at ``docs/_examples/<name>.md`` via :mod:`mkdocs_gen_files`.

It is wired into ``mkdocs.yml`` through the ``gen-files`` plugin, so it runs on
both ``mkdocs serve`` and ``mkdocs build`` — there is no separate CI step to
drift out of sync.
"""

from __future__ import annotations

import os
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import mkdocs_gen_files

# --- Configuration -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Names to skip — e.g. supporting modules imported by the examples rather than
# notebooks in their own right. Empty for now.
EXCLUDE: set[str] = set()

# Image extensions nbconvert may emit alongside the Markdown.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif"}

# Cells tagged with this are executed but their *output* is stripped from the
# rendered Markdown (e.g. ipywidgets that cannot render statically).
REMOVE_OUTPUT_TAG = "remove-output"


def _env_flag(name: str, *, default: bool) -> bool:
    """Read a boolean-ish environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


EXECUTE = _env_flag("DECIJAX_DOCS_EXECUTE", default=True)
STRICT = _env_flag("DECIJAX_DOCS_STRICT", default=True)
# Per-cell execution timeout, in seconds.
CELL_TIMEOUT = int(os.environ.get("DECIJAX_DOCS_TIMEOUT", "300"))
MAX_WORKERS = int(os.environ.get("DECIJAX_DOCS_MAX_WORKERS", "4"))


# --- Conversion --------------------------------------------------------------


@dataclass
class Converted:
    """Result of converting a single example."""

    stem: str
    markdown: str
    images: list[tuple[str, bytes]]  # (filename, bytes)


def _run(cmd: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run a command from the repo root so example imports resolve."""
    return subprocess.run(  # noqa: S603
        cmd,
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def _convert(py_file: Path) -> Converted:
    """Convert (and optionally execute) one example into Markdown + images."""
    stem = py_file.stem
    rel = py_file.relative_to(REPO_ROOT).as_posix()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        if not EXECUTE:
            # Fast path: transcribe source to Markdown without running it.
            result = _run(
                [
                    "jupytext",
                    "--to",
                    "markdown",
                    rel,
                    "--output",
                    str(out_dir / f"{stem}.md"),
                ]
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"jupytext failed to convert {rel}:\n{result.stderr}"
                )
        else:
            # jupytext (.py -> .ipynb on stdout) | nbconvert (execute -> .md).
            to_nb = _run(["jupytext", "--to", "ipynb", rel, "--output", "-"])
            if to_nb.returncode != 0:
                raise RuntimeError(
                    f"jupytext failed to convert {rel}:\n{to_nb.stderr}"
                )

            result = _run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--execute",
                    "--stdin",
                    f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}",
                    # Strip output from cells tagged for removal (they still run).
                    "--TagRemovePreprocessor.enabled=True",
                    "--TagRemovePreprocessor.remove_all_outputs_tags="
                    f'{{"{REMOVE_OUTPUT_TAG}"}}',
                    "--output",
                    stem,
                    "--output-dir",
                    str(out_dir),
                ],
                stdin=to_nb.stdout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"nbconvert failed to execute {rel}:\n{result.stderr}"
                )

        md_path = out_dir / f"{stem}.md"
        if not md_path.exists():
            raise RuntimeError(f"expected Markdown output was not produced for {rel}")
        markdown = md_path.read_text()

        images: list[tuple[str, bytes]] = []
        files_dir = out_dir / f"{stem}_files"
        if files_dir.is_dir():
            images = [
                (img.name, img.read_bytes())
                for img in sorted(files_dir.iterdir())
                if img.suffix.lower() in IMAGE_SUFFIXES
            ]

    return Converted(stem=stem, markdown=markdown, images=images)


def _write(converted: Converted) -> None:
    """Emit the Markdown and images into the virtual ``docs/_examples`` tree."""
    md_path = f"_examples/{converted.stem}.md"
    with mkdocs_gen_files.open(md_path, "w") as f:
        f.write(converted.markdown)

    for name, data in converted.images:
        img_path = f"_examples/{converted.stem}_files/{name}"
        with mkdocs_gen_files.open(img_path, "wb") as f:
            f.write(data)


def main() -> None:
    examples = sorted(
        f for f in EXAMPLES_DIR.glob("*.py") if f.name not in EXCLUDE
    )
    if not examples:
        print("convert_examples: no examples found under examples/")
        return

    mode = "executing" if EXECUTE else "transcribing (no execution)"
    print(f"convert_examples: {mode} {len(examples)} example(s)")

    # Convert in parallel (subprocess-bound); write results on the main thread
    # since mkdocs_gen_files is not designed for concurrent writes.
    failures: list[tuple[str, Exception]] = []
    results: list[Converted] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_convert, f): f for f in examples}
        for future in as_completed(futures):
            py_file = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - reported below
                failures.append((py_file.name, exc))

    for converted in results:
        _write(converted)
        print(f"convert_examples: ✓ {converted.stem}.md")

    for name, exc in failures:
        print(f"convert_examples: ✗ {name}\n{exc}")

    if failures and STRICT:
        raise SystemExit(
            f"convert_examples: {len(failures)} example(s) failed to convert "
            "(set DECIJAX_DOCS_STRICT=0 to downgrade to a warning)."
        )


main()
