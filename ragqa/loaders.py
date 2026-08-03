"""Reading documents off disk.

Plain text and Markdown need no dependency. PDF support is optional so the
package installs and its tests run without pulling a parser in.
"""

from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv"}


class UnsupportedDocument(ValueError):
    """Raised for a file type no installed loader can read."""


def load_text(path: Path | str) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise UnsupportedDocument(
                "Reading PDFs needs pypdf: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    raise UnsupportedDocument(f"No loader for {suffix or 'file with no extension'}")


def load_directory(directory: Path | str, *, pattern: str = "*") -> dict[str, str]:
    """Load every readable file under a directory, keyed by filename.

    Files that no loader handles are skipped rather than failing the batch —
    a mixed folder is the normal case, not an error.
    """
    directory = Path(directory)
    documents: dict[str, str] = {}
    for path in sorted(directory.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            documents[path.name] = load_text(path)
        except UnsupportedDocument:
            continue
    return documents
