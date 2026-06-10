from __future__ import annotations

import asyncio
import io
import re
from typing import Final

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


class PDFParseError(ValueError):
    pass


_PDF_MAGIC: Final[bytes] = b"%PDF-"
_EXCESS_NEWLINES: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
_TRAILING_SPACES: Final[re.Pattern[str]] = re.compile(r"[ \t]+\n")
_INNER_SPACES: Final[re.Pattern[str]] = re.compile(r"[ \t]{2,}")


def _extract_sync(file_bytes: bytes) -> str:
    if not file_bytes:
        raise PDFParseError("Uploaded PDF is empty.")

    if not file_bytes.lstrip().startswith(_PDF_MAGIC):
        raise PDFParseError("File does not appear to be a valid PDF document.")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as exc:
        raise PDFParseError(f"Failed to read PDF: {exc}") from exc

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages_text.append(text)

    raw = "\n\n".join(pages_text)

    cleaned = _TRAILING_SPACES.sub("\n", raw)
    cleaned = _INNER_SPACES.sub(" ", cleaned)
    cleaned = _EXCESS_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


async def extract_text_from_pdf(file_bytes: bytes) -> str:
    return await asyncio.to_thread(_extract_sync, file_bytes)
