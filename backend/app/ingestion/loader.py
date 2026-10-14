from pathlib import Path
from pypdf import PdfReader


def load_documents(folder: str) -> list[dict]:
    """Read PDFs and TXT files into page-level records with metadata."""
    pages = []
    for path in sorted(Path(folder).iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"text": text, "filename": path.name, "page": i})
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append({"text": text, "filename": path.name, "page": 1})
    return pages