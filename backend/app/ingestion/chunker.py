from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], chunk_size: int = 60, overlap: int = 10) -> list[dict]:
    """Split pages into chunks, keeping filename/page metadata for citations."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for p in pages:
        for j, piece in enumerate(splitter.split_text(p["text"])):
            non_ascii = sum(1 for ch in piece if ord(ch) > 127)
            if non_ascii / max(len(piece), 1) > 0.3:
                continue  # skip non-English (e.g. Chinese) chunks
            chunks.append({
                "id": f"{p['filename']}-p{p['page']}-c{j}",
                "text": piece,
                "filename": p["filename"],
                "page": p["page"],
            })
    return chunks