from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split pages into chunks, keeping filename/page metadata for citations."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for p in pages:
        for j, piece in enumerate(splitter.split_text(p["text"])):
            chunks.append({
                "id": f"{p['filename']}-p{p['page']}-c{j}",
                "text": piece,
                "filename": p["filename"],
                "page": p["page"],
            })
    return chunks