from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_pages

pages = load_documents("data/docs")
chunks = chunk_pages(pages)
print(f"{len(pages)} pages -> {len(chunks)} chunks")
print(chunks[0])