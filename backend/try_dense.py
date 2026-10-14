from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_pages
from app.retrieval.store import index_chunks, dense_search

chunks = chunk_pages(load_documents("data/docs"))
print("Indexed:", index_chunks(chunks))

for q in ["maximum operating voltage of ATmega328P", "LM317 output voltage formula"]:
    print("\nQUERY:", q)
    for r in dense_search(q, k=3):
        print(f"  [{r['filename']} p{r['page']}] {r['text'][:100]!r}")
        