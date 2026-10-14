from app.retrieval.store import dense_search
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

def show(title, results):
    print(f"  {title}")
    for r in results[:3]:
        print(f"    [{r['filename']} p{r['page']}] {r['text'][:70]!r}")

for q in ["ATmega328P", "GPIO25 pin function", "HTTP 404 status code meaning"]:
    print("\nQUERY:", q)
    show("Dense only", dense_search(q, 3))
    show("BM25 only", bm25_search(q, 3))
    show("Hybrid + rerank", rerank(q, hybrid_search(q), 3))
    