import json
from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_pages
from app.retrieval import store, bm25
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank


def ensure_index():
    if store.get_collection().count() == 0:
        store.index_chunks(chunk_pages(load_documents("data/docs")))
        bm25._bm25 = None


def normalize(text: str) -> str:
    """Collapse line breaks and repeated spaces so phrases split across lines still match."""
    return " ".join(text.split()).lower()


def run(top_k: int = 5) -> tuple[float, list[dict]]:
    ensure_index()
    items = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))
    results = []
    for it in items:
        top = rerank(it["q"], hybrid_search(it["q"]), top_k=top_k)
        kw = normalize(it["kw"])
        hit = any(c["filename"] == it["file"] and kw in normalize(c["text"]) for c in top)
        results.append({"q": it["q"], "hit": hit})
    return sum(r["hit"] for r in results) / len(results), results


if __name__ == "__main__":
    rate, res = run()
    for r in res:
        print("PASS" if r["hit"] else "FAIL", "-", r["q"])
    print(f"\nHit rate @5: {rate:.0%}")