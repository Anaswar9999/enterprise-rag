import re
from rank_bm25 import BM25Okapi
from app.retrieval.store import get_collection

_bm25 = None
_chunks = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _build():
    global _bm25, _chunks
    data = get_collection().get(include=["documents", "metadatas"])
    _chunks = [
        {"id": data["ids"][i], "text": data["documents"][i],
         "filename": data["metadatas"][i]["filename"], "page": data["metadatas"][i]["page"]}
        for i in range(len(data["ids"]))
    ]
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])


def bm25_search(query: str, k: int = 20) -> list[dict]:
    if _bm25 is None:
        _build()
    scores = _bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_chunks[i] for i in top if scores[i] > 0]