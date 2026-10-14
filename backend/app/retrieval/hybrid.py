from app.retrieval.store import dense_search
from app.retrieval.bm25 import bm25_search


def rrf(result_lists: list[list[dict]], k: int = 60, top_n: int = 20) -> list[dict]:
    scores, items = {}, {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (k + rank)
            items[r["id"]] = r
    best = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [items[i] for i in best]


def hybrid_search(query: str, top_n: int = 20) -> list[dict]:
    return rrf([dense_search(query, 20), bm25_search(query, 20)], top_n=top_n)