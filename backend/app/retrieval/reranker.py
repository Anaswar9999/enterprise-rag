from sentence_transformers import CrossEncoder

_model = None


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _model.predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=False)[:top_k]
    return [{**c, "score": float(s)} for c, s in ranked]