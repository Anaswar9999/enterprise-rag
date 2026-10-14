import chromadb
from sentence_transformers import SentenceTransformer

_model = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path="chroma_db")
    return _client.get_or_create_collection(
        name="docs", metadata={"hnsw:space": "cosine"}
    )


def index_chunks(chunks: list[dict], batch_size: int = 64) -> int:
    """Embed chunks and store them in ChromaDB (replaces any old index)."""
    global _client
    get_collection()
    _client.delete_collection("docs")
    col = get_collection()
    model = get_model()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = model.encode([c["text"] for c in batch]).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings,
            metadatas=[{"filename": c["filename"], "page": c["page"]} for c in batch],
        )
    return col.count()


def dense_search(query: str, k: int = 20) -> list[dict]:
    col = get_collection()
    emb = get_model().encode([query]).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    return [
        {
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "filename": res["metadatas"][0][i]["filename"],
            "page": res["metadatas"][0][i]["page"],
        }
        for i in range(len(res["ids"][0]))
    ]
