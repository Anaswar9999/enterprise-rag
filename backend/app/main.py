import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_pages
from app.retrieval import store, bm25
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.llm import generate_answer
from app.observability.metrics import log_request, summary, estimate_cost

app = FastAPI(title="Enterprise RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Welcome to Enterprise RAG!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest():
    chunks = chunk_pages(load_documents("data/docs"))
    count = store.index_chunks(chunks)
    bm25._bm25 = None  # force BM25 rebuild
    return {"chunks_indexed": count}


@app.post("/query")
def query(q: Query):
    try:
        t0 = time.perf_counter()
        candidates = hybrid_search(q.question)
        t1 = time.perf_counter()
        top = rerank(q.question, candidates, top_k=5)
        t2 = time.perf_counter()
        gen = generate_answer(q.question, top)
        t3 = time.perf_counter()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    rec = {
        "ts": time.time(),
        "question": q.question,
        "retrieval_ms": round((t1 - t0) * 1000),
        "rerank_ms": round((t2 - t1) * 1000),
        "generation_ms": round((t3 - t2) * 1000),
        "total_ms": round((t3 - t0) * 1000),
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "est_cost_usd": estimate_cost(gen["input_tokens"], gen["output_tokens"]),
        "model": gen["model"],
        "sources": [f"{c['filename']} p{c['page']}" for c in top],
    }
    log_request(rec)
    return {
        "answer": gen["answer"],
        "model": gen["model"],
        "sources": [
            {"filename": c["filename"], "page": c["page"],
             "score": c["score"], "text": c["text"]}
            for c in top
        ],
        "timings_ms": {k: rec[k] for k in ("retrieval_ms", "rerank_ms", "generation_ms", "total_ms")},
        "tokens": {"input": rec["input_tokens"], "output": rec["output_tokens"]},
        "est_cost_usd": rec["est_cost_usd"],
    }


@app.get("/metrics")
def metrics():
    return summary()
