 # Enterprise RAG with CI/CD Observability

Production-style Retrieval-Augmented Generation (Ask My Docs) with hybrid retrieval,
cross-encoder reranking, cited answers, latency/cost monitoring and a CI evaluation gate.

## Architecture
Ingestion -> Hybrid search (BM25 + dense, RRF) -> Cross-encoder rerank -> Gemini answer with citations -> CI eval gate

## Run locally
    cd backend
    python -m venv .venv && .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    # create backend/.env with GEMINI_API_KEY=your-key
    uvicorn app.main:app --reload
    # open /docs and run POST /ingest once, then in a second terminal:
    streamlit run ..\frontend\app.py

## Evaluation
    cd backend
    python -m eval.evaluate
    python -m pytest -q

Team: Anaswar B R (241150036), Adithyan M S (241150051)
Guide: Dr. Anupama JIMS