import json
import os
from pathlib import Path

LOG = Path("logs/requests.jsonl")
# Estimated paid-tier price per 1M tokens (free tier actually costs $0)
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0.30"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "2.50"))


def estimate_cost(in_tok: int, out_tok: int) -> float:
    return (in_tok * PRICE_IN + out_tok * PRICE_OUT) / 1_000_000


def log_request(rec: dict) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def summary() -> dict:
    if not LOG.exists():
        return {"requests": 0}
    recs = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return {"requests": 0}
    lat = [r["total_ms"] for r in recs]
    return {
        "requests": len(recs),
        "p50_ms": round(_pct(lat, 0.5)),
        "p95_ms": round(_pct(lat, 0.95)),
        "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in recs) / len(recs)),
        "avg_rerank_ms": round(sum(r["rerank_ms"] for r in recs) / len(recs)),
        "avg_generation_ms": round(sum(r["generation_ms"] for r in recs) / len(recs)),
        "total_input_tokens": sum(r["input_tokens"] for r in recs),
        "total_output_tokens": sum(r["output_tokens"] for r in recs),
        "total_est_cost_usd": round(sum(r["est_cost_usd"] for r in recs), 6),
    }