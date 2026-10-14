import json
from pathlib import Path

from eval.evaluate import ensure_index, normalize
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.retrieval.store import get_collection

BAD = [
    "What does status code 301 mean in HTTP?",
    "What is the CPU clock speed of the Arduino UNO R3?",
]

ensure_index()
items = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))
data = get_collection().get(include=["documents", "metadatas"])

for it in items:
    if it["q"] not in BAD:
        continue
    kw = normalize(it["kw"])
    print("\nQ:", it["q"], "| keyword:", repr(kw))
    for c in rerank(it["q"], hybrid_search(it["q"]), 5):
        print(f"  top: {c['filename']} p{c['page']}: {normalize(c['text'])[:110]}")
    where = [
        (m["filename"], m["page"])
        for d, m in zip(data["documents"], data["metadatas"])
        if m["filename"] == it["file"] and kw in normalize(d)
    ]
    print("  keyword found in index at:", where[:8] or "NOWHERE")