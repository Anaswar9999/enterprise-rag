import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
_client = None
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-2.5-flash"
).split(",")]

PROMPT = """Answer the question using ONLY the context below.
Cite sources inline as [filename, page]. If the answer is not in the context, say
"I could not find this in the documents."

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def generate_answer(question: str, chunks: list[dict]) -> dict:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    context = "\n\n".join(f"[{c['filename']}, p{c['page']}]\n{c['text']}" for c in chunks)
    prompt = PROMPT.format(context=context, question=question)
    errors = []
    for model in MODELS:
        try:
            resp = _client.models.generate_content(model=model, contents=prompt)
            u = resp.usage_metadata
            return {
                "answer": resp.text,
                "model": model,
                "input_tokens": (u.prompt_token_count or 0) if u else 0,
                "output_tokens": (u.candidates_token_count or 0) if u else 0,
            }
        except Exception as e:
            errors.append(f"{model}: {str(e)[:200]}")
    raise RuntimeError("All Gemini models failed. " + " | ".join(errors))