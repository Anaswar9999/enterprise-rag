from eval.evaluate import run

THRESHOLD = 0.80


def test_retrieval_hit_rate():
    rate, results = run()
    failed = [r["q"] for r in results if not r["hit"]]
    assert rate >= THRESHOLD, f"Hit rate {rate:.0%} below {THRESHOLD:.0%}. Failed: {failed}"