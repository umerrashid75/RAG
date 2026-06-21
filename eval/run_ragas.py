"""
RAGAS evaluation harness (§9.2, §10).
Run on demand or in CI nightly — NOT on every commit (cost + nondeterminism).

Usage:
    python eval/run_ragas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"

TARGETS = {
    "faithfulness": 0.95,
    "answer_relevancy": 0.90,
    "context_recall": 0.92,
    "context_precision": 0.88,
}


def load_golden_set() -> list[dict]:
    with GOLDEN_SET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval() -> None:
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("Install ragas and datasets: pip install ragas datasets")
        sys.exit(1)

    from app.config import get_settings
    from app.factory import build_llm, build_retriever

    cfg = get_settings()
    retriever = build_retriever(cfg)
    llm = build_llm(cfg)

    golden = load_golden_set()
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden:
        query = item["query"]
        docs = retriever.retrieve(query, top_k=10)
        context_texts = [d.chunk.content for d in docs]
        context_combined = "\n---\n".join(context_texts)

        prompt = (
            "Answer the following research question using ONLY the context below. "
            "Cite each claim by its source id.\n\n"
            f"Question: {query}\n\nContext:\n{context_combined}\n\nAnswer:"
        )
        answer = str(llm.generate(prompt))

        rows["question"].append(query)
        rows["answer"].append(answer)
        rows["contexts"].append(context_texts)
        rows["ground_truth"].append(item["ground_truth_answer"])

    dataset = Dataset.from_dict(rows)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    print("\n=== RAGAS Evaluation Results ===")
    failed = []
    for metric, target in TARGETS.items():
        score = results[metric]
        status = "PASS" if score >= target else "FAIL"
        if score < target:
            failed.append(metric)
        print(f"  {metric:25s}: {score:.4f}  (target >= {target})  [{status}]")

    if failed:
        print(f"\nFailed metrics: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll metrics passed.")


if __name__ == "__main__":
    run_eval()
