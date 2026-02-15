# src/rag/eval_retrieval.py
from __future__ import annotations

from typing import List

from src.rag.retrieve import default_retriever
from pathlib import Path

import chromadb


TEST_QUERIES: List[str] = [
    "what is an etf",
    "difference between etf and mutual fund",
    "what is diversification",
    "what is asset allocation",
    "what is tracking error",
    "what is dollar cost averaging",
    "what is rebalancing",
    "short vs long term capital gains",
    "what is a bid ask spread",
    "market vs limit order difference",
]


def main() -> None:
    retriever = default_retriever()

    # Index health check (helps detect stale or empty indexes)
    repo_root = Path(__file__).resolve().parents[2]
    persist_dir = repo_root / "src" / "data" / "chroma"
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_or_create_collection(name="finance_kb")
        print("INDEX:", persist_dir)
        print("COLLECTION:", "finance_kb")
        print("COUNT:", col.count())
    except Exception as e:
        print("INDEX CHECK FAILED:", repr(e))

    print()  # blank line

    for q in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("-" * 80)

        chunks = retriever.retrieve(q, top_k=5)

        if not chunks:
            print("No results returned.\n")
            continue

        for i, c in enumerate(chunks, start=1):
            print(
                f"{i}. {c.title} | {c.source} | score={c.score}"
            )
        print()


if __name__ == "__main__":
    main()