"""
BM25 Keyword Retriever — exact keyword matching over PySpark knowledge base.
Complements vector search for technical terms like function names and patterns.
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi

from rag.retriever import load_knowledge_base


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    Lowercases, splits on non-alphanumeric, filters short tokens.
    Preserves PySpark-specific tokens like 'groupBy', 'withColumn'.
    """
    # Split camelCase — groupBy → group by
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """
    BM25-based keyword retriever for PySpark migration patterns.
    Loads knowledge base into memory on initialization.
    Fast and deterministic — no GPU needed.
    """

    def __init__(self):
        self.documents = load_knowledge_base()
        self._build_index()

    def _build_index(self) -> None:
        """Build BM25 index from loaded documents."""
        if not self.documents:
            self.bm25 = None
            return

        tokenized = [tokenize(d["text"]) for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve top-k most keyword-relevant chunks for a query.

        Args:
            query: Search query (SQL, PySpark, or natural language)
            top_k: Number of results to return

        Returns:
            List of dicts with 'text', 'source', 'score' fields
        """
        if not query or not query.strip() or self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        output = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive score
                output.append({
                    "text": self.documents[idx]["text"],
                    "source": self.documents[idx]["source"],
                    "score": round(float(scores[idx]), 4),
                })

        return output