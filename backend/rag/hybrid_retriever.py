"""
Hybrid Retriever — combines ChromaDB vector search with BM25 keyword search.
Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.
Vector search finds semantically similar patterns.
BM25 finds exact keyword matches (function names, SQL keywords).
Together they cover both cases reliably.
"""

from typing import Any

from rag.retriever import VectorRetriever
from rag.bm25_retriever import BM25Retriever


# RRF constant — controls how much top ranks are boosted
RRF_K = 60

# Default weight split between vector and BM25
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum of (weight / (k + rank)) across both lists.
    Higher RRF score = better combined result.

    Args:
        vector_results: Results from ChromaDB vector search
        bm25_results: Results from BM25 keyword search
        vector_weight: Weight for vector search contribution
        bm25_weight: Weight for BM25 contribution

    Returns:
        Merged and re-ranked list of results
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    sources: dict[str, str] = {}

    # Score vector results
    for rank, result in enumerate(vector_results):
        key = result["text"][:100]  # Use text prefix as dedup key
        rrf_score = vector_weight / (RRF_K + rank + 1)
        scores[key] = scores.get(key, 0) + rrf_score
        texts[key] = result["text"]
        sources[key] = result["source"]

    # Score BM25 results
    for rank, result in enumerate(bm25_results):
        key = result["text"][:100]
        rrf_score = bm25_weight / (RRF_K + rank + 1)
        scores[key] = scores.get(key, 0) + rrf_score
        texts[key] = result["text"]
        sources[key] = result["source"]

    # Sort by combined RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    return [
        {
            "text": texts[k],
            "source": sources[k],
            "score": round(scores[k], 6),
        }
        for k in sorted_keys
    ]


class HybridRetriever:
    """
    Production-grade hybrid retriever combining vector and keyword search.
    Used by the Analyzer agent to fetch relevant PySpark migration context.
    """

    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks using hybrid search.

        Args:
            query: Search query — can be SQL, IR dict fields, or natural language
            top_k: Number of final results to return
            vector_weight: Weight for semantic search (default 0.6)
            bm25_weight: Weight for keyword search (default 0.4)

        Returns:
            List of dicts with 'text', 'source', 'score' sorted by relevance
        """
        if not query or not query.strip():
            return []

        # Fetch more candidates than needed from each retriever
        fetch_k = top_k * 3

        vector_results = self.vector.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)

        merged = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        return merged[:top_k]

    def retrieve_for_ir(self, ir: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Build a query from IR fields and retrieve relevant context.
        Used by the Analyzer agent directly.

        Args:
            ir: Unified IR dict from any parser
            top_k: Number of results to return

        Returns:
            List of relevant PySpark migration patterns
        """
        # Build a rich query from IR fields
        parts = []

        lang = ir.get("source_language", "sql")
        parts.append(f"migrate {lang} to pyspark")

        joins = ir.get("joins", [])
        if joins:
            join_types = list({j.get("type", "JOIN") for j in joins})
            parts.append(f"{' '.join(join_types)} join")

        aggs = ir.get("aggregations", [])
        if aggs:
            parts.append(f"aggregations {' '.join(aggs)}")

        windows = ir.get("window_functions", [])
        if windows:
            parts.append(f"window functions {' '.join(windows)}")

        ctes = ir.get("ctes", [])
        if ctes:
            parts.append("CTE with clause pyspark")

        flags = ir.get("procedural_flags", [])
        if flags:
            parts.append("procedural cursor loop exception handler")

        complexity = ir.get("complexity_label", "simple")
        parts.append(f"complexity {complexity}")

        query = " ".join(parts)
        return self.retrieve(query, top_k=top_k)