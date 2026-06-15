"""
RAG Layer Tests — 8 unit tests.
Run with: pytest tests/test_agents.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rag.bm25_retriever import BM25Retriever, tokenize
from rag.retriever import load_knowledge_base
from rag.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


def test_knowledge_base_loads_documents():
    """Knowledge base loads at least 10 document chunks."""
    docs = load_knowledge_base()
    assert len(docs) >= 10
    assert all("text" in d for d in docs)
    assert all("source" in d for d in docs)


def test_knowledge_base_has_all_sources():
    """All 7 knowledge base files are loaded."""
    docs = load_knowledge_base()
    sources = {d["source"] for d in docs}
    expected = {
        "pyspark_basics", "migration_patterns", "window_functions",
        "anti_patterns", "plsql_patterns", "sp_patterns", "pii_patterns"
    }
    assert expected == sources


def test_tokenize_splits_camel_case():
    """Tokenizer splits camelCase into separate tokens."""
    tokens = tokenize("groupBy withColumn orderBy")
    assert "group" in tokens
    assert "by" in tokens
    assert "with" in tokens
    assert "column" in tokens


def test_bm25_retriever_returns_results():
    """BM25 retriever returns results for a PySpark query."""
    retriever = BM25Retriever()
    results = retriever.retrieve("GROUP BY aggregation SUM COUNT", top_k=3)
    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)


def test_bm25_retriever_empty_query_returns_empty():
    """BM25 retriever returns empty list for empty query."""
    retriever = BM25Retriever()
    results = retriever.retrieve("")
    assert results == []


def test_bm25_retriever_pii_query():
    """BM25 retriever finds PII patterns for PAN number query."""
    retriever = BM25Retriever()
    results = retriever.retrieve("PAN number aadhaar PII masking", top_k=3)
    assert len(results) > 0
    sources = [r["source"] for r in results]
    assert "pii_patterns" in sources


def test_rrf_merges_two_lists():
    """RRF correctly merges two result lists without duplicates."""
    vector = [
        {"text": "pattern A long text here", "source": "basics", "score": 0.9},
        {"text": "pattern B long text here", "source": "basics", "score": 0.8},
    ]
    bm25 = [
        {"text": "pattern B long text here", "source": "basics", "score": 5.0},
        {"text": "pattern C long text here", "source": "basics", "score": 3.0},
    ]
    merged = reciprocal_rank_fusion(vector, bm25)
    texts = [r["text"] for r in merged]
    # pattern B appears in both lists, should rank high
    assert "pattern B long text here" in texts
    # No duplicates
    assert len(texts) == len(set(texts))


def test_hybrid_retriever_window_function_query():
    """Hybrid retriever finds window function patterns."""
    retriever = HybridRetriever()
    results = retriever.retrieve("RANK OVER PARTITION BY window function", top_k=3)
    assert len(results) > 0
    combined_text = " ".join(r["text"] for r in results).lower()
    assert "window" in combined_text or "rank" in combined_text or "partition" in combined_text