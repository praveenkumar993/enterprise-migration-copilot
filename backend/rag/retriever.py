"""
ChromaDB Vector Retriever — semantic similarity search over PySpark knowledge base.
Uses sentence-transformers for embeddings.
"""

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Knowledge base directory
KB_DIR = Path(__file__).parent / "knowledge_base"

# ChromaDB persistent store location
CHROMA_DIR = Path(__file__).parent / "chroma_store"

# Embedding model — lightweight, runs on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Collection name
COLLECTION_NAME = "pyspark_knowledge"


def load_knowledge_base() -> list[dict[str, str]]:
    """
    Load all .txt files from the knowledge base directory.
    Each file is split into chunks by double newline.

    Returns:
        List of dicts with 'text', 'source', 'chunk_id' fields.
    """
    documents = []
    chunk_id = 0

    for txt_file in sorted(KB_DIR.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8")
        source = txt_file.stem

        # Split by PATTERN: or double newline — keeps related content together
        raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

        for chunk in raw_chunks:
            if len(chunk) > 30:  # Skip very short chunks
                documents.append({
                    "text": chunk,
                    "source": source,
                    "chunk_id": str(chunk_id),
                })
                chunk_id += 1

    return documents


class VectorRetriever:
    """
    ChromaDB-based semantic retriever for PySpark migration patterns.
    Loads knowledge base on first call and persists to disk.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create or load the ChromaDB collection."""
        existing = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            return

        # Create and populate collection
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_knowledge_base()

    def _index_knowledge_base(self) -> None:
        """Index all knowledge base documents into ChromaDB."""
        documents = load_knowledge_base()
        if not documents:
            return

        texts = [d["text"] for d in documents]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids=[d["chunk_id"] for d in documents],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": d["source"]} for d in documents],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve top-k most semantically similar chunks for a query.

        Args:
            query: Natural language or code query
            top_k: Number of results to return

        Returns:
            List of dicts with 'text', 'source', 'score' fields
        """
        if not query or not query.strip():
            return []

        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
        )

        output = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                output.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(1 - dist, 4),  # Convert distance to similarity
                })

        return output

    def rebuild(self) -> int:
        """Force rebuild the ChromaDB index from knowledge base files."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_knowledge_base()
        return self.collection.count()