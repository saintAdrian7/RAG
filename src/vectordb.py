"""
ChromaDB wrapper that supports many corpora. Each corpus is a separate
ChromaDB collection identified by a caller-supplied ``corpus_id`` string,
so the same VectorDB instance can serve multiple tenants/users without
state confusion.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


DEFAULT_CORPUS = "default"


class VectorDB:
    """
    Multi-corpus vector store backed by ChromaDB + Sentence-Transformers.

    Each ``corpus_id`` maps 1:1 to a ChromaDB collection. The single
    embedding model is shared across all corpora (changing models would
    invalidate every collection's vectors).
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db_path = db_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.client = chromadb.PersistentClient(path=self.db_path)

        # Cache collection handles so we don't hit Chroma's metadata layer
        # on every request.
        self._collections: Dict[str, Any] = {}

        print(f"VectorDB initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _get_collection(self, corpus_id: str):
        """Get or create the ChromaDB collection for a corpus_id."""
        if corpus_id not in self._collections:
            self._collections[corpus_id] = self.client.get_or_create_collection(
                name=corpus_id,
                metadata={"description": f"RAG corpus: {corpus_id}"},
            )
        return self._collections[corpus_id]

    def list_corpora(self) -> List[str]:
        """Return the names of every corpus that currently exists."""
        return [c.name for c in self.client.list_collections()]

    def count(self, corpus_id: str) -> int:
        """Number of chunks (not documents) currently stored in a corpus."""
        return self._get_collection(corpus_id).count()

    def list_sources(self, corpus_id: str) -> List[str]:
        """Distinct ``metadata['source']`` values currently in a corpus."""
        collection = self._get_collection(corpus_id)
        raw = collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", []) or []
        return sorted({
            m["source"]
            for m in metadatas
            if m and m.get("source")
        })

    def delete_corpus(self, corpus_id: str) -> None:
        """Remove an entire corpus (collection) from the store."""
        try:
            self.client.delete_collection(name=corpus_id)
        finally:
            self._collections.pop(corpus_id, None)

    def delete_document(self, corpus_id: str, source: str) -> int:
        """
        Remove every chunk whose ``metadata['source']`` matches the given
        value. Returns the number of chunks removed.
        """
        collection = self._get_collection(corpus_id)
        # Chroma supports metadata filters on delete.
        existing = collection.get(where={"source": source})
        ids = existing.get("ids", []) or []
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks suitable for embedding."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        corpus_id: str = DEFAULT_CORPUS,
    ) -> Dict[str, int]:
        """
        Chunk, embed, and index a list of LangChain Document objects into
        the given corpus. Returns counts of documents and chunks added.
        """
        collection = self._get_collection(corpus_id)
        existing = collection.count()

        print(f"[{corpus_id}] Processing {len(documents)} document(s)...")
        total_chunks = 0
        docs_processed = 0

        for doc_idx, doc in enumerate(documents):
            metadata = doc.metadata or {}
            content = doc.page_content
            chunks = self.chunk_text(content)
            if not chunks:
                continue

            source = metadata.get("source", f"doc_{existing + doc_idx}")
            embeddings = self.embedding_model.encode(chunks).tolist()
            ids = [f"{source}_{i}" for i in range(len(chunks))]
            metadatas = [{**metadata, "source": source} for _ in chunks]

            collection.add(
                embeddings=embeddings,
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            total_chunks += len(chunks)
            docs_processed += 1
            print(f"  [{corpus_id}] Added {len(chunks)} chunks from {source}")

        return {"documents_processed": docs_processed, "chunks_added": total_chunks}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        corpus_id: str = DEFAULT_CORPUS,
        n_results: int = 5,
    ) -> Dict[str, List[Any]]:
        """
        Semantic search within a corpus. Returns flat lists indexed by
        rank (closest first).
        """
        collection = self._get_collection(corpus_id)
        query_vector = self.embedding_model.encode([query]).tolist()
        raw = collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma nests results as [[...]] for multi-query. Unwrap.
        return {
            "documents": raw.get("documents", [[]])[0],
            "metadatas": raw.get("metadatas", [[]])[0],
            "distances": raw.get("distances", [[]])[0],
            "ids": raw.get("ids", [[]])[0],
        }
