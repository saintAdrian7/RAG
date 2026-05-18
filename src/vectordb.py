import os
import chromadb
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        #
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""])
        chunks = text_splitter.split_text(text)
        
        return chunks
        
    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
    
        print(f"Processing {len(documents)} documents...")
        existing = self.collection.count()
        for doc_idx, doc in enumerate(documents):
            metadata = doc.metadata or {}
            content = doc.page_content
            chunks = self.chunk_text(content)
            if not chunks:
                print("No chunks returned")
                continue

            source = metadata.get("source", f"doc_{existing + doc_idx}")
            embeddings = self.embedding_model.encode(chunks).tolist()
            ids = [f"{source}_{i}" for i in range(len(chunks))]
            metadatas = [metadata for _ in chunks]

            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            print(f"Added {len(chunks)} chunks from {source}")
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_vector = self.embedding_model.encode([query]).tolist()
        raw = self.collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma nests results as [[...]] (outer list = one entry per submitted
        # query). Unwrap the single-query case so callers get flat lists.
        return {
            "documents": raw.get("documents", [[]])[0],
            "metadatas": raw.get("metadatas", [[]])[0],
            "distances": raw.get("distances", [[]])[0],
            "ids": raw.get("ids", [[]])[0],
        }
