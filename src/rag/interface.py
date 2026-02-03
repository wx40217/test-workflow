"""
RAG (Retrieval-Augmented Generation) interface for the test case generator.

This module provides a pluggable interface for RAG functionality.
The actual implementation can use various vector stores (Chroma, Pinecone, etc.)
and embedding models.

Currently provides a skeleton/interface for future implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RAGConfig:
    """
    Configuration for RAG functionality.
    
    Attributes:
        enabled: Whether RAG is enabled
        collection_name: Name of the vector store collection
        embedding_model: Model to use for embeddings
        embedding_api_key: API key for embedding model
        embedding_base_url: Base URL for embedding API
        vector_store_type: Type of vector store (chroma, pinecone, faiss, etc.)
        vector_store_config: Additional vector store configuration
        top_k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score for retrieval
    """
    enabled: bool = False
    collection_name: str = "test_case_knowledge"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    vector_store_type: str = "chroma"
    vector_store_config: dict = field(default_factory=dict)
    top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class RetrievedDocument:
    """
    A document retrieved from the vector store.
    
    Attributes:
        content: The document content
        metadata: Document metadata
        score: Similarity score
        source: Source identifier
    """
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    source: str = ""
    
    def __str__(self) -> str:
        """String representation of the document."""
        source_info = f" (Source: {self.source})" if self.source else ""
        return f"{self.content}{source_info}"


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    Implement this class to add support for different vector stores.
    """
    
    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document contents
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of retrieved documents
        """
        pass
    
    @abstractmethod
    def delete(self, ids: list[str]) -> bool:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all documents from the store.
        
        Returns:
            True if successful
        """
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store implementation.
    
    Note: This is a placeholder implementation. Install chromadb
    and implement the actual functionality when needed.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._client = None
        self._collection = None
        self._embeddings = None
    
    def _initialize(self):
        """Initialize the Chroma client and collection."""
        try:
            import chromadb
            from langchain_openai import OpenAIEmbeddings
            
            # Initialize embeddings
            self._embeddings = OpenAIEmbeddings(
                api_key=self.config.embedding_api_key,
                base_url=self.config.embedding_base_url,
                model=self.config.embedding_model
            )
            
            # Initialize Chroma client
            persist_dir = self.config.vector_store_config.get('persist_directory')
            if persist_dir:
                self._client = chromadb.PersistentClient(path=persist_dir)
            else:
                self._client = chromadb.Client()
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "Test case knowledge base"}
            )
            
        except ImportError:
            raise ImportError(
                "chromadb is required for Chroma vector store. "
                "Install it with: pip install chromadb"
            )
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> list[str]:
        if self._collection is None:
            self._initialize()
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings
        embeddings = self._embeddings.embed_documents(documents)
        
        # Add to collection
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        if self._collection is None:
            self._initialize()
        
        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Convert to RetrievedDocument objects
        documents = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
            
            documents.append(RetrievedDocument(
                content=doc,
                metadata=metadata,
                score=score,
                source=metadata.get('source', '')
            ))
        
        return documents
    
    def delete(self, ids: list[str]) -> bool:
        if self._collection is None:
            self._initialize()
        
        self._collection.delete(ids=ids)
        return True
    
    def clear(self) -> bool:
        if self._client is None:
            self._initialize()
        
        self._client.delete_collection(self.config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name
        )
        return True


class InMemoryVectorStore(BaseVectorStore):
    """
    Simple in-memory vector store for testing/development.
    
    Uses basic cosine similarity without external dependencies.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._documents: dict[str, dict] = {}
        self._embeddings = None
    
    def _initialize(self):
        """Initialize embeddings model."""
        try:
            from langchain_openai import OpenAIEmbeddings
            
            self._embeddings = OpenAIEmbeddings(
                api_key=self.config.embedding_api_key,
                base_url=self.config.embedding_base_url,
                model=self.config.embedding_model
            )
        except ImportError:
            # Fall back to no embeddings (text matching only)
            self._embeddings = None
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> list[str]:
        if self._embeddings is None:
            self._initialize()
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings if available
        embeddings = None
        if self._embeddings:
            embeddings = self._embeddings.embed_documents(documents)
        
        # Store documents
        for i, doc in enumerate(documents):
            self._documents[ids[i]] = {
                'content': doc,
                'metadata': metadatas[i] if metadatas else {},
                'embedding': embeddings[i] if embeddings else None
            }
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        if not self._documents:
            return []
        
        if self._embeddings is None:
            self._initialize()
        
        results = []
        
        if self._embeddings:
            # Vector similarity search
            query_embedding = self._embeddings.embed_query(query)
            
            for doc_id, doc_data in self._documents.items():
                # Apply metadata filter
                if filter_dict:
                    match = all(
                        doc_data['metadata'].get(k) == v
                        for k, v in filter_dict.items()
                    )
                    if not match:
                        continue
                
                if doc_data['embedding']:
                    score = self._cosine_similarity(
                        query_embedding,
                        doc_data['embedding']
                    )
                else:
                    score = 0.0
                
                results.append(RetrievedDocument(
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    score=score,
                    source=doc_data['metadata'].get('source', doc_id)
                ))
        else:
            # Simple text matching fallback
            query_lower = query.lower()
            for doc_id, doc_data in self._documents.items():
                if filter_dict:
                    match = all(
                        doc_data['metadata'].get(k) == v
                        for k, v in filter_dict.items()
                    )
                    if not match:
                        continue
                
                # Simple keyword matching score
                content_lower = doc_data['content'].lower()
                words = query_lower.split()
                matched = sum(1 for w in words if w in content_lower)
                score = matched / len(words) if words else 0.0
                
                results.append(RetrievedDocument(
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    score=score,
                    source=doc_data['metadata'].get('source', doc_id)
                ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Filter by threshold
        results = [r for r in results if r.score >= self.config.similarity_threshold]
        
        return results[:top_k]
    
    def delete(self, ids: list[str]) -> bool:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        return True
    
    def clear(self) -> bool:
        self._documents.clear()
        return True


class RAGInterface:
    """
    Main RAG interface for the test case generator.
    
    Provides a unified interface for:
    - Adding documents to the knowledge base
    - Retrieving relevant documents
    - Managing the vector store
    
    Usage:
        # Initialize with config
        config = RAGConfig(enabled=True, embedding_api_key="sk-...")
        rag = RAGInterface(config)
        
        # Add documents
        rag.add_documents([
            "Login test case requirements...",
            "Payment flow test guidelines..."
        ])
        
        # Retrieve relevant documents
        docs = rag.retrieve("user authentication tests")
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG interface.
        
        Args:
            config: RAG configuration, uses defaults if not provided
        """
        self.config = config or RAGConfig()
        self._vector_store: Optional[BaseVectorStore] = None
    
    def _get_vector_store(self) -> BaseVectorStore:
        """Get or create the vector store instance."""
        if self._vector_store is None:
            store_type = self.config.vector_store_type.lower()
            
            if store_type == "chroma":
                self._vector_store = ChromaVectorStore(self.config)
            elif store_type == "memory":
                self._vector_store = InMemoryVectorStore(self.config)
            else:
                # Default to in-memory
                self._vector_store = InMemoryVectorStore(self.config)
        
        return self._vector_store
    
    def is_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self.config.enabled
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        source: Optional[str] = None
    ) -> list[str]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document contents
            metadatas: Optional metadata for each document
            source: Optional source identifier to add to all documents
            
        Returns:
            List of document IDs
        """
        if not self.is_enabled():
            return []
        
        # Add source to metadata if provided
        if source and metadatas is None:
            metadatas = [{"source": source} for _ in documents]
        elif source and metadatas:
            for m in metadatas:
                m["source"] = source
        
        store = self._get_vector_store()
        return store.add_documents(documents, metadatas)
    
    def add_from_file(self, file_path: str) -> list[str]:
        """
        Add documents from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs
        """
        if not self.is_enabled():
            return []
        
        from src.input_handler.handlers import InputHandler
        
        handler = InputHandler()
        processed = handler.process_file(file_path)
        
        # Split content into chunks (simple approach)
        content = processed.text_content
        chunks = self._chunk_text(content)
        
        metadatas = [{"source": file_path, "chunk": i} for i in range(len(chunks))]
        
        return self.add_documents(chunks, metadatas)
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> list[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for sep in ['. ', '.\n', '\n\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[dict] = None
    ) -> list[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (uses config default if not specified)
            filter_dict: Optional metadata filter
            
        Returns:
            List of document contents
        """
        if not self.is_enabled():
            return []
        
        store = self._get_vector_store()
        docs = store.search(
            query,
            top_k=top_k or self.config.top_k,
            filter_dict=filter_dict
        )
        
        return [doc.content for doc in docs]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Optional metadata filter
            
        Returns:
            List of RetrievedDocument objects
        """
        if not self.is_enabled():
            return []
        
        store = self._get_vector_store()
        return store.search(
            query,
            top_k=top_k or self.config.top_k,
            filter_dict=filter_dict
        )
    
    def delete_documents(self, ids: list[str]) -> bool:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs
            
        Returns:
            True if successful
        """
        if not self.is_enabled():
            return False
        
        store = self._get_vector_store()
        return store.delete(ids)
    
    def clear(self) -> bool:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            True if successful
        """
        if not self.is_enabled():
            return False
        
        store = self._get_vector_store()
        return store.clear()
    
    def get_stats(self) -> dict:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with stats
        """
        if not self.is_enabled():
            return {"enabled": False}
        
        store = self._get_vector_store()
        
        # Get document count (implementation depends on store)
        if isinstance(store, InMemoryVectorStore):
            doc_count = len(store._documents)
        else:
            doc_count = -1  # Unknown
        
        return {
            "enabled": True,
            "vector_store_type": self.config.vector_store_type,
            "collection_name": self.config.collection_name,
            "document_count": doc_count,
            "embedding_model": self.config.embedding_model
        }
