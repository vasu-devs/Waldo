"""
Vector Store service using Qdrant.

Stores document vectors with Shadow Text metadata using real embeddings.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Metadata for a document element stored in Qdrant."""
    
    def __init__(
        self,
        shadow_text: str,
        original_image_path: str | None,
        element_type: str,
        source_pdf: str,
        page_number: int,
        keywords: str | None = None,
    ) -> None:
        self.shadow_text = shadow_text
        self.original_image_path = original_image_path
        self.element_type = element_type
        self.source_pdf = source_pdf
        self.page_number = page_number
        self.keywords = keywords


class VectorStore:
    """Service for storing vectors and metadata in Qdrant."""

    # all-MiniLM-L6-v2 outputs 384-dim vectors
    VECTOR_DIM = 384
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "pdf_documents",
    ) -> None:
        """
        Initialize the Qdrant vector store.

        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            collection_name: Name of the collection.
        """
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(self.EMBEDDING_MODEL)
        
        try:
            if host == ":memory:":
                self.client = QdrantClient(":memory:")
                logger.info("Connected to in-memory Qdrant")
            else:
                self.client = QdrantClient(host=host, port=port)
                logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant server: {e}")
            logger.info("Using in-memory Qdrant client for local development")
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    def _embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def upsert_document(self, metadata: DocumentMetadata) -> str:
        """
        Store a document with its metadata in Qdrant.

        Args:
            metadata: Document metadata including shadow text.

        Returns:
            The generated document ID.
        """
        doc_id = str(uuid.uuid4())

        payload = {
            "shadow_text": metadata.shadow_text,
            "original_image_path": metadata.original_image_path,
            "element_type": metadata.element_type,
            "source_pdf": metadata.source_pdf,
            "page_number": metadata.page_number,
            "keywords": metadata.keywords,
        }

        # Generate real embedding from shadow text
        vector = self._embed_text(metadata.shadow_text)

        point = PointStruct(
            id=doc_id,
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

        logger.info(
            f"Stored document {doc_id}: {metadata.element_type} from page {metadata.page_number}"
        )
        return doc_id

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text.
            limit: Maximum number of results.
            
        Returns:
            List of matching documents with scores.
        """
        query_vector = self._embed_text(query)
        print(f"DEBUG: Searching Qdrant with vector length: {len(query_vector)}")
        
        try:
            # Try qdrant-client v1.7+ API first (query_points)
            if hasattr(self.client, 'query_points'):
                from qdrant_client.models import models
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                ).points
            else:
                # Fallback to older .search() API
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as e:
            print(f"\n{'='*60}\nCRITICAL SEARCH ERROR: {e}\n{'='*60}\n")
            logger.error(f"Qdrant search failed: {e}", exc_info=True)
            return []
        
        documents = []
        for hit in results:
            documents.append({
                "id": str(hit.id),
                "score": hit.score,
                **hit.payload,
            })
        
        return documents

    def get_all_documents(self) -> list[dict]:
        """
        Retrieve all documents from the collection.

        Returns:
            List of document payloads.
        """
        result = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        documents = []
        for point in result[0]:
            documents.append({
                "id": point.id,
                **point.payload,
            })

        return documents

    def count_documents(self) -> int:
        """Get the total number of documents in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
