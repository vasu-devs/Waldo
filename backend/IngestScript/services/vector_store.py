"""
Vector Store service using Qdrant.

Stores document vectors with Shadow Text metadata.
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a document element stored in Qdrant."""
    shadow_text: str
    original_image_path: str
    element_type: str
    source_pdf: str
    page_number: int


class VectorStore:
    """Service for storing vectors and metadata in Qdrant."""

    # Dummy vector dimension (we're storing metadata primarily)
    # In production, this would be the embedding dimension
    VECTOR_DIM = 384

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
        
        try:
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

    def _generate_dummy_vector(self) -> list[float]:
        """
        Generate a placeholder vector.
        
        In production, this would use an embedding model.
        For now, we use random values as the focus is on metadata storage.
        """
        import random
        return [random.uniform(-1, 1) for _ in range(self.VECTOR_DIM)]

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
        }

        point = PointStruct(
            id=doc_id,
            vector=self._generate_dummy_vector(),
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
