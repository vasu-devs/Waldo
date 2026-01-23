import logging
from qdrant_client import QdrantClient
from backend.IngestScript.config.settings import get_settings

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reset_db")

def reset_qdrant():
    """Delete the Qdrant collection to start fresh."""
    settings = get_settings()
    
    logger.info(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    collection_name = settings.qdrant_collection_name
    
    try:
        logger.info(f"Checking if collection '{collection_name}' exists...")
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            logger.info(f"Deleting collection '{collection_name}'...")
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}'.")
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Nothing to delete.")
            
    except Exception as e:
        logger.error(f"Error resetting Qdrant: {e}")

if __name__ == "__main__":
    reset_qdrant()
