from qdrant_client import QdrantClient
import config
import logging

logger = logging.getLogger("qdrant")

_client = None
_is_mock = False

# Ephemeral in-memory fallback list to store document chunks
_mock_vector_store = []

def get_qdrant_client():
    global _client, _is_mock
    if _client is not None:
        return _client, _is_mock

    try:
        # Try connecting to the live Qdrant container using config
        _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=3.0)
        # Test connection by listing collections
        _client.get_collections()
        _is_mock = False
        logger.info(f"Connected to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}: {e}")
        logger.warning("Falling back to local in-memory mock Vector Store.")
        _client = None
        _is_mock = True

    return _client, _is_mock

def reset_mock_store():
    global _mock_vector_store
    _mock_vector_store.clear()

def get_mock_store():
    return _mock_vector_store
