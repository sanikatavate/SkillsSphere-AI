import os
import logging
from rag.retriever import ingest_documents

logger = logging.getLogger("ingestion")

def run_ingestion_pipeline():
    """
    Ingests all trusted technical documentation located in the knowledge/ folder.
    """
    base_knowledge_dir = "knowledge"
    if not os.path.exists(base_knowledge_dir):
        logger.warning(f"Knowledge directory '{base_knowledge_dir}' does not exist.")
        return {}

    topics = [d for d in os.listdir(base_knowledge_dir) if os.path.isdir(os.path.join(base_knowledge_dir, d))]
    
    report = {}
    for topic in topics:
        folder_path = os.path.join(base_knowledge_dir, topic)
        logger.info(f"Starting document ingestion for topic: {topic} from {folder_path}")
        num_chunks = ingest_documents(topic, folder_path)
        report[topic] = num_chunks
        logger.info(f"Completed ingestion for topic: {topic}. Total chunks stored: {num_chunks}")
        
    return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing Standalone Ingestion Script...")
    result = run_ingestion_pipeline()
    print("Ingestion Pipeline Summary:", result)
