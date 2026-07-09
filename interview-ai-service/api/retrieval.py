from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag.retriever import retrieve_context, ingest_raw_text
from rag.ingestion import run_ingestion_pipeline

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    topic: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    query: str
    topic: str
    results: List[str]

class TextIngestRequest(BaseModel):
    text: str
    topic: str

@router.post("/retrieve", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Expose Phase 6 Retrieval Engine as an API. Fetch the top-k technical chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
    contexts = retrieve_context(request.query, request.topic, top_k=request.top_k)
    return SearchResponse(
        query=request.query,
        topic=request.topic,
        results=contexts
    )

@router.post("/ingest")
async def trigger_ingestion():
    """
    Endpoint to manually run document chunking, embedding generation, and ingestion.
    """
    try:
        report = run_ingestion_pipeline()
        return {
            "status": "success",
            "message": "Ingestion pipeline completed successfully",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@router.post("/ingest-text")
async def ingest_custom_text(request: TextIngestRequest):
    """
    Endpoint to ingest custom document text dynamically.
    """
    if not request.text.strip() or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Text and topic cannot be empty")
        
    try:
        num_chunks = ingest_raw_text(request.topic, request.text)
        return {
            "status": "success",
            "message": f"Successfully ingested {num_chunks} chunks for topic {request.topic}",
            "chunks": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom text ingestion failed: {e}")
