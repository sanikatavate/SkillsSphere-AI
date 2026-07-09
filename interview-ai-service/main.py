from app import app
from fastapi import Request
from api.evaluation import router as evaluation_router
from api.retrieval import router as retrieval_router
from api.interview import router as interview_router
from routers.transcription import router as transcription_router

@app.get("/health")
async def health_check():
    """Health check endpoint for the Node.js backend to verify service availability."""
    return {"status": "ok", "service": "interview-ai-service"}

@app.get("/api/health")
async def api_health_check():
    """Health check endpoint under the same /api prefix used by service routers."""
    return {"status": "ok", "service": "interview-ai-service", "prefix": "/api"}

@app.middleware("http")
async def log_404_path(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        print(f"[interview-ai-service] 404 Not Found: {request.method} {request.url}")
    return response

@app.get("/api/routes")
async def routes():
    """Return the expected public routes for Node clients."""
    return {
        "service": "interview-ai-service",
        "routes": {
            "health": "/api/health",
            "evaluate": "/api/evaluate",
            "transcribe": "/api/transcribe",
            "transcription_ws": "/api/ws/transcribe",
            "retrieve": "/api/retrieve",
            "ingest": "/api/ingest",
            "generate_questions": "/api/interview/generate"
        }
    }

# Register routers under /api prefix
app.include_router(transcription_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(retrieval_router, prefix="/api")
app.include_router(interview_router, prefix="/api")
