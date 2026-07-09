from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from cors_config import get_cors_config

app = FastAPI(
    title="Interview AI Service",
    description="Python AI microservice for the SkillsSphere AI Interview Engine. Handles speech-to-text transcription and answer evaluation.",
    version="1.0.0",
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

# Allow only configured trusted browser origins.
app.add_middleware(
    CORSMiddleware,
    **get_cors_config(),
)
