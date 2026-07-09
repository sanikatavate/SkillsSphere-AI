import os
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
