from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from rag.evaluator import generate_questions_with_ai

router = APIRouter()

class GenerationRequest(BaseModel):
    topic: str
    difficulty: str

class QuestionDetail(BaseModel):
    questionText: str
    expectedAnswer: str
    expectedConcepts: List[str]

class GenerationResponse(BaseModel):
    topic: str
    difficulty: str
    questions: List[QuestionDetail]

@router.post("/interview/generate", response_model=GenerationResponse)
async def generate_interview_questions(request: GenerationRequest):
    """
    Expose Phase 7 Interview Question Engine. Generates 5 progressive difficulty questions.
    """
    if not request.topic.strip() or not request.difficulty.strip():
        raise HTTPException(status_code=400, detail="Topic and difficulty cannot be empty")
        
    result = generate_questions_with_ai(request.topic, request.difficulty)
    return GenerationResponse(
        topic=request.topic,
        difficulty=request.difficulty,
        questions=[QuestionDetail(**q) for q in result.get("questions", [])]
    )
