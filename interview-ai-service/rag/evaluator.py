import config
import google.generativeai as genai
import json
import re
import logging
from rag.prompt_builder import build_evaluation_prompt, build_question_prompt

logger = logging.getLogger("evaluator")

# Initialize Gemini from config
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY environment variable is not set. Gemini evaluator will fallback to offline mock scores.")

def evaluate_with_rag(question: str, answer: str, context: list, expected_concepts: list) -> dict:
    """
    Evaluates a candidate's answer using Gemini and retrieved RAG context.
    """
    if not config.GEMINI_API_KEY:
        return {
            "technical": 75,
            "communication": 80,
            "relevance": 70,
            "concepts": {
                "detected": [c for c in expected_concepts[:2]] if expected_concepts else [],
                "missed": [c for c in expected_concepts[2:]] if expected_concepts else []
            },
            "weakConcepts": ["Performance Optimization"],
            "feedback": "Gemini API key is not configured. Running with mock fallback evaluation.",
            "learningRecommendations": ["Study official documentation for hooks and state."]
        }

    prompt = build_evaluation_prompt(question, answer, context, expected_concepts)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\n|```$', '', text, flags=re.MULTILINE).strip()
            
        result = json.loads(text)
        return result
    except Exception as e:
        logger.error(f"Error calling Gemini API for evaluation: {e}")
        return {
            "technical": 60,
            "communication": 70,
            "relevance": 65,
            "concepts": {
                "detected": [],
                "missed": expected_concepts
            },
            "weakConcepts": ["General Concept Understanding"],
            "feedback": f"Failed to complete AI evaluation: {e}",
            "learningRecommendations": ["Review foundational topics."]
        }

def generate_questions_with_ai(topic: str, difficulty: str) -> dict:
    """
    Generates a set of questions dynamically using Gemini.
    """
    if not config.GEMINI_API_KEY:
        # Static offline fallback questions if no API key is configured
        return {
            "questions": [
                {
                    "questionText": f"Can you explain the basic principles of {topic}?",
                    "expectedAnswer": f"Foundational concepts of {topic}.",
                    "expectedConcepts": ["architecture", "basics"]
                },
                {
                    "questionText": f"What are some common use-cases or patterns of {topic}?",
                    "expectedAnswer": f"Common usage patterns of {topic}.",
                    "expectedConcepts": ["patterns", "usecases"]
                },
                {
                    "questionText": f"How do you optimize performance in a {topic} application?",
                    "expectedAnswer": f"Performance profiling and optimizations for {topic}.",
                    "expectedConcepts": ["performance", "optimization"]
                },
                {
                    "questionText": f"Explain context propagation or state management in {topic}.",
                    "expectedAnswer": f"State handling structures of {topic}.",
                    "expectedConcepts": ["state", "context"]
                },
                {
                    "questionText": f"Compare {topic} with other modern alternatives.",
                    "expectedAnswer": f"Comparative trade-off analysis of {topic}.",
                    "expectedConcepts": ["alternatives", "tradeoffs"]
                }
            ]
        }

    # Local lazy import to avoid circular dependency
    from rag.retriever import retrieve_context
    context = retrieve_context("core concepts definitions implementation details", topic, top_k=8)

    prompt = build_question_prompt(topic, difficulty, context)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\n|```$', '', text, flags=re.MULTILINE).strip()
            
        result = json.loads(text)
        return result
    except Exception as e:
        logger.error(f"Error calling Gemini API for question generation: {e}")
        # Default safety fallback
        return {
            "questions": [
                {
                    "questionText": f"Explain the basic architecture of {topic}.",
                    "expectedAnswer": f"Basic architectural overview.",
                    "expectedConcepts": ["architecture"]
                }
            ]
        }
