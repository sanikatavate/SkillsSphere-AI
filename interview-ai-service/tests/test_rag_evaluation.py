import os
from rag.chunker import chunk_document
from rag.embedder import get_embedding
from rag.retriever import ingest_documents, retrieve_context
from rag.evaluator import evaluate_with_rag, generate_questions_with_ai
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chunker():
    """Verify document chunking yields chunks with correct structure and metadata."""
    doc_path = "knowledge/react/react_docs.md"
    if not os.path.exists("knowledge/react"):
        os.makedirs("knowledge/react")
        with open(doc_path, "w") as f:
            f.write("# React\n## State\nState is local data.\n## Hooks\nuseState is a hook.")

    chunks = chunk_document(doc_path, "React")
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["topic"] == "React"
        assert "subtopic" in chunk["metadata"]

def test_embedder():
    """Verify bge embedding generation yields list of length 384."""
    emb = get_embedding("React state management", is_query=False)
    assert isinstance(emb, list)
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)

def test_retriever_mock_ingest_and_retrieve():
    """Verify document ingestion and semantic search context retrieval in mock mode."""
    num_chunks = ingest_documents("React", "knowledge/react")
    assert num_chunks > 0

    contexts = retrieve_context("What is React State?", "React", top_k=2)
    assert len(contexts) > 0
    assert len(contexts) <= 2
    assert isinstance(contexts[0], str)

def test_evaluator_fallback():
    """Verify Gemini evaluator falls back gracefully to default results if API key is absent."""
    result = evaluate_with_rag(
        question="What are hooks?",
        answer="Hooks are helper functions.",
        context=["Hooks are functions introduced in 16.8."],
        expected_concepts=["hooks", "usecontext"]
    )
    assert "technical" in result
    assert "relevance" in result
    assert "feedback" in result
    assert "weakConcepts" in result

def test_question_generator_fallback():
    """Verify Gemini question generator returns fallback sets if API key is absent."""
    result = generate_questions_with_ai("React", "easy")
    assert "questions" in result
    assert len(result["questions"]) > 0
    assert "questionText" in result["questions"][0]

def test_evaluation_api_route():
    """Verify the evaluate FastAPI endpoint works with mock database and returns expected keys."""
    ingest_documents("React", "knowledge/react")

    payload = {
        "transcript": "React state is a local data store inside component.",
        "expectedAnswer": "State holds local information of component.",
        "expectedConcepts": ["state", "render"],
        "topic": "React"
    }

    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "technical" in data
    assert "communication" in data
    assert "relevance" in data
    assert "concepts" in data
    assert "detected" in data["concepts"]
    assert "missed" in data["concepts"]
    assert "weakConcepts" in data
    assert "feedback" in data

def test_retrieval_api_route():
    """Verify search and ingestion FastAPI endpoints."""
    # Test Ingestion
    ingest_response = client.post("/api/ingest")
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"

    # Test Search
    payload = {
        "query": "virtual dom reconciliation",
        "topic": "React",
        "top_k": 3
    }
    search_response = client.post("/api/retrieve", json=payload)
    assert search_response.status_code == 200
    data = search_response.json()
    assert "results" in data
    assert len(data["results"]) > 0

def test_question_generation_api_route():
    """Verify dynamic question generation FastAPI endpoint."""
    payload = {
        "topic": "React",
        "difficulty": "medium"
    }
    response = client.post("/api/interview/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "questions" in data
    assert len(data["questions"]) > 0
    assert "questionText" in data["questions"][0]

def test_custom_notes_ingest_and_generation():
    """Verify custom note text ingestion and subsequent custom question generation."""
    # Test Ingestion
    ingest_payload = {
        "text": "This is a custom study guide about hooks. React Hooks allow functional components to manage local state and trigger side effects using useState and useEffect.",
        "topic": "custom_notes_test_123"
    }
    response = client.post("/api/ingest-text", json=ingest_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks"] > 0

    # Test custom question generation
    gen_payload = {
        "topic": "custom_notes_test_123",
        "difficulty": "easy"
    }
    gen_response = client.post("/api/interview/generate", json=gen_payload)
    assert gen_response.status_code == 200
    gen_data = gen_response.json()
    assert "questions" in gen_data
    assert len(gen_data["questions"]) > 0
    assert "questionText" in gen_data["questions"][0]
