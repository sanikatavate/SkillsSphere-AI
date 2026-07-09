def build_evaluation_prompt(question: str, answer: str, context: list, expected_concepts: list) -> str:
    """
    Constructs the prompt instructions for Gemini evaluation grounded in RAG context.
    """
    formatted_context = "\n---\n".join(context) if context else "No relevant documentation found."
    
    prompt = f"""
You are an expert technical interviewer evaluating a candidate's response.
Your evaluation MUST be strictly grounded in the official technical documentation provided below. Do not use outside knowledge or hallucinated information.

Interview Question:
"{question}"

Candidate's Answer:
"{answer}"

Official Technical Documentation (Context):
{formatted_context}

Expected Concepts list:
{expected_concepts}

Evaluate the candidate's answer based ONLY on the provided documentation. Detect which expected concepts from the list were successfully explained, which were missed, and identify any weak sub-concepts.

Return your response in raw JSON format with the following schema:
{{
  "technical": <int: score 0 to 100 based on technical correctness matching the documentation>,
  "communication": <int: score 0 to 100 based on clarity and structure>,
  "relevance": <int: score 0 to 100 based on how well they answered the specific question asked>,
  "concepts": {{
    "detected": [<list of concepts from the Expected Concepts list that were explained>],
    "missed": [<list of concepts from the Expected Concepts list that were missed>]
  }},
  "weakConcepts": [<list of specific sub-concepts or details they explained poorly, missed, or ignored, e.g. "cleanup function", "dependency array">],
  "feedback": "<string: structured feedback with strengths and areas for improvement>",
  "learningRecommendations": [<list of topics or suggested sections to study based on their weak concepts>]
}}

Do not include any markdown format tags like ```json or ``` in the response. Return raw JSON.
"""
    return prompt

def build_question_prompt(topic: str, difficulty: str, context: list = None) -> str:
    """
    Constructs the prompt instructions to generate a mock interview session question set.
    """
    context_str = ""
    if context:
        formatted_context = "\n---\n".join(context)
        context_str = f"\n\nSource Documentation Context:\n{formatted_context}\n\nYour generated questions MUST be grounded in the above source documentation context."

    prompt = f"""
You are an expert technical interviewer. Generate a set of exactly 5 interview questions for the topic "{topic}" at the "{difficulty}" difficulty level.{context_str}

The questions should:
1. Cover foundational concepts, implementation details, and practical scenarios.
2. Form a progressive sequence of difficulty.
3. Be completely unique (no duplicates).

For each question, define:
- "questionText": The text of the question asked to the candidate.
- "expectedAnswer": The ideal/expected model answer.
- "expectedConcepts": A list of 2-4 key concept strings that the candidate must hit.

Return your response in raw JSON format with the following schema:
{{
  "questions": [
    {{
      "questionText": "Question string",
      "expectedAnswer": "Model answer string",
      "expectedConcepts": ["concept1", "concept2"]
    }}
  ]
}}

Do not include any markdown format tags like ```json or ``` in the response. Return raw JSON.
"""
    return prompt
