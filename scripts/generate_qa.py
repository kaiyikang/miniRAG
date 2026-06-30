from minirag.llm_engine import OpenRouterEngine
from minirag.vector_store import ChromaVectorStore
from minirag.config import get_settings
import os
import re
import json
from pydantic import BaseModel, Field, ValidationError


class QAPair(BaseModel):
    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)


QA_GENERATION_PROMPT = """\
You are given a text chunk. Generate {n_questions} diverse question-answer pairs.

Rules:
1. Each question must be answerable using ONLY the information in this chunk.
2. Do NOT ask about titles, dates, authors, or document metadata.
3. Do NOT ask about English related information.
4. Include at least one "what/who/when" factual question and one "why/how" explanatory question.
5. Questions should be specific and natural, as if a real user asked them.

Output a JSON array:
[
  {{"question": "...", "answer": "..."}}
]

Chunk:
{text}
"""


def parse_qa_content(content: str) -> list[QAPair]:
    content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content.strip())

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    try:
        return [QAPair(**item) for item in data]
    except ValidationError as e:
        raise ValueError(f"LLM output does not match QA schema: {e}") from e


def generate_qa_for_chunk(
    text: str, llm: OpenRouterEngine, n_questions: int = 3
) -> list[dict]:

    content = llm.generate(
        messages=QA_GENERATION_PROMPT.format(n_questions=n_questions, text=text)
    )["content"]

    content = parse_qa_content(content)

    return content


if __name__ == "__main__":
    settings = get_settings()

    llm = OpenRouterEngine(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
    )
    store = ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    )

    os.makedirs("data", exist_ok=True)
    chunks = store.get_all_chunks()
    with open("data/qa_dataset.jsonl", "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            # too short, don't generate
            if len(chunk.document.split()) < 20:
                continue
            print(f"{idx}/{len(chunks)}")
            try:
                qa_pairs = generate_qa_for_chunk(chunk.document, llm)
                for qa in qa_pairs:
                    sample = {
                        "question": qa.question,
                        "expected_answer": qa.answer,
                        "expected_chunk_ids": [chunk.chunk_id],
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed on chunk {chunk.chunk_id}: {e}")
                continue
