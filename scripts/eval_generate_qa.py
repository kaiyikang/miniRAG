from minirag.adapters.llm import OpenRouterEngine
from minirag.adapters.vector_store import ChromaVectorStore
from minirag.config import get_settings
from minirag.adapters.evaluator import QA_DATASET_FILENAME, _git_info
import argparse
import random
import re
import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

PROMPT_VERSION = "v1"  # bump when QA_GENERATION_PROMPT changes, recorded in provenance
MIN_WORDS = 20  # chunks shorter than this are too thin to ask good questions
SEED = 42  # fixes the chunk sample so a regenerated dataset is reproducible


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
        messages=QA_GENERATION_PROMPT.format(n_questions=n_questions, text=text),
        span_name="qa_generation",
    )["content"]

    content = parse_qa_content(content)

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a QA eval dataset from indexed chunks."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max QA samples to write (default 10). 0 = generate from the whole corpus.",
    )
    parser.add_argument("--n-questions", type=int, default=3)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing dataset."
    )
    args = parser.parse_args()

    settings = get_settings()

    llm = OpenRouterEngine(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
    )
    store = ChromaVectorStore(
        vector_store_path=settings.vector_store_path,
        collection_name=settings.collection_name,
    )

    # Output lives next to what the evaluator reads (eval/), under one shared name.
    out_path = Path("eval") / QA_DATASET_FILENAME
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists — refusing to overwrite the ground-truth "
            f"dataset. Pass --force to regenerate, or delete it first."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample eligible chunks up front (seeded, so a regenerated dataset is
    # reproducible and not biased toward whatever happens to be first in the store).
    eligible = [
        c for c in store.get_all_chunks() if len(c.document.split()) >= MIN_WORDS
    ]
    random.Random(SEED).shuffle(eligible)

    n_chunks = 0
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in eligible:
            if args.limit and n_written >= args.limit:
                break
            n_chunks += 1
            print(f"chunk {n_chunks} | {n_written} samples so far")
            try:
                qa_pairs = generate_qa_for_chunk(chunk.document, llm, args.n_questions)
            except Exception as e:
                print(f"Failed on chunk {chunk.chunk_id}: {e}")
                continue
            for qa in qa_pairs:
                if args.limit and n_written >= args.limit:
                    break
                sample = {
                    "question": qa.question,
                    "expected_answer": qa.answer,
                    "expected_chunk_ids": [chunk.chunk_id],
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1

    # Provenance sidecar: how this dataset was produced, so eval results stay
    # interpretable months later (mirrors the git/params recorded per eval run).
    commit, dirty = _git_info()
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "model": settings.openrouter_model,
        "source_collection": settings.collection_name,
        "prompt_version": PROMPT_VERSION,
        "n_questions_per_chunk": args.n_questions,
        "min_words": MIN_WORDS,
        "seed": SEED,
        "limit": args.limit,
        "n_chunks_used": n_chunks,
        "n_samples": n_written,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {n_written} samples to {out_path}\nProvenance: {meta_path}")
