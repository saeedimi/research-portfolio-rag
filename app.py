from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import gradio as gr
import numpy as np
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

INDEX_PATH = ARTIFACT_DIR / "research_index.faiss"
CHUNKS_PATH = ARTIFACT_DIR / "chunks.json"
RETRIEVAL_TEXTS_PATH = ARTIFACT_DIR / "retrieval_texts.json"
CONFIG_PATH = ARTIFACT_DIR / "index_config.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TOP_K = 8
FAISS_K = 50
BM25_K = 20
MAX_HISTORY_MESSAGES = 6

QUERY_EXPANSIONS = {
    "gan": "generative adversarial network",
    "gans": "generative adversarial networks",
    "cnn": "convolutional neural network",
    "llm": "large language model",
    "llms": "large language models",
    "lora": "low-rank adaptation",
    "grpo": "group relative policy optimization",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
}


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class Chunk:
    text: str
    source: str
    file_type: str
    chunk_id: str

    page: int | None = None
    document_type: str | None = None
    page_header: str | None = None

    repository: str | None = None
    relative_path: str | None = None
    section: str | None = None


# =============================================================================
# Load saved artifacts
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact is missing: {path}\n"
            "Run the notebook artifact-saving section first."
        )


for required_path in (
    INDEX_PATH,
    CHUNKS_PATH,
    RETRIEVAL_TEXTS_PATH,
    CONFIG_PATH,
):
    require_file(required_path)


with CONFIG_PATH.open("r", encoding="utf-8") as file:
    index_config = json.load(file)

EMBEDDING_MODEL_NAME = index_config["embedding_model"]
RERANKER_MODEL_NAME = index_config["reranker_model"]
QUERY_INSTRUCTION = index_config["query_instruction"]

index = faiss.read_index(str(INDEX_PATH))

with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = [
        Chunk(**item)
        for item in json.load(file)
    ]

with RETRIEVAL_TEXTS_PATH.open("r", encoding="utf-8") as file:
    retrieval_texts: list[str] = json.load(file)


if index.ntotal != len(chunks):
    raise ValueError(
        f"FAISS contains {index.ntotal} vectors, "
        f"but chunks.json contains {len(chunks)} chunks."
    )

if len(chunks) != len(retrieval_texts):
    raise ValueError(
        "chunks.json and retrieval_texts.json contain "
        "different numbers of records."
    )

if index.d != index_config["embedding_dimension"]:
    raise ValueError(
        "The FAISS embedding dimension does not match index_config.json."
    )


# Load retrieval models once when the app starts.
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text while preserving technical terms and filenames.
    """

    return re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9_.+-]*\b",
        text.lower(),
    )


bm25 = BM25Okapi(
    [
        tokenize_for_bm25(text)
        for text in retrieval_texts
    ]
)


# =============================================================================
# Query processing
# =============================================================================

def expand_query(query: str) -> str:
    """
    Append full forms of common technical abbreviations.
    """

    words = re.findall(
        r"\b[\w-]+\b",
        query.lower(),
    )

    expansions = [
        QUERY_EXPANSIONS[word]
        for word in words
        if word in QUERY_EXPANSIONS
    ]

    if not expansions:
        return query

    unique_expansions = list(
        dict.fromkeys(expansions)
    )

    return query + " " + " ".join(unique_expansions)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a query for semantic retrieval using the same
    instruction used when building the index.
    """

    instructed_query = (
        QUERY_INSTRUCTION
        + query.strip()
    )

    query_embedding = embedding_model.encode(
        [instructed_query],
        normalize_embeddings=True,
    )

    return np.asarray(
        query_embedding,
        dtype="float32",
    )


# =============================================================================
# FAISS and BM25 retrieval
# =============================================================================

def retrieve_faiss_candidates(
    query: str,
    top_k: int = FAISS_K,
) -> list[dict[str, Any]]:
    """
    Retrieve semantic candidates from FAISS.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    search_k = min(
        top_k,
        index.ntotal,
    )

    scores, indices = index.search(
        encode_query(query),
        search_k,
    )

    results = []

    for rank, (score, chunk_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        if chunk_index < 0:
            continue

        results.append(
            {
                "chunk_index": int(chunk_index),
                "faiss_score": float(score),
                "faiss_rank": rank,
            }
        )

    return results


def retrieve_bm25_candidates(
    query: str,
    top_k: int = BM25_K,
) -> list[dict[str, Any]]:
    """
    Retrieve keyword candidates from BM25.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_tokens = tokenize_for_bm25(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, chunk_index in enumerate(
        top_indices,
        start=1,
    ):
        score = float(scores[chunk_index])

        if score <= 0:
            continue

        results.append(
            {
                "chunk_index": int(chunk_index),
                "bm25_score": score,
                "bm25_rank": rank,
            }
        )

    return results


def merge_candidates(
    faiss_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge FAISS and BM25 candidates by chunk index.
    """

    merged: dict[int, dict[str, Any]] = {}

    for result in faiss_results:
        chunk_index = result["chunk_index"]

        merged[chunk_index] = {
            "chunk_index": chunk_index,
            "faiss_score": result["faiss_score"],
            "faiss_rank": result["faiss_rank"],
            "bm25_score": None,
            "bm25_rank": None,
            "retrieved_by": {"faiss"},
        }

    for result in bm25_results:
        chunk_index = result["chunk_index"]

        if chunk_index not in merged:
            merged[chunk_index] = {
                "chunk_index": chunk_index,
                "faiss_score": None,
                "faiss_rank": None,
                "bm25_score": result["bm25_score"],
                "bm25_rank": result["bm25_rank"],
                "retrieved_by": {"bm25"},
            }
        else:
            merged[chunk_index]["bm25_score"] = (
                result["bm25_score"]
            )
            merged[chunk_index]["bm25_rank"] = (
                result["bm25_rank"]
            )
            merged[chunk_index]["retrieved_by"].add(
                "bm25"
            )

    merged_results = list(
        merged.values()
    )

    for result in merged_results:
        result["retrieved_by"] = sorted(
            result["retrieved_by"]
        )

    return merged_results


def retrieve_hybrid_candidates(
    query: str,
    faiss_k: int = FAISS_K,
    bm25_k: int = BM25_K,
) -> list[dict[str, Any]]:
    """
    Retrieve and combine FAISS and BM25 candidates.
    """

    faiss_results = retrieve_faiss_candidates(
        query=query,
        top_k=faiss_k,
    )

    bm25_results = retrieve_bm25_candidates(
        query=query,
        top_k=bm25_k,
    )

    return merge_candidates(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
    )


def enrich_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach chunk text and metadata to retrieval candidates.
    """

    enriched_results = []

    for candidate in candidates:
        chunk_index = candidate["chunk_index"]
        chunk = chunks[chunk_index]

        enriched_results.append(
            {
                **candidate,
                "text": chunk.text,
                "retrieval_text": retrieval_texts[
                    chunk_index
                ],
                "source": chunk.source,
                "file_type": chunk.file_type,
                "page": chunk.page,
                "document_type": chunk.document_type,
                "page_header": chunk.page_header,
                "repository": chunk.repository,
                "relative_path": chunk.relative_path,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
            }
        )

    return enriched_results


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Rerank hybrid candidates using the cross-encoder.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not candidates:
        return []

    query_chunk_pairs = [
        [
            query,
            candidate["retrieval_text"],
        ]
        for candidate in candidates
    ]

    reranker_scores = reranker.predict(
        query_chunk_pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for candidate, score in zip(
        candidates,
        reranker_scores,
    ):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        reranked_results.append(result)

    reranked_results.sort(
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True,
    )

    return reranked_results[:top_k]


def retrieve_with_reranking(
    query: str,
    top_k: int = TOP_K,
    faiss_k: int = FAISS_K,
    bm25_k: int = BM25_K,
) -> list[dict[str, Any]]:
    """
    Run query expansion, hybrid retrieval, and reranking.
    """

    search_query = expand_query(query)

    hybrid_candidates = retrieve_hybrid_candidates(
        query=search_query,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
    )

    enriched_candidates = enrich_candidates(
        hybrid_candidates
    )

    return rerank_candidates(
        query=query,
        candidates=enriched_candidates,
        top_k=top_k,
    )


# =============================================================================
# Source formatting and context construction
# =============================================================================

def format_source_location(
    result: dict[str, Any],
) -> str:
    """
    Format source metadata for the LLM context.
    """

    if result.get("repository"):
        location = (
            f"GitHub repository: "
            f"{result['repository']}"
        )

        if result.get("relative_path"):
            location += (
                f", file: "
                f"{result['relative_path']}"
            )

        if result.get("section"):
            location += (
                f", section: "
                f"{result['section']}"
            )

        return location

    location = result.get(
        "source",
        "Unknown source",
    )

    if result.get("page") is not None:
        location += (
            f", page {result['page']}"
        )

    if result.get("document_type"):
        location += (
            f", {result['document_type']}"
        )

    return location


def build_context(
    results: list[dict[str, Any]],
) -> str:
    """
    Build numbered source blocks for answer generation.
    """

    context_parts = []

    for source_number, result in enumerate(
        results,
        start=1,
    ):
        location = format_source_location(
            result
        )

        context_parts.append(
            f"[Source {source_number}: "
            f"{location}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)


def format_source(
    source: dict[str, Any],
) -> str:
    """
    Create a readable public source label.
    """

    if source.get("repository"):
        location = (
            f"GitHub: "
            f"{source['repository']}"
        )

        if source.get("relative_path"):
            location += (
                f" / "
                f"{source['relative_path']}"
            )

        if source.get("section"):
            location += (
                f" — "
                f"{source['section']}"
            )

        return location

    location = source.get(
        "source",
        "Unknown source",
    )

    if source.get("page") is not None:
        location += (
            f", page {source['page']}"
        )

    return location


def format_sources(
    sources: list[dict[str, Any]],
) -> str:
    """
    Format unique sources as Markdown.
    """

    if not sources:
        return ""

    lines = ["### Sources"]
    seen_locations = set()

    for source in sources:
        location = format_source(source)

        if location in seen_locations:
            continue

        seen_locations.add(location)
        lines.append(f"- {location}")

    return "\n".join(lines)


# =============================================================================
# Conversation history
# =============================================================================

def remove_source_section(
    text: str,
) -> str:
    """
    Remove the displayed source list from a previous response.
    """

    if not text:
        return ""

    marker = "\n\n---\n\n### Sources"

    return text.split(
        marker,
        1,
    )[0].strip()


def format_chat_history(
    history,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    """
    Convert recent Gradio history into readable text.

    Supports newer message dictionaries and older tuple-style history.
    """

    if not history:
        return ""

    lines = []

    for item in history[-max_messages:]:

        if isinstance(item, dict):
            role = str(
                item.get("role", "")
            ).strip().lower()

            content = item.get(
                "content",
                "",
            )

            if not isinstance(content, str):
                continue

            content = content.strip()

            if role == "assistant":
                content = remove_source_section(
                    content
                )

            if (
                content
                and role in {"user", "assistant"}
            ):
                lines.append(
                    f"{role.capitalize()}: "
                    f"{content}"
                )

        elif (
            isinstance(item, (list, tuple))
            and len(item) == 2
        ):
            user_message, assistant_message = item

            if (
                isinstance(user_message, str)
                and user_message.strip()
            ):
                lines.append(
                    f"User: "
                    f"{user_message.strip()}"
                )

            if isinstance(
                assistant_message,
                str,
            ):
                assistant_message = (
                    remove_source_section(
                        assistant_message
                    )
                )

                if assistant_message:
                    lines.append(
                        f"Assistant: "
                        f"{assistant_message}"
                    )

    return "\n".join(lines)


# =============================================================================
# Gemini generation
# =============================================================================

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not configured. Add it as a private Secret "
        "in the Hugging Face Space settings."
    )

gemini_client = genai.Client()


def generate_answer(
    prompt: str,
    model_name: str = GEMINI_MODEL,
) -> str:
    """
    Generate a response using the Gemini API.
    """

    interaction = gemini_client.interactions.create(
        model=model_name,
        input=prompt,
    )

    answer = interaction.output_text

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer.strip()


def rewrite_question_with_history(
    question: str,
    history,
    model_name: str = GEMINI_MODEL,
) -> str:
    """
    Rewrite a follow-up question as a standalone retrieval query.
    """

    history_text = format_chat_history(history)

    if not history_text:
        return question

    prompt = f"""
Rewrite the latest question as a standalone search query.

Use the conversation only to resolve references such as:
he, his, it, that project, that paper, or the thesis.

Do not answer the question.
Do not add unsupported facts.
Return only the rewritten query.

Conversation:
{history_text}

Latest question:
{question}

Standalone query:
""".strip()

    try:
        rewritten = generate_answer(
            prompt=prompt,
            model_name=model_name,
        ).strip()

        return rewritten or question

    except Exception:
        return question


def build_prompt(
    question: str,
    results: list[dict[str, Any]],
    history=None,
) -> str:
    """
    Build the grounded answer-generation prompt.
    """

    context = build_context(results)
    history_text = format_chat_history(history)

    if not history_text:
        history_text = "No previous conversation."

    return f"""
You are ResearchGPT, a research and portfolio assistant for Milad Saeedi.

Use only the retrieved context as factual evidence.

Citation requirements:
1. Every paragraph containing a factual claim must include at least one citation.
2. Use citations exactly in this format: [Source 1], [Source 2], etc.
3. Place citations immediately after the sentence or claim they support.
4. Use only source numbers that appear in the retrieved context.
5. Do not create a separate source list; the application adds it automatically.
6. Do not omit citations in summaries, lists, or conclusions.

Additional instructions:
1. Answer the latest question directly.
2. Use conversation history only to resolve follow-up references.
3. Do not treat conversation history as factual evidence.
4. Do not invent publications, methods, results, skills, projects, or experience.
5. If the retrieved context is insufficient, say so clearly.
6. Avoid unsupported praise or subjective claims.
7. Contact information may be provided only when explicitly requested.
8. Synthesize the evidence instead of copying long passages.

Recent conversation:
{history_text}

Retrieved context:
{context}

Latest question:
{question}

Write a grounded answer with inline citations:
""".strip()


def has_inline_citations(answer: str) -> bool:
    """
    Check whether an answer contains at least one [Source N] citation.
    """

    return bool(
        re.search(r"\[Source\s+\d+\]", answer)
    )


def answer_question(
    question: str,
    history=None,
    top_k: int = TOP_K,
    faiss_k: int = FAISS_K,
    bm25_k: int = BM25_K,
    model_name: str = GEMINI_MODEL,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve evidence and generate a grounded answer.
    """

    question = question.strip()

    if not question:
        return "Please enter a question.", []

    retrieval_query = rewrite_question_with_history(
        question=question,
        history=history,
        model_name=model_name,
    )

    results = retrieve_with_reranking(
        query=retrieval_query,
        top_k=top_k,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
    )

    if not results:
        return (
            "I could not find relevant information in the knowledge base.",
            [],
        )

    answer = generate_answer(
        prompt=build_prompt(
            question=question,
            results=results,
            history=history,
        ),
        model_name=model_name,
    )

    # Retry once if Gemini omitted the required inline citations.
    if not has_inline_citations(answer):
        retry_prompt = f"""
Revise the answer below by adding accurate inline citations.

Requirements:
- Every factual paragraph must contain at least one citation.
- Use only [Source 1] through [Source {len(results)}].
- Use the retrieved context to determine which citation supports each claim.
- Do not invent facts.
- Do not add a separate source list.
- Return only the revised answer.

Retrieved context:
{build_context(results)}

Original answer:
{answer}
""".strip()

        answer = generate_answer(
            prompt=retry_prompt,
            model_name=model_name,
        )

    return answer, results


# =============================================================================
# Gradio application
# =============================================================================

def research_chat(
    message: str,
    history,
) -> str:
    """
    Answer one Gradio message using the RAG pipeline.
    """

    message = message.strip()

    if not message:
        return "Please enter a question."

    try:
        answer, sources = answer_question(
            question=message,
            history=history,
            top_k=TOP_K,
            faiss_k=FAISS_K,
            bm25_k=BM25_K,
            model_name=GEMINI_MODEL,
        )

        response = answer

        sources_markdown = format_sources(
            sources
        )

        if sources_markdown:
            response += (
                f"\n\n---\n\n"
                f"{sources_markdown}"
            )

        return response

    except Exception as error:
        return (
            "I could not process this question because "
            "an error occurred.\n\n"
            f"`{type(error).__name__}: {error}`"
        )


with gr.Blocks(
    title="ResearchGPT — Milad Saeedi",
) as demo:

    gr.Markdown(
        """
# ResearchGPT — Milad Saeedi

Explore Milad Saeedi's research, PhD thesis, publications,
machine-learning experience, and selected GitHub projects.

Answers are grounded in retrieved portfolio documents and
include supporting sources.
"""
    )

    chatbot = gr.Chatbot(
        placeholder=(
            "Ask about Milad's research, publications, "
            "technical skills, or GitHub projects."
        ),
        height=550,
    )

    gr.ChatInterface(
        fn=research_chat,
        chatbot=chatbot,
        examples=[
            "Summarize Milad Saeedi's research.",
            "What are the main contributions of his PhD thesis?",
            "Summarize his GitHub GAN projects.",
            "Which projects involve computer vision?",
            "What experience does he have with LoRA and GRPO?",
            "What did his research show about spatial cross-validation?",
            "Tell me about his geospatial modeling experience.",
            "How can I contact Milad?",
        ],
        save_history=True,
        flagging_mode="never",
    )

    gr.Markdown(
        """
---
**ResearchGPT** uses hybrid retrieval, BGE embeddings,
FAISS, BM25, cross-encoder reranking, Gemini,
and grounded answer generation.
"""
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
