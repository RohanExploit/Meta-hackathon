"""Two-stage retriever — simulating NVIDIA NeMo Reranker on free-tier infra.

Stage 1 (Dense Retrieval):
    FAISS similarity_search with HuggingFaceEmbeddings → top-20 candidates.

Stage 2 (Cross-Encoder Reranking via LLM):
    Groq's llama-3.3-70b-versatile scores each of the 20 chunks against the
    query on a 0-10 relevance scale, then we keep the top 3-5 by score.

This two-stage approach achieves near cross-encoder recall quality without
requiring a GPU-resident reranker model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from groq import AsyncGroq

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """A document chunk with its relevance score after reranking."""

    content: str
    score: float
    metadata: dict


# ── Embedding model singleton ────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        cfg = get_config()
        _embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding.model_name,
            model_kwargs={"device": cfg.embedding.device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── FAISS vector store ───────────────────────────────────────────────

_vectorstore: FAISS | None = None


def load_vectorstore(path: str | None = None) -> FAISS:
    """Load FAISS index from disk. Creates singleton."""
    global _vectorstore
    if _vectorstore is None:
        cfg = get_config()
        index_path = path or cfg.retriever.faiss_index_path
        _vectorstore = FAISS.load_local(
            index_path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index from %s", index_path)
    return _vectorstore


def set_vectorstore(vs: FAISS) -> None:
    """Inject a pre-built vectorstore (used during ingestion or testing)."""
    global _vectorstore
    _vectorstore = vs


# ── Stage 1: Dense Retrieval ─────────────────────────────────────────

async def dense_retrieve(query: str, k: int | None = None) -> List[Document]:
    """Fetch top-k documents from FAISS using dense embedding similarity.

    Runs the CPU-bound FAISS search in a thread pool to keep the event
    loop unblocked.
    """
    cfg = get_config()
    top_k = k or cfg.retriever.dense_top_k
    vs = load_vectorstore()

    # FAISS search is CPU-bound — offload to a thread
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None,
        lambda: vs.similarity_search(query, k=top_k),
    )
    logger.info("Dense retrieval returned %d documents for query", len(docs))
    return docs


# ── Stage 2: LLM Cross-Encoder Reranking ─────────────────────────────

RERANK_PROMPT = """\
You are a relevance scoring engine. Given a user query and a text chunk, \
rate the chunk's relevance to the query on a scale of 0 to 10.

Rules:
- 0 means completely irrelevant.
- 10 means perfectly answers the query.
- Consider semantic relevance, not just keyword overlap.
- Respond with ONLY a JSON object: {{"score": <number>, "reason": "<brief reason>"}}

User Query: {query}

Text Chunk:
---
{chunk}
---

Your relevance score:"""


async def _score_chunk(
    client: AsyncGroq,
    model: str,
    query: str,
    doc: Document,
    semaphore: asyncio.Semaphore,
) -> RankedChunk:
    """Score a single chunk using the LLM reranker."""
    async with semaphore:
        prompt = RERANK_PROMPT.format(query=query, chunk=doc.page_content[:1500])
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Parse score from JSON
            try:
                data = json.loads(raw)
                score = float(data.get("score", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                # Fallback: extract first number
                import re
                match = re.search(r"(\d+(?:\.\d+)?)", raw)
                score = float(match.group(1)) if match else 0.0

            # Normalize to 0-1
            score = max(0.0, min(10.0, score)) / 10.0

        except Exception as exc:
            logger.warning("Rerank scoring failed for chunk: %s", exc)
            score = 0.0

        return RankedChunk(
            content=doc.page_content,
            score=score,
            metadata=doc.metadata or {},
        )


async def rerank(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[RankedChunk]:
    """Score and rerank documents using LLM cross-encoder pattern.

    Fires all scoring requests concurrently (bounded by semaphore) for
    minimum wall-clock latency.

    Args:
        query: The user's search query.
        documents: Candidate documents from Stage 1.
        top_k: Number of top results to return (default from config).

    Returns:
        Sorted list of RankedChunk, highest relevance first.
    """
    cfg = get_config()
    final_k = top_k or cfg.retriever.rerank_top_k
    client = AsyncGroq(api_key=cfg.groq.api_key)

    # Limit concurrent Groq API calls to avoid rate limiting
    semaphore = asyncio.Semaphore(cfg.max_concurrent_llm_calls)

    # Score all chunks concurrently
    tasks = [
        _score_chunk(client, cfg.groq.reranker_model, query, doc, semaphore)
        for doc in documents
    ]
    ranked = await asyncio.gather(*tasks)

    # Sort by score descending, filter by minimum threshold, take top_k
    ranked.sort(key=lambda r: r.score, reverse=True)
    filtered = [r for r in ranked if r.score >= cfg.retriever.rerank_min_score]

    result = filtered[:final_k]
    logger.info(
        "Reranking: %d candidates → %d after threshold → returning top %d",
        len(documents),
        len(filtered),
        len(result),
    )
    return result


# ── Combined Two-Stage Pipeline ──────────────────────────────────────

async def retrieve_and_rerank(
    query: str,
    dense_k: int | None = None,
    final_k: int | None = None,
) -> List[RankedChunk]:
    """Full two-stage retrieval: dense fetch → LLM rerank.

    Args:
        query: User's search query.
        dense_k: Number of candidates for Stage 1 (default: 20).
        final_k: Number of results after reranking (default: 5).

    Returns:
        Top-k RankedChunks sorted by relevance.
    """
    # Stage 1: Dense retrieval
    candidates = await dense_retrieve(query, k=dense_k)

    if not candidates:
        logger.warning("No documents found in dense retrieval")
        return []

    # Stage 2: LLM reranking
    return await rerank(query, candidates, top_k=final_k)
