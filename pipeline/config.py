"""Centralized configuration for the RAG pipeline.

All model names, API keys, and tuning constants live here.
Reads from environment variables with sensible defaults for local dev.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroqConfig:
    """Groq cloud API settings."""

    api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    base_url: str = "https://api.groq.com/openai/v1"

    # ── Model registry ──────────────────────────────────────────────
    guard_model: str = "llama-guard-3-8b"
    router_model: str = "llama-3.1-8b-instant"
    reranker_model: str = "llama-3.3-70b-versatile"
    generator_model: str = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class OllamaConfig:
    """Local Ollama (optional fallback for router)."""

    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    router_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_ROUTER_MODEL", "qwen2.5-coder:7b")
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local HuggingFace embedding settings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass(frozen=True)
class RetrieverConfig:
    """FAISS retrieval tuning knobs."""

    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./vectorstore")
    )
    dense_top_k: int = 20          # Stage 1: over-fetch from FAISS
    rerank_top_k: int = 5          # Stage 2: keep after LLM reranking
    rerank_min_score: float = 0.3  # Floor for relevance score (0-1)


@dataclass(frozen=True)
class IngestConfig:
    """Semantic chunking settings for the ingestion pipeline."""

    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 85.0  # Higher = fewer, larger chunks


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level config bundle."""

    groq: GroqConfig = field(default_factory=GroqConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)

    # Routing preferences
    use_ollama_router: bool = field(
        default_factory=lambda: os.getenv("USE_OLLAMA_ROUTER", "false").lower() == "true"
    )

    # Async concurrency limit
    max_concurrent_llm_calls: int = 4

    # Environment server URL (for the retail env)
    env_base_url: str = field(
        default_factory=lambda: os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
    )


# ── Singleton ────────────────────────────────────────────────────────
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Return the pipeline config singleton."""
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config
