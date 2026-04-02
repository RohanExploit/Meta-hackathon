"""Semantic Chunking ingestion pipeline.

Replaces RecursiveCharacterTextSplitter with LangChain's
SemanticChunker (backed by local HuggingFace embeddings) so that text
is split on semantic meaning boundaries — not arbitrary character counts.

Usage:
    python -m pipeline.ingest --source ./documents --output ./vectorstore

Supports: .txt, .pdf, .md, .json files.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from .config import get_config
from .retriever import get_embeddings

logger = logging.getLogger(__name__)


# ── Document loaders ─────────────────────────────────────────────────

def _load_text_file(path: Path) -> List[Document]:
    """Load a plain text or markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_pdf_file(path: Path) -> List[Document]:
    """Load a PDF file (requires pypdf)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        return loader.load()
    except ImportError:
        logger.warning("pypdf not installed — skipping %s", path)
        return []


def _load_json_file(path: Path) -> List[Document]:
    """Load a JSON file as a single document."""
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    content = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
    return [Document(page_content=content, metadata={"source": str(path)})]


LOADERS = {
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".pdf": _load_pdf_file,
    ".json": _load_json_file,
}


def load_documents(source_dir: str) -> List[Document]:
    """Recursively load all supported documents from a directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    documents: List[Document] = []

    for file_path in sorted(source_path.rglob("*")):
        if file_path.is_dir():
            continue

        loader = LOADERS.get(file_path.suffix.lower())
        if loader is None:
            logger.debug("Skipping unsupported file: %s", file_path)
            continue

        try:
            docs = loader(file_path)
            documents.extend(docs)
            logger.info("Loaded %d document(s) from %s", len(docs), file_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)

    logger.info("Total documents loaded: %d", len(documents))
    return documents


# ── Semantic Chunking ────────────────────────────────────────────────

def semantic_chunk(documents: List[Document]) -> List[Document]:
    """Split documents using SemanticChunker.

    Instead of splitting on character count, this analyzes the embedding
    distance between consecutive sentences and places breaks where the
    semantic similarity drops below a threshold. This preserves paragraph
    coherence and prevents context loss mid-sentence.
    """
    cfg = get_config()
    embeddings = get_embeddings()

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=cfg.ingest.breakpoint_threshold_type,
        breakpoint_threshold_amount=cfg.ingest.breakpoint_threshold_amount,
    )

    chunks: List[Document] = []
    for doc in documents:
        try:
            split_docs = chunker.create_documents(
                texts=[doc.page_content],
                metadatas=[doc.metadata],
            )
            chunks.extend(split_docs)
        except Exception as exc:
            logger.warning("Chunking failed for %s: %s", doc.metadata.get("source"), exc)
            # Fallback: keep the document as a single chunk
            chunks.append(doc)

    logger.info(
        "Semantic chunking: %d documents → %d chunks",
        len(documents),
        len(chunks),
    )
    return chunks


# ── FAISS Index Building ─────────────────────────────────────────────

def build_vectorstore(chunks: List[Document], output_dir: str) -> FAISS:
    """Build and persist a FAISS index from chunked documents."""
    embeddings = get_embeddings()

    logger.info("Building FAISS index from %d chunks...", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    logger.info("FAISS index saved to %s", output_dir)

    return vectorstore


# ── Full Ingestion Pipeline ──────────────────────────────────────────

def run_ingestion(source_dir: str, output_dir: str) -> FAISS:
    """End-to-end ingestion: load → semantic chunk → embed → persist.

    Args:
        source_dir: Path to directory containing source documents.
        output_dir: Path to save the FAISS index.

    Returns:
        The built FAISS vectorstore.
    """
    logger.info("Starting ingestion pipeline")
    logger.info("  Source: %s", source_dir)
    logger.info("  Output: %s", output_dir)

    # 1. Load documents
    documents = load_documents(source_dir)
    if not documents:
        raise ValueError(f"No documents found in {source_dir}")

    # 2. Semantic chunking
    chunks = semantic_chunk(documents)

    # 3. Build and persist FAISS index
    vectorstore = build_vectorstore(chunks, output_dir)

    logger.info("Ingestion complete: %d chunks indexed", len(chunks))
    return vectorstore


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    """CLI for running the ingestion pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG pipeline (SemanticChunker + FAISS)"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to directory containing source documents",
    )
    parser.add_argument(
        "--output", "-o",
        default="./vectorstore",
        help="Path to save the FAISS index (default: ./vectorstore)",
    )
    args = parser.parse_args()

    try:
        run_ingestion(args.source, args.output)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
