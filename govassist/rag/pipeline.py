"""Legacy reference pipeline.

The active runtime path uses FastAPI + LangGraph under `govassist/api/api.py`
and `govassist/agents/*`. This module is kept for reference and is not wired
into the current application flow.
"""

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from govassist.rag.embeddings import EmbeddingService, infer_tags_from_text, load_schemes
from govassist.rag.llm import GroqLLMClient
from govassist.rag.vector_store import QdrantManager
from govassist.storage.checkpointer import FileCheckpointer

logger = logging.getLogger(__name__)


def resolve_data_file() -> str:
    """Resolve scheme data path with fallback for legacy .env values."""
    configured_path = os.getenv("SCHEMES_FILE")
    project_root = Path(__file__).resolve().parents[2]

    candidates: List[Path] = []
    if configured_path:
        configured = Path(configured_path)
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.append(project_root / configured)

    candidates.extend(
        [
            Path("data/raw/scheme.json"),
            project_root / "data/raw/scheme.json",
            Path("scheme.json"),
            project_root / "scheme.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Prefer the canonical location for clear startup errors if all candidates are missing.
    return str(project_root / "data/raw/scheme.json")


class GovernmentSchemesRAG:
    def __init__(
        self,
        collection_name: str = "schemes",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "llama-3.1-8b-instant",
    ) -> None:
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.qdrant = QdrantManager(collection_name=collection_name)
        self.llm = GroqLLMClient(model_name=llm_model)
        self.checkpointer = FileCheckpointer()
        self.schemes_cache: List[Dict] = []

    def ingest_schemes(
        self,
        data_file: Optional[str] = None,
        force_recreate: bool = False,
    ) -> int:
        file_path = data_file or resolve_data_file()
        schemes = load_schemes(file_path)
        self.schemes_cache = schemes
        texts = [scheme["search_text"] for scheme in schemes]
        embeddings = self.embedding_service.embed_texts(texts)

        if force_recreate:
            self.qdrant.recreate_collection(self.embedding_service.vector_size)
        else:
            self.qdrant.ensure_collection(self.embedding_service.vector_size)

        self.qdrant.upsert_schemes(schemes, embeddings)
        return len(schemes)

    def _keyword_tokens(self, query: str) -> List[str]:
        raw_tokens = re.findall(r"[a-zA-Z]+", query.lower())
        tokens = []
        seen = set()

        for token in raw_tokens:
            if len(token) <= 2:
                continue
            candidates = {token}
            if token.endswith("s") and len(token) > 3:
                candidates.add(token[:-1])
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    tokens.append(candidate)
        return tokens

    def _fallback_search(
        self,
        query: str,
        top_k: int = 5,
        tag: Optional[str] = None,
    ) -> List[Dict]:
        if not self.schemes_cache:
            return []

        tag_value = tag.strip().lower() if tag else None
        tokens = self._keyword_tokens(query)
        ranked = []

        for scheme in self.schemes_cache:
            if tag_value and tag_value not in scheme.get("tags", []):
                continue

            haystack = " ".join(
                [
                    scheme.get("scheme_name", ""),
                    scheme.get("category", ""),
                    scheme.get("description", ""),
                    scheme.get("eligibility", ""),
                    scheme.get("benefits", ""),
                    " ".join(scheme.get("tags", [])),
                ]
            ).lower()

            score = 0
            for token in tokens:
                if token in haystack:
                    score += 1

            if score > 0:
                item = dict(scheme)
                item["score"] = float(score)
                ranked.append(item)

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def _detect_query_tag(self, query: str, tag: Optional[str]) -> Optional[str]:
        if tag:
            return tag.strip().lower()

        inferred_tags = infer_tags_from_text(query)
        if inferred_tags:
            detected = inferred_tags[0]
            logger.info("Auto-detected query tag: %s", detected)
            return detected

        return None

    def search_schemes(
        self,
        query: str,
        top_k: int = 5,
        tag: Optional[str] = None,
    ) -> List[Dict]:
        effective_tag = self._detect_query_tag(query, tag)
        query_vector = self.embedding_service.embed_query(query)
        results = self.qdrant.search(query_vector=query_vector, top_k=top_k, tag=effective_tag)

        if results:
            logger.info("Vector search returned %s results", len(results))
            return results

        logger.warning("Vector search returned no results. Falling back to keyword search.")
        return self._fallback_search(query=query, top_k=top_k, tag=effective_tag)

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        tag: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        current_session_id = session_id or str(uuid.uuid4())
        history = self.checkpointer.get_history(current_session_id)
        schemes = self.search_schemes(query=query, top_k=top_k, tag=tag)

        if not schemes:
            answer = (
                "I could not find a good matching scheme in the current database. "
                "Please try a different keyword like student, farmer, women, pension, or loan."
            )
            self.checkpointer.save_turn(current_session_id, query, answer, [])
            return {
                "session_id": current_session_id,
                "query": query,
                "answer": answer,
                "matches": [],
            }

        answer = self.llm.generate_answer(
            query=query,
            schemes=schemes,
            chat_history=history,
        )
        self.checkpointer.save_turn(current_session_id, query, answer, schemes)
        return {
            "session_id": current_session_id,
            "query": query,
            "answer": answer,
            "matches": schemes,
        }
