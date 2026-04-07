import json
import logging
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "schemes.db"
QUERY_HINTS = {
    "farmer": ["farmer", "farmers", "agriculture", "agricultural", "krishi", "crop", "dairy", "livestock"],
    "student": ["student", "students", "education", "scholarship", "school", "college"],
    "women": ["women", "woman", "girl", "girls", "female", "maternity"],
    "health": ["health", "medical", "hospital", "treatment", "insurance"],
    "pension": ["pension", "retirement", "old age", "widow"],
    "loan": ["loan", "credit", "finance", "subsidy"],
}


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schemes (
                id INTEGER PRIMARY KEY,
                scheme_name TEXT,
                category TEXT,
                description TEXT,
                eligibility TEXT,
                benefits TEXT,
                documents_required TEXT,
                application_process TEXT,
                official_link TEXT,
                raw_json TEXT
            )
            """
        )
        connection.execute(
            """
            DELETE FROM schemes
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM schemes
                GROUP BY scheme_name
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_schemes_scheme_name ON schemes (scheme_name)"
        )
        connection.commit()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def upsert_scheme(scheme: Dict[str, Any]) -> None:
    init_db()

    record = {
        "scheme_name": _stringify(scheme.get("scheme_name")),
        "category": _stringify(scheme.get("category")),
        "description": _stringify(scheme.get("description")),
        "eligibility": _stringify(scheme.get("eligibility")),
        "benefits": _stringify(scheme.get("benefits")),
        "documents_required": _stringify(scheme.get("documents_required")),
        "application_process": _stringify(scheme.get("application_process")),
        "official_link": _stringify(scheme.get("official_link")),
        "raw_json": json.dumps(scheme, ensure_ascii=True),
    }

    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO schemes (
                scheme_name,
                category,
                description,
                eligibility,
                benefits,
                documents_required,
                application_process,
                official_link,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scheme_name) DO UPDATE SET
                category = excluded.category,
                description = excluded.description,
                eligibility = excluded.eligibility,
                benefits = excluded.benefits,
                documents_required = excluded.documents_required,
                application_process = excluded.application_process,
                official_link = excluded.official_link,
                raw_json = excluded.raw_json
            """,
            (
                record["scheme_name"],
                record["category"],
                record["description"],
                record["eligibility"],
                record["benefits"],
                record["documents_required"],
                record["application_process"],
                record["official_link"],
                record["raw_json"],
            ),
        )
        connection.commit()


def insert_scheme(scheme: Dict[str, Any]) -> None:
    """Backward-compatible alias used by the scraper."""
    upsert_scheme(scheme)


def insert_schemes(schemes: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for scheme in schemes:
        upsert_scheme(scheme)
        count += 1
    return count


def fetch_schemes_from_db() -> List[Dict[str, Any]]:
    init_db()
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                scheme_name,
                category,
                description,
                eligibility,
                benefits,
                documents_required,
                application_process,
                official_link,
                raw_json
            FROM schemes
            ORDER BY id ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def _truncate(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def _query_terms(query: str) -> List[str]:
    base_terms = re.findall(r"[a-zA-Z]{3,}", (query or "").lower())
    expanded = set(base_terms)

    for key, hints in QUERY_HINTS.items():
        if key in expanded or any(hint in (query or "").lower() for hint in hints):
            expanded.update(hints)

    return sorted(expanded)


def search_schemes_in_db(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    terms = _query_terms(query)
    if not terms:
        return []

    scored_results: List[Dict[str, Any]] = []
    for scheme in fetch_schemes_from_db():
        haystack = " ".join(
            [
                scheme.get("scheme_name", ""),
                scheme.get("category", ""),
                scheme.get("description", ""),
                scheme.get("eligibility", ""),
                scheme.get("benefits", ""),
                scheme.get("documents_required", ""),
                scheme.get("application_process", ""),
            ]
        ).lower()

        score = 0.0
        for term in terms:
            if term in haystack:
                if term in (scheme.get("scheme_name", "").lower()):
                    score += 3.0
                elif term in (scheme.get("category", "").lower()):
                    score += 2.0
                else:
                    score += 1.0

        if score <= 0:
            continue

        result = dict(scheme)
        result["score"] = round(min(score / max(len(terms), 1), 1.0), 3)
        scored_results.append(result)

    scored_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return scored_results[:top_k]


def ingest_schemes_to_qdrant(force_recreate: bool | None = None) -> int:
    from govassist.rag.embeddings import EmbeddingService
    from govassist.rag.vector_store import QdrantManager

    schemes = fetch_schemes_from_db()
    if not schemes:
        logger.info("No schemes found in SQLite. Skipping Qdrant ingestion.")
        return 0

    if force_recreate is None:
        force_recreate = os.getenv("FORCE_RECREATE_COLLECTION", "false").lower() == "true"

    embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantManager(collection_name="schemes")
    if force_recreate:
        qdrant.recreate_collection(vector_size=embedding_service.vector_size)
    else:
        qdrant.ensure_collection(vector_size=embedding_service.vector_size)

    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for scheme in schemes:
        text = (
            f"Scheme: {scheme.get('scheme_name', '')}\n"
            f"Category: {scheme.get('category', '')}\n\n"
            f"Description: {_truncate(scheme.get('description', ''), 500)}\n\n"
            f"Benefits:\n{_truncate(scheme.get('benefits', ''), 500)}\n\n"
            f"Eligibility:\n{_truncate(scheme.get('eligibility', ''), 500)}\n\n"
            f"Documents Required:\n{scheme.get('documents_required', '')}\n\n"
            f"Application Process:\n{_truncate(scheme.get('application_process', ''), 300)}"
        )
        metadata = {
            "id": int(scheme["id"]),
            "scheme_name": scheme.get("scheme_name", ""),
            "category": scheme.get("category", ""),
            "description": scheme.get("description", ""),
            "eligibility": scheme.get("eligibility", ""),
            "benefits": scheme.get("benefits", ""),
            "documents_required": scheme.get("documents_required", ""),
            "application_process": scheme.get("application_process", ""),
            "official_link": scheme.get("official_link", ""),
            "source": scheme.get("official_link", ""),
            "raw_json": scheme.get("raw_json", ""),
        }
        texts.append(text)
        payloads.append(metadata)

    logger.info("Generating embeddings for %s schemes from SQLite", len(texts))
    vectors = embedding_service.embed_texts(texts)
    qdrant.upsert_schemes(schemes=payloads, embeddings=vectors)
    logger.info("Ingested %s schemes into Qdrant", len(payloads))
    return len(payloads)


def rebuild_graph_store_from_db(force_rebuild: bool = True) -> int:
    # Graph store decommissioned
    return 0


def refresh_indexes_from_db(
    force_recreate_collection: bool | None = None,
    force_rebuild_graph: bool = True,
) -> Dict[str, int]:
    qdrant_count = ingest_schemes_to_qdrant(force_recreate=force_recreate_collection)
    return {
        "qdrant_count": qdrant_count,
        "graph_count": 0,
    }
