import logging
import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


_local_clients = {}

class QdrantManager:
    """Small wrapper around Qdrant for this project."""

    def __init__(
        self,
        collection_name: str = "schemes",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        qdrant_mode = os.getenv("QDRANT_MODE", "local").lower()
        local_path = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_data")
        qdrant_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")

        if qdrant_mode == "local":
            global _local_clients
            if local_path not in _local_clients:
                logger.info("Using local Qdrant storage at: %s", local_path)
                _local_clients[local_path] = QdrantClient(path=local_path)
            self.client = _local_clients[local_path]
        else:
            logger.info("Connecting to Qdrant server: %s", qdrant_url)
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        existing = {collection.name for collection in collections}

        if self.collection_name in existing:
            logger.info("Qdrant collection '%s' already exists", self.collection_name)
            return

        logger.info("Creating Qdrant collection '%s'", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def recreate_collection(self, vector_size: int) -> None:
        logger.info("Recreating Qdrant collection '%s'", self.collection_name)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def count(self) -> int:
        response = self.client.count(collection_name=self.collection_name, exact=True)
        return response.count

    def upsert_schemes(
        self,
        schemes: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 64,
    ) -> None:
        if len(schemes) != len(embeddings):
            raise ValueError("Number of schemes and embeddings must match.")

        for start in range(0, len(schemes), batch_size):
            end = start + batch_size
            batch_schemes = schemes[start:end]
            batch_vectors = embeddings[start:end]

            points = [
                PointStruct(
                    id=scheme["id"],
                    vector=vector,
                    payload=scheme,
                )
                for scheme, vector in zip(batch_schemes, batch_vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info("Inserted %s schemes into Qdrant", len(schemes))

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_filter = None
        if tag:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tags",
                        match=MatchAny(any=[tag.strip().lower()]),
                    )
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            formatted = []
            for item in response.points:
                payload = dict(item.payload or {})
                payload["score"] = item.score
                formatted.append(payload)
            return formatted
        except Exception as e:
            logger.warning(f"Qdrant search failed (Collection might not exist): {e}")
            return []
