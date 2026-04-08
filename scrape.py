import asyncio
import os

from govassist.api.db_utils import refresh_indexes_from_db
from govassist.ingestion.scraper import main


if __name__ == "__main__":
    asyncio.run(main())
    if os.getenv("AUTO_INGEST", "true").lower() == "true":
        refreshed = refresh_indexes_from_db()
        print(
            "Post-scrape index refresh complete. "
            f"Qdrant={refreshed['qdrant_count']} Graph={refreshed['graph_count']}"
        )
