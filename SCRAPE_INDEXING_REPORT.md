# Scrape, DB, Qdrant, and Graph Store Report

## Summary

I traced the scrape pipeline end to end and found the gap:

1. the scraper was writing to SQLite
2. `/scrape` API was refreshing Qdrant after the scrape
3. standalone `scrape.py` was not refreshing Qdrant at all
4. the graph store was not bootstrapping from `schemes.db`, so it logged:
   `No graph store found and no documents provided.`

That warning was about the graph-based synergy index, not about SQLite or Qdrant directly.

I fixed the wiring so the system now has one shared post-scrape refresh path and the graph store can build itself from the scraped SQLite data.

---

## What I Found

### 1. Why the graph-store warning appeared

The warning came from `GraphStoreManager.load_or_create()` in [govassist/rag/graph_store.py](/home/dhinakaran/GovAssist-RAG/govassist/rag/graph_store.py), where it previously only did this:

- load an existing graph if `qdrant_data/graph_store/graph_store.json` existed
- otherwise create a graph only if `documents` were explicitly passed in
- otherwise log the warning

The active runtime called `graph_manager.load_or_create()` with no documents from [govassist/agents/nodes.py](/home/dhinakaran/GovAssist-RAG/govassist/agents/nodes.py), so it never used `schemes.db` as a fallback source for graph creation.

That meant:

- SQLite had data
- Qdrant might have data
- but graph-store bootstrap still said “no documents provided”

### 2. Why `/scrape` behaved differently from standalone `scrape.py`

The old behavior was split:

- `/scrape` API:
  - ran `scrape.py`
  - then called `ingest_schemes_to_qdrant()` in the API process
- standalone `scrape.py`:
  - only ran the scraper
  - only wrote to SQLite
  - never refreshed Qdrant

So your doubt was correct: the standalone scraper path and the API path were not equivalent.

Relevant files:

- [scrape.py](/home/dhinakaran/GovAssist-RAG/scrape.py)
- [govassist/api/api.py](/home/dhinakaran/GovAssist-RAG/govassist/api/api.py)
- [govassist/api/db_utils.py](/home/dhinakaran/GovAssist-RAG/govassist/api/db_utils.py)

### 3. Why source links are showing now

Source links are built from `_build_sources()` in [govassist/agents/nodes.py](/home/dhinakaran/GovAssist-RAG/govassist/agents/nodes.py#L141), which uses:

- `scheme_name`
- `official_link` or `source`

The current SQLite rows already contain `official_link`, and the Qdrant ingestion path also stores both:

- `official_link`
- `source`

from [govassist/api/db_utils.py](/home/dhinakaran/GovAssist-RAG/govassist/api/db_utils.py#L263)

That is why links are now surfacing in responses more reliably.

---

## Changes Made

### 1. Shared index refresh flow

Added shared helpers in [govassist/api/db_utils.py](/home/dhinakaran/GovAssist-RAG/govassist/api/db_utils.py):

- `ingest_schemes_to_qdrant(force_recreate: bool | None = None)`
- `rebuild_graph_store_from_db(force_rebuild: bool = True)`
- `refresh_indexes_from_db(...)`

This now gives one place to refresh:

- Qdrant from `schemes.db`
- graph store from `schemes.db`

### 2. Graph store now bootstraps from SQLite

Updated [govassist/rag/graph_store.py](/home/dhinakaran/GovAssist-RAG/govassist/rag/graph_store.py) so it can:

- load scheme rows from SQLite
- convert them into LlamaIndex `Document` objects
- create the property graph from DB content when no persisted graph exists
- rebuild the graph when requested

This resolves the specific warning cause in normal populated environments.

### 3. Standalone `scrape.py` now refreshes indexes

Updated [scrape.py](/home/dhinakaran/GovAssist-RAG/scrape.py) so standalone runs now do:

1. scrape into SQLite
2. if `AUTO_INGEST=true`, refresh Qdrant and graph store from SQLite

This closes the previous standalone gap.

### 4. `/scrape` API avoids duplicate auto-ingestion in the subprocess

Updated [govassist/api/api.py](/home/dhinakaran/GovAssist-RAG/govassist/api/api.py#L468) so the API:

- runs `scrape.py` with `AUTO_INGEST=false`
- then refreshes indexes in the API process via `refresh_indexes_from_db()`

This avoids duplicate ingest work and keeps the API-owned scrape flow deterministic.

---

## Important Note About Local Qdrant Mode

Your repo uses local Qdrant storage:

- `QDRANT_MODE=local`
- `QDRANT_LOCAL_PATH=./qdrant_data`

Local Qdrant uses a filesystem lock. I confirmed this in this workspace when I tried to open `qdrant_data` from another Python process:

- it raised a lock error saying the storage folder was already accessed by another instance

This means:

- one process can hold the local Qdrant store at a time
- API-process refresh is safer than subprocess refresh when the API already owns the client
- standalone `scrape.py` refresh works best when no other process is actively holding the local Qdrant store

If you need true concurrent access, the correct fix is to run a real Qdrant server instead of local file mode.

---

## Verified State

I confirmed:

- `schemes.db` currently has 47 rows
- rows include `official_link`
- API import still succeeds after the changes
- the new shared refresh function imports correctly

---

## Files Changed

- [govassist/api/api.py](/home/dhinakaran/GovAssist-RAG/govassist/api/api.py)
- [govassist/api/db_utils.py](/home/dhinakaran/GovAssist-RAG/govassist/api/db_utils.py)
- [govassist/rag/graph_store.py](/home/dhinakaran/GovAssist-RAG/govassist/rag/graph_store.py)
- [scrape.py](/home/dhinakaran/GovAssist-RAG/scrape.py)

---

## Bottom Line

Your suspicion was right:

- scraped data was reaching SQLite
- `/scrape` API refreshed Qdrant afterward
- standalone `scrape.py` did not
- graph store was never using the scraped DB as a bootstrap source

That gap is now resolved in code by:

- making standalone scrape refresh indexes
- making `/scrape` use the same shared refresh logic
- letting the graph store build itself from `schemes.db`

If you want, the next step I can take is to run a controlled end-to-end scrape/index smoke test and then fix the remaining direct-response graph bug from the earlier audit.
