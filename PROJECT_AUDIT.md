# GovAssist Project Audit

## Scope

This audit covers the active backend, frontend, retrieval stack, ingestion flow, persistence layer, and obviously stale or dead code in the repository.

Review date: 2026-04-07 UTC

Workspace inspected:

- `main.py`
- `govassist/api/*`
- `govassist/agents/*`
- `govassist/rag/*`
- `govassist/integrations/*`
- `govassist/ingestion/*`
- `frontend/src/app/page.tsx`
- root dependency and readme files

Verification performed:

- Imported the API successfully with `.venv/bin/python -c "from govassist.api.api import app; print('api-import-ok')"`
- Ran `.venv/bin/python test_graph.py`
- Observed a real logic defect in the graph response for `"hello"`

---

## 1. Project Context

### What the project is

GovAssist is a multimodal government-scheme assistant with:

- FastAPI backend
- LangGraph orchestration for the blocking `/chat` path
- manual streaming orchestration for `/chat/stream`
- Groq-backed LLM stages
- Sarvam speech and translation integration
- SQLite for scheme storage and chat session storage
- Qdrant for semantic retrieval
- Next.js frontend in `frontend/`

### Active request flow

#### Blocking path

`frontend -> /chat -> govassist_graph -> optional DocumentAgent -> MainAgent -> RAGAgent/LLMAgent -> response`

Relevant files:

- `govassist/api/api.py`
- `govassist/agents/graph.py`
- `govassist/agents/nodes.py`

#### Streaming path

`frontend -> /chat/stream -> _parse_chat_request -> manual calls to main_agent / llm_agent / rag_agent -> streamed chunks`

Important note:

The streaming path is not using the same graph wiring as the blocking path. This is the source of one of the biggest functional mismatches in the repo.

---

## 2. Database Usage

### Active databases

#### Chat history DB

File:

- `govassist/api/chat_history.db`

Used by:

- `govassist/api/db.py`
- session endpoints in `govassist/api/api.py`
- frontend chat history persistence in `frontend/src/app/page.tsx`

Storage pattern:

- whole message arrays are serialized as JSON and saved per session row

Observations:

- simple and workable for small scale
- synchronous sqlite access in request handlers
- no WAL mode, no pooling, no retry strategy
- every message update from the frontend writes the full session again

#### Scheme DB

File:

- `govassist/api/schemes.db`

Used by:

- `govassist/api/db_utils.py`
- scraper writes to it
- RAG fallback reads from it
- Qdrant ingestion is built from it

Storage pattern:

- normalized fields plus a `raw_json` copy

Observations:

- it is the canonical structured source for retrieval fallback and vector ingestion
- current upsert logic is application-side, not enforced by database uniqueness

### DB design concerns

1. `schemes` has no `UNIQUE` constraint on `scheme_name`, but `upsert_scheme()` assumes it is unique by doing `SELECT` then `UPDATE/INSERT`.
   Evidence: `govassist/api/db_utils.py:33-44`, `govassist/api/db_utils.py:73-132`
   Risk: concurrent ingests or repeated scraper runs can create duplicate logical records.

2. chat sessions are rewritten on every `messages` state change from the frontend.
   Evidence: `frontend/src/app/page.tsx:148-169`
   Risk: unnecessary DB churn and avoidable write amplification.

3. both SQLite DBs live under the source tree and are present in the repo workspace.
   Evidence: `govassist/api/chat_history.db`, `govassist/api/schemes.db`
   Risk: environment-specific state leaks into development and review workflows.

---

## 3. Confirmed High-Risk Findings

### F1. Streaming document chat is wired incorrectly and skips `DocumentAgent`

Severity: Critical

Why it matters:

- the frontend uses `/chat/stream`
- uploaded documents are therefore handled by the streaming path
- the streaming path never runs `document_agent`
- document extraction context is missing for streamed document requests

Evidence:

- blocking path uses the graph: `govassist/api/api.py:337-355`
- graph routes documents through `DocumentAgent`: `govassist/agents/graph.py:8-16`, `govassist/agents/graph.py:38-42`
- streaming path manually calls only `main_agent`, `llm_agent`, and `rag_agent`: `govassist/api/api.py:381-407`

Impact:

- document uploads in the active frontend do not actually go through the document extraction node
- the UI claims document intelligence, but the streaming implementation bypasses the only extraction stage

Recommended fix:

- either run the graph for `/chat/stream` too, or explicitly call `document_agent` before `main_agent` when `input_type == "document"`

### F2. Graph routing overrides direct small-talk responses and produces the wrong answer

Severity: Critical

Why it matters:

- direct responses are supposed to end immediately
- instead, the graph always routes `route == "respond"` to `LLMAgent`
- when `rag_completed=True`, `llm_agent()` switches into post-RAG mode and can replace the already-correct reply

Evidence:

- `main_agent()` marks greetings/out-of-scope/empty input as `route="respond"` and `rag_completed=True`: `govassist/agents/nodes.py:383-428`
- graph still routes any `respond` state to `LLMAgent`: `govassist/agents/graph.py:18-21`
- `llm_agent()` treats `rag_completed=True` as post-RAG synthesis: `govassist/agents/nodes.py:461-466`

Verified behavior:

- `.venv/bin/python test_graph.py` returned `SUCCESS`
- for input `"hello"`, final output was the fallback scheme-miss answer instead of the small-talk greeting

Impact:

- the graph’s direct-answer path is logically broken
- greetings and other non-RAG responses can be replaced by irrelevant fallback text

Recommended fix:

- in `route_after_main`, exit when `rag_completed` is already true
- or make `llm_agent()` respect an already-populated `final_package` before post-RAG synthesis

### F3. The frontend creates a brand-new chat session on every initial load

Severity: High

Evidence:

- history is fetched: `frontend/src/app/page.tsx:114-121`
- immediately after loading history, `createNewSession(loadedHistory)` is always called: `frontend/src/app/page.tsx:122`

Impact:

- every refresh inserts a new `"New Conversation"` row
- DB and sidebar history will accumulate noise quickly
- users cannot naturally resume the latest existing conversation on reload

Recommended fix:

- only create a new session when no session exists or when the user explicitly clicks new chat

### F4. Frontend is hardcoded to `http://127.0.0.1:8000`

Severity: High

Evidence:

- repeated hardcoded fetch URLs in `frontend/src/app/page.tsx:103`, `117`, `163`, `208`, `220`, `270`, `466`

Impact:

- breaks deployment behind another hostname, port, proxy, or HTTPS origin
- prevents same-origin deployment without code edits
- makes preview/staging environments brittle

Recommended fix:

- use a single configurable base URL from env, or use relative URLs when frontend and backend are proxied together

### F5. CORS configuration is internally inconsistent

Severity: High

Evidence:

- `allow_origins=["*"]` with `allow_credentials=True`: `govassist/api/api.py:257-263`

Impact:

- wildcard origins with credentials are not a valid browser CORS combination
- can cause confusing frontend failures once cookies/auth headers are introduced

Recommended fix:

- use explicit allowed origins when credentials are enabled

### F6. `requirements.txt` does not match the actual runtime dependencies

Severity: High

Evidence:

- minimal `requirements.txt`: `requirements.txt:1-9`
- actual runtime deps listed in `pyproject.toml`: `pyproject.toml:7-27`
- system `python3` import failed on `langchain_core` during checks

Impact:

- installs from `requirements.txt` will produce a broken runtime
- onboarding and deployment can fail depending on which install path is used

Missing examples from `requirements.txt`:

- `langchain-core`
- `langchain-groq`
- `langgraph`
- `llama-index-*`
- `python-multipart`
- `pypdf`
- `easyocr`
- `twilio`

Recommended fix:

- choose one source of truth for Python deps and generate the other from it

---

## 4. Medium-Risk Logic and Wiring Issues

### M1. Document-intelligence claims in UI are stronger than the active implementation

Evidence:

- UI says document uploads use EasyOCR and Llama 3.2 Vision: `frontend/src/app/page.tsx:675-680`
- active document path in `document_agent()` uses PDF text extraction or EasyOCR plus a text-only LLM parsing step: `govassist/agents/nodes.py:492-539`
- `govassist/integrations/vision.py` exists but is not wired into the active runtime

Impact:

- the UI currently overstates the active document-processing capability

### M2. WhatsApp onboarding data is hardcoded in the UI

Evidence:

- join code and number are hardcoded in `frontend/src/app/page.tsx:852-859`

Impact:

- if the Twilio sandbox code changes, the UI becomes wrong immediately
- this also makes non-sandbox production rollout harder

### M3. TTS fallback behavior is inconsistent with STT fallback behavior

Evidence:

- STT returns a mock transcript when `SARVAM_API_KEY` is missing: `govassist/integrations/sarvam.py:145-152`
- TTS returns empty bytes when `SARVAM_API_KEY` is missing: `govassist/integrations/sarvam.py:251-256`
- `/tts` turns empty bytes into HTTP 502: `govassist/api/api.py:447-465`

Impact:

- audio input appears to degrade gracefully
- audio output does not
- README wording can mislead people into expecting similar graceful fallback for TTS

### M4. Scraper config constant `OUTPUT_FILE` is unused

Evidence:

- declared in `govassist/ingestion/scraper.py:6`
- final write path only inserts into SQLite: `govassist/ingestion/scraper.py:341-347`

Impact:

- configuration is misleading
- suggests a JSON export path that the active code does not use

### M5. Scraper contains many debug prints and broad bare `except` blocks

Evidence:

- debug prints: `govassist/ingestion/scraper.py:263-274`, `303`, `326`, `345-347`
- bare `except`: `govassist/ingestion/scraper.py:244-252`, `280-287`

Impact:

- noisy production logs
- hidden failures during extraction
- harder debugging because exceptions are swallowed

### M6. Frontend save strategy can race against load/session switching

Evidence:

- save-on-any-message-change effect: `frontend/src/app/page.tsx:148-169`
- loadChat sets session first, then asynchronously loads messages: `frontend/src/app/page.tsx:204-215`

Impact:

- there is a window where a different session id and stale messages can coexist
- this can produce accidental overwrites in some timing scenarios

---

## 5. Dead Code, Legacy Code, and Stale Artifacts

### Legacy modules not on the active runtime path

These are documented as legacy or effectively unused in the active app flow:

- `govassist/rag/pipeline.py`
- `govassist/rag/llm.py`
- `govassist/storage/checkpointer.py`
- `govassist/integrations/vision.py`
- `govassist/integrations/twilio.py`

Notes:

- `vision.py` is especially important because the frontend copy implies capabilities that this inactive module provides, but the runtime does not use it.
- `twilio.py` appears to be a prepared integration module, but there is no active FastAPI webhook wiring for it in the reviewed code.

### Stale generated artifacts in repo workspace

- `govassist/__pycache__/`
- `govassist/agents/__pycache__/`
- `govassist/api/__pycache__/`
- `govassist/ingestion/__pycache__/`
- `govassist/integrations/__pycache__/`
- `govassist/rag/__pycache__/`

These are not source and should not usually live in the repository tree.

### Checked-in mutable data

- `govassist/api/chat_history.db`
- `govassist/api/schemes.db`

These are environment-generated state files, not stable source artifacts.

---

## 6. Hardcoded and Environment-Coupled Parts

### Backend / runtime

- local host/port defaults in `main.py`
- CORS open to all origins in `govassist/api/api.py`
- Qdrant local path default `./qdrant_data` in `govassist/rag/vector_store.py`
- graph store persist dir `qdrant_data/graph_store` in `govassist/rag/graph_store.py`

### Frontend

- hardcoded backend origin `http://127.0.0.1:8000`
- hardcoded WhatsApp sandbox number and join code
- UI copy assumes capabilities that are only partly active

### Scraper

- category URLs are hardcoded in `govassist/ingestion/scraper.py`
- sleeps/timeouts are hardcoded

These are not always wrong, but they should be deliberate config rather than hidden assumptions.

---

## 7. Architecture Notes

### Good parts

- backend structure is understandable
- state shape is explicit via `AgentState`
- retrieval has both Qdrant and SQLite fallback
- upload temp files are cleaned up in both blocking and streaming paths
- README already distinguishes active runtime from legacy modules better than many repos do

### Structural weakness

The project currently has two execution models for chat:

1. graph-driven blocking path
2. hand-wired streaming path

That split is causing behavior drift:

- different node wiring
- document extraction mismatch
- higher chance of future regressions

This is the single biggest maintainability risk in the current architecture.

---

## 8. Recommended Cleanup Order

1. Fix graph routing so direct responses end correctly.
2. Make `/chat/stream` reuse the same graph behavior as `/chat`.
3. Stop auto-creating a new session on every page load.
4. Replace hardcoded frontend backend URLs with env-driven config.
5. Reconcile `requirements.txt` and `pyproject.toml`.
6. Add DB-level uniqueness for schemes if `scheme_name` is intended to be unique.
7. Remove stale artifacts from source control: `__pycache__`, `.db` files if they are not intentionally versioned.
8. Decide whether `vision.py` and `twilio.py` are future roadmap modules or dead code; either wire them properly or remove/archive them.
9. Reduce broad exception swallowing in the scraper and replace debug prints with logging.

---

## 9. Bottom Line

The repo has a workable core idea and a mostly coherent structure, but the active user experience is currently undermined by two major runtime wiring bugs:

- the graph direct-response path is logically broken
- the streaming path bypasses document extraction

After those, the next most important problems are deployment rigidity from hardcoded frontend URLs, uncontrolled session creation, and dependency drift between `requirements.txt` and `pyproject.toml`.

If needed, this audit can be turned into a concrete fix checklist or implemented directly in a follow-up pass.
