# GovAssist-RAG

GovAssist is a multimodal government-schemes assistant for Indian public-benefit discovery. It combines:

- FastAPI for the backend API
- LangGraph for orchestration
- Groq for query refinement and final answer generation
- Sarvam for speech-to-text
- EasyOCR and `pypdf` for document extraction
- SQLite for scheme storage and chat history
- Qdrant for vector retrieval
- Next.js for the active frontend

## Current Runtime Architecture

The active request path is:

1. User sends text, audio, or a document.
2. FastAPI builds an `AgentState`.
3. LangGraph routes through:
   - document extraction when a document is uploaded
   - LLM query normalization
   - RAG retrieval
   - final LLM answer synthesis
4. The frontend renders the final answer, with streaming supported for text, audio, and document chat.

Main runtime files:

- [main.py](/home/slakshman2004/GovAssist-RAG/main.py)
- [govassist/api/api.py](/home/slakshman2004/GovAssist-RAG/govassist/api/api.py)
- [govassist/agents/graph.py](/home/slakshman2004/GovAssist-RAG/govassist/agents/graph.py)
- [govassist/agents/nodes.py](/home/slakshman2004/GovAssist-RAG/govassist/agents/nodes.py)
- [frontend/src/app/page.tsx](/home/slakshman2004/GovAssist-RAG/frontend/src/app/page.tsx)

## Features Implemented

- Text chat for government-scheme discovery
- Small-talk handling for greeting, thanks, and goodbye messages
- Streaming assistant responses
- Audio upload and in-browser microphone recording
- Speech-to-text using Sarvam before Groq inference
- On-demand text-to-speech for assistant responses using Sarvam
- PDF and image document extraction
- Semantic retrieval with Qdrant
- SQLite fallback retrieval when vector search fails
- Graph-based synergy lookup
- Chat history persistence in SQLite
- Scheme ingestion into SQLite and Qdrant

## API

### Core endpoints

- `GET /`
  - lightweight API landing page
- `GET /health`
  - health check
- `POST /chat`
  - blocking chat for text, audio, and document requests
- `POST /chat/stream`
  - streamed chat for text, audio, and document requests
- `POST /tts`
  - synthesize assistant text to playable audio
- `GET /api/sessions`
  - list chat sessions
- `GET /api/sessions/{session_id}`
  - fetch one session’s saved messages
- `POST /api/sessions`
  - save a session snapshot
- `DELETE /api/sessions/{session_id}`
  - delete a session
- `POST /scrape`
  - run the scraping pipeline in the background

### Input modes

- `application/json`
  - text chat
- `multipart/form-data`
  - audio file chat
  - document chat with optional query text

## Frontend

The active frontend is the Next.js app in [frontend/](/home/slakshman2004/GovAssist-RAG/frontend).

It currently supports:

- chat session history
- streamed responses
- microphone recording in the chat composer
- document upload in the chat composer
- theme toggle
- WhatsApp onboarding modal

Run it with:

```bash
cd frontend
npm install
npm run dev
```

## Backend Setup

Create a `.env` file in the repo root:

```env
GROQ_API_KEY=
SARVAM_API_KEY=
LOG_LEVEL=INFO
SCHEMES_FILE=scheme.json
AUTO_INGEST=true
FORCE_RECREATE_COLLECTION=false
QDRANT_MODE=local
QDRANT_LOCAL_PATH=./qdrant_data
API_BASE_URL=http://127.0.0.1:8000
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

### Required vs optional credentials

- `GROQ_API_KEY`
  - required for the real LLM path
- `SARVAM_API_KEY`
  - optional, but required for real speech-to-text and text-to-speech
- `TWILIO_*`
  - optional, only needed for WhatsApp integration

If `SARVAM_API_KEY` is missing, the current Sarvam client returns a mock transcription for audio requests.

## Running the backend

```bash
source .venv/bin/activate
python3 main.py
```

The backend runs at `http://127.0.0.1:8000`.

## Data Stores

- [govassist/api/schemes.db](/home/slakshman2004/GovAssist-RAG/govassist/api/schemes.db)
  - normalized scheme records in SQLite
- [govassist/api/chat_history.db](/home/slakshman2004/GovAssist-RAG/govassist/api/chat_history.db)
  - chat session snapshots in SQLite
- `qdrant_data/`
  - local vector database and graph artifacts
- [data/raw/scheme.json](/home/slakshman2004/GovAssist-RAG/data/raw/scheme.json)
  - scraped scheme source data

## Legacy Modules

There are older modules still present for reference:

- [govassist/rag/pipeline.py](/home/slakshman2004/GovAssist-RAG/govassist/rag/pipeline.py)
- [govassist/rag/llm.py](/home/slakshman2004/GovAssist-RAG/govassist/rag/llm.py)
- [govassist/storage/checkpointer.py](/home/slakshman2004/GovAssist-RAG/govassist/storage/checkpointer.py)

They are not on the active FastAPI + LangGraph runtime path.

## Notes

- The backend root no longer attempts to serve a stale `web/index.html`.
- The active UI is the Next.js app, not the older Vue-based description from earlier project docs.
- Streaming now supports text, audio, and document requests through the same API route.
