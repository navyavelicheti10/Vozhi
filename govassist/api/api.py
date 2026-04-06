import logging
import os
import shutil
import subprocess
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from govassist.agents.graph import govassist_graph
from govassist.agents import nodes as agent_nodes
from govassist.api.db import delete_session, get_all_sessions, get_session, init_db as init_chat_db, save_session
from govassist.api.db_utils import ingest_schemes_to_qdrant, init_db as init_schemes_db
from govassist.config import load_env_file
from govassist.integrations.sarvam import sarvam_client

load_env_file()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
TEMP_DIR = PROJECT_ROOT / "temp_uploads"

TEXT_MIME_TYPES = {"application/json"}
DOCUMENT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"}


class SaveSessionRequest(BaseModel):
    session_id: str
    title: str
    messages: list


class ChatJsonRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


def _ensure_temp_dir() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _detect_input_type(upload_file: UploadFile | None) -> str:
    if upload_file is None:
        return "text"

    suffix = Path(upload_file.filename or "").suffix.lower()
    content_type = (upload_file.content_type or "").lower()

    if content_type.startswith("audio/") or suffix in AUDIO_SUFFIXES:
        return "audio"
    if content_type.startswith("image/") or suffix == ".pdf" or suffix in DOCUMENT_SUFFIXES:
        return "document"

    raise HTTPException(status_code=400, detail="Unsupported file type. Use audio or image/pdf documents.")


async def _persist_upload(upload_file: UploadFile, session_id: str) -> Path:
    _ensure_temp_dir()
    suffix = Path(upload_file.filename or "").suffix
    safe_name = f"{session_id}_{upload_file.filename or 'upload'}".replace("/", "_")
    file_path = TEMP_DIR / safe_name
    if suffix and file_path.suffix != suffix:
        file_path = file_path.with_suffix(suffix)

    with file_path.open("wb") as target:
        shutil.copyfileobj(upload_file.file, target)

    return file_path


def _build_state(
    input_type: str,
    query_text: str,
    session_id: str,
    uploaded_file_path: Optional[Path] = None,
    transcribed_text: str = "",
) -> dict[str, Any]:
    initial_message = query_text or "Process the uploaded document and retrieve relevant government schemes."
    return {
        "messages": [HumanMessage(content=initial_message)],
        "input_type": input_type,
        "raw_query": query_text,
        "current_query": query_text,
        "transcribed_text": transcribed_text,
        "uploaded_file_path": str(uploaded_file_path) if uploaded_file_path else None,
        "user_profile": {},
        "documents_extracted": {},
        "retrieved_schemes": [],
        "synergy_schemes": [],
        "route": "",
        "rag_completed": False,
        "final_package": "",
        "confidence_score": 0.0,
        "citations": [],
        "sources": [],
    }


def _format_chat_response(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "input_type": state.get("input_type"),
        "query": state.get("current_query") or state.get("raw_query", ""),
        "raw_query": state.get("raw_query", ""),
        "transcribed_text": state.get("transcribed_text", ""),
        "answer": state.get("final_package", ""),
        "confidence": state.get("confidence_score", 0.0),
        "matches": state.get("retrieved_schemes", []),
        "synergy_schemes": state.get("synergy_schemes", []),
        "documents_extracted": state.get("documents_extracted", {}),
        "citations": state.get("citations", []),
        "sources": state.get("sources", []),
    }


def _merge_state(base_state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_state)
    merged.update(updates)
    return merged


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting GovAssist API")
    init_chat_db()
    init_schemes_db()
    _ensure_temp_dir()
    yield
    logger.info("Stopping GovAssist API")


app = FastAPI(
    title="GovAssist API",
    description="Minimal multimodal multi-agent RAG backend for government schemes.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_web_app() -> FileResponse:
    index_file = WEB_ROOT / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(index_file)


@app.get("/api/sessions")
def api_get_sessions():
    return get_all_sessions()


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    messages = get_session(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": messages}


@app.post("/api/sessions")
def api_save_session(req: SaveSessionRequest):
    save_session(req.session_id, req.title, req.messages)
    return {"status": "ok"}


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    session_id = "default-session"
    query_text = ""
    transcribed_text = ""
    uploaded_file: UploadFile | None = None
    uploaded_file_path: Path | None = None

    try:
        if any(text_type in content_type for text_type in TEXT_MIME_TYPES):
            payload = ChatJsonRequest.model_validate(await request.json())
            session_id = payload.session_id or session_id
            query_text = payload.query.strip()
            input_type = "text"
        elif "multipart/form-data" in content_type:
            form = await request.form()
            session_id = str(form.get("session_id") or session_id)
            query_text = str(form.get("query") or form.get("text") or "").strip()
            uploaded_file = form.get("file")
            if uploaded_file is None:
                raise HTTPException(status_code=400, detail="Multipart chat requests require a file.")

            input_type = _detect_input_type(uploaded_file)
            uploaded_file_path = await _persist_upload(uploaded_file, session_id)
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported content type. Use application/json for text or multipart/form-data for files.",
            )

        if input_type == "audio":
            if uploaded_file_path is None:
                raise HTTPException(status_code=400, detail="Audio upload is missing.")
            transcribed_text = sarvam_client.speech_to_text(str(uploaded_file_path)).strip()
            if not transcribed_text:
                raise HTTPException(status_code=502, detail="Audio transcription failed.")
            query_text = transcribed_text

        state = _build_state(
            input_type=input_type,
            query_text=query_text,
            session_id=session_id,
            uploaded_file_path=uploaded_file_path,
            transcribed_text=transcribed_text,
        )
        config = {"configurable": {"thread_id": session_id}}
        result_state = govassist_graph.invoke(state, config=config)
        return _format_chat_response(session_id, result_state)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"Chat orchestration failed: {exc}") from exc
    finally:
        if uploaded_file_path and uploaded_file_path.exists():
            uploaded_file_path.unlink(missing_ok=True)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    if not any(text_type in content_type for text_type in TEXT_MIME_TYPES):
        raise HTTPException(status_code=415, detail="Streaming chat currently supports application/json text input only.")

    payload = ChatJsonRequest.model_validate(await request.json())
    session_id = payload.session_id or "default-session"
    query_text = payload.query.strip()

    state = _build_state(
        input_type="text",
        query_text=query_text,
        session_id=session_id,
    )

    async def event_stream():
        try:
            routed_state = _merge_state(state, agent_nodes.main_agent(state))

            if routed_state.get("rag_completed") and routed_state.get("final_package"):
                final_payload = _format_chat_response(session_id, routed_state)
                yield json.dumps({"type": "chunk", "content": final_payload["answer"]}) + "\n"
                yield json.dumps({"type": "final", "data": final_payload}) + "\n"
                return

            llm_state = _merge_state(routed_state, agent_nodes.llm_agent(routed_state))

            if llm_state.get("rag_completed") and llm_state.get("final_package"):
                final_payload = _format_chat_response(session_id, llm_state)
                yield json.dumps({"type": "chunk", "content": final_payload["answer"]}) + "\n"
                yield json.dumps({"type": "final", "data": final_payload}) + "\n"
                return

            rag_state = _merge_state(llm_state, agent_nodes.rag_agent(llm_state))
            agent_nodes._ensure_llm()
            if agent_nodes.llm is None:
                raise RuntimeError("LLM client could not be initialized for streaming.")

            messages = agent_nodes.build_post_rag_messages(rag_state)
            chunks: list[str] = []
            async for chunk in agent_nodes.llm.astream(messages):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                else:
                    text = str(content or "")

                if not text:
                    continue

                chunks.append(text)
                yield json.dumps({"type": "chunk", "content": text}) + "\n"

            final_text = "".join(chunks).strip()
            metadata = agent_nodes.build_post_rag_metadata(rag_state)
            final_state = _merge_state(
                rag_state,
                {
                    "final_package": final_text,
                    "citations": metadata["citations"],
                    "sources": metadata["sources"],
                    "confidence_score": metadata["confidence_score"],
                },
            )
            yield json.dumps({"type": "final", "data": _format_chat_response(session_id, final_state)}) + "\n"
        except Exception as exc:
            logger.exception("Streaming chat request failed")
            yield json.dumps({"type": "error", "detail": f"Chat orchestration failed: {exc}"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/scrape")
def scrape(background_tasks: BackgroundTasks, ingest_after_scrape: bool = True):
    def run_scrape_pipeline() -> None:
        logger.info("Starting scraper pipeline")
        try:
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scrape.py")],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            logger.info("Scraper completed and SQLite was updated")

            if ingest_after_scrape:
                ingested = ingest_schemes_to_qdrant()
                logger.info("Qdrant ingestion completed for %s schemes", ingested)
        except Exception:
            logger.exception("Scrape pipeline failed")

    background_tasks.add_task(run_scrape_pipeline)
    return {
        "status": "accepted",
        "message": "Scraper started. Results will be stored in SQLite.",
        "ingest_after_scrape": ingest_after_scrape,
    }
