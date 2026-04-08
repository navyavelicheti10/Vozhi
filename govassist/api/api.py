import asyncio
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
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from govassist.agents.graph import govassist_graph
from govassist.api.db import delete_session, get_all_sessions, get_session, init_db as init_chat_db, save_session
from govassist.api.db_utils import init_db as init_schemes_db
from govassist.config import load_env_file
from govassist.integrations.sarvam import sarvam_client

load_env_file()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = PROJECT_ROOT / "temp_uploads"

TEXT_MIME_TYPES = {"application/json"}
DOCUMENT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"}


def _allowed_origins() -> list[str]:
    raw_value = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    if origins:
        return origins
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


class SaveSessionRequest(BaseModel):
    session_id: str
    title: str
    messages: list


class ChatJsonRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ParsedChatRequest(BaseModel):
    session_id: str
    input_type: str
    query_text: str
    raw_query: str = ""
    transcribed_text: str = ""
    uploaded_document_path: Optional[Path] = None
    uploaded_audio_path: Optional[Path] = None
    uploaded_document_name: str = ""
    query_language_code: str = "en-IN"
    response_language_code: str = "en-IN"


class TTSRequest(BaseModel):
    text: str
    language_code: str = "en-IN"
    speaker: str = "shubh"


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


def _combine_query_inputs(*values: str) -> str:
    normalized = [value.strip() for value in values if value and value.strip()]
    return "\n".join(normalized)


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
    raw_query: str = "",
    query_language_code: str = "en-IN",
    response_language_code: str = "en-IN",
) -> dict[str, Any]:
    initial_message = query_text or "Process the uploaded document and retrieve relevant government schemes."
    return {
        "messages": [HumanMessage(content=initial_message)],
        "input_type": input_type,
        "raw_query": raw_query or query_text,
        "current_query": query_text,
        "transcribed_text": transcribed_text,
        "uploaded_file_path": str(uploaded_file_path) if uploaded_file_path else None,
        "query_language_code": query_language_code,
        "response_language_code": response_language_code,
        "user_profile": {},
        "documents_extracted": {},
        "route": "",
        "retrieved_schemes": [],
        "synergy_schemes": [],
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
        "query_language_code": state.get("query_language_code", "en-IN"),
        "response_language_code": state.get("response_language_code", "en-IN"),
        "answer": state.get("final_package", ""),
        "confidence": state.get("confidence_score", 0.0),
        "matches": state.get("retrieved_schemes", []),
        "synergy_schemes": state.get("synergy_schemes", []),
        "documents_extracted": state.get("documents_extracted", {}),
        "citations": state.get("citations", []),
        "sources": state.get("sources", []),
    }


def _iter_answer_chunks(answer: str, chunk_size: int = 120) -> list[str]:
    normalized = (answer or "").strip()
    if not normalized:
        return []

    words = normalized.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current + " ")
        current = word

    if current:
        chunks.append(current)
    return chunks


async def _parse_chat_request(request: Request) -> ParsedChatRequest:
    content_type = (request.headers.get("content-type") or "").lower()
    session_id = "default-session"
    query_text = ""
    raw_query = ""
    transcribed_text = ""
    uploaded_file_path: Path | None = None
    uploaded_document_path: Path | None = None
    uploaded_audio_path: Path | None = None
    uploaded_document_name = ""
    query_language_code = "en-IN"
    response_language_code = "en-IN"
    input_type = "text"

    try:
        if any(text_type in content_type for text_type in TEXT_MIME_TYPES):
            payload = ChatJsonRequest.model_validate(await request.json())
            session_id = payload.session_id or session_id
            raw_query = payload.query.strip()
            query_text = raw_query
            input_type = "text"
        elif "multipart/form-data" in content_type:
            form = await request.form()
            session_id = str(form.get("session_id") or session_id)
            raw_query = _combine_query_inputs(
                str(form.get("query") or "").strip(),
                str(form.get("text") or "").strip(),
            )
            query_text = raw_query
            primary_file = form.get("file")
            secondary_audio = form.get("audio_file") or form.get("voice_file")

            if primary_file is None and secondary_audio is None:
                raise HTTPException(status_code=400, detail="Multipart chat requests require a document or audio file.")

            primary_type = _detect_input_type(primary_file) if primary_file is not None else ""
            secondary_type = _detect_input_type(secondary_audio) if secondary_audio is not None else ""

            if primary_type == "document":
                uploaded_document_path = await _persist_upload(primary_file, session_id)
                uploaded_document_name = primary_file.filename or uploaded_document_path.name
            elif primary_type == "audio":
                uploaded_audio_path = await _persist_upload(primary_file, session_id)
            elif primary_file is not None:
                raise HTTPException(status_code=400, detail="Unsupported primary upload type.")

            if secondary_type == "audio":
                uploaded_audio_path = await _persist_upload(secondary_audio, session_id)
            elif secondary_audio is not None:
                raise HTTPException(status_code=400, detail="The extra upload must be an audio file.")

            if uploaded_document_path is not None:
                uploaded_file_path = uploaded_document_path
                input_type = "document"
            elif uploaded_audio_path is not None:
                uploaded_file_path = uploaded_audio_path
                input_type = "audio"
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported content type. Use application/json for text or multipart/form-data for files.",
            )

        if uploaded_audio_path is not None:
            stt_payload = sarvam_client.speech_to_text_with_metadata(str(uploaded_audio_path))
            transcribed_text = stt_payload.get("transcript", "").strip()
            if not transcribed_text:
                raise HTTPException(status_code=502, detail="Audio transcription failed.")
            query_language_code = stt_payload.get("language_code", "en-IN")
            response_language_code = query_language_code
            raw_query = _combine_query_inputs(raw_query, transcribed_text)
            if query_language_code != "en-IN":
                translated = sarvam_client.translate_text(
                    text=transcribed_text,
                    source_language_code=query_language_code,
                    target_language_code="en-IN",
                )
                translated_stt = translated.get("translated_text", transcribed_text).strip() or transcribed_text
            else:
                translated_stt = transcribed_text
            query_text = _combine_query_inputs(query_text, translated_stt)
        elif query_text:
            translated = sarvam_client.translate_text(
                text=query_text,
                source_language_code="auto",
                target_language_code="en-IN",
            )
            query_language_code = translated.get("source_language_code", "en-IN")
            response_language_code = query_language_code
            if query_language_code != "en-IN":
                query_text = translated.get("translated_text", query_text).strip() or query_text
    except Exception:
        for temp_path in (uploaded_document_path, uploaded_audio_path, uploaded_file_path):
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise

    return ParsedChatRequest(
        session_id=session_id,
        input_type=input_type,
        query_text=query_text,
        raw_query=raw_query,
        transcribed_text=transcribed_text,
        uploaded_document_path=uploaded_document_path,
        uploaded_audio_path=uploaded_audio_path,
        uploaded_document_name=uploaded_document_name,
        query_language_code=query_language_code,
        response_language_code=response_language_code,
    )


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
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_root() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>GovAssist API</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f4; color: #18181b; }
              .wrap { max-width: 760px; margin: 48px auto; padding: 0 20px; }
              .card { background: white; border: 1px solid #e4e4e7; border-radius: 16px; padding: 24px; }
              h1 { margin: 0 0 8px; }
              p, li { line-height: 1.5; }
              code { background: #f4f4f5; padding: 2px 6px; border-radius: 6px; }
              a { color: #2563eb; text-decoration: none; }
            </style>
          </head>
          <body>
            <div class="wrap">
              <div class="card">
                <h1>GovAssist API</h1>
                <p>This backend serves the GovAssist orchestration and chat APIs.</p>
                <p>The active web UI is the Next.js app in <code>frontend/</code>, typically run on <code>http://localhost:3000</code> during development.</p>
                <ul>
                  <li><a href="/docs">Open API Docs</a></li>
                  <li><a href="/health">Health Check</a></li>
                  <li><code>POST /chat</code> for blocking text/audio/document chat</li>
                  <li><code>POST /chat/stream</code> for streamed text/audio/document chat</li>
                </ul>
              </div>
            </div>
          </body>
        </html>
        """
    )


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
    parsed_request: ParsedChatRequest | None = None
    try:
        parsed_request = await _parse_chat_request(request)

        state = _build_state(
            input_type=parsed_request.input_type,
            query_text=parsed_request.query_text,
            session_id=parsed_request.session_id,
            uploaded_file_path=parsed_request.uploaded_document_path,
            transcribed_text=parsed_request.transcribed_text,
            raw_query=parsed_request.raw_query,
            query_language_code=parsed_request.query_language_code,
            response_language_code=parsed_request.response_language_code,
        )
        config = {"configurable": {"thread_id": parsed_request.session_id}}
        result_state = govassist_graph.invoke(state, config=config)
        return _format_chat_response(parsed_request.session_id, result_state)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"Chat orchestration failed: {exc}") from exc
    finally:
        if parsed_request:
            for temp_path in (parsed_request.uploaded_document_path, parsed_request.uploaded_audio_path):
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    parsed_request = await _parse_chat_request(request)

    state = _build_state(
        input_type=parsed_request.input_type,
        query_text=parsed_request.query_text,
        session_id=parsed_request.session_id,
        uploaded_file_path=parsed_request.uploaded_document_path,
        transcribed_text=parsed_request.transcribed_text,
        raw_query=parsed_request.raw_query,
        query_language_code=parsed_request.query_language_code,
        response_language_code=parsed_request.response_language_code,
    )

    async def event_stream():
        try:
            if parsed_request.transcribed_text:
                yield json.dumps({"type": "transcript", "content": parsed_request.transcribed_text}) + "\n"
            config = {"configurable": {"thread_id": parsed_request.session_id}}
            result_state = await asyncio.to_thread(govassist_graph.invoke, state, config)
            final_payload = _format_chat_response(parsed_request.session_id, result_state)

            for chunk in _iter_answer_chunks(final_payload["answer"]):
                yield json.dumps({"type": "chunk", "content": chunk}) + "\n"

            yield json.dumps({"type": "final", "data": final_payload}) + "\n"
        except Exception as exc:
            logger.exception("Streaming chat request failed")
            yield json.dumps({"type": "error", "detail": f"Chat orchestration failed: {exc}"}) + "\n"
        finally:
            for temp_path in (parsed_request.uploaded_document_path, parsed_request.uploaded_audio_path):
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/tts")
def text_to_speech(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for TTS.")

    audio_bytes = sarvam_client.text_to_speech_bytes(
        text=text,
        language_code=req.language_code,
        speaker=req.speaker,
    )
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="TTS generation failed.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="vozhi-response.wav"'},
    )


@app.post("/scrape")
def scrape(background_tasks: BackgroundTasks, ingest_after_scrape: bool = True):
    def run_scrape_pipeline() -> None:
        logger.info("[SCRAPE] Starting scraper pipeline (subprocess)")
        logger.info("[SCRAPE] Script: %s", str(PROJECT_ROOT / 'scrape.py'))
        logger.info("[SCRAPE] AUTO_INGEST=%s", 'true' if ingest_after_scrape else 'false')
        try:
            proc = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scrape.py")],
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "AUTO_INGEST": "true" if ingest_after_scrape else "false"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout
                text=True,
                bufsize=1,
            )

            # Stream subprocess output line-by-line into the parent logger
            scraper_logger = logging.getLogger("govassist.ingestion.scraper")
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    scraper_logger.info("%s", stripped)

            proc.wait()
            if proc.returncode == 0:
                logger.info("[SCRAPE] Scraper pipeline completed successfully (exit code 0)")
            else:
                logger.error("[SCRAPE] Scraper pipeline exited with code %d", proc.returncode)
        except Exception:
            logger.exception("[SCRAPE] Scrape pipeline launcher failed")

    background_tasks.add_task(run_scrape_pipeline)
    return {
        "status": "accepted",
        "message": "Scraper started. Results will be stored in SQLite.",
        "ingest_after_scrape": ingest_after_scrape,
    }


async def _parse_twilio_to_chat_request(msg_data: dict) -> ParsedChatRequest:
    """Convert a parsed Twilio message dict into a ParsedChatRequest using the
    exact same logic as _parse_chat_request, so WhatsApp and web share one pipeline."""
    from govassist.integrations.twilio import twilio_client

    session_id   = msg_data.get("from", "default")
    text         = (msg_data.get("body") or "").strip()
    media_url    = msg_data.get("media_url", "")
    media_type   = (msg_data.get("media_type") or "").lower()

    input_type               = "text"
    query_text               = text
    raw_query                = text
    transcribed_text         = ""
    query_language_code      = "en-IN"
    response_language_code   = "en-IN"
    uploaded_document_path: Path | None = None
    uploaded_audio_path:    Path | None = None
    uploaded_file_path:     Path | None = None
    uploaded_document_name  = ""

    # ── 1. Download media if present ────────────────────────────────────────
    if media_url:
        _ensure_temp_dir()
        # Derive a file extension from the content-type so OCR/PDF routing works correctly
        ext = (
            ".pdf"  if media_type == "application/pdf" else
            ".jpg"  if "jpeg" in media_type else
            ".png"  if "png"  in media_type else
            ".webp" if "webp" in media_type else
            ".ogg"  if "ogg"  in media_type else
            ".mp3"  if "mp3"  in media_type else
            ".wav"  if "wav"  in media_type else
            ".m4a"  if "m4a"  in media_type else
            ""
        )
        safe_id  = session_id.replace("+", "").replace(":", "")
        filename = f"{safe_id}_twilio_media{ext}"
        save_path = str(TEMP_DIR / filename)

        dl_path = twilio_client.download_media(media_url, save_path)
        logger.info("[TWILIO] Downloaded media → %s (type=%s)", dl_path, media_type)

        if dl_path:
            dl_file = Path(dl_path)
            if media_type.startswith("audio/"):
                # ── Audio: STT → translate → treat like /chat audio upload ──
                uploaded_audio_path = dl_file
                uploaded_file_path  = dl_file
                input_type          = "audio"

                stt_payload      = sarvam_client.speech_to_text_with_metadata(dl_path)
                transcribed_text = stt_payload.get("transcript", "").strip()
                if not transcribed_text:
                    raise HTTPException(status_code=502, detail="Audio transcription failed.")

                query_language_code    = stt_payload.get("language_code", "en-IN")
                response_language_code = query_language_code
                raw_query              = _combine_query_inputs(text, transcribed_text)

                if query_language_code != "en-IN":
                    translated   = sarvam_client.translate_text(
                        text=transcribed_text,
                        source_language_code=query_language_code,
                        target_language_code="en-IN",
                    )
                    translated_stt = translated.get("translated_text", transcribed_text).strip() or transcribed_text
                else:
                    translated_stt = transcribed_text

                query_text = _combine_query_inputs(text, translated_stt)

            elif media_type.startswith("image/") or media_type == "application/pdf":
                # ── Image/PDF: treat like /chat document upload ───────────────
                uploaded_document_path = dl_file
                uploaded_document_name = dl_file.name
                uploaded_file_path     = dl_file
                input_type             = "document"
                query_text             = text  # any caption the user typed

            else:
                logger.warning("[TWILIO] Unsupported media type '%s'; treating as text", media_type)

    # ── 2. Translate text query (same as _parse_chat_request) ───────────────
    if not media_url and query_text:
        translated = sarvam_client.translate_text(
            text=query_text,
            source_language_code="auto",
            target_language_code="en-IN",
        )
        query_language_code    = translated.get("source_language_code", "en-IN")
        response_language_code = query_language_code
        if query_language_code != "en-IN":
            query_text = translated.get("translated_text", query_text).strip() or query_text

    logger.info(
        "[TWILIO] Parsed request — session=%s input_type=%s lang=%s",
        session_id, input_type, response_language_code,
    )

    return ParsedChatRequest(
        session_id=session_id,
        input_type=input_type,
        query_text=query_text,
        raw_query=raw_query,
        transcribed_text=transcribed_text,
        uploaded_document_path=uploaded_document_path,
        uploaded_audio_path=uploaded_audio_path,
        uploaded_document_name=uploaded_document_name,
        query_language_code=query_language_code,
        response_language_code=response_language_code,
    )


async def _process_twilio_message_bg(msg_data: dict) -> None:
    """Background task: runs the full GovAssist pipeline for a WhatsApp message.
    Uses the exact same orchestration path as POST /chat.
    """
    from govassist.integrations.twilio import twilio_client

    user_number   = msg_data.get("from", "")
    parsed_request: ParsedChatRequest | None = None

    try:
        logger.info("[TWILIO] Processing message from %s", user_number)

        # ── Step 1: Parse (same helpers as web /chat) ────────────────────────
        parsed_request = await _parse_twilio_to_chat_request(msg_data)

        # ── Step 2: Build LangGraph state (identical to web chat) ────────────
        state = _build_state(
            input_type=parsed_request.input_type,
            query_text=parsed_request.query_text,
            session_id=parsed_request.session_id,
            uploaded_file_path=parsed_request.uploaded_document_path,
            transcribed_text=parsed_request.transcribed_text,
            raw_query=parsed_request.raw_query,
            query_language_code=parsed_request.query_language_code,
            response_language_code=parsed_request.response_language_code,
        )

        # ── Step 3: Run LangGraph agents (identical to web chat) ─────────────
        config       = {"configurable": {"thread_id": parsed_request.session_id}}
        logger.info("[TWILIO] Invoking govassist_graph for session=%s", parsed_request.session_id)
        result_state = await asyncio.to_thread(govassist_graph.invoke, state, config)

        # ── Step 4: Format response (identical to web chat) ──────────────────
        final_payload = _format_chat_response(parsed_request.session_id, result_state)
        answer        = final_payload.get("answer", "").strip()

        if not answer:
            answer = "I could not find a matching scheme. Please try a different query."

        logger.info(
            "[TWILIO] Response ready for %s — %d chars, confidence=%.2f",
            user_number, len(answer), final_payload.get("confidence", 0.0),
        )

        # ── Step 5: Send WhatsApp reply ───────────────────────────────────────
        twilio_client.send_proactive_message(user_number, answer)

    except HTTPException as exc:
        logger.warning("[TWILIO] Chat parsing rejected for %s: %s", user_number, exc.detail)
        twilio_client.send_proactive_message(
            user_number,
            f"Sorry, I couldn't process your message: {exc.detail}"
        )
    except Exception:
        logger.exception("[TWILIO] Pipeline failed for %s", user_number)
        twilio_client.send_proactive_message(
            user_number,
            "Sorry, I am having trouble connecting to the Vozhi Orchestrator right now. Please try again.",
        )
    finally:
        # Clean up temp files exactly as the web chat endpoint does
        if parsed_request:
            for temp_path in (parsed_request.uploaded_document_path, parsed_request.uploaded_audio_path):
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                    logger.debug("[TWILIO] Cleaned up temp file: %s", temp_path)


@app.post("/webhook/twilio")
async def twilio_webhook(request: Request, background_tasks: BackgroundTasks):
    """Twilio WhatsApp webhook.
    Returns an empty TwiML immediately (Twilio's 15-second timeout),
    then processes the full GovAssist pipeline in the background.
    """
    from govassist.integrations.twilio import twilio_client

    form     = await request.form()
    msg_data = twilio_client.parse_incoming_message(dict(form))

    logger.info(
        "[TWILIO] Webhook received — from=%s has_media=%s",
        msg_data.get("from"), bool(msg_data.get("media_url")),
    )

    # Acknowledge immediately so Twilio doesn't time out
    fast_twiml = twilio_client.generate_twiml_response("")
    background_tasks.add_task(_process_twilio_message_bg, msg_data)

    return Response(content=fast_twiml, media_type="application/xml")
