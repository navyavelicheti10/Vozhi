import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from govassist.config import load_env_file
from govassist.agents.graph import vozhi_orchestrator
from langchain_core.messages import HumanMessage
from fastapi import Request, BackgroundTasks
from pydantic import BaseModel, Field

from govassist.api.db import init_db, get_all_sessions, get_session, save_session

class SaveSessionRequest(BaseModel):
    session_id: str
    title: str
    messages: list

# Ensure env vars are loaded
load_env_file()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question about schemes")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of results to retrieve")
    session_id: Optional[str] = Field(
        default=None,
        description="Pass the same session_id on follow-up messages to continue the chat",
    )


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    matches: list


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Vozhi API (LangGraph Edition)")
    init_db()
    yield
    logger.info("Shutting down Vozhi API")


app = FastAPI(
    title="Government Schemes Assistant API",
    description="RAG backend using FastAPI, Qdrant, sentence-transformers, and Groq.",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"


@app.get("/health")
def health_check() -> dict:
    return {"status": "OK"}


@app.get("/", include_in_schema=False)
def serve_web_app() -> FileResponse:
    index_file = WEB_ROOT / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI not found. Create web/index.html.")
    return FileResponse(index_file)

@app.get("/api/sessions")
def api_get_sessions():
    """Returns a list of all chat sessions for the UI sidebar."""
    return get_all_sessions()

@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    """Returns the full message history array for a session."""
    msgs = get_session(session_id)
    if msgs is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": msgs}

@app.post("/api/sessions")
def api_save_session(req: SaveSessionRequest):
    """Saves or updates the UI chat array into the SQLite database."""
    save_session(req.session_id, req.title, req.messages)
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or "default-session"
    config = {"configurable": {"thread_id": session_id}}
    
    # Run the langgraph via streaming or invoke
    state_input = {
        "messages": [HumanMessage(content=request.query)],
        "current_query": request.query
    }
    
    # Get final state after execution
    result_state = vozhi_orchestrator.invoke(state_input, config=config)
    
    # Format response for Web UI
    final_text = result_state.get("final_package", "I am having trouble processing that right now.")
    confidence = result_state.get("confidence_score", 0.0)
    matches = result_state.get("retrieved_schemes", [])
    
    return {
        "session_id": session_id,
        "query": request.query,
        "answer": final_text,
        "confidence": confidence,
        "matches": matches,
    }

from govassist.integrations.twilio import twilio_client
from fastapi import Response

@app.post("/whatsapp")
async def twilio_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handles incoming Twilio WhatsApp messages instantly to avoid 15s timeouts."""
    form_data = dict(await request.form())
    parsed = twilio_client.parse_incoming_message(form_data)
    
    sender = parsed.get("from")
    body = parsed.get("body", "").strip()
    
    if not sender:
        return Response(content="<Response></Response>", media_type="application/xml")
        
    def process_and_reply(sender_id: str, query: str):
        session_id = sender_id.replace("whatsapp:", "")
        config = {"configurable": {"thread_id": session_id}}
        state_input = {"messages": [HumanMessage(content=query)], "current_query": query}
        
        try:
            result_state = vozhi_orchestrator.invoke(state_input, config=config)
            reply_text = result_state.get("final_package", "System Error while processing.")
        except Exception as e:
            logger.error(f"Error in graph: {e}")
            reply_text = "Sorry, I am currently undergoing maintenance."
            
        # Send back to Twilio async using REST API to prevent HTTP timeout
        twilio_client.send_proactive_message(to_number=sender_id, message=reply_text)

    # Queue the heavy LangGraph task
    background_tasks.add_task(process_and_reply, sender, body)
    
    # Return empty TwiML instantly
    return Response(content="<Response></Response>", media_type="application/xml")
