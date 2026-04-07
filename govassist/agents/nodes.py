import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from govassist.agents.state import AgentState
from govassist.api.db_utils import search_schemes_in_db
from govassist.config import load_env_file
from govassist.integrations.sarvam import sarvam_client
from govassist.rag.embeddings import EmbeddingService, clean_text
from govassist.rag.vector_store import QdrantManager

logger = logging.getLogger(__name__)

llm: ChatGroq | None = None
embedding_service: EmbeddingService | None = None
qdrant: QdrantManager | None = None
ocr_reader = None

GREETING_PATTERNS = (
    re.compile(r"^\s*(hi|hii+|hello|hey|heyy+|namaste|namaskaram|good (morning|afternoon|evening))[\s!.?]*$", re.I),
    re.compile(r"^\s*(bye+|goodbye|see you|thanks+|thank you|ok thanks|okay thanks|thx)[\s!.?]*$", re.I),
)

LANGUAGE_LABELS = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
}

SCHEME_KEYWORDS = {
    "scheme",
    "schemes",
    "yojana",
    "subsidy",
    "benefit",
    "benefits",
    "eligibility",
    "eligible",
    "apply",
    "application",
    "loan",
    "scholarship",
    "pension",
    "grant",
    "farmer",
    "student",
    "women",
    "widow",
    "disabled",
    "startup",
    "housing",
    "government",
}


def _is_small_talk(query: str) -> bool:
    candidate = clean_text(query or "").strip()
    if not candidate:
        return False
    return any(pattern.fullmatch(candidate) for pattern in GREETING_PATTERNS)


def _build_small_talk_response(query: str) -> str:
    normalized = clean_text(query or "").strip().lower()
    if any(token in normalized for token in ("bye", "goodbye", "see you")):
        return "Bye. Reach out anytime if you want help finding the right government scheme."
    if any(token in normalized for token in ("thanks", "thank you", "thx")):
        return "You’re welcome. If you want, I can help you check eligibility or find a scheme by state, profession, or benefit type."
    if "your name" in normalized or "who are you" in normalized:
        return "I’m Vozhi. I can chat naturally, help you understand documents, and look up relevant government schemes only when your question needs that."
    if "what can you do" in normalized or "help me" in normalized:
        return (
            "I can chat with you normally, answer quick questions about the assistant, "
            "and help find government schemes when you share your state, profile, or goal."
        )
    return "Hello. I can help you find relevant government schemes, check eligibility, or review uploaded documents."


def _localize_text(text: str, state: AgentState) -> str:
    target_language = state.get("response_language_code", "en-IN")
    if not text or target_language == "en-IN":
        return text

    translated = sarvam_client.translate_text(
        text=text,
        source_language_code="en-IN",
        target_language_code=target_language,
    )
    return translated.get("translated_text", text).strip() or text


def _is_assistant_meta_query(query: str) -> bool:
    normalized = clean_text(query or "").strip().lower()
    if not normalized:
        return False

    assistant_phrases = (
        "who are you",
        "what is your name",
        "your name",
        "what can you do",
        "help me",
        "how can you help",
        "can you help",
        "what do you do",
    )
    return any(phrase in normalized for phrase in assistant_phrases)


def _build_out_of_scope_response() -> str:
    return (
        "I’m mainly here to help with government schemes, eligibility, documents, and application guidance. "
        "If you want, tell me your state and what kind of benefit you need, and I’ll help with that."
    )


def _looks_like_scheme_query(query: str) -> bool:
    normalized = clean_text(query or "").lower()
    if not normalized:
        return False
    if _is_small_talk(normalized):
        return False
    return any(keyword in normalized for keyword in SCHEME_KEYWORDS)


def _build_sources(schemes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    sources: List[Dict[str, str]] = []

    for scheme in schemes:
        name = clean_text(scheme.get("scheme_name", ""))
        url = clean_text(scheme.get("official_link") or scheme.get("source", ""))
        if not name or not url:
            continue
        key = (name, url)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"title": name, "url": url})

    return sources


def _safe_json_loads(payload: str) -> Dict[str, Any]:
    candidate = (payload or "").strip()
    if not candidate:
        return {}

    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0]

    try:
        parsed = json.loads(candidate.strip())
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON structured output. Falling back to raw text.")
        return {}


def _ensure_llm() -> None:
    global llm

    if llm is not None:
        return

    load_env_file()
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2, max_retries=10)


def _ensure_rag_clients() -> None:
    global embedding_service, qdrant

    _ensure_llm()

    if embedding_service is None:
        embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

    if qdrant is None:
        qdrant = QdrantManager(collection_name="schemes")


def _get_or_init_ocr_reader():
    global ocr_reader

    if ocr_reader is not None:
        return ocr_reader

    try:
        import easyocr

        ocr_reader = easyocr.Reader(["en", "hi"], gpu=False)
    except Exception as exc:
        logger.warning("EasyOCR initialization failed: %s", exc)
        ocr_reader = False

    return ocr_reader


def _extract_pdf_text(file_path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return clean_text("\n".join(pages))
    except Exception as exc:
        logger.warning("PDF text extraction failed for %s: %s", file_path, exc)
        return ""


def _extract_image_text(file_path: Path) -> str:
    reader = _get_or_init_ocr_reader()
    if not reader:
        return ""

    try:
        results = reader.readtext(str(file_path), detail=0)
        return clean_text(" ".join(results))
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", file_path, exc)
        return ""


def _build_document_context(state: AgentState) -> str:
    documents_extracted = state.get("documents_extracted") or {}
    raw_text = clean_text(documents_extracted.get("raw_text", ""))
    structured_fields = documents_extracted.get("structured_fields") or {}
    parts: List[str] = []

    if state.get("raw_query"):
        parts.append(f"User request: {clean_text(state['raw_query'])}")
    if structured_fields:
        parts.append(f"Document fields: {json.dumps(structured_fields, ensure_ascii=True)}")
    if raw_text:
        parts.append(f"Document text: {raw_text[:1200]}")

    return "\n".join(parts).strip()


def _calculate_confidence(schemes: List[Dict[str, Any]]) -> float:
    average_score = sum(float(scheme.get("score", 0.0) or 0.0) for scheme in schemes) / max(len(schemes), 1)
    return round(min(max(average_score, 0.0), 1.0), 3)


def _build_post_rag_messages(state: AgentState) -> List[SystemMessage | HumanMessage]:
    schemes = state.get("retrieved_schemes", [])
    synergies = state.get("synergy_schemes", [])
    documents_extracted = state.get("documents_extracted", {})
    user_profile = state.get("user_profile", {})
    current_query = state.get("current_query", "")
    response_language_code = state.get("response_language_code", "en-IN")
    response_language = LANGUAGE_LABELS.get(response_language_code, "English")

    trimmed_schemes = [
        {
            "scheme_name": scheme.get("scheme_name", ""),
            "category": scheme.get("category", ""),
            "description": clean_text(scheme.get("description", ""))[:400],
            "eligibility": clean_text(scheme.get("eligibility", ""))[:350],
            "benefits": clean_text(scheme.get("benefits", ""))[:350],
            "documents_required": scheme.get("documents_required", ""),
            "application_process": clean_text(scheme.get("application_process", ""))[:240],
            "official_link": scheme.get("official_link") or scheme.get("source", ""),
            "score": scheme.get("score"),
        }
        for scheme in schemes
    ]

    system_prompt = (
        "You are Vozhi, a helpful and natural conversational assistant for Indian government schemes.\n"
        "Answer using only the supplied retrieval context. Do not invent or assume missing facts.\n"
        "Keep the response tight, readable, and decision-oriented.\n\n"
        "Formatting rules:\n"
        "- Use short Markdown sections with these headings when relevant: Best Match, Eligibility, Benefits, Documents Required, How to Apply, Also Consider, Next Step.\n"
        "- Start with one sentence that directly answers the user.\n"
        "- Focus on the single best-fit scheme first.\n"
        "- Mention synergy schemes only if they are genuinely relevant and label them under Also Consider.\n"
        "- Use plain language and short bullets under sections when useful.\n"
        "- Add inline citations in this exact format: [Source: Scheme Name].\n"
        "- If retrieval is weak or partial, say so briefly instead of filling gaps.\n"
        "- Do not output JSON.\n"
        f"- Write the final answer entirely in {response_language}."
    )
    
    messages = [SystemMessage(content=system_prompt)]
    
    chat_history = state.get("messages", [])
    if len(chat_history) > 1:
        # Pass up to the last 6 messages (excluding the current one) to supply context
        messages.extend(chat_history[-7:-1])

    human_prompt = (
        f"User query: {current_query}\n"
        f"User profile: {json.dumps(user_profile, ensure_ascii=True)}\n"
        f"Document context: {json.dumps(documents_extracted, ensure_ascii=True)}\n"
        f"Retrieved schemes: {json.dumps(trimmed_schemes, ensure_ascii=True)}\n"
        f"Synergy schemes: {json.dumps(synergies, ensure_ascii=True)}"
    )
    
    messages.append(HumanMessage(content=human_prompt))
    return messages


def build_post_rag_messages(state: AgentState) -> List[SystemMessage | HumanMessage]:
    _ensure_llm()
    return _build_post_rag_messages(state)


def build_post_rag_metadata(state: AgentState) -> Dict[str, Any]:
    schemes = state.get("retrieved_schemes", [])
    citations = [
        scheme.get("scheme_name")
        for scheme in schemes
        if clean_text(scheme.get("scheme_name"))
    ]
    return {
        "citations": citations,
        "sources": _build_sources(schemes),
        "confidence_score": _calculate_confidence(schemes),
    }


def _post_rag_response(state: AgentState) -> Dict[str, Any]:
    _ensure_llm()

    schemes = state.get("retrieved_schemes", [])

    if not schemes:
        fallback = (
            "I could not find a confident scheme match in the current database for this request. "
            "Please try a more specific query such as your state, profession, income bracket, or target benefit. "
            "If you uploaded a document, I can also use a clearer scan or an extra text hint about what you want to check."
        )
        fallback = _localize_text(fallback, state)
        return {
            "final_package": fallback,
            "confidence_score": 0.25,
            "citations": [],
            "sources": [],
            "messages": [AIMessage(content=fallback)],
        }

    try:
        response = llm.invoke(_build_post_rag_messages(state))
        content = response.content.strip()
    except Exception as exc:
        logger.warning("LLM post-RAG synthesis failed (possibly Rate Limit): %s", exc)
        fallback_lines = ["I found some relevant schemes for you (Note: Intelligent synthesis delayed due to high load):\n"]
        for scheme in schemes:
            name = scheme.get("scheme_name", "Unknown Scheme")
            fallback_lines.append(f"- **{name}**: {scheme.get('description', '')[:200]}...")
        content = "\n".join(fallback_lines)

    citations = [
        scheme.get("scheme_name")
        for scheme in schemes
        if clean_text(scheme.get("scheme_name"))
    ]

    return {
        "final_package": content,
        "confidence_score": _calculate_confidence(schemes),
        "citations": citations,
        "sources": _build_sources(schemes),
        "messages": [AIMessage(content=content)],
    }


def main_agent(state: AgentState) -> Dict[str, Any]:
    """Primary router that decides whether to answer directly or run RAG."""
    _ensure_llm()

    seed_query = clean_text(
        state.get("raw_query", "") or state.get("transcribed_text", "") or state.get("current_query", "")
    )
    if not seed_query:
        reply = _localize_text(
            "Tell me what you need, and I’ll help. If you're looking for a government scheme, share your state or purpose.",
            state,
        )
        return {
            "route": "respond",
            "current_query": "",
            "final_package": reply,
            "confidence_score": 1.0,
            "citations": [],
            "sources": [],
            "rag_completed": True,
            "messages": [AIMessage(content=reply)],
        }

    if _is_small_talk(seed_query):
        reply = _localize_text(_build_small_talk_response(seed_query), state)
        return {
            "route": "respond",
            "current_query": seed_query,
            "final_package": reply,
            "confidence_score": 1.0,
            "citations": [],
            "sources": [],
            "rag_completed": True,
            "messages": [AIMessage(content=reply)],
        }

    if _is_assistant_meta_query(seed_query):
        return {"route": "respond"}

    if _looks_like_scheme_query(seed_query):
        return {"route": "retrieve"}

    reply = _localize_text(_build_out_of_scope_response(), state)
    return {
        "route": "respond",
        "current_query": seed_query,
        "final_package": reply,
        "confidence_score": 1.0,
        "citations": [],
        "sources": [],
        "rag_completed": True,
        "messages": [AIMessage(content=reply)],
    }


def _pre_rag_query_refinement(state: AgentState) -> Dict[str, Any]:
    _ensure_llm()

    raw_query = clean_text(state.get("raw_query", ""))
    transcribed_text = clean_text(state.get("transcribed_text", ""))
    current_query = clean_text(state.get("current_query", ""))
    document_context = _build_document_context(state)

    seed_query = current_query or raw_query or transcribed_text or document_context
    if not seed_query:
        seed_query = "Find relevant government schemes using the uploaded document details."

    prompt = (
        "You normalize conversational queries into semantic search keywords for a Vector Database.\n"
        "Return only a short, generic string of keywords (DO NOT write SQL, JSON, or code).\n"
        "Preserve important facts such as state, demographics, document hints, and benefit intent.\n"
        "Do not add explanations or formatting."
    )
    
    chat_history = state.get("messages", [])
    history_str = ""
    if len(chat_history) > 1:
        history_str = "Conversation History:\n" + "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" 
            for m in chat_history[-5:-1]
        ) + "\n\n"
        
    human_content = seed_query if not document_context else f"{seed_query}\n\n{document_context}"
    human_content = history_str + f"Latest User Query: {human_content}"

    try:
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=human_content),
            ]
        )
        normalized_query = clean_text(response.content).strip("\"'")
    except Exception as exc:
        logger.warning("LLM query refinement failed, falling back to original query: %s", exc)
        normalized_query = seed_query

    return {"current_query": normalized_query or seed_query}


def llm_agent(state: AgentState) -> Dict[str, Any]:
    """Dual-role LLM node: query normalization before RAG and answer generation after RAG."""
    if state.get("rag_completed"):
        logger.info("Running LLM agent in post-RAG synthesis mode")
        return _post_rag_response(state)

    seed_query = clean_text(
        state.get("raw_query", "") or state.get("transcribed_text", "") or state.get("current_query", "")
    )
    if state.get("route") == "respond":
        logger.info("Answering directly without RAG for query: %s", seed_query)
        
        system_prompt = (
            "You are Vozhi, a conversational AI for Indian government schemes.\n"
            "The user asked a general question, made small talk, or provided personal context.\n"
            "Use the conversation history to give a short, friendly, and helpful response. If they are asking out-of-scope questions, gently steer them back to schemes."
        )
        
        chat_history = state.get("messages", [])
        history_str = ""
        if len(chat_history) > 1:
            history_str = "Conversation History:\n" + "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:500]}" 
                for m in chat_history[-6:-1]
            ) + "\n\n"
            
        human_content = history_str + f"Latest User Query: {seed_query}"
        
        try:
            _ensure_llm()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_content)
            ])
            reply = clean_text(response.content)
        except Exception as exc:
            logger.warning("LLM direct answer failed: %s", exc)
            reply = (
                _build_small_talk_response(seed_query)
                if (_is_small_talk(seed_query) or _is_assistant_meta_query(seed_query))
                else _build_out_of_scope_response()
            )
        reply = _localize_text(reply, state)
        return {
            "current_query": seed_query,
            "final_package": reply,
            "confidence_score": 0.9,
            "citations": [],
            "sources": [],
            "rag_completed": True,
            "messages": [AIMessage(content=reply)],
        }

    logger.info("Running LLM agent in pre-RAG query refinement mode")
    return _pre_rag_query_refinement(state)


def document_agent(state: AgentState) -> Dict[str, Any]:
    """Extract text and basic structured fields from uploaded image/PDF documents."""
    _ensure_llm()
    file_path_value = state.get("uploaded_file_path")
    if not file_path_value:
        return {"documents_extracted": {}}

    file_path = Path(file_path_value)
    if not file_path.exists():
        logger.warning("Uploaded document path does not exist: %s", file_path)
        return {
            "documents_extracted": {
                "file_name": file_path.name,
                "raw_text": "",
                "structured_fields": {},
                "error": "Uploaded file was not found.",
            }
        }

    suffix = file_path.suffix.lower()
    extracted_text = ""
    structured_fields: Dict[str, Any] = {}
    
    if suffix == ".pdf":
        extracted_text = _extract_pdf_text(file_path)
    else:
        logger.info("Routing image %s to EasyOCR extraction.", file_path.name)
        extracted_text = _extract_image_text(file_path)

    if extracted_text:
        parser_prompt = (
            "Extract basic structured fields from the document text.\n"
            "Return only a JSON object with keys when available: "
            "document_type, name, dob, gender, state, district, income_amount, occupation, id_number, notes."
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=parser_prompt),
                    HumanMessage(content=extracted_text[:5000]),
                ]
            )
            structured_fields = _safe_json_loads(response.content)
        except Exception as e:
            logger.warning("LLM structured parsing failed: %s", e)

    if not extracted_text and not structured_fields:
        logger.warning("No text could be extracted from document %s", file_path)

    return {
        "documents_extracted": {
            "file_name": file_path.name,
            "file_type": suffix.lstrip("."),
            "raw_text": extracted_text,
            "structured_fields": structured_fields,
        }
    }


def rag_agent(state: AgentState) -> Dict[str, Any]:
    """Hybrid semantic retrieval with optional graph-based synergy lookup."""
    _ensure_rag_clients()

    query = clean_text(state.get("current_query", ""))
    if not query:
        logger.warning("RAG agent received an empty query")
        return {
            "retrieved_schemes": [],
            "rag_completed": True,
        }

    logger.info("Running RAG agent for query: %s", query)
    semantic_results: List[Dict[str, Any]] = []
    try:
        query_vector = embedding_service.embed_query(query)
        semantic_results = qdrant.search(query_vector=query_vector, top_k=5)
    except Exception as exc:
        logger.warning("Vector retrieval failed for query '%s': %s", query, exc)

    if not semantic_results:
        semantic_results = search_schemes_in_db(query=query, top_k=5)

    return {
        "retrieved_schemes": semantic_results,
        "rag_completed": True,
    }
