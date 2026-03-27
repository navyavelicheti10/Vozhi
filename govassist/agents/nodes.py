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
from govassist.rag.embeddings import EmbeddingService, clean_text
from govassist.rag.graph_store import GraphStoreManager
from govassist.rag.vector_store import QdrantManager

logger = logging.getLogger(__name__)

llm: ChatGroq | None = None
embedding_service: EmbeddingService | None = None
qdrant: QdrantManager | None = None
graph_manager: GraphStoreManager | None = None
ocr_reader = None

GREETING_PATTERNS = (
    re.compile(r"^\s*(hi|hii+|hello|hey|heyy+|namaste|namaskaram|good (morning|afternoon|evening))[\s!.?]*$", re.I),
    re.compile(r"^\s*(bye|goodbye|see you|thanks|thank you|ok thanks|okay thanks|thx)[\s!.?]*$", re.I),
)


def _is_small_talk(query: str) -> bool:
    candidate = clean_text(query or "").strip()
    if not candidate:
        return False
    return any(pattern.fullmatch(candidate) for pattern in GREETING_PATTERNS)


def _build_small_talk_response(query: str) -> str:
    normalized = clean_text(query or "").strip().lower()
    if any(token in normalized for token in ("bye", "goodbye", "see you")):
        return "Goodbye. Reach out anytime if you want help finding the right government scheme."
    if any(token in normalized for token in ("thanks", "thank you", "thx")):
        return "You’re welcome. If you want, I can help you check eligibility or find a scheme by state, profession, or benefit type."
    return "Hello. I can help you find relevant government schemes, check eligibility, or review uploaded documents."


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


def _get_or_init_clients() -> None:
    global llm, embedding_service, qdrant, graph_manager

    if llm is not None:
        return

    load_env_file()
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
    embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantManager(collection_name="schemes")
    graph_manager = GraphStoreManager()

    try:
        graph_manager.load_or_create()
    except Exception as exc:
        logger.warning("Graph store could not be loaded. Synergy retrieval will be skipped: %s", exc)


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
        "You are GovAssist, a precise assistant for Indian government schemes.\n"
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
        "- Do not output JSON."
    )
    human_prompt = (
        f"User query: {current_query}\n"
        f"User profile: {json.dumps(user_profile, ensure_ascii=True)}\n"
        f"Document context: {json.dumps(documents_extracted, ensure_ascii=True)}\n"
        f"Retrieved schemes: {json.dumps(trimmed_schemes, ensure_ascii=True)}\n"
        f"Synergy schemes: {json.dumps(synergies, ensure_ascii=True)}"
    )
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]


def build_post_rag_messages(state: AgentState) -> List[SystemMessage | HumanMessage]:
    _get_or_init_clients()
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
        "confidence_score": _calculate_confidence(schemes),
    }


def _post_rag_response(state: AgentState) -> Dict[str, Any]:
    _get_or_init_clients()

    schemes = state.get("retrieved_schemes", [])
    synergies = state.get("synergy_schemes", [])
    documents_extracted = state.get("documents_extracted", {})
    user_profile = state.get("user_profile", {})
    current_query = state.get("current_query", "")

    if not schemes:
        fallback = (
            "I could not find a confident scheme match in the current database for this request. "
            "Please try a more specific query such as your state, profession, income bracket, or target benefit. "
            "If you uploaded a document, I can also use a clearer scan or an extra text hint about what you want to check."
        )
        return {
            "final_package": fallback,
            "confidence_score": 0.25,
            "citations": [],
            "messages": [AIMessage(content=fallback)],
        }

    response = llm.invoke(_build_post_rag_messages(state))
    citations = [
        scheme.get("scheme_name")
        for scheme in schemes
        if clean_text(scheme.get("scheme_name"))
    ]
    normalized_confidence = _calculate_confidence(schemes)

    return {
        "final_package": response.content.strip(),
        "confidence_score": normalized_confidence,
        "citations": citations,
        "messages": [AIMessage(content=response.content.strip())],
    }


def _pre_rag_query_refinement(state: AgentState) -> Dict[str, Any]:
    _get_or_init_clients()

    raw_query = clean_text(state.get("raw_query", ""))
    transcribed_text = clean_text(state.get("transcribed_text", ""))
    current_query = clean_text(state.get("current_query", ""))
    document_context = _build_document_context(state)

    seed_query = raw_query or transcribed_text or current_query or document_context
    if not seed_query:
        seed_query = "Find relevant government schemes using the uploaded document details."

    prompt = (
        "You normalize queries for government-scheme retrieval.\n"
        "Return only one short retrieval query.\n"
        "Preserve important facts such as state, profile, document hints, category, and benefit intent.\n"
        "Do not add explanations or formatting."
    )
    human_content = seed_query if not document_context else f"{seed_query}\n\n{document_context}"
    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=human_content),
        ]
    )
    normalized_query = clean_text(response.content).strip("\"'")

    return {"current_query": normalized_query or seed_query}


def llm_agent(state: AgentState) -> Dict[str, Any]:
    """Dual-role LLM node: query normalization before RAG and answer generation after RAG."""
    if state.get("rag_completed"):
        logger.info("Running LLM agent in post-RAG synthesis mode")
        return _post_rag_response(state)

    seed_query = clean_text(
        state.get("raw_query", "") or state.get("transcribed_text", "") or state.get("current_query", "")
    )
    if _is_small_talk(seed_query):
        logger.info("Short-circuiting small-talk query without RAG: %s", seed_query)
        reply = _build_small_talk_response(seed_query)
        return {
            "current_query": seed_query,
            "final_package": reply,
            "confidence_score": 1.0,
            "citations": [],
            "rag_completed": True,
            "messages": [AIMessage(content=reply)],
        }

    logger.info("Running LLM agent in pre-RAG query refinement mode")
    return _pre_rag_query_refinement(state)


def document_agent(state: AgentState) -> Dict[str, Any]:
    """Extract text and basic structured fields from uploaded image/PDF documents."""
    _get_or_init_clients()
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
    if suffix == ".pdf":
        extracted_text = _extract_pdf_text(file_path)
    else:
        extracted_text = _extract_image_text(file_path)

    structured_fields: Dict[str, Any] = {}
    if extracted_text:
        parser_prompt = (
            "Extract basic structured fields from the document text.\n"
            "Return only a JSON object with keys when available: "
            "document_type, name, dob, gender, state, district, income_amount, occupation, id_number, notes."
        )
        response = llm.invoke(
            [
                SystemMessage(content=parser_prompt),
                HumanMessage(content=extracted_text[:5000]),
            ]
        )
        structured_fields = _safe_json_loads(response.content)

    if not extracted_text:
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
    _get_or_init_clients()

    query = clean_text(state.get("current_query", ""))
    if not query:
        logger.warning("RAG agent received an empty query")
        return {
            "retrieved_schemes": [],
            "synergy_schemes": [],
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
        logger.info("Falling back to SQLite retrieval for query: %s", query)
        semantic_results = search_schemes_in_db(query=query, top_k=5)

    synergy_payload: List[Dict[str, Any]] = []
    try:
        raw_synergies = graph_manager.search_synergies(query, top_k=3) if graph_manager else []
        synergy_payload = [
            {"summary": clean_text(item)}
            for item in raw_synergies
            if clean_text(item)
        ]
    except Exception as exc:
        logger.warning("Graph synergy retrieval failed: %s", exc)

    return {
        "retrieved_schemes": semantic_results,
        "synergy_schemes": synergy_payload,
        "rag_completed": True,
    }
