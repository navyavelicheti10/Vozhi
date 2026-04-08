from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state for the minimal multimodal multi-agent workflow."""

    messages: Annotated[List[BaseMessage], add_messages]

    # Input definition
    input_type: str                    # text | audio | document
    transcribed_text: Optional[str]    # STT output (audio inputs)
    raw_query: Optional[str]           # original user text (pre-translation)
    current_query: str                 # normalized / translated query for RAG
    uploaded_file_path: Optional[str]  # temp path to uploaded file
    query_language_code: str           # detected input language
    response_language_code: str        # language to respond in

    # Agent outputs
    user_profile: Dict[str, Any]       # extracted profile fields from documents
    documents_extracted: Dict[str, Any]# OCR + structured fields from uploaded file
    route: str                         # routing decision: respond | retrieve

    # RAG retrievals
    retrieved_schemes: List[Dict[str, Any]]   # top-K schemes from Qdrant / SQLite
    synergy_schemes: List[Dict[str, Any]]     # complementary / synergy schemes
    rag_completed: bool                       # True once RAG + synthesis is done

    # Final response
    final_package: str          # markdown answer to the user
    confidence_score: float     # 0.0 – 1.0 retrieval confidence
    citations: List[str]        # scheme names cited
    sources: List[Dict[str, str]]  # [{title, url}] for official links
