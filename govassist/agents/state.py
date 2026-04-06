from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state for the minimal multimodal multi-agent workflow."""

    messages: Annotated[List[BaseMessage], add_messages]

    # Input definition
    input_type: str
    transcribed_text: Optional[str]
    raw_query: Optional[str]
    current_query: str
    uploaded_file_path: Optional[str]
    query_language_code: str
    response_language_code: str

    # Agent outputs
    user_profile: Dict[str, Any]
    documents_extracted: Dict[str, Any]
    route: str
    # RAG retrievals
    retrieved_schemes: List[Dict[str, Any]]
    synergy_schemes: List[Dict[str, Any]]
    rag_completed: bool

    # Final response
    final_package: str
    confidence_score: float
    citations: List[str]
    sources: List[Dict[str, str]]
