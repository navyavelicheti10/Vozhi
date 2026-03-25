import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from govassist.agents.state import AgentState
from govassist.rag.embeddings import EmbeddingService
from govassist.rag.vector_store import QdrantManager
from govassist.rag.graph_store import GraphStoreManager
from govassist.config import load_env_file

logger = logging.getLogger(__name__)

# Global clients (lazy loaded to prevent Uvicorn multiprocessing lock issues)
llm = None
embedding_service = None
qdrant = None
graph_manager = None

def _get_or_init_clients():
    global llm, embedding_service, qdrant, graph_manager
    if llm is None:
        load_env_file()
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="llama-3.1-8b-instant")
        embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
        qdrant = QdrantManager(collection_name="schemes")
        graph_manager = GraphStoreManager()
        try:
            graph_manager.load_or_create()
        except Exception as e:
            logger.error(f"Could not load graph store: {e}")

def profile_agent(state: AgentState) -> Dict[str, Any]:
    """Extracts user profile details from the conversation."""
    _get_or_init_clients()
    logger.info("Executing Profile Agent")
    # Simple profile extraction
    sys_msg = SystemMessage(content=(
        "You are an expert profile extractor. Read the conversation and extract demographic details "
        "like state, age, income, gender, and occupation as a JSON object. Only output JSON."
    ))
    resp = llm.invoke([sys_msg] + state.get("messages", []))
    
    try:
        profile_updates = json.loads(resp.content)
    except Exception:
        profile_updates = {}
        
    current_profile = state.get("user_profile", {})
    current_profile.update(profile_updates)
    
    return {"user_profile": current_profile}

def document_agent(state: AgentState) -> Dict[str, Any]:
    """Reads uploaded documents (if any) using OCR/Vision."""
    logger.info("Executing Document Agent")
    # For hackathon: assuming document data is sometimes passed in the state directly via API layer 
    # (or we process base64 images here).
    # This will be fully implemented in Phase 2 with EasyOCR fallback.
    return {}

def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """Hybrid Retrieval + Graph Traversal."""
    _get_or_init_clients()
    logger.info("Executing Retrieval Agent")
    query = state.get("current_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content
        
    # Semantic Search
    query_vector = embedding_service.embed_query(query)
    # Metadata filters can be derived from user_profile
    profile = state.get("user_profile", {})
    state_filter = profile.get("state")
    
    # Qdrant search, reduce top_k to avoid Groq 6000 TPM limit
    semantic_results = qdrant.search(query_vector=query_vector, top_k=2)
    
    # Graph Traversal for Synergies
    synergies = graph_manager.search_synergies(query, top_k=1)
    
    return {
        "retrieved_schemes": semantic_results,
        "synergy_schemes": synergies
    }

def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    """Generates the final bundled response package."""
    _get_or_init_clients()
    logger.info("Executing Synthesis Agent")
    
    schemes = state.get("retrieved_schemes", [])
    synergies = state.get("synergy_schemes", [])
    profile = state.get("user_profile", {})
    query = state.get("current_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content
        
    def _trim(s: Dict) -> Dict:
        return {
            "name": s.get("scheme_name", ""),
            "benefits": str(s.get("benefits", ""))[:500],
            "eligibility": str(s.get("eligibility", ""))[:500]
        }
        
    trimmed_schemes = [_trim(s) for s in schemes]
    
    prompt = f"""
    You are Vozhi, India's Intelligent Benefits Orchestrator.
    User Profile: {json.dumps(profile)}
    
    Retrieved Core Schemes: {json.dumps(trimmed_schemes, default=str)}
    Potential Synergies (Graph RAG): {json.dumps(synergies, default=str)}
    
    Generate a highly accurate, bundled benefit package for the user based.
    Answer the user's latest query using the provided context and previous chat history.
    1. Acknowledge their profile.
    2. Explain the primary scheme.
    3. Suggest bundle synergies to maximize benefits.
    4. Provide clear citations (Source: [Scheme Name]).
    5. Be compassionate but authoritative.
    """
    
    sys_msg = SystemMessage(content=prompt)
    recent_history = state.get("messages", [])[-6:] # Keep context window small to avoid TPM errors
    resp = llm.invoke([sys_msg] + recent_history)
    
    # Mock confidence score calculation based on result density
    confidence = 92.5 if schemes else 45.0
    
    # Extract citations naively for now
    citations = [s.get("scheme_name") for s in schemes if s.get("scheme_name")]
    
    return {
        "final_package": resp.content,
        "confidence_score": confidence,
        "citations": citations,
        "messages": [AIMessage(content=resp.content)]
    }
