import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from govassist.agents.nodes import document_agent, llm_agent, main_agent, rag_agent
from govassist.agents.state import AgentState

logger = logging.getLogger(__name__)


def route_from_start(_: AgentState) -> str:
    """MainAgent is always the entry point regardless of input type."""
    logger.info("[GRAPH] START → MainAgent")
    return "MainAgent"


def route_after_document(_: AgentState) -> str:
    """After document extraction, always return to MainAgent for routing."""
    logger.info("[GRAPH] DocumentAgent → MainAgent (extraction complete)")
    return "MainAgent"


def route_after_main(state: AgentState) -> str:
    if state.get("rag_completed"):
        logger.info("[GRAPH] MainAgent → END (fast-path: small-talk / out-of-scope / empty)")
        return END
    route = state.get("route", "retrieve")
    if route == "document":
        logger.info("[GRAPH] MainAgent → DocumentAgent (document extraction needed)")
        return "DocumentAgent"
    logger.info("[GRAPH] MainAgent → LLMAgent (route=%s)", route)
    return "LLMAgent"


def route_after_llm(state: AgentState) -> str:
    if state.get("rag_completed"):
        logger.info("[GRAPH] LLMAgent → END (synthesis complete)")
        return END
    logger.info("[GRAPH] LLMAgent → RAGAgent (pre-RAG query refinement done)")
    return "RAGAgent"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("MainAgent", main_agent)
    builder.add_node("LLMAgent", llm_agent)
    builder.add_node("RAGAgent", rag_agent)
    builder.add_node("DocumentAgent", document_agent)

    builder.add_conditional_edges(START, route_from_start)
    builder.add_conditional_edges("DocumentAgent", route_after_document)
    builder.add_conditional_edges("MainAgent", route_after_main)
    builder.add_conditional_edges("LLMAgent", route_after_llm)
    builder.add_edge("RAGAgent", "LLMAgent")

    return builder.compile(checkpointer=MemorySaver())


govassist_graph = build_graph()
vozhi_orchestrator = govassist_graph
