import sys
from govassist.agents.state import AgentState
from govassist.agents.graph import vozhi_orchestrator
from langchain_core.messages import HumanMessage

state_input = {
    "messages": [HumanMessage(content="hello")],
    "input_type": "text",
    "raw_query": "hello",
    "current_query": "hello",
    "transcribed_text": "",
    "user_profile": {},
    "documents_extracted": {},
    "retrieved_schemes": [],
    "synergy_schemes": [],
    "citations": []
}

config = {"configurable": {"thread_id": "test_session"}}

try:
    result = vozhi_orchestrator.invoke(state_input, config=config)
    print("SUCCESS")
    print(result)
except Exception as e:
    import traceback
    traceback.print_exc()
