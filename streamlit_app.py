import os
from typing import Any, Dict, List
from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).strip()
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
SESSION_ENDPOINT = f"{API_BASE_URL}/sessions"


def fetch_chat_response(query: str, top_k: int, session_id: str | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
    }
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def fetch_session_history(session_id: str) -> List[Dict[str, Any]]:
    response = requests.get(f"{SESSION_ENDPOINT}/{session_id}", timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("history", [])


def render_matches(matches: List[Dict[str, Any]]) -> None:
    if not matches:
        return

    with st.expander("Relevant Schemes"):
        for match in matches:
            st.markdown(f"**{match.get('scheme_name', 'Unknown Scheme')}**")
            st.write(f"Category: {match.get('category', 'N/A')}")
            st.write(f"Score: {round(match.get('score', 0), 3)}")
            if match.get("official_link"):
                st.markdown(f"[Official Link]({match['official_link']})")
            st.divider()


st.set_page_config(
    page_title="Government Schemes Assistant",
    page_icon="🏛️",
    layout="wide",
)

st.title("Government Schemes Assistant")
st.caption("A Streamlit chatbot powered by FastAPI, Qdrant, BGE embeddings, and Groq.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "archived_chats" not in st.session_state:
    st.session_state.archived_chats = []

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI URL", value=API_BASE_URL).strip() or DEFAULT_API_BASE_URL
    top_k = st.slider("Top K Schemes", min_value=1, max_value=5, value=3)

    if st.button("Start New Chat"):
        if st.session_state.messages:
            st.session_state.archived_chats.append(
                {
                    "session_id": st.session_state.session_id,
                    "messages": st.session_state.messages.copy(),
                }
            )
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    if st.session_state.archived_chats:
        st.markdown("### Previous Chats")
        for index, chat in enumerate(reversed(st.session_state.archived_chats), start=1):
            session_label = chat.get("session_id") or "No session"
            with st.expander(f"Chat {index} • Session {session_label}", expanded=False):
                for message in chat.get("messages", []):
                    role_label = "You" if message["role"] == "user" else "Assistant"
                    st.markdown(f"**{role_label}:** {message['content']}")
                    if message["role"] == "assistant":
                        render_matches(message.get("matches", []))

    if st.session_state.session_id:
        st.write(f"Session ID: `{st.session_state.session_id}`")
        if st.button("Reload Saved History"):
            try:
                history = fetch_session_history(st.session_state.session_id)
                st.session_state.messages = []
                for turn in history:
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": turn.get("user", ""),
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": turn.get("assistant", ""),
                            "matches": turn.get("matches", []),
                        }
                    )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load history: {exc}")

CHAT_ENDPOINT = f"{api_url.rstrip('/')}/chat"
SESSION_ENDPOINT = f"{api_url.rstrip('/')}/sessions"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_matches(message.get("matches", []))

user_query = st.chat_input("Ask about scholarships, farmers, loans, pensions, women schemes...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Finding relevant schemes..."):
            try:
                result = fetch_chat_response(
                    query=user_query,
                    top_k=top_k,
                    session_id=st.session_state.session_id,
                )

                st.session_state.session_id = result.get("session_id")
                answer = result.get("answer", "No answer returned.")
                matches = result.get("matches", [])

                st.markdown(answer)
                render_matches(matches)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "matches": matches,
                    }
                )
            except requests.HTTPError as exc:
                error_text = exc.response.text if exc.response is not None else str(exc)
                st.error(f"API error: {error_text}")
            except requests.ConnectionError:
                st.error(
                    "Could not reach the FastAPI backend. Start it with "
                    "`uvicorn main:app --host 127.0.0.1 --port 8000` "
                    "or update the FastAPI URL in the sidebar."
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
