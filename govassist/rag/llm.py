"""Legacy Sarvam wrapper used by the older non-LangGraph pipeline.

The active backend now synthesizes answers through `govassist/agents/nodes.py`.
This module remains in the repo as reference code.
"""

import logging
import os
from typing import Dict, List

from govassist.integrations.sarvam import sarvam_client

logger = logging.getLogger(__name__)
MAX_FIELD_CHARS = 280
MAX_DOC_ITEMS = 4
MAX_HISTORY_TURNS = 3


def _shorten(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def format_scheme_context(scheme: Dict) -> str:
    documents = scheme.get("documents_required", [])[:MAX_DOC_ITEMS]
    documents_text = ", ".join(documents) if documents else "Not specified"

    return (
        f"Scheme Name: {scheme.get('scheme_name', 'N/A')}\n"
        f"Category: {scheme.get('category', 'N/A')}\n"
        f"Description: {_shorten(scheme.get('description', 'N/A'))}\n"
        f"Eligibility: {_shorten(scheme.get('eligibility', 'N/A'))}\n"
        f"Benefits: {_shorten(scheme.get('benefits', 'N/A'))}\n"
        f"Documents Required: {_shorten(documents_text, 220)}\n"
        f"Application Process: {_shorten(scheme.get('application_process', 'N/A'), 220)}\n"
        f"Official Link: {scheme.get('official_link', 'N/A')}\n"
        f"Tags: {', '.join(scheme.get('tags', [])) or 'None'}"
    )


class SarvamLLMClient:
    """Wraps the Sarvam chat completion API for answer generation."""

    def __init__(self, model_name: str = "sarvam-m", api_key: str | None = None) -> None:
        self.model_name = os.getenv("SARVAM_CHAT_MODEL", model_name)
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is missing. Add it to your .env file.")

    def build_prompt(self, query: str, schemes: List[Dict], chat_history: List[Dict] | None = None) -> str:
        history_text = "No previous conversation."
        if chat_history:
            recent_turns = chat_history[-MAX_HISTORY_TURNS:]
            history_text = "\n\n".join(
                f"User: {_shorten(turn.get('user', ''), 140)}\nAssistant: {_shorten(turn.get('assistant', ''), 220)}"
                for turn in recent_turns
            )

        retrieved_context = "\n\n".join(
            f"Scheme {index + 1}:\n{format_scheme_context(scheme)}"
            for index, scheme in enumerate(schemes)
        )

        return f"""You are a helpful government schemes assistant.
Answer the user's query using ONLY the provided scheme data.

User Query:
{query}

Previous Conversation:
{history_text}

Relevant Schemes:
{retrieved_context}

Instructions:
- Explain in simple language
- Suggest best matching schemes
- Mention eligibility clearly
- Provide actionable guidance
- Continue the conversation naturally if the user is asking a follow-up question
- Do NOT hallucinate
- If the data is insufficient, say so clearly
- Keep the answer concise and practical
"""

    def generate_answer(
        self,
        query: str,
        schemes: List[Dict],
        chat_history: List[Dict] | None = None,
    ) -> str:
        if not schemes:
            return (
                "I could not find a matching government scheme in the current database. "
                "Try rephrasing your query or using a broader search."
            )

        logger.info("Generating response from Sarvam model: %s", self.model_name)

        # Retry with fewer schemes if the request size becomes too large.
        for scheme_count in (min(len(schemes), 5), min(len(schemes), 3), 1):
            subset = schemes[:scheme_count]
            prompt = self.build_prompt(query, subset, chat_history=chat_history)
            try:
                response = sarvam_client.chat_completion(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You answer only from the retrieved government scheme data.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1200,
                )
                return response.strip()
            except Exception as exc:
                message = str(exc)
                if "Request too large" in message or "tokens per minute" in message or "413" in message:
                    logger.warning(
                        "Sarvam request too large with %s schemes. Retrying with fewer schemes.",
                        scheme_count,
                    )
                    continue
                raise

        return (
            "I found relevant schemes, but the model request became too large. "
            "Please try a more specific question like eligibility, benefits, or documents."
        )


# Backward-compatible alias for older imports.
GroqLLMClient = SarvamLLMClient
