"""Legacy file-based chat history storage.

The active app persists chat sessions in SQLite through `govassist/api/db.py`.
This module is retained for the older pipeline and local reference usage.
"""

import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List


class FileCheckpointer:
    """
    Stores chat sessions in a local JSON file.
    This is simple, beginner-friendly, and survives server restarts.
    """

    def __init__(self, file_path: str = "checkpoints/chat_sessions.json") -> None:
        self.file_path = file_path
        self._lock = Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as file:
                json.dump({}, file, indent=2)

    def _read(self) -> Dict[str, Any]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, data: Dict[str, Any]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read()
            session = data.get(session_id, {})
            return session.get("turns", [])

    def save_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        matches: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            data = self._read()
            session = data.setdefault(
                session_id,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "turns": [],
                },
            )

            session["updated_at"] = datetime.now(timezone.utc).isoformat()
            session["turns"].append(
                {
                    "user": user_query,
                    "assistant": assistant_answer,
                    "matches": [
                        {
                            "scheme_name": match.get("scheme_name"),
                            "category": match.get("category"),
                            "official_link": match.get("official_link"),
                            "score": match.get("score"),
                        }
                        for match in matches
                    ],
                }
            )
            self._write(data)
