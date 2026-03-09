"""Conversation manager — saves session history to JSON."""

import json
from pathlib import Path
from datetime import datetime


class ConversationManager:
    """Manages and persists conversation history for a user session."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history = []
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def add(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def save(self) -> str:
        """Save conversation to JSON file. Returns the file path."""
        out_dir = Path(__file__).parent.parent / "data" / "conversations"
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = out_dir / f"{self.user_id}_{self.started_at}.json"
        payload = {
            "user_id": self.user_id,
            "started_at": self.started_at,
            "messages": self.history,
        }
        filename.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(filename)