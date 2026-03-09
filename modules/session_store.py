import json
from pathlib import Path

SESSION_PATH = Path("data/session_context.json")

def load_sessions() -> dict:
    if not SESSION_PATH.exists():
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_PATH.write_text("{}", encoding="utf-8")
    return json.loads(SESSION_PATH.read_text(encoding="utf-8") or "{}")

def save_sessions(sessions: dict) -> None:
    SESSION_PATH.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")

def get_session(user_id: str) -> dict:
    sessions = load_sessions()
    return sessions.get(user_id, {})

def update_session(user_id: str, **kwargs) -> None:
    sessions = load_sessions()
    sessions.setdefault(user_id, {})
    sessions[user_id].update(kwargs)
    save_sessions(sessions)
