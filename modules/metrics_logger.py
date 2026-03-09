import json
from datetime import datetime
from pathlib import Path

METRICS_PATH = Path("data/metrics.jsonl")


def log_turn(
    user_id: str,
    study_buddy_style: str,
    assertiveness: str,
    session: dict,
    user_text: str,
    assistant_text: str,
) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    topic = (session or {}).get("topic", "")
    goal = (session or {}).get("goal", "")
    exam = (session or {}).get("exam", "")
    deadline = (session or {}).get("deadline", "")
    material = (session or {}).get("material", "")

    context_missing = not all([topic, goal, exam, deadline])

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "study_buddy_style": study_buddy_style,
        "assertiveness": assertiveness,
        "context_missing": context_missing,
        "has_material": bool(material),
        "user_len_chars": len(user_text),
        "assistant_len_chars": len(assistant_text),
        "assistant_question_marks": assistant_text.count("?"),
    }

    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
