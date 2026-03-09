import json
from pathlib import Path

PROFILES_PATH = Path("data/user_profiles.json")

ALLOWED_ASSERTIVENESS = {"gar nicht", "mittel", "sehr"}
ALLOWED_STYLE = {"kollege", "tutor", "dozent"}

def load_profiles() -> dict:
    if not PROFILES_PATH.exists():
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILES_PATH.write_text("{}", encoding="utf-8")
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8") or "{}")

def save_profiles(profiles: dict) -> None:
    PROFILES_PATH.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")

def get_profile(user_id: str) -> dict:
    profiles = load_profiles()
    return profiles.get(user_id, {})

def set_assertiveness(user_id: str, level: str) -> None:
    if level not in ALLOWED_ASSERTIVENESS:
        raise ValueError(f"assertiveness must be one of: {sorted(ALLOWED_ASSERTIVENESS)}")
    profiles = load_profiles()
    profiles.setdefault(user_id, {})
    profiles[user_id]["assertiveness"] = level
    save_profiles(profiles)

def set_style(user_id: str, style: str) -> None:
    if style not in ALLOWED_STYLE:
        raise ValueError(f"study_buddy_style must be one of: {sorted(ALLOWED_STYLE)}")
    profiles = load_profiles()
    profiles.setdefault(user_id, {})
    profiles[user_id]["study_buddy_style"] = style
    save_profiles(profiles)