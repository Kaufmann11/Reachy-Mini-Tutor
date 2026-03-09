"""User profile store for tutor characteristics."""

import json
from pathlib import Path

# Pfad relativ zur Datei selbst — funktioniert unabhängig vom Arbeitsverzeichnis
PROFILES_PATH = Path(__file__).parent.parent / "data" / "user_profiles.json"

ALLOWED_ASSERTIVENESS = {"gar nicht", "mittel", "sehr"}
ALLOWED_STYLE = {"kollege", "tutor", "dozent"}


def load_profiles() -> dict:
    """Load all user profiles from JSON."""
    if not PROFILES_PATH.exists():
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILES_PATH.write_text("{}", encoding="utf-8")
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8") or "{}")


def save_profiles(profiles: dict) -> None:
    """Save all user profiles to JSON."""
    PROFILES_PATH.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")


def get_profile(user_id: str) -> dict:
    """Get profile for a specific user."""
    profiles = load_profiles()
    return profiles.get(user_id, {})


def set_assertiveness(user_id: str, level: str) -> None:
    """Set assertiveness level for a user."""
    if level not in ALLOWED_ASSERTIVENESS:
        raise ValueError(f"assertiveness must be one of: {sorted(ALLOWED_ASSERTIVENESS)}")
    profiles = load_profiles()
    profiles.setdefault(user_id, {})
    profiles[user_id]["assertiveness"] = level
    save_profiles(profiles)


def set_style(user_id: str, style: str) -> None:
    """Set tutor style for a user."""
    if style not in ALLOWED_STYLE:
        raise ValueError(f"study_buddy_style must be one of: {sorted(ALLOWED_STYLE)}")
    profiles = load_profiles()
    profiles.setdefault(user_id, {})
    profiles[user_id]["study_buddy_style"] = style
    save_profiles(profiles)


def ensure_default_profile(user_id: str) -> dict:
    """Create default profile if user doesn't exist yet."""
    profile = get_profile(user_id)
    if not profile:
        profiles = load_profiles()
        profiles[user_id] = {
            "assertiveness": "mittel",
            "study_buddy_style": "tutor",
            "onboarded": False,
        }
        save_profiles(profiles)
        return profiles[user_id]
    return profile