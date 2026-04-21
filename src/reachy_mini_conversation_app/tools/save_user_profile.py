import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# Module-level storage — picked up by openai_realtime.py after the tool call
_pending_profile_context: str = ""


class SaveUserProfile(Tool):
    """Store the onboarding profile so it can be injected into the conversation context."""

    name = "save_user_profile"
    description = (
        "Call this ONCE after all 7 onboarding questions are answered. "
        "Saves the student profile so it stays visible throughout the session."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Student's name"},
            "study": {"type": "string", "description": "Field of study and semester"},
            "motivation": {"type": "string", "description": "What motivates the student"},
            "learning_style": {"type": "string", "description": "How the student prefers to learn"},
            "hobbies": {"type": "string", "description": "Student's hobbies and interests"},
            "humor": {"type": "string", "description": "Whether humor is welcome: ja or nein"},
            "goal": {"type": "string", "description": "What the student wants to achieve today"},
        },
        "required": ["name", "study", "hobbies", "humor", "goal"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        global _pending_profile_context

        name = kwargs.get("name", "?")
        study = kwargs.get("study", "?")
        motivation = kwargs.get("motivation", "?")
        learning_style = kwargs.get("learning_style", "?")
        hobbies = kwargs.get("hobbies", "?")
        humor = kwargs.get("humor", "?")
        goal = kwargs.get("goal", "?")

        _pending_profile_context = (
            f"[LERNPROFIL: Name={name} | Studium={study} | "
            f"Motivation={motivation} | Lernstil={learning_style} | "
            f"Hobbys={hobbies} | Humor={humor} | Ziel={goal}]"
        )

        logger.info("User profile saved: %s", _pending_profile_context)
        return {"status": "saved"}
