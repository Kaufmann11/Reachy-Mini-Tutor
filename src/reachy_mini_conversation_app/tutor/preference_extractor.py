"""Preference extractor — detects style/assertiveness changes from natural language."""

import json
from openai import OpenAI

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_preferences",
            "description": "Update user tutoring preferences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_buddy_style": {
                        "type": "string",
                        "enum": ["kollege", "tutor", "dozent"],
                        "description": "Preferred interaction role",
                    },
                    "assertiveness": {
                        "type": "string",
                        "enum": ["gar nicht", "mittel", "sehr"],
                        "description": "Guidance directness level",
                    },
                },
                "additionalProperties": False,
            },
        },
    }
]

SYSTEM = """
You detect if the user wants to change tutoring preferences.

Only call the function if the user clearly expresses a preference.
If there is no clear preference change, do NOT call any function.
"""


def extract_preferences(user_text: str) -> dict:
    """Detect preference changes from user input. Returns dict with new values or None."""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )

    msg = resp.choices[0].message

    if not msg.tool_calls:
        return {"study_buddy_style": None, "assertiveness": None}

    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)

    return {
        "study_buddy_style": args.get("study_buddy_style"),
        "assertiveness": args.get("assertiveness"),
    }