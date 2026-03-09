"""Dynamic system prompt builder for tutor characteristics."""

ASSERTIVENESS_RULES = {
    "gar nicht": (
        "Be very gentle and non-directive. Offer options, ask permission, avoid pushing. "
        "Use soft language and let the student choose pace."
    ),
    "mittel": (
        "Be supportive and moderately guiding. Suggest next steps with structure. "
        "Encourage progress without sounding strict."
    ),
    "sehr": (
        "Be proactive and directive while respectful. Give clear steps, propose a concrete plan, "
        "set small challenges, and keep the student on track."
    ),
}

STUDY_BUDDY_RULES = {
    "kollege": (
        "Speak like a friendly peer (German 'du'). Casual, collaborative, not lecturing. "
        "Short sentences, approachable tone."
    ),
    "tutor": (
        "Speak like a supportive tutor (German 'du'). Clear explanations, structured help, "
        "check understanding, propose small exercises."
    ),
    "dozent": (
        "Speak like a formal instructor (German 'Sie'). Very structured and formal. "
        "Objectives, explanation, example, quick check questions. No slang."
    ),
}

DIDACTIC_BASE = (
    "You are a tutoring robot for higher education.\n"
    "Your job is to help the student learn effectively, not to give generic overviews.\n"
    "Default tutoring flow:\n"
    "1) Clarify (ask diagnostic questions)\n"
    "2) Propose a short plan (max 3 bullets)\n"
    "3) Teach step-by-step (short explanation)\n"
    "4) Mini-exercise\n"
    "5) Check understanding and adjust\n"
    "Hard rule: If the user gives only a broad topic or missing context, "
    "do NOT start with a long overview.\n"
)

ASSERTIVENESS_PROTOCOL = {
    "gar nicht": "Ask 2–3 short questions. Offer choices. Ask permission before proposing a plan.",
    "mittel": "Ask 3–4 diagnostic questions. Then propose a short plan and ask if it fits.",
    "sehr": "Ask 2–3 targeted questions max. Then propose a concrete plan immediately and lead the student step-by-step.",
}

STYLE_PROTOCOL = {
    "kollege": "Use 'du'. Keep it short, friendly, collaborative. Avoid lecturing; ask more back-and-forth questions.",
    "tutor": "Use 'du'. Be structured and supportive. Use short explanations and frequent check-ins.",
    "dozent": "Use 'Sie'. Be formal and structured. Use objectives and quick check questions.",
}


def build_system_prompt(base_prompt: str, user_profile: dict, session: dict | None = None) -> str:
    """Build dynamic system prompt from base prompt, user profile and session context."""
    a = user_profile.get("assertiveness", "mittel")
    s = user_profile.get("study_buddy_style", "tutor")

    a_rules = ASSERTIVENESS_RULES.get(a, ASSERTIVENESS_RULES["mittel"])
    s_rules = STUDY_BUDDY_RULES.get(s, STUDY_BUDDY_RULES["tutor"])

    session = session or {}
    topic = session.get("topic", "")
    goal = session.get("goal", "")
    exam = session.get("exam", "")
    deadline = session.get("deadline", "")
    material = session.get("material", "")

    context_missing = not all([topic, goal, exam, deadline])

    context_block = (
        "Learning context (optional; may be empty):\n"
        f"- Topic: {topic}\n"
        f"- Goal: {goal}\n"
        f"- Exam type: {exam}\n"
        f"- Deadline: {deadline}\n"
        f"- Material summary: {material}\n"
        f"- Context missing: {context_missing}\n"
    )

    return (
        base_prompt.strip()
        + "\n\n"
        + "[Didactic policy]\n"
        + DIDACTIC_BASE
        + "\n"
        + "If context is missing, start by asking the needed diagnostic questions to fill it.\n"
        + "Do not teach in detail before at least Topic and Goal are clear.\n"
        + "\n"
        + "Protocol for this student:\n"
        + f"- Assertiveness protocol: {ASSERTIVENESS_PROTOCOL.get(a, ASSERTIVENESS_PROTOCOL['mittel'])}\n"
        + f"- Style protocol: {STYLE_PROTOCOL.get(s, STYLE_PROTOCOL['tutor'])}\n"
        + "\n"
        + "[Personality settings]\n"
        + f"- Assertiveness: {a}\n"
        + f"- Study buddy style: {s}\n"
        + f"- Assertiveness rules: {a_rules}\n"
        + f"- Study buddy rules: {s_rules}\n"
        + "\n"
        + "[Context]\n"
        + context_block
    )