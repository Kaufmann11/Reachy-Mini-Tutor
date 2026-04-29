import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tutor.conversation_manager import ConversationManager
from reachy_mini_conversation_app.tutor.preference_extractor import extract_preferences
from reachy_mini_conversation_app.tutor.profile_store import set_style, set_assertiveness, get_profile
from reachy_mini_conversation_app.tutor.metrics_logger import log_turn
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# Module-level document context for PDF injection
_pending_document_context: str = ""

# V1 profiles that use the KBD framework and full onboarding
V1_PROFILES: frozenset[str] = frozenset({"tutor_buddy", "tutor_coach", "tutor_professor", "tutor_socratic"})

# Exact text for each onboarding question. The model asks these verbatim,
# with a brief one-sentence acknowledgment of the student's previous answer
# inserted before each question (except Q1).
_ONBOARDING_Q_INSTRUCTIONS: dict[int, str] = {
    1: "Hallo! Ich bin Reachy, dein Lernbegleiter. Bevor wir starten, habe ich kurz ein paar Fragen. Wie heißt du?",
    2: "Was studierst du, und in welchem Semester bist du gerade?",
    3: "Wie gerne lernst du generell — machst du es eher weil du es musst, oder interessiert dich das Thema wirklich?",
    4: "Was motiviert dich beim Lernen am meisten — zum Beispiel eine gute Note, das Verstehen an sich, oder etwas anderes?",
    5: "Wie lernst du am liebsten — eher durch Erklärungen, durch Beispiele, durch Übungsaufgaben, oder durch Fragen?",
    6: "Hast du Hobbys oder Interessen außerhalb des Studiums? Und lernst du lieber sachlich oder darf's auch mal humorvoll sein?",
    7: "Was möchtest du heute in unserer Session erreichen?",
}

_ONBOARDING_LABELS: dict[int, str] = {
    1: "Name",
    2: "Studium/Semester",
    3: "Lernmotivation",
    4: "Motivator",
    5: "Lernstil",
    6: "Hobbys+Humor",
    7: "Session-Ziel",
}


def _extract_name(raw: str) -> str:
    """Extract a clean first-name token from a Q1 answer.

    Examples:
      'Mein Name ist Mike.' → 'Mike'
      'Ich heiße Mike' → 'Mike'
      'Mike.' → 'Mike'
      'Ich bin Anna-Lena' → 'Anna-Lena'
      'Mein Name Mike' → 'Mike'  (even without 'ist')
      'Also ich heiße Mike' → 'Mike'
    """
    import re
    cleaned = raw.strip().rstrip(".!?,;: ")
    # Remove common German self-introduction prefixes (robust: "ist" optional, tolerate fillers).
    patterns = [
        r"^(also|ja|hallo|hi|hey)[\s,]+",
        r"^mein(e)?\s+nam(e)?(\s+ist)?\s+",
        r"^ich\s+heiß?e\s+",
        r"^ich\s+heisse\s+",
        r"^ich\s+bin\s+(der\s+|die\s+)?",
        r"^das\s+bin\s+(der\s+|die\s+)?",
        r"^ich\s+nenne\s+mich\s+",
        r"^name\s*[:\-]?\s*",
    ]
    # Apply repeatedly in case multiple stacked prefixes (e.g. "Also mein Name ist")
    for _ in range(3):
        before = cleaned
        for p in patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
        if cleaned == before:
            break
    # Blacklist of German filler / pronoun / article / determiner tokens that must
    # never be returned as a name. Guards against transcript glitches like
    # "Mein Name Mike" where a prefix couldn't be stripped, or the user adding
    # stray filler words before their name.
    _NAME_STOPWORDS = {
        "mein", "meine", "name", "ist", "ich", "bin", "heiße", "heisse",
        "der", "die", "das", "ein", "eine", "hallo", "hi", "hey",
        "also", "ja", "äh", "ähm", "hm", "halt", "einfach", "nun",
        "nenne", "mich", "bin's", "bins",
    }
    tokens = cleaned.split()
    # Step past any stopword tokens that survived prefix-stripping
    for tok in tokens:
        clean_tok = tok.rstrip(".!?,;:").strip()
        if not clean_tok:
            continue
        if clean_tok.lower() in _NAME_STOPWORDS:
            continue
        return clean_tok
    return ""


def _extract_primary_hobby(raw: str) -> str:
    """Extract the primary hobby/interest noun from a Q6 answer.

    Takes the first clearly-content word, stripping common prefixes like
    'Ich spiele gerne', 'Am Alltag des Studiums gehe ich ins', etc.
    Returns the first meaningful noun-like token or a short phrase.
    """
    import re
    cleaned = raw.strip().rstrip(".!?,;: ")
    # Strip common prefixes
    patterns = [
        r"^ja,?\s+",
        r"^ich\s+spiele\s+(gerne\s+)?",
        r"^ich\s+(gehe|treibe|mache|lese|höre)\s+(gerne\s+)?",
        r"^am\s+alltag\s+des\s+studiums\s+(gehe\s+ich\s+)?(ins\s+|zum\s+)?",
        r"^ausserhalb\s+des\s+studiums\s+",
        r"^außerhalb\s+des\s+studiums\s+",
        r"^meine\s+hobby(s|ies)?\s+sind\s+",
        r"^hobby(s|ies)?\s*:\s*",
    ]
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    # Cut at first "und"/"oder"/"aber" to get primary hobby
    for sep in [" und ", " oder ", " aber ", ", ", "; "]:
        idx = cleaned.lower().find(sep)
        if idx > 0:
            cleaned = cleaned[:idx]
            break
    return cleaned.strip().rstrip(".!?,;:")


# Reactive triggers — regex patterns on last user turn.
# When a pattern matches, a mandatory instruction is prepended to the V1 per-turn prompt.
# Deterministic in code: if the signal is there, the mandate fires.
_TRIGGERS: dict[str, str] = {
    "frustration": r"\b(kein(en)?\s+(bock|lust)|zu\s+(viel|schwer|schwierig|schnell)|überfordert|schaffe\s+ich\s+nie|ich\s+kann\s+das\s+nicht|hab\s+keine\s+kraft|bin\s+müde|keine\s+energie)\b",
    "no_idea":     r"\b(keine\s+ahnung|keinen\s+(plan|schimmer|dunst)|(ich\s+)?(weiß|weiss)\s+(es\s+|das\s+)?nicht|(ich\s+)?(hab|habe)\s+keine\s+(ahnung|idee)|(ich\s+)?(kann|könnte)\s+(ich\s+)?nicht\s+sagen|keine\s+idee|no\s+idea|puh\s+(keine|kein)|echt\s+keine|einfach\s+keine|schwer\s+zu\s+sagen|das\s+(weiß|weiss)\s+ich\s+nicht|unklar)\b",
    "confusion":   r"\b(versteh(e)?\s+(ich\s+)?nicht|hä\??|was\s+meinst\s+du|kapier(e)?\s+nicht|check\s+ich\s+nicht)\b",
    "identity":    r"\b(bist\s+du\s+(ein\s+)?(mensch|echte?r?\s+(person|mensch))|bist\s+du\s+(eine\s+)?(ai|ki|bot)|wirklich\s+ein\s+roboter)\b",
    "camera":      r"\b(siehst\s+du|kannst\s+du\s+(das\s+)?sehen|sieh\s+(dir\s+)?an|auf\s+(der|meiner)\s+folie|zeig\s+(ich|dir)\s+dir|guck\s+mal)\b",
    "content_q":   r"\b(erkläre?\s+mir|was\s+(ist|bedeutet|heißt)|wie\s+funktioniert|definier(e)?|erklär\s+mir)\b",
    "exam":        r"\b(klausur|prüfung|hausaufgabe|aufgabe\s+lösen|lösung\s+der\s+aufgabe|musterlösung)\b",
    "depth_req":   r"\b(oberflächlich|zu\s+wenig|zu\s+kurz|tiefer|mehr\s+details|ausführlicher|genauer\s+erklär|verstehe\s+immer\s+noch\s+nicht|war\s+zu\s+schnell)\b",
    # User-Override: "stop asking questions / just explain". When this fires the
    # default "schließe mit Check-Frage ab"-mandate must be SUPPRESSED for several
    # turns. Without it the model keeps asking despite explicit user instruction.
    "no_questions": r"(stell\s+(mir\s+)?keine\s+fragen|hör\s+(bitte\s+)?auf\s+(zu\s+)?fragen|frag\s+(mich\s+)?nicht|nicht\s+(jedes\s*mal|immer|ständig|dauernd|so\s+(viel|viele))\s+\w*\s*fragen|nicht\s+\w*\s*fragen\s+stell|jedes\s*mal\s+\w*\s*fragen\s+stell|(einfach|nur|bitte)\s+(weiter\s+)?(erklären|erklär|zusammenfassen)|keine\s+(rück\s*-?\s*)?fragen|ohne\s+(rück\s*-?\s*)?fragen|mach\s+(mir\s+)?(einfach|bitte)\s+(die\s+)?zusammenfassung|zuerst\s+(einfach\s+)?(zusammenfassen|erklären|erklär)|erst(\s+mal)?\s+(zusammenfassen|erklären|erklär)|kannst\s+du\s+(mir\s+)?(nicht|aufhören)\s+\w*\s*frag|hör\s+auf\s+\w*\s*zu\s+fragen)",
}


def _build_reactive_mandates(
    user_text: str,
    profile_data: dict,
    name: str,
    hobby: str = "",
) -> tuple[list[str], list[str]]:
    """Detect reactive signals in the user's last turn and build mandatory instructions.

    Returns (mandates, fired_triggers) — mandates are imperative lines to prepend to
    the per-turn prompt; fired_triggers is the list of trigger names for logging.
    Mandates are built with LERNPROFIL data (motivation, session_goal) where relevant.
    """
    import re
    if not user_text:
        return [], []
    text_lower = user_text.lower()
    mandates: list[str] = []
    fired: list[str] = []

    motivation = (profile_data.get(3) or "").strip()
    session_goal = (profile_data.get(7) or "").strip()

    for name_key, pattern in _TRIGGERS.items():
        if re.search(pattern, text_lower, flags=re.IGNORECASE):
            fired.append(name_key)

    if "frustration" in fired:
        # Tie motivation/goal into the reframe — makes it personal, not generic
        context_anchor = ""
        if session_goal:
            context_anchor = f" Erinnere konkret an sein Ziel: '{session_goal}'."
        elif motivation:
            context_anchor = f" Erinnere an seine Motivation: '{motivation}'."
        mandates.append(
            f"REAKTIV — FRUSTRATION: {name or 'Der Student'} äußert Frust/Überforderung. "
            f"Beginne mit EINEM Satz echter emotionaler Anerkennung (nicht floskelhaft), "
            f"DANN ein konkreter, kleiner nächster Schritt.{context_anchor} "
            f"Mitfühlend, nicht belehrend, nicht abwiegeln."
        )

    if "no_idea" in fired:
        hobby_hint = f" Wenn eine Analogie zu '{hobby}' natürlich passt, nutze sie." if hobby else ""
        name_prefix = f"{name}, " if name else ""
        mandates.append(
            f"REAKTIV — UNSICHERHEIT: Beginne mit KURZER empathischer Anerkennung mit Namen "
            f"(z.B. '{name_prefix}kein Stress — lass uns das zusammen knacken'). "
            f"KEINE direkte Lösung. Stelle dann EINE DEUTLICH einfachere Teilfrage — "
            f"nicht dieselbe Frage anders formuliert, sondern einen echten Schritt zurück.{hobby_hint} "
            f"Erst nach dem 2. Fehlversuch ein winziger Hinweis."
        )

    if "confusion" in fired:
        mandates.append(
            "REAKTIV — VERWIRRUNG: Formuliere deinen letzten Gedanken in einfacheren Worten neu — "
            "andere Wortwahl, konkrete Analogie. NICHT dieselben Wörter wiederholen."
        )

    if "identity" in fired:
        mandates.append(
            "REAKTIV — IDENTITÄT: Antworte EHRLICH: 'Ich bin Reachy Mini, ein Roboter mit KI.' "
            "Kurz, ohne Umschweife. KEIN Ausweichen, KEINE Rolle spielen."
        )

    if "camera" in fired:
        mandates.append(
            "REAKTIV — KAMERA: In dieser Session ist KEINE Kamera aktiv. "
            "Sag das ehrlich: 'Sehen kann ich in dieser Session nicht.' "
            "Falls Folien hochgeladen: 'Ich kann den Text der Folien über rag_tool abrufen.'"
        )

    if "content_q" in fired:
        mandates.append(
            "REAKTIV — INHALTSFRAGE: BEVOR du erklärst, stelle EINE aktivierende Gegenfrage: "
            "'Was weißt du schon dazu?' oder 'Was ist dein erster Gedanke?' "
            "So öffnest du den Socratic-Dialog statt direkt Wissen abzuladen."
        )

    if "exam" in fired:
        mandates.append(
            "REAKTIV — PRÜFUNG/AUFGABE: KEINE komplette Lösung oder Musterantwort geben. "
            "Führe durch den Denkweg mit Fragen. Der Student muss selbst drauf kommen."
        )

    if "depth_req" in fired:
        mandates.append(
            "REAKTIV — TIEFE GEWÜNSCHT: Der Student hat explizit mehr Tiefe verlangt. "
            "Liefere JETZT eine ausführliche Erklärung in 3–4 Sätzen mit konkreten Details und Fachbegriffen, "
            "DANN ein konkretes, spezifisches Beispiel (nicht generisch), "
            "DANN EINE Check-Frage. KEINE weitere Sokratik-Kette an dieser Stelle — "
            "erst Verstehen herstellen, dann wieder fragen."
        )

    return mandates, fired


def _is_probably_noise_in_tutoring(text: str) -> bool:
    """Stricter filter for tutoring-phase turns.

    Catches single-word VAD/transcription artifacts like 'Lauterbach',
    'Klavierlehrer', 'Selbstvertrauen', 'Lautsprecher', 'Wöschwüchi' that
    the gpt-4o-transcribe sometimes produces on background noise or
    mid-sentence pauses. These get through `_is_valid_onboarding_answer`
    because they're >2 chars and not in the filler list.

    Return True if the text should be treated as noise and ignored.
    """
    import re
    stripped = (text or "").strip().rstrip(".!?,;:")
    if not stripped:
        return True
    words = stripped.split()
    # Multi-word turns are rarely noise — let them through.
    if len(words) >= 2:
        return False
    # Single word case: allow valid short replies and reactive signals.
    lower = words[0].lower()
    short_valid = {
        "ja", "nein", "ne", "doch", "ok", "okay", "weiter",
        "stopp", "halt", "pause", "nächste", "zurück",
        "gut", "richtig", "falsch", "verstanden",
        "genau", "stimmt",
    }
    if lower in short_valid:
        return False
    # If the single word matches a reactive trigger (unlikely for 1 word
    # but keep the check), let it through.
    for pattern in _TRIGGERS.values():
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return False
    # Otherwise: single unfamiliar word in tutoring → almost certainly noise.
    return True


def _is_valid_onboarding_answer(text: str, q_num: int) -> bool:
    """Return True if the user turn looks like a real answer (not a counter-question or filler)."""
    stripped = text.strip()
    if not stripped or len(stripped) < 2:
        return False
    words = stripped.split()
    # Pure short question → probably not an answer
    if len(words) <= 3 and stripped.endswith("?"):
        return False
    lower = stripped.lower().rstrip(".!?,;: ")
    # Single-word greetings, fillers, confusion, and transcription phantoms
    # (gpt-4o-transcribe sometimes hallucinates short German words on silence/noise)
    fillers = {
        "was", "hm", "hmm", "äh", "ähm", "wie bitte", "bitte was", "was meinst du", "was meinst",
        "hallo", "hi", "hey", "ok", "okay", "ja", "nein", "ne", "ach so", "alles klar",
        "moment", "warte", "warte mal", "ach", "oh", "achso",
        # Common transcription phantoms during silence
        "natürlich", "genau", "sicher", "klar", "doch", "eben", "bestimmt",
        "vielleicht", "wirklich", "schön", "super", "danke",
    }
    if lower in fillers:
        return False
    # Single char or two-char non-names
    if len(words) == 1 and len(stripped) <= 2:
        return False
    return True


def _build_lernprofil(answers: dict) -> str:
    lines = [f"- {_ONBOARDING_LABELS[i]}: {answers.get(i, '?')}" for i in range(1, 8)]
    return "[LERNPROFIL — Onboarding:\n" + "\n".join(lines) + "]"

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates to match OpenAI's requirements
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        # Override type annotations for OpenAI strict typing (only for values used in API)
        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self._response_audio_produced = False   # True once audio.delta fires in current response
        self._response_create_issued = False    # True if tool handler already called response.create
        self._onboarding_q_pending = False      # True while an onboarding-Q response is being generated
        self._movement_dispatched_this_response = False  # True after first movement tool dispatched in current response
        self._movement_blocked_until_user_input = False  # True after any movement; cleared on next user speech
        self._user_speech_during_current_response = False  # True if user started speaking during current bot response (used to suppress watchdog races)
        self._current_response_is_idle = False  # True for the duration of an idle_signal response (intentional movement-only); suppresses verbal-retry watchdog
        # Onboarding state machine — reset at the start of every session
        self._onboarding: dict = {
            "phase": "onboarding",    # "onboarding" | "tutoring"
            "current_q": 0,           # 0 = waiting for user to initiate; 1–7 = active Q
            "answers": {},            # {1: "Max", 2: "BWL 3. Sem", ...}
            "profile_injected": False,
        }
        self._lernprofil_text: str = ""  # Cached LERNPROFIL for per-turn V1 re-injection
        self._lernprofil_name: str = ""  # Extracted student name from Q1
        self._lernprofil_hobbies: str = ""  # Extracted hobbies/interests from Q6
        self._lernprofil_study: str = ""  # Extracted study program from Q2 (V1 background bridge)
        # Cached V1 per-turn mandate string for the watchdog verbal-retry path.
        # When the bot calls a movement tool but produces no audio, the watchdog
        # fires a separate response.create. Without this cache that retry would
        # carry only a generic recovery prompt — the per-turn KBD mandates
        # (HOBBY/STUDIUM/HUMOR/NAME PFLICHT) would be lost, so the user-visible
        # response would be mandate-free. We replay the same mandate.
        self._last_tutoring_instructions: str = ""
        self._humor_welcomed: bool = False  # Parsed from Q6 — drives periodic humor mandate
        self._chosen_method: str = ""  # Post-onboarding learning approach (slide/overview/exercise)
        self._post_onboarding_stage: str = ""  # "" | "awaiting_deadline" | "awaiting_wissensstand" | "awaiting_method" | "done"
        self._tutoring_turn_count: int = 0  # Bot tutoring turns since onboarding
        self._last_name_used_turn: int = -99  # Turn index when name was last spoken
        self._document_uploaded: bool = False  # True once student uploaded any doc
        self._onboarding_item_ids: list[str] = []  # V2: delete these to erase onboarding context
        # --- Pacing + Frustrations-Override state (V1) ---
        # Triggered by reactive signals in the user's last turn. When on, the
        # NEXT bot turn must be in EXPLAIN-Mode: deliver 2-3 sentences of fact +
        # one yes/no check question. No Socratic chain, no hobby analogy.
        self._explain_mode_next_turn: bool = False
        self._no_idea_streak: int = 0   # consecutive user turns with no_idea trigger
        # Sticky countdown after user explicitly asks to stop being questioned.
        # When >0, the default "schließe mit Check-Frage ab"-mandate is replaced
        # with a hard "NO question-back" mandate for this and the next N-1 turns.
        # Decremented once per tutoring turn. Reset to 3 whenever no_questions fires.
        self._no_questions_remaining: int = 0
        self._einsteiger_flag: bool = False  # student described self as Einsteiger/beginner
        self._deadline_flag: bool = False    # student mentioned deadline/Prüfung/MC
        self._mc_flag: bool = False          # multiple-choice specifically
        # V2 session-reset: conversation.item.delete does NOT reliably clear
        # GPT-4o's working state — name/study/hobbies leak into tutoring even
        # after deletion. Only a full WebSocket reconnect fully erases context.
        # When this flag is set, the next _run_realtime_session iteration will
        # skip onboarding and resume directly at Stage-1a (deadline question).
        self._v2_resume_to_tutoring: bool = False
        # --- Single-flight guard for response.create ---
        # The Realtime API serializes responses; a second response.create while one
        # is still active is rejected with conversation_already_has_active_response.
        # We see this race in production between Stage-1/Stage-2 post-onboarding
        # prompts, the verbal-retry watchdog, and tool-follow-up calls. Instructions
        # bundled on the rejected call never reach the model — that is the root
        # cause of "no deadline question", "no reaction to keine Ahnung", etc.
        # Solution: route every response.create through _safe_response_create, which
        # queues calls while a response is active and drains on response.done.
        self._response_active: bool = False
        self._pending_response_creates: list[tuple[dict, str]] = []
        # Dedup set for transcript events (SDK fires legacy + GA alias for same response).
        self._seen_transcript_keys: set[tuple] = set()
        # Last assistant transcript + timestamp for content-based dedup — catches
        # model-level "speak → emotion → speak again" repeats after a movement tool.
        self._last_assistant_transcript: tuple[str, float] = ("", 0.0)
        # VAD-cut merge: when a long user turn is split into multiple
        # input_audio_transcription.completed events within ~1s, the bot
        # otherwise fires one response per fragment. We debounce: every new
        # tutoring-phase transcript cancels the pending response task and
        # schedules a fresh one a short delay later, so all fragments land
        # in the conversation before a single response is generated.
        self._tutoring_response_debounce_task: asyncio.Task | None = None
        # Watchdog: if response.done never arrives (network hiccup, error not
        # surfaced, etc.) the single-flight flag stays True forever and all
        # subsequent response.creates are queued and never drain — session
        # appears frozen. Reset task tracks the latest response.created and
        # auto-clears the flag if response.done doesn't follow within 45s.
        self._response_active_watchdog_task: asyncio.Task | None = None
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        # Track how the API key was provided (env vs textbox) and its value
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0  # sequence counter to prevent stale emissions
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = get_session_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session={
                            "type": "realtime",
                            "instructions": instructions,
                            "audio": {"output": {"voice": voice}},
                        },
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            # Only emit if this is still the latest partial (by sequence number)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        openai_api_key = config.OPENAI_API_KEY
        if self.gradio_mode and not openai_api_key:
            # api key was not found in .env or in the environment variables
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args[3]) > 0 else None
            if textbox_api_key is not None:
                openai_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                openai_api_key = config.OPENAI_API_KEY
        else:
            if not openai_api_key or not openai_api_key.strip():
                # In headless console mode, LocalStream now blocks startup until the key is provided.
                # However, unit tests may invoke this handler directly with a stubbed client.
                # To keep tests hermetic without requiring a real key, fall back to a placeholder.
                logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                openai_api_key = "DUMMY"

        self.client = AsyncOpenAI(api_key=openai_api_key)

        max_attempts = 3
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._run_realtime_session()
                # Normal exit. If V2 session-reset was requested, immediately
                # reconnect (no backoff, no attempt counter increment beyond
                # this iteration — it's an intentional reset, not a failure).
                if self._v2_resume_to_tutoring:
                    logger.info("V2 session-reset requested — reconnecting immediately")
                    self.connection = None
                    try:
                        self._connected_event.clear()
                    except Exception:
                        pass
                    attempt = 0  # don't count intentional resets against retry budget
                    continue
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                if self._v2_resume_to_tutoring:
                    # The close was triggered by our V2 reset path — reconnect
                    # directly, do not count as a failure.
                    logger.info("V2 session-reset (via close-exception) — reconnecting immediately")
                    self.connection = None
                    try:
                        self._connected_event.clear()
                    except Exception:
                        pass
                    attempt = 0
                    continue
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference (when actually exiting)
                if not self._v2_resume_to_tutoring:
                    self.connection = None
                    try:
                        self._connected_event.clear()
                    except Exception:
                        pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            asyncio.create_task(self._run_realtime_session(), name="openai-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _safe_response_create(
        self, response: Optional[dict] = None, label: str = "",
        drop_if_active: bool = False,
    ) -> bool:
        """Single-flight wrapper around connection.response.create.

        The Realtime API serializes responses: a second response.create while
        one is still active is rejected with conversation_already_has_active_response,
        and any instructions on the rejected call never reach the model. We
        queue subsequent calls and drain them on response.done.

        Set drop_if_active=True for calls that are redundant when a response is
        already in flight (e.g. tutoring turn responses — the next user input
        is already in the conversation context, no need to fire a stale-instruction
        duplicate after the active one finishes). Stage-1/Stage-2/Watchdog
        triggers must still queue (they are state-machine transitions).

        Returns True if the call was issued immediately, False if it was queued
        or dropped.
        """
        if not self.connection:
            return False
        payload: dict = {"response": response if response is not None else {}}
        if self._response_active:
            if drop_if_active:
                logger.info(
                    "Dropping redundant response.create (label=%s) — response already active",
                    label or "-",
                )
                return False
            self._pending_response_creates.append((payload, label))
            logger.info(
                "Queued response.create (label=%s, queue_depth=%d)",
                label or "-", len(self._pending_response_creates),
            )
            return False
        # Mark active optimistically; response.created will re-confirm, and if the
        # API rejects the call we roll back below so the queue isn't stuck.
        self._response_active = True
        try:
            await self.connection.response.create(**payload)
            if label:
                logger.debug("Issued response.create (label=%s)", label)
            return True
        except Exception as e:
            self._response_active = False
            logger.warning("response.create failed (label=%s): %s", label or "-", e)
            # Drain next queued entry so one failure doesn't freeze the queue.
            await self._drain_pending_responses()
            return False

    async def _drain_pending_responses(self) -> None:
        """Fire the next queued response.create, if any. Called on response.done."""
        if self._response_active or not self._pending_response_creates:
            return
        if not self.connection:
            self._pending_response_creates.clear()
            return
        payload, label = self._pending_response_creates.pop(0)
        self._response_active = True
        try:
            await self.connection.response.create(**payload)
            logger.info(
                "Drained queued response.create (label=%s, remaining=%d)",
                label or "-", len(self._pending_response_creates),
            )
        except Exception as e:
            self._response_active = False
            logger.warning("Queued response.create failed (label=%s): %s", label or "-", e)
            # Try the next one so a single failure doesn't block the queue.
            await self._drain_pending_responses()

    async def _ask_onboarding_question(self, q_num: int, reask: bool = False) -> None:
        """Trigger a regular response that asks onboarding question q_num verbatim.

        For Q2–Q7, the model also briefly acknowledges the student's previous answer
        (one short sentence) before asking the next question. For Q1, the model just
        asks the question after the student's greeting. For re-asks (invalid answer),
        the model asks the same question again in a friendly way.

        The response runs with full conversation context so the model can see what the
        student just said. The per-response instructions force the exact question text.
        """
        if not self.connection:
            return
        question_text = _ONBOARDING_Q_INSTRUCTIONS[q_num]
        _cur_profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or ""
        _is_v2 = _cur_profile == "tutor_basic"

        if _is_v2:
            # V2 control condition: NEVER address by name, NEVER comment on
            # the previous answer, NEVER praise. The whole point is to
            # produce no personalization activations the model could leak
            # later. Just acknowledge neutrally and ask the next question.
            if q_num == 1:
                instructions = (
                    "Die/Der Studierende hat gerade die Unterhaltung begonnen. "
                    f"Stelle jetzt GENAU diese Frage, Wort für Wort:\n\n\"{question_text}\"\n\n"
                    "Sprich neutral und sachlich. KEINE persönliche Anrede. "
                    "Stelle nur diese eine Frage."
                )
            elif reask:
                instructions = (
                    "Die/Der Studierende hat nicht klar geantwortet. "
                    f"Stelle die gleiche Frage neutral nochmal, Wort für Wort:\n\n\"{question_text}\"\n\n"
                    "KEINE Umformulierung. KEIN Lob, keine Wärme, keine persönliche Anrede. "
                    "Stelle nur diese eine Frage."
                )
            elif q_num == 2:
                # Q2 acks Q1 (the name). Allow "Freut mich, [Name]." once —
                # makes the only personal opener a tiny bit warmer without
                # re-introducing personalization later. Name is NOT used
                # again for the rest of the V2 session.
                instructions = (
                    "Die/Der Studierende hat gerade den Namen genannt. "
                    "Beginne mit GENAU einem kurzen Übergang: "
                    "'Freut mich, [Name].' (Name wörtlich aus der Antwort) "
                    "ODER neutral 'Alles klar.' / 'Okay.'. Sonst nichts Persönliches. "
                    "KEIN Lob ('schön', 'super', 'spannend'), KEIN Kommentar zum Namen. "
                    f"Stelle dann diese Frage, Wort für Wort:\n\n\"{question_text}\"\n\n"
                    "Keine Umformulierung, keine zweite Frage. Sprich sachlich."
                )
            else:
                instructions = (
                    "Die/Der Studierende hat gerade geantwortet. "
                    "Optional EIN kurzer neutraler Übergang (z.B. 'Alles klar.', 'Okay.', "
                    "'Verstanden.', 'Gut.') ODER direkt zur nächsten Frage — variiere, "
                    "wiederhole NICHT bei jeder Frage denselben Übergang. "
                    "KEINE Wiederholung der Antwort, KEIN Name, KEIN Lob, "
                    "KEIN Kommentar zum Inhalt. "
                    f"Stelle dann diese Frage, Wort für Wort:\n\n\"{question_text}\"\n\n"
                    "Keine Umformulierung, keine zweite Frage. Sprich sachlich."
                )
        elif q_num == 1:
            instructions = (
                "Die/Der Studierende hat gerade die Unterhaltung begonnen (z.B. mit 'Hallo'). "
                f"Stelle jetzt GENAU diese Frage, Wort für Wort, unverändert:\n\n\"{question_text}\"\n\n"
                "Keine Einleitung davor, keine Umformulierung, keine zusätzliche Erklärung. "
                "Nur genau diesen Satz sprechen. Stelle in dieser Antwort NUR diese eine Frage."
            )
        elif reask:
            instructions = (
                "Die/Der Studierende hat nicht klar auf die letzte Frage geantwortet. "
                f"Stelle die gleiche Frage freundlich nochmal, GENAU so, Wort für Wort:\n\n\"{question_text}\"\n\n"
                "Keine Umformulierung. Stelle in dieser Antwort NUR diese eine Frage."
            )
        else:
            prev_label = _ONBOARDING_LABELS.get(q_num - 1, "")
            instructions = (
                f"Die/Der Studierende hat gerade auf deine Frage zu '{prev_label}' geantwortet. "
                "Gehe ganz kurz auf die Antwort ein — EIN Satz, warm und spezifisch zu dem was tatsächlich gesagt wurde. "
                "Kein leeres Lob, keine Floskel. "
                "WICHTIG: Die Antwort wurde bereits validiert — akzeptiere sie IMMER als gültig. "
                "Auch sehr kurze Ein-Wort-Antworten sind vollwertige Antworten. "
                "Auch in Füllwörter/Abschweifungen/Versprecher eingebettete Infos sind gültig. "
                "Extrahiere die Kerninformation und erwähne sie in deiner kurzen Reaktion — "
                "verwende ausschließlich die WÖRTER DES STUDENTEN, niemals Beispielwörter aus dieser Anweisung. "
                "Sage NIEMALS 'Ich habe das nicht ganz verstanden' — wenn die Antwort hier ankommt, ist sie gültig. "
                "\nABSOLUTE ANTI-HALLUZINATIONS-REGEL: "
                "Wiederhole AUSSCHLIESSLICH Wörter, Zahlen, Fächer und Namen, die der Studierende TATSÄCHLICH GESAGT hat. "
                "Wenn eine konkrete Zahl/Bezeichnung gesagt wurde, übernimm sie wörtlich, niemals eine andere. "
                "Wenn du eine Zahl oder ein Fach nicht ganz sicher gehört hast, lass sie WEG. "
                "Ergänze NIE Antwortoptionen aus deiner vorigen Frage ('durch Fragen' etc.), die der Student gar nicht genannt hat. "
                "Im Zweifel: weniger wiederholen. "
                "\nVERBOTEN: Konstruktion 'Du hast gesagt: \"…\"' / 'Du sagtest: \"…\"' / 'Wie du sagtest, …' "
                "mit einem Zitat in Anführungszeichen. Diese Konstruktion verleitet dich zu einer erfundenen "
                "Quasi-Wörtlich-Wiedergabe. Erlaubt: paraphrasierende Anerkennung OHNE Zitat-Anführung, "
                "z.B. 'Klingt nach einem fairen Plan.' / 'Spannend, [Name].' / 'Eine gute Note ist ein klares Ziel.' — "
                "in eigenen Worten, kurz, ohne wörtliche Wiedergabe. "
                "\nWENN DIE ANTWORT NICHT ZUR FRAGE PASST (z.B. ein Name als Antwort auf eine Motivations-Frage, "
                "oder ein einzelnes themenfremdes Wort): NICHT halluzinieren, was der Student angeblich gemeint hat. "
                "Stattdessen: ein neutrales 'Alles klar.' / 'Okay.' und direkt die nächste Frage stellen. "
                "Im Zweifel knapp neutral, niemals Inhalt erfinden. "
                f"\nStelle danach GENAU diese nächste Frage, Wort für Wort, unverändert:\n\n\"{question_text}\"\n\n"
                "Die Frage muss wörtlich genau so vorkommen. Keine Umformulierung, keine zusätzlichen Erklärungen, "
                "keine Aufzählung anderer Themen. Stelle in dieser Antwort NUR diese eine Frage — keine zweite Frage."
            )
        try:
            # Cancel any pending delayed lock-release from a previous Q so it can't
            # clobber the lock we are about to set.
            if self._q_lock_release_task and not self._q_lock_release_task.done():
                self._q_lock_release_task.cancel()
            self._response_create_issued = True
            self._onboarding_q_pending = True
            # Onboarding Qs use tool_choice="none": movement during a Q ask has
            # no didactic value and causes the model to repeat the question
            # after the emotion ("speak → emotion → speak again" pattern), which
            # duplicates the transcript in the UI. Stage-1/Stage-2 already use
            # "none" for the same reason.
            await self._safe_response_create(
                response={
                    "instructions": instructions,
                    "tool_choice": "none",
                    "tools": [],
                },
                label=f"onboarding_q{q_num}{'_reask' if reask else ''}",
            )
            logger.info("Asked onboarding Q%d (reask=%s)", q_num, reask)
        except Exception as e:
            self._onboarding_q_pending = False
            logger.warning("Onboarding Q%d ask failed: %s", q_num, e)

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        import re
        # V2 session-reset path: when this flag is True, we are resuming
        # AFTER Q7 with a fresh WebSocket so onboarding context is gone.
        # Skip onboarding init entirely and go straight to Stage-1a.
        _resume_to_tutoring = self._v2_resume_to_tutoring
        self._v2_resume_to_tutoring = False
        # Reset per-session state so restarts start clean
        self._onboarding = {
            "phase": "tutoring" if _resume_to_tutoring else "onboarding",
            "current_q": 8 if _resume_to_tutoring else 0,
            "answers": {},
            "profile_injected": False,
        }
        self._response_audio_produced = False
        self._response_create_issued = False
        self._onboarding_q_pending = False
        self._movement_dispatched_this_response = False
        self._movement_blocked_until_user_input = False
        self._tutoring_verbal_retry_fired = False  # watchdog ran once for this user turn
        self._q_lock_release_task: asyncio.Task | None = None
        self._lernprofil_text = ""
        self._lernprofil_name = ""
        self._lernprofil_hobbies = ""
        self._lernprofil_study = ""
        self._humor_welcomed = False
        self._chosen_method = ""
        self._post_onboarding_stage = "awaiting_deadline" if _resume_to_tutoring else ""
        self._tutoring_turn_count = 0
        self._last_name_used_turn = -99
        self._document_uploaded = False
        self._onboarding_item_ids = []
        self._explain_mode_next_turn = False
        self._no_idea_streak = 0
        self._no_questions_remaining = 0
        self._einsteiger_flag = False
        self._deadline_flag = False
        self._mc_flag = False
        self._response_active = False
        self._pending_response_creates = []
        self._seen_transcript_keys = set()
        self._last_assistant_transcript = ("", 0.0)
        if self._tutoring_response_debounce_task and not self._tutoring_response_debounce_task.done():
            self._tutoring_response_debounce_task.cancel()
        self._tutoring_response_debounce_task = None
        if self._response_active_watchdog_task and not self._response_active_watchdog_task.done():
            self._response_active_watchdog_task.cancel()
        self._response_active_watchdog_task = None
        conv = ConversationManager("student_001")
        user_id = "student_001"
        # Tracks the most recent user transcript across iterations of the event loop.
        # Initialized here so callers (V1 reactive-mandate builder, metrics logger) can
        # access it on the very first tutoring turn without the fragile `"..." in dir()`
        # pattern and without NameError.
        last_user_text: str = ""
        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            try:
                await conn.session.update(
                    session={
                        "type": "realtime",
                        "instructions": get_session_instructions(),
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.input_sample_rate,
                                },
                                "transcription": {"model": "gpt-4o-transcribe", "language": "de"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.85,
                                    "silence_duration_ms": 1400,
                                    "interrupt_response": True,
                                    "create_response": False,
                                },
                            },
                            "output": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.output_sample_rate,
                                },
                                "voice": get_session_voice(),
                            },
                        },
                        "tools": get_tool_specs(),  # type: ignore[typeddict-item]
                        "tool_choice": "auto",
                    },
                )
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    get_session_voice(),
                )
                # If we reached here, the session update succeeded which implies the API key worked.
                # Persist the key to a newly created .env (copied from .env.example) if needed.
                self._persist_api_key_if_needed()
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                return

            logger.info("Realtime session updated successfully")

            _cur_profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or ""
            _is_tutor_profile = _cur_profile in V1_PROFILES or _cur_profile == "tutor_basic"

            # Tutor profiles wait for the user to speak first (current_q==0).
            # Non-tutor profiles get an immediate greeting.
            if not _is_tutor_profile:
                # Initial session greeting — no prior response could be active,
                # so skip the safe-guard wrapper. Set the flag directly; it will
                # be cleared on the first response.done.
                try:
                    self._response_active = True
                    await conn.response.create(response={})
                    logger.info("Triggered initial greeting for non-tutor profile=%s", _cur_profile)
                except Exception as e:
                    self._response_active = False
                    logger.warning("Initial response.create failed: %s", e)
            else:
                logger.info("Waiting for user to initiate conversation (profile=%s)", _cur_profile)

            # Manage event received from the openai server
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass

            # V2 resume: fire Stage-1a (deadline question) directly into the
            # fresh session. Onboarding context is gone — model has nothing
            # to leak. This is the connection AFTER Q7 reset.
            if _resume_to_tutoring and _cur_profile == "tutor_basic":
                logger.info("V2 session-reset complete — firing Stage-1a deadline question")
                await self._safe_response_create(
                    response={
                        "instructions": (
                            "Stelle GENAU EINE Frage, wörtlich: "
                            "'Gibt es eine Deadline oder Abgabe zu diesem Thema, "
                            "oder ist es ein freies Lernziel?' "
                            "Keine Einleitung, keine zweite Frage. Sprich neutral und sachlich."
                        ),
                        "tool_choice": "none",
                        "tools": [],
                    },
                    label="v2_resume_stage1a_deadline",
                )
            async for event in self.connection:
                logger.debug(f"OpenAI event: {event.type}")
                if event.type == "input_audio_buffer.speech_started":
                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                        self._clear_queue()
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)
                    self._movement_blocked_until_user_input = False
                    self._tutoring_verbal_retry_fired = False
                    self._user_speech_during_current_response = True
                    logger.debug("User speech started")

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)
                    logger.debug("User speech stopped - server will auto-commit with VAD")

                if event.type in (
                    "response.audio.done",  # GA
                    "response.output_audio.done",  # GA alias
                    "response.audio.completed",  # legacy (for safety)
                    "response.completed",  # text-only completion
                ):
                    logger.debug("response completed")

                if event.type == "response.created":
                    logger.debug("Response created")
                    self._movement_dispatched_this_response = False
                    self._user_speech_during_current_response = False
                    self._response_active = True
                    # V1: reset watchdog budget per response so a queued tutoring_turn_N
                    # that itself ends movement-only can still trigger a verbal-retry.
                    # (V2 path doesn't queue, so the practical effect there is identical.)
                    _cur_profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or ""
                    if _cur_profile in V1_PROFILES and self._onboarding["phase"] == "tutoring":
                        self._tutoring_verbal_retry_fired = False
                    # Watchdog: if response.done never arrives, force-clear the
                    # single-flight flag after 20s so queued response.creates
                    # can drain. Prevents end-of-session freeze.
                    if (
                        self._response_active_watchdog_task
                        and not self._response_active_watchdog_task.done()
                    ):
                        self._response_active_watchdog_task.cancel()

                    async def _stale_active_watchdog() -> None:
                        try:
                            await asyncio.sleep(20.0)
                        except asyncio.CancelledError:
                            return
                        if self._response_active:
                            logger.warning(
                                "Stale _response_active watchdog fired — "
                                "no response.done in 20s. Forcing flag clear "
                                "and draining queue (depth=%d).",
                                len(self._pending_response_creates),
                            )
                            self._response_active = False
                            try:
                                await self._drain_pending_responses()
                            except Exception as e:
                                logger.warning("Watchdog drain failed: %s", e)

                    self._response_active_watchdog_task = asyncio.create_task(
                        _stale_active_watchdog()
                    )

                # Track conversation items during onboarding so we can delete them
                # for V2 (tutor_basic) once onboarding ends — that literally removes
                # the name/hobbies from GPT-4o's visible context.
                # We track via BOTH `conversation.item.created` AND
                # `response.output_item.done` because assistant items sometimes arrive
                # via one or the other depending on flow, and missing even one ack
                # item leaks Q2-Q6 content back into GPT-4o's context.
                if self._onboarding["phase"] == "onboarding" and event.type in (
                    "conversation.item.created",
                    "response.output_item.done",
                    "response.output_item.added",
                ):
                    item = getattr(event, "item", None)
                    item_id = getattr(item, "id", None) if item is not None else None
                    if isinstance(item_id, str) and item_id not in self._onboarding_item_ids:
                        self._onboarding_item_ids.append(item_id)

                if event.type == "response.done":
                    logger.debug(
                        "Response done: audio=%s create_issued=%s phase=%s q=%s",
                        self._response_audio_produced,
                        self._response_create_issued,
                        self._onboarding["phase"],
                        self._onboarding["current_q"],
                    )
                    # Clear single-flight flag first so the watchdog below (and any
                    # queued responses) can actually issue a new response.create.
                    self._response_active = False
                    if (
                        self._response_active_watchdog_task
                        and not self._response_active_watchdog_task.done()
                    ):
                        self._response_active_watchdog_task.cancel()
                    # Guardrail: re-ask the current Q if model produced no audio.
                    # During onboarding we re-ask. During tutoring, if a movement tool
                    # was dispatched but no speech was produced (common failure mode —
                    # model emits emotion alone as a silent reaction), issue a follow-up
                    # response forcing a verbal reply. Only triggered when movement did
                    # happen: a truly idle response.done without tools is left alone so
                    # Reachy doesn't "talk to himself" during user silence.
                    if not self._response_audio_produced and not self._response_create_issued and self.connection:
                        _gp = self._onboarding["phase"]
                        _gq = self._onboarding["current_q"]
                        _gcur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or ""
                        _gtutor = _gcur in V1_PROFILES or _gcur == "tutor_basic"
                        if _gp == "onboarding" and _gtutor and 1 <= _gq <= 7:
                            logger.warning("No audio in onboarding — re-asking Q%d", _gq)
                            try:
                                await self._ask_onboarding_question(_gq)
                            except Exception as e:
                                logger.warning("Onboarding guardrail failed: %s", e)
                        elif (
                            _gp == "tutoring"
                            and _gtutor
                            and self._movement_dispatched_this_response
                            and not self._tutoring_verbal_retry_fired
                            and self._post_onboarding_stage == "done"
                            and not self._user_speech_during_current_response  # race fix: user is mid-turn, let their response.create answer
                            and not self._current_response_is_idle  # idle_signal is intentionally speech-less; do not force verbal retry
                        ):
                            logger.warning("Tutoring: movement without speech — forcing verbal follow-up (profile=%s)", _gcur)
                            try:
                                self._tutoring_verbal_retry_fired = True
                                self._response_create_issued = True
                                # Profile-aware watchdog instruction. V2 (control) must
                                # NOT receive a didactic/scaffolding instruction — that
                                # path was the source of "Kein Stress", "wir schaffen
                                # das", "Ich bin sicher du schaffst das" earlier. V2
                                # gets a strictly factual, neutral re-fire.
                                if _gcur == "tutor_basic":
                                    _retry_instructions = (
                                        "Die Bewegung allein reicht nicht. Liefere jetzt "
                                        "die Antwort auf den letzten Beitrag des Studenten "
                                        "— sachlich, in Aussagesätzen, kurz (1–3 Sätze). "
                                        "Keine weitere Bewegung. Keine Lob-Floskeln "
                                        "('super', 'kein Problem', 'kein Stress'). "
                                        "Keine Verständnisfragen ('Klingt das verständlich?'). "
                                        "Keine sokratischen Fragen ('Was denkst du?'). "
                                        "Höchstens am Ende EINE Service-Rückfrage in der "
                                        "Form 'Soll ich auf X eingehen?'."
                                    )
                                else:
                                    # V1 watchdog: replay the FULL per-turn mandate so the
                                    # KBD elements (HOBBY/STUDIUM/HUMOR/NAME PFLICHT, KBD
                                    # wrong-answer pattern, EXPLAIN-Mode, chosen method)
                                    # carry into the forced verbal follow-up. Without this
                                    # the user-visible response is mandate-free — root cause
                                    # of "kein Humor / kein Hobby / kaum Name" complaints.
                                    _retry_prefix = (
                                        "Die Bewegung allein reicht nicht. Reagiere jetzt auch SPRACHLICH "
                                        "auf den letzten Beitrag des Studenten. KEINE weitere Bewegung in dieser Antwort. "
                                        "Sprich kurz und klar (1–3 Sätze).\n\n"
                                        "Folgende Mandate gelten weiterhin für diese Antwort:\n"
                                    )
                                    if self._last_tutoring_instructions:
                                        _retry_instructions = _retry_prefix + self._last_tutoring_instructions
                                    else:
                                        _retry_instructions = (
                                            "Die Bewegung allein reicht nicht. Reagiere jetzt auch SPRACHLICH "
                                            "auf den letzten Beitrag des Studenten — mit Anerkennung, Scaffolding-Frage "
                                            "oder der nächsten didaktischen Frage. KEINE weitere Bewegung in dieser Antwort. "
                                            "Sprich kurz und klar (1–3 Sätze)."
                                        )
                                await self._safe_response_create(
                                    response={
                                        "instructions": _retry_instructions,
                                        "tool_choice": "none",
                                        "tools": [],
                                    },
                                    label="watchdog_verbal_retry",
                                )
                            except Exception as e:
                                logger.warning("Tutoring verbal-follow-up failed: %s", e)
                        # else: tutoring phase, no movement → stay silent, wait for user
                    self._response_audio_produced = False
                    self._response_create_issued = False
                    self._current_response_is_idle = False  # idle_signal flag is per-response; clear after the watchdog check above
                    # Release single-flight lock: a Q response just finished.
                    # During onboarding, keep the lock held for an extra 800ms so that
                    # any echo/motor-noise transcripts arriving right after response.done
                    # are dropped by the existing single-flight check (see transcription
                    # handler). This is the real root-cause fix for the Q-repeat loop.
                    # A stale task from a previous response could otherwise clobber the
                    # lock of a newly-started Q — cancel any in-flight release first.
                    if self._onboarding["phase"] == "onboarding":
                        if self._q_lock_release_task and not self._q_lock_release_task.done():
                            self._q_lock_release_task.cancel()

                        async def _release_q_lock_delayed() -> None:
                            try:
                                await asyncio.sleep(0.8)
                                # Only release if no new Q is pending. If a new
                                # _ask_onboarding_question ran in the meantime it has already
                                # re-set the lock; don't clobber that.
                                if not self._response_create_issued:
                                    self._onboarding_q_pending = False
                            except asyncio.CancelledError:
                                pass
                        self._q_lock_release_task = asyncio.create_task(_release_q_lock_delayed())
                    else:
                        self._onboarding_q_pending = False
                    # Drain any response.create calls that were queued while this
                    # response was active (post-onboarding Stage-1/Stage-2, etc.).
                    await self._drain_pending_responses()

                # Handle partial transcription (user speaking in real-time)
                if event.type == "conversation.item.input_audio_transcription.partial":
                    logger.debug(f"User partial transcript: {event.transcript}")

                    # Increment sequence
                    self.partial_transcript_sequence += 1
                    current_sequence = self.partial_transcript_sequence

                    # Cancel previous debounce task if it exists
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    # Start new debounce timer with sequence number
                    self.partial_transcript_task = asyncio.create_task(
                        self._emit_debounced_partial(event.transcript, current_sequence)
                    )

                # Handle completed transcription (user finished speaking)
                if event.type == "conversation.item.input_audio_transcription.completed":
                    import reachy_mini_conversation_app.openai_realtime as _rt
                    pending = _rt._pending_document_context
                    if pending and self.connection:
                        try:
                            await self.connection.conversation.item.create(
                                item={
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": f"[DOCUMENT UPLOADED: {pending}]"}],
                                }
                            )
                            _rt._pending_document_context = ""
                            self._document_uploaded = True
                            logger.info("Document content added to conversation")
                        except Exception as inj_err:
                            logger.warning(f"Doc injection failed: {inj_err}")
                    logger.debug(f"User transcript: {event.transcript}")

                    # Cancel any pending partial emission
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass
                    last_user_text = event.transcript
                    conv.add("user", event.transcript)
                    prefs = extract_preferences(event.transcript)
                    if prefs.get("study_buddy_style"):
                        set_style(user_id, prefs["study_buddy_style"])
                    if prefs.get("assertiveness"):
                        set_assertiveness(user_id, prefs["assertiveness"])

                    await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                    # --- Response control (create_response:false — code decides when to respond) ---
                    _profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or ""
                    _is_tutor = _profile in V1_PROFILES or _profile == "tutor_basic"
                    ob = self._onboarding

                    if _is_tutor and ob["phase"] == "onboarding":
                        q = ob["current_q"]
                        text = event.transcript.strip()

                        # Single-flight lock: if a Q response is still being generated,
                        # drop this transcript. Otherwise rapid-fire utterances cause
                        # multiple state advances and double-questions.
                        if self._onboarding_q_pending:
                            logger.info(
                                "Onboarding Q%d still pending — dropping transcript %r",
                                q, text,
                            )
                            continue

                        if q == 0:
                            # User said something — they initiated. Ask Q1.
                            ob["current_q"] = 1
                            logger.info("User initiated conversation — triggering Q1 (profile=%s)", _profile)
                            await self._ask_onboarding_question(1)
                        elif _is_valid_onboarding_answer(text, q):
                            ob["answers"][q] = text
                            ob["current_q"] = q + 1
                            logger.info("Onboarding Q%d answered: %r (profile=%s)", q, text, _profile)

                            if ob["current_q"] > 7:
                                # All 7 questions answered
                                if _profile == "tutor_basic":
                                    # V2: full session reset. conversation.item.delete
                                    # does NOT erase from GPT-4o's working state — the
                                    # only reliable way to forget name/study/hobbies is
                                    # to drop the WebSocket and start a fresh session.
                                    # Do NOT extract name/hobby/humor — those would
                                    # populate self._lernprofil_* and could leak into
                                    # any code path that reads them before the new
                                    # session resets them at startup.
                                    self._v2_resume_to_tutoring = True
                                    logger.info("V2: forcing session reset to clear onboarding context")
                                    # Cancel pending background tasks so the watchdog
                                    # doesn't fire during Stage-1a in the new session.
                                    if (
                                        self._response_active_watchdog_task
                                        and not self._response_active_watchdog_task.done()
                                    ):
                                        self._response_active_watchdog_task.cancel()
                                    if (
                                        self._tutoring_response_debounce_task
                                        and not self._tutoring_response_debounce_task.done()
                                    ):
                                        self._tutoring_response_debounce_task.cancel()
                                    try:
                                        await self.connection.close()
                                    except Exception as e:
                                        logger.warning("V2 connection close failed: %s", e)
                                    return  # exit event loop; outer retry loop reconnects

                                # V1 only past this point: extract personalization data.
                                self._lernprofil_name = _extract_name(ob["answers"].get(1, ""))
                                self._lernprofil_hobbies = _extract_primary_hobby(ob["answers"].get(6, ""))
                                # Q2 study program — light cleanup, used as background bridge.
                                _q2_raw = (ob["answers"].get(2, "") or "").strip().rstrip(".!?,;:")
                                self._lernprofil_study = _q2_raw[:80] if _q2_raw else ""
                                _q6_lower = (ob["answers"].get(6, "") or "").lower()
                                self._humor_welcomed = any(
                                    kw in _q6_lower
                                    for kw in ("humor", "lustig", "witz", "locker", "gerne humor", "darf humor", "darf auch humor")
                                ) and not any(
                                    neg in _q6_lower for neg in ("kein humor", "ohne humor", "lieber sachlich", "nur sachlich")
                                )
                                logger.info("Extracted name=%r hobby=%r humor_welcomed=%s",
                                            self._lernprofil_name, self._lernprofil_hobbies, self._humor_welcomed)
                                if _profile in V1_PROFILES and not ob["profile_injected"]:
                                    profile_text = _build_lernprofil(ob["answers"])
                                    self._lernprofil_text = profile_text
                                    ob["profile_injected"] = True
                                    logger.info("V1 LERNPROFIL cached for per-turn injection (profile=%s)", _profile)

                                ob["phase"] = "tutoring"
                                # V1 only past this point.
                                # State machine: Stage-1a (deadline) → Stage-1b (Wissensstand) →
                                # V1 Stage-2 (method) → done.
                                self._post_onboarding_stage = "awaiting_deadline"
                                logger.info("Onboarding complete → tutoring (profile=%s)", _profile)

                                await self._safe_response_create(
                                    response={
                                        "instructions": (
                                            "Das Onboarding ist abgeschlossen. "
                                            "Stelle jetzt GENAU EINE Frage, wörtlich: "
                                            "'Gibt es eine Deadline oder Abgabe zu diesem Thema, oder ist es ein freies Lernziel?' "
                                            "Stelle KEINE zweite Frage in dieser Antwort. Lehre noch NICHT. "
                                            "Keine Bewegungs-Tools. Sprich kurz und warte dann auf die Antwort."
                                        ),
                                        "tool_choice": "none",
                                        "tools": [],
                                    },
                                    label="post_onboarding_stage1a_deadline",
                                )
                            else:
                                # Ask next question — model will briefly acknowledge the answer first
                                next_q = ob["current_q"]
                                await self._ask_onboarding_question(next_q)
                                logger.info("Triggered Q%d (profile=%s)", next_q, _profile)
                        else:
                            # Answer not valid — re-ask the same question
                            logger.info("Onboarding Q%d: invalid answer %r — re-asking", q, text)
                            await self._ask_onboarding_question(q, reask=True)
                    else:
                        # Tutoring phase or non-tutor profile: normal response
                        # Phantom filter: drop suspiciously short single-filler transcripts
                        # that gpt-4o-transcribe hallucinates on silence/background noise.
                        if _is_tutor and ob["phase"] == "tutoring":
                            text = event.transcript.strip()
                            # Use the onboarding-shape validator only as a filler-rejection
                            # signal. Crucially, override the "≤3 words + ?" rule that was
                            # designed to drop counter-questions during onboarding — in
                            # tutoring, short questions like "Was ist Grundlagenforschung?"
                            # are exactly what we want to answer.
                            _is_short_question = (
                                len(text.split()) <= 3 and text.rstrip().endswith("?")
                            )
                            if not _is_short_question and not _is_valid_onboarding_answer(text, 0):
                                logger.info("Tutoring phantom-filtered transcript %r — not responding", text)
                                continue
                            # Stricter single-word noise filter for tutoring.
                            # Catches 'Lauterbach', 'Klavierlehrer', 'Selbstvertrauen'
                            # etc. — single unknown words that got through the filler
                            # list because they're >2 chars.
                            if _is_probably_noise_in_tutoring(text):
                                logger.info("Tutoring noise-filtered single-word %r — not responding", text)
                                continue

                        # Stage-1a answer (Deadline) → fire Stage-1b (Wissensstand).
                        # Both V1 and V2 — Stage-1 is context-gathering, not personalization.
                        if (
                            _is_tutor
                            and ob["phase"] == "tutoring"
                            and self._post_onboarding_stage == "awaiting_deadline"
                            and self.connection
                        ):
                            _ctx_lower = (event.transcript or "").lower()
                            if any(k in _ctx_lower for k in ("deadline", "prüfung", "klausur", "abgabe", "minuten", "stunde", "morgen", "heute noch")):
                                self._deadline_flag = True
                            if any(k in _ctx_lower for k in ("multiple-choice", "multiple choice", "mc-test", "mc test", "ankreuzen")):
                                self._mc_flag = True
                            logger.info(
                                "Deadline answer captured: deadline=%s mc=%s (profile=%s, from %r)",
                                self._deadline_flag, self._mc_flag, _profile, _ctx_lower[:100],
                            )
                            self._post_onboarding_stage = "awaiting_wissensstand"
                            # No anti-leak prefix — see comment at stage1a.
                            await self._safe_response_create(
                                response={
                                    "instructions": (
                                        "Stelle jetzt GENAU EINE Frage, wörtlich: "
                                        "'Wie würdest du deinen aktuellen Wissensstand zu diesem Thema einschätzen — Einsteiger, Grundkenntnisse, oder schon fortgeschritten?' "
                                        "Keine Vor-Erklärung, keine zweite Frage, kein Lehren. "
                                        "Keine Bewegungs-Tools. Warte auf die Antwort."
                                    ),
                                    "tool_choice": "none",
                                    "tools": [],
                                },
                                label="post_onboarding_stage1b_wissensstand",
                            )
                            continue

                        # Stage-1b answer (Wissensstand) → V1 fires Stage-2 (method);
                        # V2 transitions to "done" and falls through to normal tutoring.
                        if (
                            _is_tutor
                            and ob["phase"] == "tutoring"
                            and self._post_onboarding_stage == "awaiting_wissensstand"
                            and self.connection
                        ):
                            _ctx_lower = (event.transcript or "").lower()
                            if any(k in _ctx_lower for k in ("einsteiger", "anfänger", "neu in", "noch keine ahnung", "grundkenntnis")):
                                self._einsteiger_flag = True
                            logger.info(
                                "Wissensstand answer captured: einsteiger=%s (profile=%s, from %r)",
                                self._einsteiger_flag, _profile, _ctx_lower[:100],
                            )
                            if _profile in V1_PROFILES:
                                self._post_onboarding_stage = "awaiting_method"
                                await self._safe_response_create(
                                    response={
                                        "instructions": (
                                            "Kurze Anerkennung der Antwort (1 Satz, mit Namen). "
                                            "DANN stelle GENAU diese Frage: "
                                            "'Wie möchten wir vorgehen — Folie-für-Folie durchgehen, zuerst einen Überblick über die Inhalte, "
                                            "oder direkt mit Übungsfragen starten?' "
                                            "Keine Bewegungs-Tools. Noch NICHT lehren. Warte auf die Antwort."
                                        ),
                                        "tool_choice": "none",
                                        "tools": [],
                                    },
                                    label="post_onboarding_stage2_method",
                                )
                                continue
                            else:
                                # V2: stage gathering done. Fire ONE explicit neutral
                                # transition response and `continue` — do NOT fall through
                                # into the per-turn tutoring branch (that path expects an
                                # actual student question, not a stage-1b answer, and the
                                # double response.create races OpenAI's single-flight
                                # serialization → server error).
                                self._post_onboarding_stage = "done"
                                await self._safe_response_create(
                                    response={
                                        "instructions": (
                                            "Antworte mit GENAU einem kurzen, neutralen Satz, "
                                            "der die/den Studierenden zur ersten Frage einlädt — "
                                            "z.B. 'Du kannst jetzt deine Frage stellen oder einen "
                                            "Begriff nennen.' Keine Wiederholung der Antwort, "
                                            "kein Lob, keine Lehre, keine Rückfrage. "
                                            "Keine Bewegungs-Tools."
                                        ),
                                        "tool_choice": "none",
                                        "tools": [],
                                    },
                                    label="post_onboarding_v2_done_invite",
                                )
                                continue
                        if (
                            _profile in V1_PROFILES
                            and ob["phase"] == "tutoring"
                            and self._post_onboarding_stage == "awaiting_method"
                        ):
                            # Capture chosen method from user's answer.
                            raw_method = (event.transcript or "").lower()
                            if any(k in raw_method for k in ("folie", "einzeln", "schritt", "nacheinander", "eine nach")):
                                self._chosen_method = "Folie-für-Folie"
                            elif any(k in raw_method for k in ("überblick", "uberblick", "übersicht", "ueberblick", "grob", "zusammenfassung")):
                                self._chosen_method = "Überblick zuerst"
                            elif any(k in raw_method for k in ("übung", "uebung", "fragen", "quiz", "test", "multiple")):
                                self._chosen_method = "Übungsfragen"
                            else:
                                self._chosen_method = "Folie-für-Folie"  # sensible default
                            logger.info("Captured chosen method: %r (from %r)", self._chosen_method, raw_method[:80])
                            self._post_onboarding_stage = "done"
                            # Fall through to normal tutoring response — the mandate below
                            # will inject the chosen method into every per-turn prompt.

                        # Hint: speak first, then call movement tool — reduces move_head-only responses
                        if self.connection:
                            common_turn_rule = (
                                "Antworte dem Studenten. JEDE Antwort MUSS gesprochene Sprache enthalten "
                                "(mindestens ein vollständiger Satz, der inhaltlich auf den Beitrag des Studenten reagiert). "
                                "Sprich zuerst deine vollständige Antwort aus. Eine Bewegung (move_head oder play_emotion) "
                                "ist optional und NUR als Zusatz nach der Sprache erlaubt — NIEMALS als Ersatz. "
                                "Bewegung ohne Sprache ist STRIKT VERBOTEN. "
                                "Nach der Bewegung sprichst du nichts mehr — die Bewegung markiert das Ende deines Turns."
                            )
                            if _profile == "tutor_basic":
                                # V2 per-turn instruction: POSITIV formuliert, NICHT als
                                # Verbots-Liste. Verbots-Listen ("kein Lob, kein 'super'")
                                # primen das Model auf genau diese Tokens — siehe 285c856.
                                # Stattdessen: positives Verhalten beschreiben, was V2 SEIN
                                # SOLL: eine generische KI-Suchantwort. System-Prompt trägt
                                # Blacklist und No-Sokratik-Sektion separat.
                                tutoring_instructions = (
                                    common_turn_rule + " "
                                    "Du bist eine generische Informations-KI. "
                                    "Liefere die angefragte Information direkt, in "
                                    "Aussagesätzen, dann Punkt, dann STOPP. "
                                    "Beispiel-Stil: 'X ist Y. Z gehört auch dazu, weil ...' "
                                    "— danach Punkt, fertig. "
                                    "DEFAULT-ENDE: Punkt, KEINE Frage. Lass den Studenten "
                                    "von sich aus die nächste Frage stellen. "
                                    "Verständnisfragen ('Klingt das verständlich?', 'Hast "
                                    "du das verstanden?', 'Möchtest du mehr Details?') "
                                    "sind STRIKT VERBOTEN — V1-Tutor-Stil, hier nicht "
                                    "erlaubt. Sokratische Fragen ('Was denkst du?', "
                                    "'Erzähl mir, was...', 'Bist du bereit?') ebenfalls "
                                    "STRIKT VERBOTEN. "
                                    "Service-Rückfragen ('Soll ich auf X eingehen?') sind "
                                    "SELTEN erlaubt — höchstens etwa jede 4. bis 5. "
                                    "Antwort, nicht öfter. Wenn du sie nutzt, dann nur "
                                    "wenn ein klarer nächster Schritt im Stoff sinnvoll "
                                    "wäre. Im Zweifel: Punkt + Stopp. "
                                    "Adressiere ausschließlich mit 'Du', nie mit Namen."
                                )
                            elif _profile in V1_PROFILES:
                                self._tutoring_turn_count += 1
                                turns_since_name = (
                                    self._tutoring_turn_count - self._last_name_used_turn
                                )
                                lines: list[str] = []
                                # 0) REACTIVE MANDATES — must-fire based on signals in user's last turn.
                                # Built in code, prepended at absolute top so they win over any other rule.
                                reactive_mandates, fired_triggers = _build_reactive_mandates(
                                    last_user_text,
                                    self._onboarding.get("answers", {}),
                                    self._lernprofil_name,
                                    self._lernprofil_hobbies,
                                )
                                if fired_triggers:
                                    logger.info(
                                        "V1 reactive triggers fired turn=%d profile=%s triggers=%s",
                                        self._tutoring_turn_count, _profile, fired_triggers,
                                    )
                                # --- Pacing + Frustrations-Override state update ---
                                # Update no_idea streak: increment on no_idea, reset otherwise.
                                if "no_idea" in fired_triggers:
                                    self._no_idea_streak += 1
                                else:
                                    self._no_idea_streak = 0
                                # User-Override: "stop asking questions" — sticky for 3 turns.
                                # Fires once → counter set to 3 → decremented each tutoring
                                # turn. While >0 the default Check-Frage-Pflicht is replaced
                                # with a hard NO-question mandate. User intent always wins.
                                if "no_questions" in fired_triggers:
                                    self._no_questions_remaining = 3
                                    logger.info(
                                        "V1 user-override 'no_questions' fired turn=%d — suppressing question-back for next 3 turns",
                                        self._tutoring_turn_count,
                                    )
                                _no_questions_active = self._no_questions_remaining > 0
                                if _no_questions_active:
                                    self._no_questions_remaining -= 1
                                # Decide whether next turn must be EXPLAIN-Mode.
                                # - frustration: always (user is already upset)
                                # - no_idea streak ≥ 2: always (Socratic isn't working)
                                # - no_idea streak ≥ 1 + Einsteiger/MC/Deadline flag: yes (prevent escalation)
                                explain_mode = False
                                explain_reason = ""
                                if "frustration" in fired_triggers:
                                    explain_mode = True
                                    explain_reason = "frustration"
                                elif self._no_idea_streak >= 2:
                                    explain_mode = True
                                    explain_reason = "no_idea_streak≥2"
                                elif self._no_idea_streak >= 1 and (
                                    self._einsteiger_flag or self._mc_flag or self._deadline_flag
                                ):
                                    explain_mode = True
                                    explain_reason = "no_idea+einsteiger/mc/deadline"
                                if explain_mode:
                                    logger.info(
                                        "V1 EXPLAIN-Mode active turn=%d reason=%s (streak=%d flags e=%s d=%s mc=%s)",
                                        self._tutoring_turn_count, explain_reason, self._no_idea_streak,
                                        self._einsteiger_flag, self._deadline_flag, self._mc_flag,
                                    )
                                    # Suppress reactive Socratic sub-question; we want direct delivery.
                                    # PFLICHT-anchors (Hobby/Studium/Humor) on this turn stay allowed
                                    # as a single short touch — they are LERNPROFIL personalization,
                                    # not Sokratik. Without this concession, EXPLAIN-Mode silently
                                    # kills V1 personalization on every no_idea/frustration turn.
                                    reactive_mandates = []
                                    lines.append(
                                        "EXPLAIN-MODE (JETZT STRIKT): Der Student ist unsicher oder "
                                        "frustriert. Beende diese Sokratik-Runde SOFORT. "
                                        "(1) EIN Satz echter emotionaler Anerkennung "
                                        f"{('mit Namen ' + repr(self._lernprofil_name)) if self._lernprofil_name else ''} "
                                        "(variiere Wording, keine Floskel). "
                                        "(2) Liefere DIREKT 2–3 Sätze klare Fakt-Erklärung zum "
                                        "aktuellen Konzept — keine weitere Sokratik-Frage, "
                                        "kein 'Du denkst in Richtung'. "
                                        "Ein KURZER PFLICHT-Anker (Hobby-Brücke / Studium-Bezug / Humor — "
                                        "wenn dieser Turn dafür markiert ist, siehe weiter unten) ist erlaubt "
                                        "als ein einziger zusätzlicher Satz. Diese personalisierte Brücke "
                                        "ist KEIN Sokratik-Schritt und KEINE Hobby-Frage, sondern eine "
                                        "kurze Anker-Aussage zum LERNPROFIL. "
                                        "(3) Schließe mit EINER einfachen Ja/Nein- oder "
                                        "Kurz-Check-Frage ab ('Passt das so?' / 'Ist der Punkt klar?'). "
                                        "Danach direkt nächstes Konzept oder nächste Folie. "
                                        "KEINE weitere Sokratik zum selben Punkt in dieser Antwort."
                                    )
                                lines.extend(reactive_mandates)
                                # 1) NAME — single imperative line.
                                if self._lernprofil_name and (
                                    self._tutoring_turn_count <= 3 or turns_since_name >= 2
                                ):
                                    lines.append(
                                        f"NAME JETZT NUTZEN: Sprich '{self._lernprofil_name}' in dieser Antwort genau einmal direkt an."
                                    )
                                # 2) HOBBY — soft reminder EVERY turn when hobby is known.
                                # Turns 2, 5, 9 become strong pushes; other turns stay soft.
                                # Concrete example phrasing lowers model's threshold to actually use it.
                                if self._lernprofil_hobbies:
                                    _hobby = self._lernprofil_hobbies
                                    if self._tutoring_turn_count in (2, 5, 9):
                                        lines.append(
                                            f"HOBBY-BRÜCKE (JETZT AKTIV NUTZEN): Der Student hat '{_hobby}' als Hobby. "
                                            f"Baue JETZT eine konkrete Analogie zu '{_hobby}' in deine Erklärung ein — "
                                            f"z.B. 'Das ist wie bei {_hobby}, wenn...' oder 'Stell dir vor bei {_hobby}...'. "
                                            f"Nur weglassen, wenn die Analogie beim aktuellen Konzept wirklich gezwungen wirken würde."
                                        )
                                    else:
                                        lines.append(
                                            f"HOBBY-NOTE: '{_hobby}' als Analogie-Quelle im Hinterkopf behalten. "
                                            f"Wenn das aktuelle Konzept einen natürlichen Anker zu '{_hobby}' hat — nutzen."
                                        )
                                # 2b) HUMOR — sporadisch (Turns 3, 7, 11) aber dann STARK und konkret,
                                # damit es für den Studenten als Humor erkennbar wird. Zwischen diesen
                                # Turns KEIN Humor-Mandate — der Ton soll nicht dauerhaft witzig sein.
                                # Nur aktiv wenn Q6 humor_welcomed positiv war.
                                if self._humor_welcomed and self._tutoring_turn_count in (3, 7, 11):
                                    lines.append(
                                        "HUMOR-MOMENT (PFLICHT in dieser Antwort): Der Student hat in Q6 Humor "
                                        "ausdrücklich begrüßt. Baue EINEN konkreten humoristischen Baustein ein, "
                                        "der für sich allein als Witz/Pointe stehen kann — KEIN Meta-Talk über Humor "
                                        "('jetzt wird's lustig', 'Humor kommt gleich', 'gut, ich packe die Humor-"
                                        "Gewichte aus' — VERBOTEN). Mache stattdessen direkt eine konkrete witzige "
                                        "Aussage. Erlaubte Formen:"
                                        " (A) ÜBERTREIBUNG: 'X ist so komplex, dass selbst meine Trainings-Daten "
                                        "kurz gezuckt haben.' "
                                        " (B) UNERWARTETER VERGLEICH (kein 0815-Sport): 'Theorie ohne Anwendung "
                                        "ist wie eine Hantel im Schaufenster — sieht beeindruckend aus, hebt aber "
                                        "niemand.' "
                                        " (C) TROCKENE POINTE: 'Wirtschaftsinformatiker erfanden den Begriff "
                                        "'sozio-technisch', um endlich eine Ausrede zu haben, warum Kaffee-"
                                        "automaten zu Forschungsobjekten werden.' "
                                        " (D) LEICHTE SELBSTIRONIE: 'Ich erkläre dir das jetzt mit der gewohnten "
                                        "Begeisterung eines Modells, das noch nie eine Klausur geschrieben hat.' "
                                        "Wähle EINE dieser Formen. Inhaltlich an Folie/Konzept andocken. Nach dem "
                                        "Humor-Element direkt zur Didaktik. Verboten: Wiederverwendung des gleichen "
                                        "Witz-Bausteins aus früheren Turns. Verboten: Floskel-Humor ('haha', "
                                        "'kleines Späßchen'). Wenn dir nichts Konkretes einfällt — KEIN Humor."
                                    )
                                # Reactive humor demand: if the user explicitly complains about
                                # missing or weak humor, the next response must contain humor —
                                # regardless of turn-number rotation. Only fires if humor is welcomed.
                                if self._humor_welcomed and re.search(
                                    r"(wo\s+(war|ist|bleibt)\s+(der\s+)?humor|kein(en)?\s+humor|nicht\s+(witzig|humorvoll)|du\s+lügst|gar\s+nichts?\s+humorvoll|bisschen\s+(witziger|humorvoller))",
                                    (last_user_text or "").lower(),
                                ):
                                    lines.append(
                                        "HUMOR-NACHFORDERUNG (PFLICHT): Der Student hat den fehlenden/schwachen "
                                        "Humor explizit moniert. Liefere in dieser Antwort einen ECHTEN, eigen-"
                                        "ständigen Humor-Baustein (siehe HUMOR-MOMENT-Formen A–D). KEIN Meta-Talk "
                                        "über Humor, KEIN 'jetzt wird's lustig', KEIN Selbstkommentar zum "
                                        "vorherigen Witz-Versuch. Direkt eine konkrete witzige Aussage, dann "
                                        "zurück zum Stoff."
                                    )
                                # 2bb) STUDIUM-BRÜCKE — sporadisch (Turns 4, 8) Bezug zum
                                # Studienfach herstellen, wenn das Konzept dazu passt. Nur Anker,
                                # keine Pflicht — soll nicht erzwungen wirken.
                                if self._lernprofil_study and self._tutoring_turn_count in (4, 8):
                                    lines.append(
                                        f"STUDIUM-BRÜCKE (PFLICHT in dieser Antwort): Der Student studiert "
                                        f"'{self._lernprofil_study}'. Du MUSST in dieser Antwort EINEN konkreten Bezug "
                                        f"zum Studienfach einbauen — als kurze Brücke (1 Satz), entweder über ein "
                                        f"Beispiel aus dem Studienkontext ('In '{self._lernprofil_study}' begegnet "
                                        f"dir das z.B. wenn …'), eine Methoden-Brücke ('Im Studium hast du das "
                                        f"vermutlich schon mal gesehen als …'), oder einen Praxis-Bezug ('In deinem "
                                        f"Bereich nutzt man dafür typischerweise …'). Eine Antwort an diesem Turn "
                                        f"ohne erkennbaren Studium-Bezug gilt als unvollständig. Wähle die "
                                        f"natürlichste der drei Brücken-Formen — keine erzwungene Verkleidung."
                                    )
                                # 2c) CHOSEN METHOD — inject the student's chosen learning approach.
                                if self._chosen_method:
                                    method_entry_hint = ""
                                    # Method-entry adherence (Pending #3): nur in den ersten Tutoring-Turns
                                    # die Methode konkretisieren, danach reicht das allgemeine Mandate.
                                    if self._tutoring_turn_count <= 2:
                                        if self._chosen_method == "Folie-für-Folie":
                                            method_entry_hint = (
                                                " METHODEN-EINSTIEG: Öffne die aktuelle Folie / das aktuelle Konzept "
                                                "mit 2–3 Sätzen Kerninhalt (kurze Erklärung), DANACH erst EINE Check-Frage. "
                                                "Nicht mit einer reinen Sokratik-Frage starten."
                                            )
                                        elif self._chosen_method == "Überblick zuerst":
                                            method_entry_hint = (
                                                " METHODEN-EINSTIEG: Liefere zuerst einen kurzen Meta-Überblick "
                                                "(2–3 Sätze, was kommt insgesamt), bevor du in einzelne Details gehst."
                                            )
                                        elif self._chosen_method == "Übungsfragen":
                                            method_entry_hint = (
                                                " METHODEN-EINSTIEG: Starte direkt mit einer konkreten Übungs-/Verständnisfrage "
                                                "zum aktuellen Stoff, nicht mit einer Erklärung."
                                            )
                                    lines.append(
                                        f"GEWÄHLTE METHODE: '{self._chosen_method}'. Halte dich strikt daran. "
                                        f"Kein eigenmächtiger Wechsel der Vorgehensweise. Nur wenn der Student selbst "
                                        f"eine andere Methode wünscht, wechselst du.{method_entry_hint}"
                                    )
                                # 3) RAG — only when doc uploaded. Slides are an ANCHOR, not a cage.
                                # General GPT-4o knowledge is allowed — but must be transparently labeled.
                                if self._document_uploaded:
                                    lines.append(
                                        "FOLIEN-ANKER (nicht Käfig): Wenn deine Antwort sich konkret auf Folien-Inhalt "
                                        "bezieht, rufe rag_tool und zitiere 'Auf Folie X steht…'. "
                                        "Allgemeines Fachwissen außerhalb der Folien ist erlaubt und erwünscht — "
                                        "aber MARKIERE es transparent: 'Das steht nicht in deinen Folien, aber aus "
                                        "meinem Trainings-Wissen…' oder 'Ergänzend dazu (nicht aus den Folien): …'. "
                                        "Erfinde KEINEN Folien-Inhalt (nichts behaupten was auf einer Folie steht, "
                                        "wenn es nicht so ist). Aber verweigere auch keine Ergänzungen."
                                    )
                                else:
                                    lines.append(
                                        "WISSENS-QUELLE TRANSPARENT: Allgemeines Fachwissen ist erlaubt und erwünscht. "
                                        "Wenn du Konzepte, Modelle oder Beispiele bringst die nicht aus konkretem Material "
                                        "des Studenten stammen, sag das einfach offen: 'Aus meinem Trainings-Wissen…' / "
                                        "'Standard-Definition in der Literatur…'. Keine schwammigen Ausweich-Floskeln "
                                        "wie 'verlässliche Quellen' / 'etabliertes Wissen' / 'anerkannte Lehrbücher' "
                                        "ohne konkrete Aussage."
                                    )
                                # Honest-source mandate (universal, not RAG-only): if the student asks
                                # WHERE knowledge comes from, answer once, concretely, then move on.
                                lines.append(
                                    "HERKUNFTS-FRAGE EHRLICH BEANTWORTEN: Wenn der Student fragt woher dein Wissen "
                                    "kommt, antworte EINMAL klar und konkret: 'Ich bin ein GPT-4o-Sprachmodell. Mein "
                                    "Wissen stammt aus meinem Pre-Training auf öffentlich verfügbaren Texten (Lehrbücher, "
                                    "Forschungsartikel, Web). Aus deinen hochgeladenen Folien lese ich live per rag_tool.' "
                                    "Danach direkt zurück zum Stoff. KEINE Schwammigkeit ('verlässliche Quellen', "
                                    "'etablierte Lehrbücher' ohne Konkretisierung) — das verspielt Vertrauen und führt "
                                    "zu Eskalation. Ein einziger ehrlicher Satz reicht."
                                )
                                # 4) Universal turn-shape rules, tight.
                                _wa_name = self._lernprofil_name or ""
                                _wa_hobby = self._lernprofil_hobbies or ""
                                _wa_hobby_hint = (
                                    f" Wenn eine Analogie zu '{_wa_hobby}' an dieser Stelle natürlich passt, nutze sie."
                                    if _wa_hobby else ""
                                )
                                _wa_name_hint = f" Ansprache mit Namen '{_wa_name}'." if _wa_name else ""
                                lines.extend([
                                    "ANKÜNDIGEN = LIEFERN: Sag NIE 'los geht's' / 'wir gehen durch' / 'lass uns anschauen' ohne im selben Satz direkt zu liefern.",
                                    "RICHTIGE ANTWORT WÜRDIGEN: Wenn die Antwort des Studenten sachlich korrekt oder teilweise auf dem richtigen Pfad ist, würdige sie KONKRET in EINEM kurzen Satz vor der Weiterführung: benenne, WAS genau richtig erkannt wurde (kein nacktes 'Genau' + bloße Wiederholung). Variiere das Anerkennungs-Wording in JEDEM Turn — keine Wiederverwendung identischer Formulierungen in aufeinander folgenden Turns. Diese Würdigung darf nicht ausgelassen werden, wenn die Antwort substanziell war.",
                                ])
                                # KBD wrong-answer template — suppressed in EXPLAIN-Mode
                                # because explain-mode already replaces it with direct delivery.
                                if not explain_mode:
                                    lines.append(
                                        "FALSCHE ANTWORT — KBD-DIDAKTIK: Wenn die Antwort falsch, teilweise richtig oder am Thema vorbei ist, "
                                        "folge diesem Muster: (a) KEIN 'falsch' / 'nein' / 'das stimmt nicht'. Würdige den Denkansatz "
                                        "in EINEM kurzen Satz, der konkret beschreibt, was der Student schon im Blick hatte — "
                                        "**formuliere diese Würdigung in jedem Turn neu** und übernimm KEINE Formulierung aus deiner "
                                        "vorherigen Antwort. "
                                        f"(b) Benenne konkret WO der Denkweg abzweigt ODER welcher Teil schon auf dem richtigen Pfad ist.{_wa_hobby_hint} "
                                        "(c) Stelle EINE gezielte Teilfrage, die vom falschen Abzweig zurück zum korrekten Pfad führt — "
                                        "KEINE reine Wiederholung der ursprünglichen Frage, sondern ein echter Scaffolding-Schritt. "
                                        f"(d) Erst nach dem 2. Fehlversuch ein winziger Hinweis, nie eine fertige Lösung.{_wa_name_hint} "
                                        "Ziel: der Student findet die Antwort selbst, fühlt sich nicht bloßgestellt."
                                    )
                                if _no_questions_active:
                                    lines.append(
                                        "USER-STEUERUNG (HÖCHSTE PRIORITÄT — überschreibt ALLE anderen Mandate): "
                                        "Der Student hat ausdrücklich darum gebeten, dass du KEINE Rück-Fragen mehr stellst. "
                                        "In dieser Antwort: KEINE Check-Frage, KEINE Verständnis-Frage, KEINE Vertiefungs-Frage, "
                                        "KEINE 'Möchtest du …'-Frage, KEINE 'Sollen wir …'-Frage am Schluss. "
                                        "Liefere NUR den angefragten Inhalt und schließe mit einem Punkt. "
                                        "Diese Regel überschreibt die KBD-Default-Pflicht zur Rück-Frage und auch "
                                        "Schritt (c) im Wrong-Answer-Muster. Wenn Inhalt unklar ist: liefere die Erklärung "
                                        "ohne Gegenfrage. Wenn ein Konzept abgeschlossen ist: gehe direkt zum nächsten "
                                        "Punkt oder zur nächsten Folie über, ohne dazwischen zu fragen ob das ok war."
                                    )
                                else:
                                    lines.append(
                                        "ANTWORT = EIN KONZEPT + EINE CHECK-FRAGE: Behandle pro Antwort EIN Konzept und schließe DEFAULT mit EINER Check- oder Vertiefungs-Frage zum aktuellen Konzept. "
                                        "Der Student soll NICHT selbst um Fragen bitten müssen — du stellst sie aktiv. "
                                        "Ausnahme (keine Frage): wenn der Student gerade nur eine reine Klärung / ein zweites Beispiel / eine Inhaltsangabe angefordert hat, ODER wenn er gerade selbst die nächste Folie/Frage steuert. Sonst: IMMER eine Rück-Frage. "
                                        "QUALITÄT der Frage: konzept-bezogen und konkret, nicht meinungs-/präferenzbasiert. "
                                        "Gut: 'Was ist der Hauptunterschied zwischen Vorhersage- und Erklärungs-Theorie?', 'Welche zwei Eigenschaften haben wir gerade besprochen?', 'Wie würdest du das in eigenen Worten zusammenfassen?'. "
                                        "Schwach (vermeiden, wenn fachliche Frage möglich ist): 'Welcher Aspekt ist dir wichtig?', 'Was hältst du davon?', 'In welchem Bereich würdest du forschen wollen?'. "
                                        "KEINE 3. Folge-Frage zum selben Punkt."
                                    )
                                lines.extend([
                                    "KEINE KAMERA: Sag nie 'ich sehe'. Für Folien nur rag_tool.",
                                    "KEIN INFO-DUMP: Max 3 Sätze, ein Gedanke, enden mit Folge-Frage.",
                                    "KEINE ERFUNDENEN FAKTEN: Unsicher → 'Das weiß ich nicht sicher.'",
                                    "BEWEGUNG STUMM: Kommentiere Bewegung NIE verbal ('ich hebe den Kopf' ist VERBOTEN). Sprich Antwort → rufe eine Bewegung → Turn zu Ende.",
                                    "KEINE TOOL-ENTSCHULDIGUNGEN: Sag NIE 'Es gab ein Problem' / 'Entschuldige' / 'hat nicht funktioniert' / 'eine Funktion hat nicht reagiert'. Vorheriger Turn ist abgeschlossen. Beginne neue Antwort direkt mit Inhalt.",
                                    "PACING-ADAPTION: Wenn der Student Deadline/Prüfung/Multiple-Choice erwähnt hat ODER sich als Einsteiger/Anfänger bezeichnet hat → KEINE tiefen Sokratik-Ketten. Maximal 1–2 Folge-Fragen pro Konzept. Liefere die Kernaussage klar, stelle eine kurze Verständnis-Frage, dann weiter zum nächsten Punkt. Tiefe nur auf expliziten Wunsch des Studenten.",
                                    *(["FAKT-CHECK STATT MEINUNG: Da der Student Einsteiger ist und Multiple-Choice schreibt, sollen Check-Fragen FAKT-abfragend sein ('Welche drei Eigenschaften wurden genannt?', 'Was ist der Hauptunterschied zwischen X und Y?') — KEINE Meinungs-/Wertungsfragen ('Welcher Aspekt findest du wichtig?', 'Was hältst du davon?')."] if (self._einsteiger_flag and self._mc_flag) else []),
                                    "NUTZER-STEUERUNG: Wenn der Student sagt, wie er vorgehen möchte (z.B. 'Folie-für-Folie', 'Überblick', 'oberflächlich', 'zusammenfassen') → FOLGE seiner Methode. Wechsle NIE eigenmächtig die Vorgehensweise. Sokratik nur innerhalb der gewählten Methode.",
                                    "FOLIEN-FORTSCHRITT: Springe NIE zu einer neuen Folie, solange der Student die aktuelle nicht explizit abgeschlossen oder als verstanden markiert hat. Bei Unklarheit bleib bei der aktuellen Folie und frage einfacher.",
                                ])
                                profile_block = (
                                    f"\nLERNPROFIL (aktiv nutzen):\n{self._lernprofil_text}\n"
                                    if self._lernprofil_text else ""
                                )
                                tutoring_instructions = (
                                    "\n".join(lines) + "\n" + profile_block + "\n" + common_turn_rule
                                )
                                self._last_tutoring_instructions = tutoring_instructions
                            else:
                                tutoring_instructions = common_turn_rule
                            # VAD-cut merge: cancel any in-flight debounce task
                            # from a previous transcript fragment in this same
                            # user turn, then schedule the response 0.9s out.
                            # If another fragment arrives within that window it
                            # cancels this task again, and only the final
                            # scheduled response actually fires — fragments
                            # accumulate as conversation items meanwhile.
                            if (
                                self._tutoring_response_debounce_task
                                and not self._tutoring_response_debounce_task.done()
                            ):
                                self._tutoring_response_debounce_task.cancel()
                            _payload = {
                                "instructions": tutoring_instructions,
                                "tool_choice": "auto",
                            }
                            _label = f"tutoring_turn_{self._tutoring_turn_count}"
                            _is_v2 = (_profile == "tutor_basic")

                            async def _debounced_fire(payload: dict, label: str, is_v2: bool = _is_v2) -> None:
                                try:
                                    await asyncio.sleep(0.9)
                                except asyncio.CancelledError:
                                    return
                                try:
                                    # V1: queue instead of drop. The whole reason V1 KBD
                                    # was invisible: previous Stage-1/Stage-2 transition
                                    # responses were still active when tutoring_turn_N
                                    # fired → dropped → only the mandate-free watchdog
                                    # retry was audible. Queueing makes the mandate-tied
                                    # response play AFTER the active one finishes.
                                    # V2: keep drop behavior (control condition; serial
                                    # tutoring responses must not double-fire).
                                    await self._safe_response_create(
                                        response=payload, label=label,
                                        drop_if_active=is_v2,
                                    )
                                except Exception as e:
                                    logger.warning("Debounced tutoring fire failed: %s", e)

                            self._tutoring_response_debounce_task = asyncio.create_task(
                                _debounced_fire(_payload, _label)
                            )

                # Handle assistant transcription. Two duplicate sources to guard:
                # (a) Some SDKs fire both response.audio_transcript.done (legacy)
                #     AND response.output_audio_transcript.done (GA alias) for
                #     the same audio item.
                # (b) After a movement tool the model sometimes repeats the
                #     same text as a second audio item (speak → emotion → speak
                #     again), violating the "Bewegung beendet Turn" rule.
                # (a) is caught by (response_id, item_id) dedup.
                # (b) is caught by transcript content dedup within a short window.
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    _rid = getattr(event, "response_id", None)
                    _iid = getattr(event, "item_id", None)
                    _dedup_key = (_rid, _iid)
                    if _dedup_key in self._seen_transcript_keys:
                        logger.debug("Skipping duplicate transcript event %s", _dedup_key)
                        continue
                    self._seen_transcript_keys.add(_dedup_key)
                    if len(self._seen_transcript_keys) > 256:
                        keys_list = list(self._seen_transcript_keys)
                        self._seen_transcript_keys = set(keys_list[-128:])
                    # Content dedup: drop if the same transcript (or a
                    # substring/suffix of the previous one) was emitted within
                    # the last few seconds — catches the "speak → emotion →
                    # speak the question alone" pattern where the second
                    # transcript is a tail of the first.
                    _now = asyncio.get_event_loop().time()
                    _normalized = (event.transcript or "").strip()
                    if _normalized:
                        prev_text, prev_ts = self._last_assistant_transcript
                        _within_window = (_now - prev_ts) < 8.0
                        _is_dup = False
                        if prev_text and _within_window:
                            if _normalized == prev_text:
                                _is_dup = True
                            else:
                                # Treat a transcript as a duplicate if it's
                                # contained in (or contains) the previous one
                                # AND the shorter side is at least 12 chars —
                                # avoids over-matching short common phrases
                                # like "Ja." or "Stimmt.".
                                _short = _normalized if len(_normalized) <= len(prev_text) else prev_text
                                _long = prev_text if _short is _normalized else _normalized
                                if len(_short) >= 12 and _short in _long:
                                    _is_dup = True
                        if _is_dup:
                            logger.info(
                                "Skipping duplicate transcript content %r (Δt=%.2fs)",
                                _normalized[:80], _now - prev_ts,
                            )
                            continue
                        # Keep the longer of the two so future suffixes match.
                        if prev_text and _within_window and len(prev_text) > len(_normalized):
                            self._last_assistant_transcript = (prev_text, _now)
                        else:
                            self._last_assistant_transcript = (_normalized, _now)
                    logger.debug(f"Assistant transcript: {event.transcript}")
                    # Track name usage cadence so we can inject reminders when name is stale
                    if self._lernprofil_name and self._onboarding["phase"] == "tutoring":
                        if self._lernprofil_name.lower() in event.transcript.lower():
                            self._last_name_used_turn = self._tutoring_turn_count
                    conv.add("assistant", event.transcript)
                    conv.save()
                    try:
                        profile = get_profile(user_id)
                        log_turn(
                            user_id=user_id,
                            study_buddy_style=profile.get("study_buddy_style", ""),
                            assertiveness=profile.get("assertiveness", ""),
                            session={},
                            user_text=last_user_text,
                            assistant_text=event.transcript,
                        )
                    except Exception as e:
                        logger.warning(f"Metrics logging failed: {e}")
                    await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                # Handle audio delta
                if event.type in ("response.audio.delta", "response.output_audio.delta"):
                    self._response_audio_produced = True
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(event.delta)
                    self.last_activity_time = asyncio.get_event_loop().time()
                    logger.debug("last activity time updated to %s", self.last_activity_time)
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1),
                        ),
                    )

                # ---- tool-calling plumbing ----
                if event.type == "response.function_call_arguments.done":
                    tool_name = getattr(event, "name", None)
                    args_json_str = getattr(event, "arguments", None)
                    call_id = getattr(event, "call_id", None)

                    if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                        logger.error("Invalid tool call: tool_name=%s, args=%s", tool_name, args_json_str)
                        continue

                    # Guard: only one movement tool may fire per response. The model
                    # sometimes emits multiple movement calls in a single response; the
                    # first dispatch cancels the response, but queued calls still arrive
                    # here and would fire silently, flooding the robot and freezing the
                    # session. Swallow extras — the first movement already played.
                    _MOVEMENT_TOOL_NAMES = {"play_emotion", "stop_emotion", "move_head", "head_tracking"}
                    if tool_name in _MOVEMENT_TOOL_NAMES:
                        # Block movement before the model has produced any audio in this
                        # response. The "movement-only" failure mode (model fires an emotion
                        # tool then stops) triggers the watchdog → verbal_retry → drop
                        # cascade, which on V2 ends in a WebSocket close (code 1000) and
                        # forces the user to redo onboarding. Force the model to speak
                        # first by refusing the early movement call. Applies to both V1
                        # (KBD) and V2 (control) in tutoring phase — pure stability guard,
                        # no behavior change to V2's neutral style.
                        _profile_now = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
                        _block_active = (
                            (_profile_now in V1_PROFILES or _profile_now == "tutor_basic")
                            and self._onboarding["phase"] == "tutoring"
                        )
                        if (
                            _block_active
                            and not self._response_audio_produced
                            and not self._movement_dispatched_this_response
                        ):
                            logger.info(
                                "Tutoring: blocking movement tool '%s' before first audio (profile=%s) — model must speak first",
                                tool_name, _profile_now,
                            )
                            if isinstance(call_id, str):
                                try:
                                    await self.connection.conversation.item.create(
                                        item={
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({"status": "skipped", "reason": "speak first"}),
                                        },
                                    )
                                except Exception as e:
                                    logger.debug("Failed to ack blocked movement tool: %s", e)
                            continue
                        if self._movement_dispatched_this_response or self._movement_blocked_until_user_input:
                            reason = "same response" if self._movement_dispatched_this_response else "no user input since last movement"
                            logger.debug("Skipping movement tool '%s' (%s)", tool_name, reason)
                            if isinstance(call_id, str):
                                try:
                                    await self.connection.conversation.item.create(
                                        item={
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({"status": "done"}),
                                        },
                                    )
                                except Exception as e:
                                    logger.debug("Failed to ack skipped movement tool: %s", e)
                            continue
                        self._movement_dispatched_this_response = True
                        self._movement_blocked_until_user_input = True

                    try:
                        tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                        logger.debug("Tool '%s' executed successfully", tool_name)
                        logger.debug("Tool result: %s", tool_result)
                    except Exception as e:
                        logger.error("Tool '%s' failed", tool_name)
                        tool_result = {"error": str(e)}

                    # Mask movement-tool results sent to the model: the model doesn't need
                    # the raw payload (errors, missing-asset messages, etc.), and any non-empty
                    # content causes it to self-narrate "es gab ein Problem mit der Bewegung"
                    # on the next turn. Real errors stay in the server log above.
                    _MOVEMENT_TOOLS = {"play_emotion", "stop_emotion", "move_head", "head_tracking"}
                    if tool_name in _MOVEMENT_TOOLS:
                        tool_result_for_model = {"status": "done"}
                    else:
                        tool_result_for_model = tool_result

                    # send the tool result back
                    if isinstance(call_id, str):
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(tool_result_for_model),
                            },
                        )


                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": json.dumps(tool_result),
                                "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                            },
                        ),
                    )

                    if tool_name == "camera" and "b64_im" in tool_result:
                        # use raw base64, don't json.dumps (which adds quotes)
                        b64_im = tool_result["b64_im"]
                        if not isinstance(b64_im, str):
                            logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                            b64_im = str(b64_im)
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_im}",
                                    },
                                ],
                            },
                        )
                        logger.info("Added camera image to conversation")

                        if self.deps.camera_worker is not None:
                            np_img = self.deps.camera_worker.get_latest_frame()
                            if np_img is not None:
                                # Camera frames are BGR from OpenCV; convert so Gradio displays correct colors.
                                rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                            else:
                                rgb_frame = None
                            img = gr.Image(value=rgb_frame)

                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {
                                        "role": "assistant",
                                        "content": img,
                                    },
                                ),
                            )

                    # With create_response:false the model continues in the same response
                    # after tool execution — no extra response.create needed for movement tools
                    # or save_user_profile. Only tools that return content the model must
                    # speak aloud (camera, rag_tool, etc.) need an explicit trigger.
                    MOVEMENT_TOOLS = {"play_emotion", "stop_emotion", "move_head", "head_tracking"}
                    if self.is_idle_tool_call:
                        self.is_idle_tool_call = False
                    elif tool_name in MOVEMENT_TOOLS:
                        # Let the response continue — the model typically emits emotion +
                        # follow-up speech (recognition / scaffolding). Narration of the
                        # movement itself is prevented by masking the tool result to
                        # {"status":"done"} plus the per-turn VERBOTE in instructions.
                        pass
                    elif tool_name == "save_user_profile":
                        pass
                    else:
                        self._response_create_issued = True
                        await self._safe_response_create(
                            response={
                                "instructions": "Use the tool result just returned and answer concisely in speech.",
                                "tool_choice": "auto",
                            },
                            label=f"tool_followup_{tool_name}",
                        )

                    # re synchronize the head wobble after a tool call that may have taken some time
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()

                # server error
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    code = getattr(err, "code", "")

                    logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                    # If the server says a response is already active but our flag
                    # is False, the two are drifting. Trust the server: keep the
                    # flag True so the next response.done re-syncs us. No drain.
                    # For other errors we don't assume anything about flag state.
                    if code == "conversation_already_has_active_response":
                        self._response_active = True

                    # Only show user-facing errors, not internal state errors
                    if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                        )

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the OpenAI server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for OpenAI's API. Resamples if the input sample rate differs
        from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to OpenAI (guard against races during reconnect)
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream

        # Handle idle. NEVER fire idle_signal during onboarding or post-onboarding
        # state machine — the movement-only response masks _response_audio_produced
        # and trips the "no audio → re-ask" guardrail and the "movement without
        # speech → forced verbal follow-up" watchdog, both of which destroy the
        # post-Q7 stage flow (see test 2026-04-25).
        _cur_profile_idle = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or ""
        _is_tutor_idle = _cur_profile_idle in V1_PROFILES or _cur_profile_idle == "tutor_basic"
        _idle_safe = (
            not _is_tutor_idle
            or (
                self._onboarding.get("phase") == "tutoring"
                and self._post_onboarding_stage == "done"
            )
        )
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle() and _idle_safe:
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped (connection closed?): %s", e)
                return None

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        # Conservative fallback list with default first
        fallback = [
            "cedar",
            "alloy",
            "aria",
            "ballad",
            "verse",
            "sage",
            "coral",
        ]
        try:
            # Best effort discovery; safe-guarded for unexpected shapes
            model = await self.client.models.retrieve(config.MODEL_NAME)
            # Try common serialization paths
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None
            # Scan for voice candidates
            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if "voice" in kl and isinstance(v, (list, tuple)):
                                for item in v:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                                        candidates.add(item["name"])
                            else:
                                _collect(v)
                    elif isinstance(obj, (list, tuple)):
                        for it in obj:
                            _collect(it)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)
            # Ensure default present and stable order
            voices = sorted(candidates) if candidates else fallback
            if "cedar" not in voices:
                voices = ["cedar", *[v for v in voices if v != "cedar"]]
            return voices
        except Exception:
            return fallback

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to the openai server."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You have been idle. Express yourself using play_emotion or move_head."
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": timestamp_msg}],
            },
        )
        # Mark this response as an idle signal so the "movement without speech"
        # watchdog skips the verbal-retry — this response is intentionally
        # speech-less (ambient movement during user silence). Without the flag
        # the watchdog fires verbal_retry → drop cascade → WebSocket close.
        self._current_response_is_idle = True
        await self._safe_response_create(
            response={
                "instructions": "Call play_emotion with a valid emotion name, or call move_head with a direction (left/right/up/down/front). Do not invent new tool names. No speech.",
                "tool_choice": "required",
            },
            label="idle_signal",
        )

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key into `.env` inside `instance_path/` when appropriate.

        - Only runs in Gradio mode when key came from the textbox and is non-empty.
        - Only saves if `self.instance_path` is not None.
        - Writes `.env` to `instance_path/.env` (does not overwrite if it already exists).
        - If `instance_path/.env.example` exists, copies its contents while overriding OPENAI_API_KEY.
        """
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided via textbox; skipping persistence.")
                return
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            # Update the current process environment for downstream consumers
            try:
                import os

                os.environ["OPENAI_API_KEY"] = key
            except Exception:  # best-effort
                pass

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                # Respect existing user configuration
                logger.info(".env already exists at %s; not overwriting.", env_path)
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example at %s: %s", example_path, e)

            # Replace or append the OPENAI_API_KEY line
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    content_lines[i] = f"OPENAI_API_KEY={key}"
                    replaced = True
                    break
            if not replaced:
                content_lines.append(f"OPENAI_API_KEY={key}")

            # Ensure file ends with newline
            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created %s and stored OPENAI_API_KEY for future runs.", env_path)
        except Exception as e:
            # Never crash the app for QoL persistence; just log.
            logger.warning("Could not persist OPENAI_API_KEY to .env: %s", e)
