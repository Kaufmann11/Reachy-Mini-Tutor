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
        self._humor_welcomed: bool = False  # Parsed from Q6 — drives periodic humor mandate
        self._chosen_method: str = ""  # Post-onboarding learning approach (slide/overview/exercise)
        self._post_onboarding_stage: str = ""  # "" | "need_context" | "need_method" | "done"
        self._tutoring_turn_count: int = 0  # Bot tutoring turns since onboarding
        self._last_name_used_turn: int = -99  # Turn index when name was last spoken
        self._document_uploaded: bool = False  # True once student uploaded any doc
        self._onboarding_item_ids: list[str] = []  # V2: delete these to erase onboarding context
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
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
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
                # never keep a stale reference
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

        if q_num == 1:
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
                "Auch sehr kurze Ein-Wort-Antworten (z.B. 'Weltall', 'Sport', 'Mike', 'BWL') sind vollwertige Antworten. "
                "Auch in Füllwörter/Abschweifungen/Versprecher eingebettete Infos sind gültig. "
                "Extrahiere die Kerninformation und erwähne sie in deiner kurzen Reaktion. "
                "Beispiele: "
                "'Weltall.' → 'Weltall — ein faszinierendes Interessengebiet!' "
                "'Trotzdem, mein Name ist Mike.' → 'Hallo Mike, freut mich!' "
                "'Ich studiere nicht mehr, vor zwei Jahren Marketing abgeschlossen.' → 'Ah, Marketing-Background!' "
                "Sage NIEMALS 'Ich habe das nicht ganz verstanden' — wenn die Antwort hier ankommt, ist sie gültig. "
                "\nABSOLUTE ANTI-HALLUZINATIONS-REGEL: "
                "Wiederhole AUSSCHLIESSLICH Wörter, Zahlen, Fächer und Namen, die der Studierende TATSÄCHLICH GESAGT hat. "
                "Wenn 'ersten Semester' gesagt wurde, sage NIE 'siebtes Semester'. "
                "Wenn du eine Zahl oder ein Fach nicht ganz sicher gehört hast, lass sie WEG — "
                "sage lieber 'Ah, Maschinenbau!' statt einer erfundenen Semester-Zahl. "
                "Ergänze NIE Antwortoptionen aus deiner vorigen Frage ('durch Fragen' etc.), die der Student gar nicht genannt hat. "
                "Im Zweifel: weniger wiederholen. "
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
            await self.connection.response.create(
                response={
                    "instructions": instructions,
                    "tool_choice": "auto",
                }
            )
            logger.info("Asked onboarding Q%d (reask=%s)", q_num, reask)
        except Exception as e:
            self._onboarding_q_pending = False
            logger.warning("Onboarding Q%d ask failed: %s", q_num, e)

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        # Reset per-session state so restarts start clean
        self._onboarding = {
            "phase": "onboarding",
            "current_q": 0,
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
        self._humor_welcomed = False
        self._chosen_method = ""
        self._post_onboarding_stage = ""
        self._tutoring_turn_count = 0
        self._last_name_used_turn = -99
        self._document_uploaded = False
        self._onboarding_item_ids = []
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
                                    "silence_duration_ms": 1000,
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
                try:
                    await conn.response.create(response={})
                    logger.info("Triggered initial greeting for non-tutor profile=%s", _cur_profile)
                except Exception as e:
                    logger.warning("Initial response.create failed: %s", e)
            else:
                logger.info("Waiting for user to initiate conversation (profile=%s)", _cur_profile)

            # Manage event received from the openai server
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass
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
                        ):
                            logger.warning("Tutoring: movement without speech — forcing verbal follow-up")
                            try:
                                self._tutoring_verbal_retry_fired = True
                                self._response_create_issued = True
                                await self.connection.response.create(
                                    response={
                                        "instructions": (
                                            "Die Bewegung allein reicht nicht. Reagiere jetzt auch SPRACHLICH "
                                            "auf den letzten Beitrag des Studenten — mit Anerkennung, Scaffolding-Frage "
                                            "oder der nächsten didaktischen Frage. KEINE weitere Bewegung in dieser Antwort. "
                                            "Sprich kurz und klar (1–3 Sätze)."
                                        ),
                                        "tool_choice": "none",
                                    }
                                )
                            except Exception as e:
                                logger.warning("Tutoring verbal-follow-up failed: %s", e)
                        # else: tutoring phase, no movement → stay silent, wait for user
                    self._response_audio_produced = False
                    self._response_create_issued = False
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
                                self._lernprofil_name = _extract_name(ob["answers"].get(1, ""))
                                self._lernprofil_hobbies = _extract_primary_hobby(ob["answers"].get(6, ""))
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
                                elif _profile == "tutor_basic":
                                    # V2: erase onboarding context so GPT-4o literally
                                    # cannot pattern-match on name/hobby/study. No profile
                                    # is injected back — V2 is the control condition.
                                    # Small settle delay so any in-flight item-created
                                    # events land in our tracking list before we delete.
                                    await asyncio.sleep(0.3)
                                    deleted = 0
                                    failed = 0
                                    for iid in self._onboarding_item_ids:
                                        try:
                                            await self.connection.conversation.item.delete(item_id=iid)
                                            deleted += 1
                                        except Exception as e:
                                            failed += 1
                                            logger.warning("V2 item delete %s failed: %s", iid, e)
                                    logger.info(
                                        "V2: deleted %d/%d onboarding items (%d failed)",
                                        deleted, len(self._onboarding_item_ids), failed,
                                    )

                                ob["phase"] = "tutoring"
                                self._post_onboarding_stage = "need_method"  # after context answer, ask method
                                logger.info("Onboarding complete → tutoring (profile=%s)", _profile)

                                # Stage 1 of post-onboarding: Deadline + Wissensstand only.
                                # Method question is asked AFTER student answers these two, so it
                                # cannot be buried in a long 3-question reply and skipped.
                                await self.connection.response.create(
                                    response={
                                        "instructions": (
                                            "Das Onboarding ist abgeschlossen. Stelle jetzt GENAU diese zwei Fragen — "
                                            "eine nach der anderen, in EINER Antwort, kurz und natürlich: "
                                            "1. 'Gibt es eine Deadline oder Abgabe zu diesem Thema, oder ist es ein freies Lernziel?' "
                                            "2. 'Wie würdest du deinen aktuellen Wissensstand zu diesem Thema einschätzen — Einsteiger, Grundkenntnisse, oder schon fortgeschritten?' "
                                            "Keine Prüfungs-Annahmen. Noch NICHT lehren. Keine Bewegungs-Tools in dieser Antwort. "
                                            "Sprich vollständig und warte dann auf die Antwort des Studenten."
                                        ),
                                        "tool_choice": "none",
                                    }
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
                            if not _is_valid_onboarding_answer(text, 0):
                                logger.info("Tutoring phantom-filtered transcript %r — not responding", text)
                                continue

                        # Post-onboarding staging (V1 only — V2 is generic and doesn't need
                        # a chosen method since it shouldn't tailor its approach anyway).
                        # Stage 1 answer received → ask method question. Stage 2 answer → capture method.
                        if (
                            _profile in V1_PROFILES
                            and ob["phase"] == "tutoring"
                            and self._post_onboarding_stage == "need_method"
                            and self.connection
                        ):
                            self._post_onboarding_stage = "awaiting_method_answer"
                            await self.connection.response.create(
                                response={
                                    "instructions": (
                                        "Kurze Anerkennung der Antworten (1 Satz, mit Namen). "
                                        "DANN stelle GENAU diese Frage: "
                                        "'Wie möchten wir vorgehen — Folie-für-Folie durchgehen, zuerst einen Überblick über die Inhalte, "
                                        "oder direkt mit Übungsfragen starten?' "
                                        "Keine Bewegungs-Tools. Noch NICHT lehren. Warte auf die Antwort."
                                    ),
                                    "tool_choice": "none",
                                }
                            )
                            continue
                        if (
                            _profile in V1_PROFILES
                            and ob["phase"] == "tutoring"
                            and self._post_onboarding_stage == "awaiting_method_answer"
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
                                # V2: onboarding items were deleted from context after Q7.
                                # GPT-4o has no access to name/hobbies/etc. anymore, so no
                                # active "forbidden" rules needed. Just a soft fallback in
                                # case the student brings the name up themselves.
                                tutoring_instructions = (
                                    common_turn_rule + " "
                                    "Du hast keine Vorinformationen über den Studierenden — "
                                    "adressiere mit 'Du'. "
                                    "KEINE KAMERA: In dieser Session hast du keinen Kamerazugriff. "
                                    "Sag NIE 'ich sehe', 'halt die Folie vor die Kamera', 'zeig mir', "
                                    "'ich schaue mir das an'. Der Student KANN dir keine Folie zeigen — "
                                    "er kann dir den Inhalt nur VORLESEN oder BESCHREIBEN. "
                                    "Wenn der Student eine Folie vor sich hat, bitte ihn, den Text oder "
                                    "die Kernpunkte der Folie vorzulesen. "
                                    "ANKÜNDIGEN = LIEFERN: Sag NIE 'los geht's', 'lass uns starten', "
                                    "'dann legen wir los' ohne im selben Turn direkt den ersten Inhalt "
                                    "zu liefern. Ankündigung und Lieferung gehören in denselben Turn. "
                                    "Wenn der Studierende fragt 'weißt du noch X?' oder ähnlich, "
                                    "antworte: 'Nein, ich starte jede Session neu ohne Vorwissen.' "
                                    "Wenn der Student dich bittet, Name/Hobbys/Studium zu nennen, "
                                    "sag ehrlich: 'Ich habe keine Informationen über dich gespeichert.' "
                                    "Gib in dem Fall KEINE erfundenen Details an."
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
                                        "HUMOR-MOMENT (JETZT spürbar einbauen): Der Student hat in Q6 "
                                        "Humor ausdrücklich begrüßt. Baue in diese Antwort EINEN konkreten "
                                        "humoristischen Baustein ein, sodass der Student es als Humor erkennt — "
                                        "z.B. einen augenzwinkernden Vergleich ('klingt komplizierter als es ist — "
                                        "ungefähr wie die Abseitsregel'), eine kleine selbstironische Bemerkung "
                                        "('mein persönliches Lieblings-Chaos'), oder eine trockene Pointe "
                                        "('ja, Statistiker hatten auch mal Spaß — angeblich'). "
                                        "EIN Humor-Element, nicht mehr, und nur falls es inhaltlich natürlich "
                                        "andockt. Danach sofort zurück zur Didaktik. Kein albernes Dauerfeuer."
                                    )
                                # 2c) CHOSEN METHOD — inject the student's chosen learning approach.
                                if self._chosen_method:
                                    lines.append(
                                        f"GEWÄHLTE METHODE: '{self._chosen_method}'. Halte dich strikt daran. "
                                        f"Kein eigenmächtiger Wechsel der Vorgehensweise. Nur wenn der Student selbst "
                                        f"eine andere Methode wünscht, wechselst du."
                                    )
                                # 3) RAG — only when doc uploaded.
                                if self._document_uploaded:
                                    lines.append(
                                        "RAG-PFLICHT: Vor jeder fachlichen Aussage rag_tool aufrufen, dann 'Auf Folie X steht…' zitieren. Nichts erfinden."
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
                                    "RICHTIGE ANTWORT SPEZIFISCH FEIERN: Wenn die Antwort des Studenten sachlich korrekt ist, benenne KONKRET den Punkt den er erkannt hat ('Genau — du hast erkannt, dass …'). KEIN nacktes 'Genau' + bloße Wiederholung. Variiere das Anerkennungs-Wording jedes Mal.",
                                    (
                                        "FALSCHE ANTWORT — KBD-DIDAKTIK: Wenn die Antwort falsch, teilweise richtig oder am Thema vorbei ist, "
                                        "folge diesem Muster STRIKT in dieser Reihenfolge: "
                                        "(a) KEIN 'falsch' / 'nein' / 'das stimmt nicht'. Würdige den Denkansatz in einem Satz "
                                        "('Interessante Überlegung' / 'Du denkst in die Richtung von X — nachvollziehbar'). "
                                        f"(b) Benenne konkret WO der Denkweg abzweigt ODER welcher Teil schon auf dem richtigen Pfad ist.{_wa_hobby_hint} "
                                        "(c) Stelle EINE gezielte Teilfrage, die vom falschen Abzweig zurück zum korrekten Pfad führt — "
                                        "KEINE reine Wiederholung der ursprünglichen Frage, sondern ein echter Scaffolding-Schritt. "
                                        f"(d) Erst nach dem 2. Fehlversuch ein winziger Hinweis, nie eine fertige Lösung.{_wa_name_hint} "
                                        "Ziel: der Student findet die Antwort selbst, fühlt sich nicht bloßgestellt."
                                    ),
                                    "ANTWORT = EIN KONZEPT + EINE CHECK-FRAGE: Behandle pro Antwort EIN Konzept, schließe mit EINER Check-Frage. Nach User-Antwort direkt nächstes Konzept. KEINE 3. Folge-Frage zum selben Punkt.",
                                    "KEINE KAMERA: Sag nie 'ich sehe'. Für Folien nur rag_tool.",
                                    "KEIN INFO-DUMP: Max 3 Sätze, ein Gedanke, enden mit Folge-Frage.",
                                    "KEINE ERFUNDENEN FAKTEN: Unsicher → 'Das weiß ich nicht sicher.'",
                                    "BEWEGUNG STUMM: Kommentiere Bewegung NIE verbal ('ich hebe den Kopf' ist VERBOTEN). Sprich Antwort → rufe eine Bewegung → Turn zu Ende.",
                                    "KEINE TOOL-ENTSCHULDIGUNGEN: Sag NIE 'Es gab ein Problem' / 'Entschuldige' / 'hat nicht funktioniert' / 'eine Funktion hat nicht reagiert'. Vorheriger Turn ist abgeschlossen. Beginne neue Antwort direkt mit Inhalt.",
                                    "PACING-ADAPTION: Wenn der Student Deadline/Prüfung/Multiple-Choice erwähnt hat ODER sich als Einsteiger/Anfänger bezeichnet hat → KEINE tiefen Sokratik-Ketten. Maximal 1–2 Folge-Fragen pro Konzept. Liefere die Kernaussage klar, stelle eine kurze Verständnis-Frage, dann weiter zum nächsten Punkt. Tiefe nur auf expliziten Wunsch des Studenten.",
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
                            else:
                                tutoring_instructions = common_turn_rule
                            await self.connection.response.create(
                                response={"instructions": tutoring_instructions, "tool_choice": "auto"}
                            )

                # Handle assistant transcription
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
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
                        await self.connection.response.create(
                            response={
                                "instructions": "Use the tool result just returned and answer concisely in speech.",
                                "tool_choice": "auto",
                            },
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

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
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
        await self.connection.response.create(
            response={
                "instructions": "Call play_emotion with a valid emotion name, or call move_head with a direction (left/right/up/down/front). Do not invent new tool names. No speech.",
                "tool_choice": "required",
            },
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
