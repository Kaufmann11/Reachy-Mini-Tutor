from modules.conversation_manager import ConversationManager
from modules.llm_client import get_response
from modules.profile_store import get_profile, set_assertiveness, set_style
from modules.prompt_builder import build_system_prompt
from modules.metrics_logger import log_turn
from modules.preference_extractor import extract_preferences


USER_ID = "student_001"


def load_base_prompt() -> str:
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def print_help():
    print("\nCommands:")
    print("  /setup (Setup erneut ausführen)")
    print("  /assertiveness gar nicht|mittel|sehr")
    print("  /style kollege|tutor|dozent")
    print("  /topic <Thema/Kapitel/Fragestellung>")
    print("  /goal <Ziel: Klausur/Projekt/Verständnis...>")
    print("  /exam <Prüfungsformat: MC/offen/...>")
    print("  /deadline <Datum/Zeitrahmen>")
    print("  /material <kurze Zusammenfassung oder Ausschnitt>")
    print("  /status   (zeigt aktuellen Kontext)")
    print("  /reset_context")
    print("  /help")
    print("  /quit\n")


def onboarding_if_needed(user_id: str) -> None:
    """
    Kein Default: Der User muss Style/Assertiveness beim ersten Mal aktiv wählen.
    Danach werden die Werte im Profil gespeichert und nicht erneut abgefragt.
    """
    profile = get_profile(user_id)

    needs_style = not profile.get("study_buddy_style")
    needs_assertiveness = not profile.get("assertiveness")

    if not (needs_style or needs_assertiveness):
        return

    print("\nBevor wir starten, kurz dein Setup (nur einmal nötig).")

    if needs_style:
        print("\nWelche Rolle soll ich einnehmen?")
        print("  1) kollege (locker, du)")
        print("  2) tutor (unterstützend, du)")
        print("  3) dozent (formal, Sie)")
        choice = input("Wähle 1/2/3: ").strip()
        style = {"1": "kollege", "2": "tutor", "3": "dozent"}.get(choice)
        while style is None:
            choice = input("Bitte 1, 2 oder 3 wählen: ").strip()
            style = {"1": "kollege", "2": "tutor", "3": "dozent"}.get(choice)
        set_style(user_id, style)
        print(f"Gespeichert: style = {style}")

    if needs_assertiveness:
        print("\nWie direkt soll ich sein?")
        print("  1) gar nicht (sehr sanft)")
        print("  2) mittel (ausgewogen)")
        print("  3) sehr (klar und führend)")
        choice = input("Wähle 1/2/3: ").strip()
        level = {"1": "gar nicht", "2": "mittel", "3": "sehr"}.get(choice)
        while level is None:
            choice = input("Bitte 1, 2 oder 3 wählen: ").strip()
            level = {"1": "gar nicht", "2": "mittel", "3": "sehr"}.get(choice)
        set_assertiveness(user_id, level)
        print(f"Gespeichert: assertiveness = {level}")

    print("\nSetup abgeschlossen. Du kannst es später ändern mit /setup, /style oder /assertiveness.\n")


def main():
    base_prompt = load_base_prompt()
    conv = ConversationManager(USER_ID)

    onboarding_if_needed(USER_ID)

    session = {
        "topic": "",
        "goal": "",
        "exam": "",
        "deadline": "",
        "material": "",
    }

    print("Reachy Study Buddy gestartet. Tippe /help für Befehle.\n")

    while True:
        user_text = input("Du: ").strip()
        if not user_text:
            continue
        # Natural language preference function calling (if not command)
        if not user_text.startswith("/"):
            prefs = extract_preferences(user_text)
            new_style = prefs.get("study_buddy_style")
            new_assert = prefs.get("assertiveness")

            if new_style:
                try:
                    set_style(USER_ID, new_style)
                    print(f"(Preference erkannt) style = {new_style}")
                except Exception:
                    pass

            if new_assert:
                try:
                    set_assertiveness(USER_ID, new_assert)
                    print(f"(Preference erkannt) assertiveness = {new_assert}")
                except Exception:
                    pass

        if user_text.startswith("/"):
            if user_text == "/quit":
                path = conv.save()
                print(f"Gespräch gespeichert: {path}")
                break

            if user_text == "/help":
                print_help()
                continue

            if user_text == "/setup":
                onboarding_if_needed(USER_ID)
                continue

            if user_text.startswith("/assertiveness "):
                level = user_text.split(" ", 1)[1].strip()
                try:
                    set_assertiveness(USER_ID, level)
                    print(f"OK: assertiveness = {level}")
                except Exception as e:
                    print(f"Fehler: {e}")
                continue

            if user_text.startswith("/style "):
                style = user_text.split(" ", 1)[1].strip()
                try:
                    set_style(USER_ID, style)
                    print(f"OK: style = {style}")
                except Exception as e:
                    print(f"Fehler: {e}")
                continue

            if user_text.startswith("/topic "):
                session["topic"] = user_text.split(" ", 1)[1].strip()
                print(f"OK: topic = {session['topic']}")
                continue

            if user_text.startswith("/goal "):
                session["goal"] = user_text.split(" ", 1)[1].strip()
                print(f"OK: goal = {session['goal']}")
                continue

            if user_text.startswith("/exam "):
                session["exam"] = user_text.split(" ", 1)[1].strip()
                print(f"OK: exam = {session['exam']}")
                continue

            if user_text.startswith("/deadline "):
                session["deadline"] = user_text.split(" ", 1)[1].strip()
                print(f"OK: deadline = {session['deadline']}")
                continue

            if user_text.startswith("/material "):
                session["material"] = user_text.split(" ", 1)[1].strip()
                print("OK: material gesetzt.")
                continue

            if user_text == "/status":
                print("\nAktueller Kontext:")
                print(f"  topic: {session.get('topic','') or '-'}")
                print(f"  goal: {session.get('goal','') or '-'}")
                print(f"  exam: {session.get('exam','') or '-'}")
                print(f"  deadline: {session.get('deadline','') or '-'}")
                print(f"  material: {'gesetzt' if session.get('material') else '-'}\n")
                continue

            if user_text == "/reset_context":
                session = {"topic": "", "goal": "", "exam": "", "deadline": "", "material": ""}
                print("OK: Kontext zurückgesetzt.\n")
                continue

            print("Unbekannter Command. /help")
            continue

        conv.add("user", user_text)

        profile = get_profile(USER_ID)
        system_prompt = build_system_prompt(base_prompt, profile, session)

        reply = get_response(system_prompt, conv.history)
        print(f"Reachy: {reply}\n")

        conv.add("assistant", reply)
        conv.save()

        # Metrics logging (no sensitive contents)
        style = profile.get("study_buddy_style", "")
        assertiveness = profile.get("assertiveness", "")
        log_turn(USER_ID, style, assertiveness, session, user_text, reply)


if __name__ == "__main__":
    main()

