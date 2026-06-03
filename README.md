# Reachy Mini Tutor
Tutoring robot software for university students built on Reachy Mini (Pollen Robotics).
Part of the research project: "Knowledge-based Generative Social Robots for Tutoring in Higher Education" at Zurich University of Applied Science, Institue of Information Systems. 

---

## Prerequisites

- Hardware: Reachy Mini platform: https://huggingface.co/reachy-mini
- Operating System: macOS 14 or newer
- Browser: Google Chrome (required — Safari does not support WebRTC audio)
- OpenAI API Key with Realtime API access
- USB-C cable to connect Reachy Mini to laptop

---

## Installation

    git clone https://github.com/Kaufmann11/Reachy-Mini-Tutor.git
    cd Reachy-Mini-Tutor
    uv sync
    cp .env.example .env

Then open `.env` and add your API key:

    OPENAI_API_KEY=sk-...

---

## Running with the Physical Reachy Mini

1. Connect the Reachy Mini to your laptop via USB-C.
2. Power the robot on and wait until the antenna LED is steady.
3. In the project folder, start the app:

       ./start_tutor_no_sim.sh

4. Open Google Chrome and go to:

       http://127.0.0.1:7860

5. Wait until the Gradio interface has fully loaded (audio device should show "Reachy Mini" as input).

---

## Running in Simulation (without physical robot)

For development or testing without the robot connected:

    ./start_tutor.sh --no-camera

The same Chrome URL applies: `http://127.0.0.1:7860`.

---

## Tutor Profiles

| Profile          | Condition     | Style                       |
|------------------|---------------|-----------------------------|
| tutor_buddy      | V1 — KBD      | Warm peer, informal (du)    |
| tutor_professor  | V1 — KBD      | Formal (Sie)                |
| tutor_coach      | V1 — KBD      | Energetic coach (du)        |
| tutor_socratic   | V1 — KBD      | Socratic method (du)        |
| tutor_basic      | V2 — Basic    | Neutral AI tutor            |

**How to start a session:**

1. Select the desired profile from the dropdown.
2. Click **Apply personality**.
3. For V1 profiles: upload the lecture PDF (drag and drop, wait for upload confirmation).
4. Click **Aufnehmen** and start speaking.

---

## Experiment Design

**V1 (Knowledge-Based Didactics):**
Combines three knowledge layers — Self-Knowledge (assertiveness and style), User-Knowledge (personalization derived from the onboarding dialogue), and Context-Knowledge (PDF/RAG content plus didactic methods: scaffolding, nudging, Socratic questioning).

**V2 (Basic):**
Identical onboarding flow, but no KBD layer is applied. The robot behaves as a neutral AI tutor without personalization or scaffolding logic.

---

## Stopping the App

Stop the running terminals with `Ctrl+C`. If a terminal does not respond:

    pkill -f mjpython && pkill -f reachy-mini-conversation-app

---

## Troubleshooting

| Problem                       | Solution                                                       |
|-------------------------------|----------------------------------------------------------------|
| Browser shows nothing         | Wait 10 seconds, then refresh Chrome at http://127.0.0.1:7860  |
| No audio output               | Use Chrome; in macOS Sound Settings choose Reachy Mini output  |
| Robot does not move           | Check USB-C connection; ensure robot is powered on             |
| "Address already in use"      | Run: pkill -f mjpython && pkill -f reachy-mini-conversation-app |
| "Camera not found"            | Normal — audio and conversation still work                     |
| API errors                    | Verify OPENAI_API_KEY in .env; check Realtime API access       |

---

## Repository

https://github.com/Kaufmann11/Reachy-Mini-Tutor

## Acknowledgements
This project is based on the Reachy Mini Conversation App by Pollen Robotics.

Original repository:
https://github.com/pollen-robotics/reachy_mini_conversation_app

This project extends/modifies the original implementation for educational and research purposes.
