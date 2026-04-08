#!/bin/bash
# Start Reachy Mini Tutor (Physical Robot)
# Usage: ./start_tutor_no_sim.sh [--no-camera] [--profile tutor_buddy]

PROFILE=${2:-tutor_buddy}
CAMERA_FLAG=""

for arg in "$@"; do
    if [ "$arg" == "--no-camera" ]; then
        CAMERA_FLAG="--no-camera"
    fi
done

echo "🤖 Starting Reachy Mini Tutor (profile: $PROFILE)"
echo "📋 Terminal 1: Simulation | Terminal 2: Conversation App"

osascript -e 'tell app "Terminal" to do script "cd /Users/karimkrimo/Projects/reachy_mini/reachy_mini_conversation_app && uv run mjpython $(uv run which reachy-mini-daemon) --scene minimal --no-localhost-only"'

sleep 3

osascript -e "tell app \"Terminal\" to do script \"cd /Users/karimkrimo/Projects/reachy_mini/reachy_mini_conversation_app && uv run reachy-mini-conversation-app --gradio $CAMERA_FLAG --profile $PROFILE\""

echo "✅ Both terminals started. Open http://127.0.0.1:7860"
