#!/bin/bash

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

# Check for API Key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY is not set."
    echo "Please run: export GEMINI_API_KEY='your_key_here'"
    exit 1
fi

echo "Starting Math Agent Service (Port 8000)..."
conda run -n math-agent python -m math_agent.server > "$PROJECT_ROOT/math_agent.log" 2>&1 &
MATH_PID=$!

echo "Services started in background."
echo "Math Agent PID: $MATH_PID (Logs: math_agent.log)"

# Function to stop services on exit
cleanup() {
    echo "Stopping services..."
    kill $MATH_PID
    exit
}

trap cleanup SIGINT SIGTERM

# Keep script running to monitor or wait
echo "Press Ctrl+C to stop both services."
wait
