#!/bin/bash

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

# Check for API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "Please run: export OPENAI_API_KEY='your_key_here'"
    exit 1
fi

echo "Starting Math Agent Service (Port 8000)..."
conda run -n math-agent python -m math_agent.server > "$PROJECT_ROOT/math_agent.log" 2>&1 &
MATH_PID=$!

echo "Starting Self-Consistency Dispatcher Service (Port 8001)..."
conda run -n agent0-curriculum python -m curriculum.self_consistency_dispatcher.server > "$PROJECT_ROOT/dispatcher.log" 2>&1 &
DISPATCHER_PID=$!

echo "Services started in background."
echo "Math Agent PID: $MATH_PID (Logs: math_agent.log)"
echo "Dispatcher PID: $DISPATCHER_PID (Logs: dispatcher.log)"

# Function to stop services on exit
cleanup() {
    echo "Stopping services..."
    kill $MATH_PID $DISPATCHER_PID
    exit
}

trap cleanup SIGINT SIGTERM

# Keep script running to monitor or wait
echo "Press Ctrl+C to stop both services."
wait
