#!/bin/bash

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting Self-Consistency Dispatcher Service (Port 8001)..."
python -m curriculum.self_consistency_dispatcher.server > "$PROJECT_ROOT/dispatcher.log" 2>&1 &
DISPATCHER_PID=$!

echo "Services started in background."
echo "Dispatcher PID: $DISPATCHER_PID (Logs: dispatcher.log)"

# Function to stop services on exit
cleanup() {
    echo "Stopping services..."
    kill $DISPATCHER_PID
    exit
}

trap cleanup SIGINT SIGTERM

# Keep script running to monitor or wait
echo "Press Ctrl+C to stop both services."
wait
