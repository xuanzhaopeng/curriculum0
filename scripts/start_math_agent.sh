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
nohup python -m math_agent.server > "$PROJECT_ROOT/math_agent.log" 2>&1 &
MATH_PID=$!

echo "Math Agent Service started in background with PID: $MATH_PID"
echo "Logs are being written to: $PROJECT_ROOT/math_agent.log"
