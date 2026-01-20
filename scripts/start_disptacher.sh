#!/bin/bash

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting Self-Consistency Dispatcher Service (Port 8001)..."
nohup python -m curriculum.self_consistency_dispatcher.server > "$PROJECT_ROOT/dispatcher.log" 2>&1 &
DISPATCHER_PID=$!

echo "Dispatcher Service started in background with PID: $DISPATCHER_PID"
echo "Logs are being written to: $PROJECT_ROOT/dispatcher.log"
