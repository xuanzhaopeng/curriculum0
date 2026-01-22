#!/bin/bash

# Usage: ./download_remote_questions.sh [remote_user@host] [remote_path] [local_path] [port]

REMOTE_TARGET=${1:-"root@194.68.245.18"}
REMOTE_PATH=${2:-"/workspace/Agent0-curriculum/questions"}
LOCAL_PATH=${3:-"./"}
PORT=${4:-"22138"}

echo "🚀 Starting download from ${REMOTE_TARGET}:${REMOTE_PATH} to ${LOCAL_PATH}..."

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

# Use scp for transfer (more compatible with minimized systems)
# -P: port
# -i: identity file
# -r: recursive
# -C: compress
scp -P $PORT -i ~/.ssh/id_ed25519 -r -C "$REMOTE_TARGET":"$REMOTE_PATH" "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Download complete!"
else
    echo "❌ Download failed."
    echo "💡 Tip: Ensure the remote path exists and your SSH key is correct."
fi
