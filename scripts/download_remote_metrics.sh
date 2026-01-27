#!/bin/bash

# Usage: ./download_remote_metrics.sh [remote_user@host] [remote_path] [local_path] [port]

REMOTE_TARGET=${1:-"root@194.68.245.20"}
REMOTE_PATH=${2:-"/workspace/Agent0-curriculum/metric"}
LOCAL_PATH=${3:-"./"}
PORT=${4:-"22034"}

echo "🚀 Starting download from ${REMOTE_TARGET}:${REMOTE_PATH} to ${LOCAL_PATH}..."

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

# Use scp for transfer (rsync is missing on remote)
# -r: recursive
# -C: compress
# -P: port
# Note: scp will NOT delete existing local files; it only adds or overwrites.
scp -P $PORT -i ~/.ssh/id_ed25519 -r -C "$REMOTE_TARGET":"$REMOTE_PATH" "./"

if [ $? -eq 0 ]; then
    echo "✅ Download complete!"
else
    echo "❌ Download failed."
    echo "💡 Tip: Ensure the remote path exists and your SSH key is correct."
fi
