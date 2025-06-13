#!/bin/bash
#
# This script sets up the environment and launches the main
# Python application. It assumes the Conda environment has
# already been unpacked in the same directory.
# ------------------------------------------------------------------

set -euo pipefail
exec &> "$HOME/VideoIndexer-launch.log"   # ✅ capture stdout+stderr
set -x       

# Resolve the directory this script lives in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Increase the open file limit to handle large models
ulimit -n 4096

# ---- Config -------------------------------------------------------
ENV_NAME="video-indexer"
ENV_DIR="$DIR/${ENV_NAME}"
PY_ENTRY="$DIR/launch_app.py"
# ------------------------------------------------------------------

# 1) Put env/bin first on PATH so ffmpeg & friends resolve here
export PATH="$ENV_DIR/bin:$PATH"

# 2) Launch your application
exec "$ENV_DIR/bin/python" "$PY_ENTRY" "$@"
