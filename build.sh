#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt
# check ffmpeg, if not present Render's image likely has it; otherwise you must use a custom Docker service that installs ffmpeg via apt.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found. You must ensure ffmpeg is installed on the host."
fi
