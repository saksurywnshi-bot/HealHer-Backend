#!/usr/bin/env bash
# Helper script to compress video for web streaming with faststart.
# Run by hand, never run by the app.
# Usage: ./scripts/compress_video.sh input.mp4 output.mp4

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_video> <output_video>"
  exit 1
fi

ffmpeg -i "$1" -vf scale=-2:720 -c:v libx264 -crf 28 -preset slow -c:a aac -b:a 96k -movflags +faststart "$2"

