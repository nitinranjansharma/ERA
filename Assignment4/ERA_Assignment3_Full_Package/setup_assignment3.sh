#!/usr/bin/env bash
set -e

TARGET_DIR="/Users/nitinranjansharma/Documents/codes/ERA/Assignment3"
echo "================================================================"
echo "ERA V5 Session 4 — Assignment Setup & Execution"
echo "Target Directory: $TARGET_DIR"
echo "================================================================"

mkdir -p "$TARGET_DIR/data" "$TARGET_DIR/netlify_widget"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$SCRIPT_DIR" != "$TARGET_DIR" ]; then
  cp -R "$SCRIPT_DIR/"* "$TARGET_DIR/"
fi

cd "$TARGET_DIR"
echo "[1/3] Files staged in $TARGET_DIR:"
ls -la "$TARGET_DIR"

echo ""
echo "[2/3] Running Hugging Face dataset download & Session 4 cleanup pipeline..."
python3 "$TARGET_DIR/clean_and_dedup_pipeline.py"

echo ""
echo "[3/3] Ready for Netlify deployment!"
echo "Option A (Drag & Drop): Open https://app.netlify.com/drop and drag '$TARGET_DIR/netlify_widget'"
echo "Option B (Netlify CLI): Run: npx netlify-cli deploy --prod --dir=\"$TARGET_DIR/netlify_widget\""
