#!/usr/bin/env bash
while inotifywait -r -e modify,create,delete --exclude '\.git' .; do
  git add .
  git commit -m "Auto‑sync at $(date +'%Y-%m-%d\ %H:%M:%S')" || true
  git push origin main || true
done
