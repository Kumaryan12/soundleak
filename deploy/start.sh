#!/bin/sh
set -eu
export SOUNDLEAK_HOSTED=1
exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true \
  --server.fileWatcherType=none \
  --server.maxUploadSize=50 \
  --browser.gatherUsageStats=false
