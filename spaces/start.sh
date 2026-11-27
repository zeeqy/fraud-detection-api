#!/bin/sh
# Start the API in the background, wait until it is healthy, then run the demo in the foreground.
set -e
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
python - <<'PY'
import time, urllib.request
for _ in range(60):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health").status == 200:
            break
    except OSError:
        time.sleep(1)
PY
exec streamlit run demo/app.py --server.address 0.0.0.0 --server.port 7860 \
    --server.headless true --browser.gatherUsageStats false
