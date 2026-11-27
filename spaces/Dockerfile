# Hugging Face Spaces image: FastAPI on 127.0.0.1:8000 (internal) + Streamlit demo on :7860 (public).
# Built by scripts/push_space.py, which copies this file to the Space root as `Dockerfile`.
# Local test from the project root:  docker build -f spaces/Dockerfile -t fraud-space .
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group demo --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --group demo --no-editable

FROM python:3.12-slim-bookworm AS runtime
# libgomp1: OpenMP runtime LightGBM links against. Spaces runs containers as uid 1000.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY api ./api
COPY models ./models
COPY demo ./demo
COPY spaces/start.sh ./start.sh
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/model.joblib \
    MPLCONFIGDIR=/tmp/mpl API_URL=http://127.0.0.1:8000 HOME=/home/user
USER user
EXPOSE 7860
CMD ["sh", "/app/start.sh"]
