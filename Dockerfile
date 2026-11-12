# ---- build: resolve deps with uv into /app/.venv --------------------------------------------
# Same base as runtime so the venv's python path matches; uv binary pinned from its own image.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---- runtime: slim python, non-root user, only what serving needs ----------------------------
FROM python:3.12-slim-bookworm AS runtime
# libgomp1: OpenMP runtime LightGBM links against
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY api ./api
COPY models ./models
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/model.joblib \
    MPLCONFIGDIR=/tmp/mpl
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
