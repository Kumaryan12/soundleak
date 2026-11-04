ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SOUNDLEAK_HOSTED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
COPY pyproject.toml README.md ./
COPY soundleak ./soundleak
RUN python -m pip install --no-deps .
COPY --chown=appuser:appuser . .
RUN chmod +x deploy/start.sh
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python deploy/healthcheck.py
CMD ["sh", "deploy/start.sh"]
