FROM python:3.10-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install -e ".[dev,gcp,musdb]"

FROM base AS test
RUN ruff check .
RUN pytest -q

FROM base AS runtime
CMD ["python", "-m", "hs_tasnet", "--help"]
