FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY shared/ shared/
COPY collector/ collector/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/data/hf_cache

RUN groupadd --gid 1000 d2oracle && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash d2oracle
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY shared/ shared/
COPY collector/ collector/

RUN mkdir -p /app/data/hf_cache && chown -R d2oracle:d2oracle /app

USER d2oracle

ENTRYPOINT ["d2oracle-collector"]
