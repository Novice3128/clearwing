FROM python:3.12-slim AS builder

WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache --extra sourcehunt -r pyproject.toml

COPY . .
RUN uv pip install --system --no-cache --no-deps .

# -------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# A real, user-owned home is required: clearwing_home() writes
# ~/.clearwing (sessions, memory, reports). --no-create-home used to leave
# HOME=/nonexistent, which made every SessionStore touch 500 (#7).
RUN adduser --system --group --home /home/clearwing clearwing

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

USER clearwing

ENTRYPOINT ["clearwing"]
