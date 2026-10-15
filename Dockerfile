FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY evalh evalh
COPY gateway gateway
COPY configs configs
RUN uv sync --frozen --no-dev
EXPOSE 8080
CMD ["uv", "run", "--no-sync", "uvicorn", "gateway.app:build_default_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
