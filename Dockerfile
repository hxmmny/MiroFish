FROM python:3.11

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app
COPY . .

WORKDIR /app/backend
RUN uv sync --frozen --no-dev \
    && uv pip install --python .venv/bin/python gunicorn==23.0.0

ENV PATH="/app/backend/.venv/bin:$PATH"
ENV FLASK_DEBUG=False

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 300 render_app:app"]
