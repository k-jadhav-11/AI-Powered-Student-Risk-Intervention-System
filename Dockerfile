# syntax=docker/dockerfile:1

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .

# Keep the large ML runtime CPU-only; install the remaining app dependencies
# afterwards so pip reuses these already-installed wheels.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.14.0" "torchvision==0.29.0"
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app /data /home/app/.cache/huggingface \
    && chown app:app /app /data /home/app/.cache /home/app/.cache/huggingface

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface

COPY --from=builder /venv /venv

WORKDIR /app
COPY --chown=app:app . .
RUN mkdir -p /app/data /app/logs /app/reports \
    && chown app:app /app/data /app/logs /app/reports

USER app

EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
