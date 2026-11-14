FROM python:3.14-slim

WORKDIR /app

# Python runtime settings.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Copy API-specific dependencies.
COPY requirements-api.txt .

# Install only production API requirements.
RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip \
    && python -m pip install \
    --no-cache-dir \
    -r requirements-api.txt

# Copy application source.
COPY src ./src

# Copy the trained model and metadata.
COPY artifacts ./artifacts

# Create a non-root application user.
RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

# Start FastAPI using Uvicorn.
CMD ["sh", "-c", "uvicorn ipl_mlops.api:app --host 0.0.0.0 --port ${PORT:-8000}"]