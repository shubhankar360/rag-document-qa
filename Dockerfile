FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# anthropic is optional: without ANTHROPIC_API_KEY the answerer is extractive
RUN pip install --no-cache-dir "numpy>=1.24" "scikit-learn>=1.3" "fastapi[standard]>=0.110" "anthropic>=0.40"

COPY ragqa ragqa
COPY serve.py .
COPY sample_docs /app/docs

RUN useradd --create-home --uid 10001 app
USER app

# Cloud Run sends PORT; the documents are indexed once at startup.
# INDEX_ROOT is left unset, so POST /index is disabled in the container.
ENV HOST=0.0.0.0 PORT=8080 DOCS_PATH=/app/docs
EXPOSE 8080
CMD ["python", "serve.py"]
