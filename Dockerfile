# Serves a saved run. The model is mounted at run time, so no weights or data live in the image.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[serve]"

ENV MELD_RUN=/model MELD_DEVICE=cpu HF_HUB_OFFLINE=1
USER nobody
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "meld_emotion.api:app", "--host", "0.0.0.0", "--port", "8000"]
