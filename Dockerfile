FROM python:3.11-slim
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV PYTHONUNBUFFERED=1 PORT=7860 EMBED_MODEL=$EMBED_MODEL FASTEMBED_CACHE=/app/.cache/fastembed
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY static static
COPY data/index data/index
# bake the embedding model into the image so startup needs no download
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('$EMBED_MODEL', cache_dir='/app/.cache/fastembed')"
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
