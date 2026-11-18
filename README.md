---
title: DAR Global & Wasalt Chatbot
sdk: docker
app_port: 7860
---

# DAR Global & Wasalt RAG Chatbot

Playwright scrapes JS-rendered pages -> chunks -> local embeddings (fastembed, bge-small) -> numpy cosine search
-> free OpenRouter model (with fallbacks and auto-discovery of free models) -> FastAPI + chat UI, in Docker.

## Build the data (locally, once)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-scrape.txt
playwright install chromium
python -m scraper.scrape          # writes data/raw/*.jsonl  (use --site wasalt, --headful to debug)
python -m scraper.ingest          # writes data/index/
```

## Run locally
```bash
export OPENROUTER_API_KEY=sk-or-...
docker build -t dg-chatbot . && docker run -p 7860:7860 -e OPENROUTER_API_KEY dg-chatbot
# open http://localhost:7860
```

## Deploy (Hugging Face Spaces)
Create a Docker Space, push this repo (including `data/index/`), and add `OPENROUTER_API_KEY` under
Settings -> Secrets. Optional vars: `OPENROUTER_MODEL`, `OPENROUTER_FALLBACK_MODELS`, `MIN_SCORE`, `TOP_K`.
