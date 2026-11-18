# DAR Global & Wasalt Chatbot

A retrieval-augmented (RAG) chatbot that answers questions about **DAR Global** and **Wasalt** using data scraped from their public websites. Containerised with Docker and deployed on Render.

**Live demo:** https://YOUR-SERVICE.onrender.com
*(Free tier: if the app has been idle, the first request can take about a minute while it wakes up.)*

## How it works

```
Playwright scraper  ->  clean + chunk  ->  local embeddings  ->  numpy index
                                                                     |
Browser chat UI  <-  FastAPI (SSE streaming)  <-  retrieval  <-  user question
                                  |
                          OpenRouter free model (with fallbacks)
```

1. **Scrape (`scraper/scrape.py`)**: Playwright (headless Chromium) renders the JavaScript-heavy sites, scrolls to trigger lazy-loaded content and extracts the visible text. It reads each site's sitemap, follows robots.txt, skips Arabic duplicates and caps pages per site.
2. **Ingest (`scraper/ingest.py`)**: drops repeated boilerplate (menus, footers keep one copy), splits text into overlapping chunks of about 180 words, and embeds them with `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU only).
3. **Retrieve (`app/rag.py`)**: embeds the question, runs an exact cosine search (one numpy matrix multiply) and returns the top chunks. If the question names one company, the search is filtered to that company.
4. **Generate**: sends the retrieved context to a free OpenRouter model and streams the answer back, with source links.

## Features

- Works on JavaScript-rendered pages (Playwright), not just static HTML.
- Answers only from the scraped data. If retrieval is weak, it says it doesn't have the information and skips the LLM call.
- Company-aware retrieval, so "What is Wasalt?" is not answered from DAR Global pages.
- Streaming responses, so text appears in about a second once a model responds.
- Reliability on free models: a first-token timeout, automatic fallback to other models, remembering the last working model, and auto-discovery of currently free models on OpenRouter.
- Answer cache for repeated questions (saves latency and free-tier quota).
- Per-IP rate limiting, input size limits, and a prompt that ignores instructions embedded in retrieved content.
- Timing logs per request (`retrieval_ms`, `first_token_ms`, model used) and a `/health` endpoint.

## Project structure

```
app/            FastAPI app (main.py), RAG logic (rag.py), settings (config.py)
scraper/        Playwright crawler (scrape.py), chunk + embed (ingest.py), site config (sites.py)
static/         Chat UI (index.html)
data/index/     Built index (chunks.json, embeddings.npy), committed so the image needs no scraping
Dockerfile      Slim Python image; embedding model baked in at build time
```

## Run it yourself

### 1. Build the data (once, locally)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-scrape.txt
playwright install chromium
python -m scraper.scrape            # writes data/raw/*.jsonl  (options: --site wasalt, --headful)
python -m scraper.ingest            # writes data/index/
```

### 2. Run locally
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...                    # Windows PowerShell: $env:OPENROUTER_API_KEY="sk-or-..."
python -m uvicorn app.main:app --port 7860
# open http://localhost:7860
```

### 3. Run with Docker
```bash
docker build -t dg-chatbot .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY dg-chatbot
```

### 4. Deploy on Render
Push the repo (including `data/index/`) to GitHub, create a **Web Service** on Render, choose **Docker**, pick the free instance, add the `OPENROUTER_API_KEY` environment variable, and set the health check path to `/health`.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | OpenRouter API key. Set as a secret; never commit it |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Preferred free model |
| `OPENROUTER_FALLBACK_MODELS` | two other free models | Comma-separated fallbacks |
| `TOP_K` | `4` | Chunks sent to the model |
| `MIN_SCORE` | `0.40` | Minimum similarity before the bot answers |
| `FIRST_TOKEN_TIMEOUT` | `6` | Seconds to wait for a model's first token before trying the next |
| `CACHE_TTL` / `CACHE_MAX` | `3600` / `300` | Answer cache lifetime (seconds) and size |
| `CHUNK_CHARS` | `900` | Max characters per chunk sent to the model |
| `RATE_PER_MIN` | `20` | Requests per IP per minute |

## Design decisions

- **Exact search instead of ANN.** The corpus is a few thousand chunks, so a brute-force matrix multiply takes about a millisecond and returns the true nearest neighbours. An ANN index (HNSW) would add memory and complexity for no measurable gain at this size.
- **Local embeddings.** fastembed runs on CPU with no extra API dependency, so only the generation step relies on OpenRouter.
- **Index built ahead of time.** Playwright and Chromium stay out of the runtime image, which keeps it small and quick to start.
- **Refuse rather than guess.** A similarity threshold stops the bot answering from weakly related text.

## Limitations

- Answers come from a snapshot of the sites taken at scrape time. Re-run the scrape and ingest steps to refresh it.
- English pages only (Arabic pages are skipped).
- Free OpenRouter models are shared and rate-limited, so latency varies and a model can briefly be unavailable.
- The cache and rate limiter are in memory, so they apply per instance.
- The free hosting tier sleeps when idle, so the first request after a pause is slow.

## Production roadmap

- **Observability:** structured JSON logs with request IDs, a Prometheus `/metrics` endpoint (latency, error and fallback rates), tracing, alerting, and an automated evaluation set for answer quality.
- **Scaling:** Redis for shared rate limiting and caching, multiple workers or replicas, and a paid model or self-hosted endpoint to remove free-tier limits.
- **Data:** scheduled re-scraping with incremental re-indexing, and a vector database (pgvector or Qdrant) with HNSW once the corpus grows past tens of thousands of chunks.
- **Retrieval quality:** hybrid search (BM25 plus vectors), a reranker, and query rewriting for follow-up questions.
