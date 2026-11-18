import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.getenv("FASTEMBED_CACHE", str(ROOT / ".cache" / "fastembed"))

OPENROUTER_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
FALLBACKS = [m.strip() for m in os.getenv(
    "OPENROUTER_FALLBACK_MODELS",
    "google/gemma-3-27b-it:free,mistralai/mistral-small-3.2-24b-instruct:free",
).split(",") if m.strip()]

TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.40"))   # cosine; tune after testing
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))
