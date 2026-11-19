import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.getenv("FASTEMBED_CACHE", str(ROOT / ".cache" / "fastembed"))

OPENROUTER_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
FALLBACKS = [m.strip() for m in os.getenv(
    "OPENROUTER_FALLBACK_MODELS",
    "nvidia/nemotron-3.5-lightning:free,dots-studio/dots-3-note-preview:free,nvidia/nemotron-3.5-super-vl:free",
).split(",") if m.strip()]

TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.40"))   # cosine; tune after testing
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MAX = int(os.getenv("CACHE_MAX", "300"))
FIRST_TOKEN_TIMEOUT = float(os.getenv("FIRST_TOKEN_TIMEOUT", "6"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "900"))
