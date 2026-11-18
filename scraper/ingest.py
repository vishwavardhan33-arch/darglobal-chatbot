"""Clean, chunk, embed.  Run: python -m scraper.ingest"""
import json
import numpy as np
from fastembed import TextEmbedding
from app import config as C


def chunk_lines(lines, max_words=180, overlap_words=30):
    chunks, cur, n = [], [], 0
    for line in lines:
        w = len(line.split())
        if cur and n + w > max_words:
            chunks.append("\n".join(cur))
            keep, k = [], 0
            for l in reversed(cur):
                lw = len(l.split())
                if k + lw > overlap_words:
                    break
                keep.insert(0, l); k += lw
            cur, n = keep, k
        cur.append(line); n += w
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def main():
    chunks = []
    for path in sorted(C.RAW_DIR.glob("*.jsonl")):
        seen_lines = set()  # keep first copy of repeated boilerplate (menus, footers) per site
        for raw in path.read_text(encoding="utf-8").split("\n"):
            if not raw.strip():
                continue
            pg = json.loads(raw)
            lines = []
            for line in (l.strip() for l in pg["text"].splitlines()):
                if len(line) < 3 or line.lower() in seen_lines:
                    continue
                seen_lines.add(line.lower()); lines.append(line)
            for t in chunk_lines(lines):
                if len(t.split()) >= 15:
                    chunks.append({"company": pg["company"], "url": pg["url"], "title": pg["title"], "text": t})
    assert chunks, "No chunks - run `python -m scraper.scrape` first."
    model = TextEmbedding(C.EMBED_MODEL, cache_dir=C.EMBED_CACHE)
    emb = np.array(list(model.passage_embed([f"{c['title']}\n{c['text']}" for c in chunks])), dtype="float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.INDEX_DIR / "embeddings.npy", emb)
    (C.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks -> {C.INDEX_DIR}")


if __name__ == "__main__":
    main()
