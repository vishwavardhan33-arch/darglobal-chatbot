import json, logging, time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from . import config as C
from .rag import Retriever, answer_stream

logging.basicConfig(level=logging.INFO)
STATE = {}
HITS = defaultdict(deque)


@asynccontextmanager
async def lifespan(app):
    if not (C.INDEX_DIR / "chunks.json").exists():
        raise RuntimeError("Index missing. Run `python -m scraper.scrape` then `python -m scraper.ingest`.")
    STATE["r"] = Retriever()
    STATE["r"].search("warm up", None, 1)   # load the embedding model before the first request
    async with httpx.AsyncClient() as client:
        STATE["client"] = client
        yield


app = FastAPI(title="DAR Global & Wasalt Chatbot", lifespan=lifespan)


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatReq(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=20)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(STATE["r"].chunks), "key_set": bool(C.API_KEY)}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq, request: Request):
    ip = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
    q, now = HITS[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= C.RATE_PER_MIN:
        raise HTTPException(429, "Too many requests - please wait a minute.")
    q.append(now)
    if req.messages[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user.")
    history = [m.model_dump() for m in req.messages]

    async def events():
        try:
            async for kind, val in answer_stream(STATE["r"], STATE["client"], history):
                yield f"data: {json.dumps({kind: val})}\n\n"
        except RuntimeError:
            yield f"data: {json.dumps({'error': 'The free language models are busy right now. Please retry in a moment.'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
