"""
FastAPI backend for the Zákoník práce RAG assistant.

Exposes:
  POST /api/chat   -> {"question": "..."}  =>  {"answer": "...", "sources": [...]}
  GET  /            -> serves the static chat UI

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os
import shutil
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Zákoník práce – RAG asistent")

# --- Vector DB bootstrap ----------------------------------------------------
# On hosts with an ephemeral disk (e.g. Render free tier) data/chroma does not
# exist after a deploy/restart. We build it in a background thread so the server
# starts listening immediately; /api/chat answers 503 until the DB is ready.
_db_ready = threading.Event()
_db_error: str | None = None


def _bootstrap_db() -> None:
    global _db_error
    if os.path.isfile(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        _db_ready.set()
        return
    try:
        import ingest

        ingest.build_database()
        _db_ready.set()
    except Exception as e:  # noqa: BLE001 - report any failure via the API
        _db_error = str(e)
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


@app.on_event("startup")
def _start_bootstrap() -> None:
    threading.Thread(target=_bootstrap_db, daemon=True).start()


# --- Rate limiting (in-memory, per process) -----------------------------------
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "5"))
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "300"))
_hits: dict[str, deque] = defaultdict(deque)
_daily = {"day": time.strftime("%Y-%m-%d"), "count": 0}
_rl_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    now = time.time()
    ip = _client_ip(request)
    with _rl_lock:
        today = time.strftime("%Y-%m-%d")
        if _daily["day"] != today:
            _daily.update(day=today, count=0)
        if _daily["count"] >= DAILY_LIMIT:
            raise HTTPException(429, "Denní limit dotazů pro demo byl vyčerpán. Zkus to zítra.")
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(429, "Příliš mnoho dotazů, zkus to za minutu.")
        q.append(now)
        _daily["count"] += 1

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT_TEMPLATE = """Jsi asistent, který odpovídá na otázky o českém zákoníku práce
(zákon č. 262/2006 Sb.) výhradně na základě poskytnutého kontextu níže.

Pravidla:
- Odpovídej stručně a věcně, v češtině.
- Pokud odpověď v kontextu není, řekni to a nevymýšlej si ji.

Kontext:
{context}

Otázka: {question}

Odpověď:"""

_qa_chain = None


def get_qa_chain() -> RetrievalQA:
    global _qa_chain
    if _qa_chain is not None:
        return _qa_chain

    if not _db_ready.is_set():
        raise RuntimeError(
            f"Vektorová databáze selhala: {_db_error}"
            if _db_error
            else "Vektorová databáze se právě připravuje (cca 1–2 minuty), zkus to znovu."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model_name=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),  # type: ignore[arg-type]
            temperature=0,
            timeout=None,
            stop=None,
        )
    else:
        raise RuntimeError(
            f"Neznámý LLM_PROVIDER '{provider}'. Použij 'anthropic' nebo 'openai'."
        )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return _qa_chain


class ChatRequest(BaseModel):
    question: str = Field(max_length=500)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Prázdná otázka.")

    try:
        chain = get_qa_chain()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    check_rate_limit(request)

    result = chain.invoke({"query": req.question})
    sources = sorted(
        {doc.metadata.get("section", "?") for doc in result.get("source_documents", [])}
    )
    return ChatResponse(answer=result["result"], sources=sources)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Serve the static frontend (index.html, etc.)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
