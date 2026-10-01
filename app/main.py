"""
FastAPI backend for the Zákoník práce RAG assistant.

Exposes:
  POST /api/chat   -> {"question": "..."}  =>  {"answer": "...", "sources": [...]}
  GET  /            -> serves the static chat UI

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel

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

    if not os.path.isdir(PERSIST_DIR):
        raise RuntimeError(
            "Vektorová databáze neexistuje. Nejdřív spusť `python ingest.py`."
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
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Prázdná otázka.")

    try:
        chain = get_qa_chain()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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
