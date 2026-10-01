"""
Ingest script for the Zákoník práce RAG demo.

Fetches the current full text of the Czech Labour Code (zákon č. 262/2006 Sb.)
from a public legal-text portal, splits it into per-paragraph (§) chunks,
embeds them, and stores them in a local Chroma vector database.

Legal texts issued by Czech state bodies (laws, court decisions, official
documents) are explicitly excluded from copyright protection under
§ 3 of the Czech Copyright Act (zákon č. 121/2000 Sb.), so redistributing
and processing the law's own text is not a copyright issue. The website
formatting / commentary around it is not scraped here, only the raw
legal text.

Usage:
    python ingest.py
"""

import os
import re
import sys

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# If you prefer local, free embeddings instead of OpenAI, swap the import
# above for:
#   from langchain_community.embeddings import HuggingFaceEmbeddings
# and use HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

SOURCE_URL = "https://www.podnikatel.cz/zakony/zakon-c-262-2006-sb-zakonik-prace/uplne/"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma")


def fetch_law_text() -> str:
    print(f"Stahuji text zákona z {SOURCE_URL} ...")
    resp = requests.get(SOURCE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The law text lives in the main article body. We grab all paragraph-like
    # elements after the first "§ 1" heading appears, and drop navigation/
    # table-of-contents noise.
    article = soup.find("article") or soup.find("main") or soup
    text = article.get_text("\n", strip=True)

    # Cut off everything before the actual law text starts (skip the ToC)
    start_marker = "Parlament se usnesl na tomto zákoně"
    idx = text.find(start_marker)
    if idx != -1:
        text = text[idx:]

    return text


def split_into_sections(raw_text: str) -> list[dict]:
    """
    Split the raw law text into chunks, one per § (paragraph/section).
    Returns a list of {"id": "§ 35", "text": "..."} dicts.
    """
    # Sections start with "§" followed by a number (and optional letter suffix)
    pattern = re.compile(r"(§\s?\d+[a-z]?)\b")
    parts = pattern.split(raw_text)

    sections = []
    # parts alternates: [preamble, "§ 1", body, "§ 1a", body, "§ 2", body, ...]
    for i in range(1, len(parts) - 1, 2):
        section_id = parts[i].replace(" ", " ").strip()
        body = parts[i + 1].strip()
        if len(body) < 20:
            continue
        sections.append({"id": section_id, "text": f"{section_id}\n{body}"})

    print(f"Nalezeno {len(sections)} paragrafů (§).")
    return sections


def build_vectorstore(sections: list[dict]) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    docs_text = []
    metadatas = []
    for sec in sections:
        chunks = splitter.split_text(sec["text"])
        for chunk in chunks:
            docs_text.append(chunk)
            metadatas.append({"section": sec["id"], "source": SOURCE_URL})

    print(f"Vytvářím {len(docs_text)} chunků pro embedding ...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectordb = Chroma.from_texts(
        texts=docs_text,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"Hotovo. Vektorová databáze uložena do: {PERSIST_DIR}")


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Chybí OPENAI_API_KEY v prostředí. Nastav ho v .env nebo exportuj.")
        sys.exit(1)

    raw_text = fetch_law_text()
    sections = split_into_sections(raw_text)
    if not sections:
        print("Nepodařilo se rozparsovat žádné paragrafy — zkontroluj SOURCE_URL / strukturu stránky.")
        sys.exit(1)

    build_vectorstore(sections)


if __name__ == "__main__":
    main()
