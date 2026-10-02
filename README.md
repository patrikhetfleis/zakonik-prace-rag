# Zákoník práce – RAG asistent

Malá demo aplikace: chatbot, který odpovídá na dotazy nad **aktuálním zněním
českého zákoníku práce** (zákon č. 262/2006 Sb.) pomocí RAG (Retrieval-Augmented
Generation).

Cíl projektu: ukázat prakticky RAG pipeline v Pythonu (embeddings, vektorová
databáze, LangChain, LLM).

## Jak to funguje

1. `ingest.py` stáhne aktuální text zákona z veřejně dostupného portálu,
   rozdělí ho po jednotlivých paragrafech (§), vytvoří embeddingy a uloží
   je do lokální vektorové databáze (Chroma).
2. `app/main.py` (FastAPI) přijme dotaz, najde nejrelevantnější paragrafy
   a pošle je spolu s dotazem do LLM (Claude), který odpoví jen na základě
   nalezeného kontextu a uvede, ze kterých paragrafů čerpal.
3. `static/index.html` je jednoduché chatovací rozhraní.

**Právní poznámka:** znění zákonů vydávaných státem není v ČR chráněno
autorským právem (§ 3 zákona č. 121/2000 Sb.), takže zpracování textu
samotného zákona není problém. Aplikace navíc nejde nad rámec citování
zákona a výslovně upozorňuje, že nejde o právní poradenství.

## Lokální spuštění

```bash
# 1. Vytvoř virtuální prostředí a nainstaluj závislosti
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Nastav API klíče
cp .env.example .env
# a doplň OPENAI_API_KEY (embeddings) a ANTHROPIC_API_KEY (chat model)

# 3. Vytvoř vektorovou databázi (stáhne a zpracuje zákon, cca 1-2 minuty)
python ingest.py

# 4. Spusť server
uvicorn app.main:app --reload --port 8000
```

Otevři `http://localhost:8000` v prohlížeči.

## Nasazení na Azure (App Service)

```bash
# Přihlášení a vytvoření resource group + plánu (pokud ještě neexistují)
az login
az group create --name rg-zakonik-prace --location westeurope
az appservice plan create --name plan-zakonik-prace --resource-group rg-zakonik-prace --sku B1 --is-linux

# Vytvoření Web App z Dockerfile (buildne se lokálně a pushne do ACR,
# nebo použij az webapp up pro rychlé demo nasazení)
az webapp create --resource-group rg-zakonik-prace --plan plan-zakonik-prace \
  --name zakonik-prace-rag-demo --deployment-container-image-name <tvuj-acr>/zakonik-prace-rag:latest

# Nastavení proměnných prostředí
az webapp config appsettings set --resource-group rg-zakonik-prace \
  --name zakonik-prace-rag-demo \
  --settings OPENAI_API_KEY="..." ANTHROPIC_API_KEY="..."
```

Alternativa pro rychlejší demo bez Dockeru: `az webapp up --runtime "PYTHON:3.11"`
přímo z tohoto adresáře (Azure sám rozpozná `requirements.txt`); v tom případě
je potřeba `ingest.py` spustit jako jednorázový deployment/startup task, protože
vektorová databáze se needitovaně neukládá persistentně mezi restarty na
Basic tier (na produkci by šla nahradit Azure AI Search nebo Azure Cosmos DB
for PostgreSQL s pgvector).

## Možná rozšíření

- Nahradit lokální Chroma za **Azure AI Search** (vector index).
- Přidat **evaluaci** odpovědí (např. sadu testovacích otázek + kontrola, že model
  cituje správné paragrafy).
- Přidat streamování odpovědi (SSE) pro plynulejší UX.
- Verzovat zákon (při novele přegenerovat index) a zobrazit datum platnosti znění.

## Struktura projektu

```
zakonik-prace-rag/
├── ingest.py           # stažení zákona + tvorba vektorové DB
├── app/
│   └── main.py          # FastAPI backend s /api/chat
├── static/
│   └── index.html        # chat UI
├── data/chroma/          # hotová vektorová DB (obnovení: make ingest)
├── requirements.txt
├── Dockerfile
└── .env.example
```
