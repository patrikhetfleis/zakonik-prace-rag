PY      := venv/bin/python
PORT    ?= 8000

.PHONY: run ingest stop restart

# Dev server; automaticky se restartuje při změně .py i .env souborů.
run:
	$(PY) -m uvicorn app.main:app --reload --reload-include '.env' --port $(PORT)

# Naplnění vektorové databáze (po změně ingest.py / zdroje dat).
ingest:
	$(PY) ingest.py

# Ruční restart (např. když běží bez --reload): ukončí server na $(PORT) a spustí znovu.
stop:
	-@fuser -k $(PORT)/tcp 2>/dev/null || true

restart: stop run
