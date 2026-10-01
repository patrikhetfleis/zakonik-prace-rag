FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Vector DB is built at build time so the container is ready to serve
# immediately (requires OPENAI_API_KEY as a build secret / build arg).
# For a simpler setup, you can instead run ingest.py once as a startup task.

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
