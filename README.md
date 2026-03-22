# ELTE RAG Assistant

Retrieval-augmented FAQ assistant for ELTE policy and administration questions.

## Stack
- Backend: FastAPI + LangChain + FAISS + BM25 + LLM reranker
- Frontend: Vite + React + TypeScript + Tailwind (chat + admin)
- Ingestion: Docling for PDFs and normalized JSON for scraped news
- Deployment: Docker Compose (backend + frontend)

## Local Development

### Backend
```bash
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend uses `VITE_API_BASE_URL` (`frontend/.env.example`).

## Docker
```bash
docker compose up --build
```
- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:8001/docs](http://localhost:8001/docs)

## Admin Flow
1. Upload/delete source PDFs in **Admin → Embeddings and Files**.
2. Run **Scrape ELTE IK** to fetch new official links/news metadata and normalize news text.
3. Run **Reindex Vector Store** to rebuild FAISS from PDFs + normalized news.

Scrape and reindex are intentionally separate operations.

## Citation Note
Page-level citations depend on chunk metadata captured during ingestion. After ingestion logic changes, run a full reindex to refresh stored metadata.
