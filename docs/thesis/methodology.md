# Methodology Draft

## Problem and Objectives
The ELTE RAG Assistant is designed to reduce the effort students spend searching campus regulations and policy documents. The system objective is to return concise, grounded answers with explicit source attribution for each response.

## Data Collection and Preparation
Source documents are collected from:
- Official ELTE PDF policy materials stored under `data/raw`.
- Targeted scraping of official ELTE IK pages for new documents and news.

PDFs are parsed through Docling and chunked with `HybridChunker`. Scraped news pages are normalized into JSON records containing URL, title, publication metadata (if available), plain text body, and scrape timestamp.

## Retrieval-Augmented Generation Pipeline
The runtime pipeline follows these stages:
1. Hybrid retrieval: FAISS MMR dense retrieval + BM25 keyword retrieval.
2. Fusion: Reciprocal Rank Fusion with configurable weighting.
3. Reranking: LLM-based reranker scores candidate chunks.
4. Generation: LLM produces structured output with answer, confidence, reasoning, and cited sources.

## Citation Grounding
Each retrieved chunk carries metadata (`source`, `title`, and page when available). Prompted citation IDs (`C1`, `C2`, ...) are mapped back to document metadata, and inline references are converted to human-readable citations.

## System Architecture and Deployment
Backend services run on FastAPI. Frontend is a Vite React application with chat and admin interfaces. The deployment target uses Docker Compose with separate backend and frontend containers.

## Evaluation Plan
Evaluation combines:
- Retrieval and answer quality checks on representative student questions.
- Citation correctness checks (document and page-level when present).
- Usability feedback from student/admin interaction with chat and admin panel.
