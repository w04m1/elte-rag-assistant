# Methodology

## Problem Context and Objective
ELTE students often need fast answers about academic rules, deadlines, and procedures, but relevant information is distributed across multiple documents and formats. The objective of this thesis project is to provide a lightweight retrieval-augmented generation (RAG) assistant that answers policy questions with explicit source attribution.

## Data Acquisition and Normalization
The system uses two primary source streams:
- Official policy and administrative documents (`.pdf`) synchronized into `data/raw`.
- Official ELTE IK news content normalized into JSON records under `data/news/items`.

Document parsing uses Docling. Chunks are generated with token-aware splitting and persisted into FAISS indexes. Metadata required for citation grounding is preserved through ingestion, including source path/URL, title, and page (where available).

## Retrieval and Generation Pipeline
At query time, the runtime pipeline performs:
1. Follow-up detection using lightweight heuristics (short query and/or referential language, with at least one prior turn available).
2. Conditional query rewrite for likely follow-ups, using the last 6 chat turns to produce a standalone retrieval query.
3. Dense retrieval (FAISS, MMR) and sparse retrieval (BM25), then fusion with RRF.
4. News retrieval from the separate news index and fusion into the same candidate pool.
5. Citation carry-over from only the most recent assistant turn (maximum 2 cited sources), merged into candidates and deduplicated by source + page + snippet.
6. Optional LLM reranking of the merged candidate set.
7. Answer generation constrained by retrieved context and system prompt rules, with chat history passed to the generator.

This design preserves conversational continuity while keeping retrieval scope focused: full chat history is not indexed or searched in vector retrieval.

Responses are structured and include answer text, confidence, reasoning field, and cited sources. Inline citations are normalized to stable source references for UI rendering.

## Conversation Context Transport
The backend remains stateless per request. The frontend sends chat history with role/text and optional assistant `cited_sources` metadata.  
This allows lightweight carry-over of recent evidence without introducing full-history retrieval.

## Runtime Configuration and Administration
The FastAPI backend exposes admin operations for:
- Runtime model/prompt configuration.
- Document upload and deletion.
- Document synchronization and vector reindex operations.
- News index bootstrap/sync.
- Usage analytics review.

The React admin panel provides corresponding controls and status displays.

## Usage Logging Protocol
To support reproducible evaluation, every `/ask` request is logged to `data/runtime/usage_log.jsonl` with:
- Request identifier (`request_id`) for traceability
- UTC timestamp
- Raw query text
- Answer length
- Confidence
- Generator/reranker model identifiers
- Latency (ms)
- Cited-source counts and source-type mix
- User feedback (`helpful` / `not helpful`) and feedback timestamp when provided
- Request status (`ok`/`error`)

Feedback is submitted through `POST /feedback` and persisted into the matching usage-log record by `request_id`.  
This log is consumed by admin analytics endpoints and by thesis evaluation workflows.

## Evaluation Protocol
Evaluation is executed with a fixed ELTE question set in `data/eval/questions.json` via `scripts/run_evaluation.py`. The runner queries the live API and computes core metrics:
- Citation presence rate
- Non-empty answer rate
- Average latency
- Confidence distribution
- Source mix (`pdf` vs `news`)

Outputs are saved to:
- `data/eval/latest_metrics.json`
- `docs/thesis/evaluation.md`

## Deployment and Reproducibility
The system is containerized with Docker Compose (`backend` + `frontend`). Reproducibility is validated through a fixed command gate: backend tests, frontend tests/build, compose build/start, and health endpoint checks.

## Milestone Timing Note
Milestone 3 deadline was March 25, 2026. Final integration, logging instrumentation, and quantitative evaluation artifacts were completed after this date as part of the recovery run.
