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
6. Optional local cross-encoder reranking of the merged candidate set.
7. Answer generation constrained by retrieved context and system prompt rules, with chat history passed to the generator.

This design preserves conversational continuity while keeping retrieval scope focused: full chat history is not indexed or searched in vector retrieval.

Responses are structured and include answer text, confidence, reasoning field, and cited sources. Inline citations are normalized to stable source references for UI rendering.

## Pipeline Modes and Reranker Modes
The runtime now supports two pipeline modes:
- `baseline_v1`: original production behavior for stable comparison.
- `enhanced_v2`: enables additional context controls (currently document diversity caps and post-answer verification checks).

Reranking is runtime-selectable:
- `cross_encoder`: local cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with strict candidate caps.
- `off`: no reranking.

These controls allow direct A/B comparisons without removing baseline behavior.

Historical experiment note: an `llm` reranker mode was also included during benchmark sweeps (March 29, 2026) as part of the trial-and-error methodology. It was later retired from runtime mode options after ablation results showed no quality advantage over `off` and worse latency/token efficiency.

## Conversation Context Transport
The backend remains stateless per request. The frontend sends chat history with role/text and optional assistant `cited_sources` metadata.  
This allows lightweight carry-over of recent evidence without introducing full-history retrieval.

## Runtime Configuration and Administration
The FastAPI backend exposes admin operations for:
- Runtime model/prompt configuration.
- Document upload and deletion.
- Document synchronization and vector reindex operations.
- News index bootstrap/sync (manual trigger only).
- Usage analytics review.

There is no periodic background scheduler for news or document sync. Refresh operations are explicitly user-triggered from the admin API/UI.

Admin runtime settings include embedding profile selection (`local_minilm`, `local_mpnet`, `openai_small`, `openai_large`), pipeline mode, reranker mode, and chunk/parser profiles used for indexed snapshots.

## Snapshot-Based Index Isolation
To prevent embedding-profile collisions, document indexes are versioned snapshots stored under `data/indexes/<snapshot-id>`.

Snapshot key components:
- Corpus fingerprint hash
- Embedding profile
- Chunk profile
- Parser profile

Each reindex writes:
- FAISS index artifacts
- `manifest.json` with build metadata (timestamp, source/chunk counts, model/profile information)

Active index selection is profile-scoped and stored in `data/runtime/active_indexes.json`.  
Switching embedding profile updates the active pointer rather than overwriting prior indexes.

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

Benchmarking beyond single-run evaluation is executed via `scripts/run_benchmarks.py`, which supports:
- Stage A family comparison (pipeline mode × reranker mode with anchor embeddings)
- Stage B embedding sweep for selected families
- Single-turn and multi-turn chat-history scenario sets
- Cost instrumentation outputs (token/cost estimates with explicit pricing assumptions)

## Deployment and Reproducibility
The system is containerized with Docker Compose (`backend` + `frontend`). Reproducibility is validated through a fixed command gate: backend tests, frontend tests/build, compose build/start, and health endpoint checks.

## Milestone Timing Note
Milestone 3 deadline was March 25, 2026. Final integration, logging instrumentation, and quantitative evaluation artifacts were completed after this date as part of the recovery run.
