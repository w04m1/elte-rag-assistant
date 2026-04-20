# ELTE RAG Assistant

[![Tests](https://github.com/w04m1/elte-rag-assistant/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/w04m1/elte-rag-assistant/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/w04m1/elte-rag-assistant/branch/main/graph/badge.svg?token=8V3W1VBQBH)](https://codecov.io/gh/w04m1/elte-rag-assistant)
[![Tests (dev)](https://github.com/w04m1/elte-rag-assistant/actions/workflows/tests.yml/badge.svg?branch=dev)](https://github.com/w04m1/elte-rag-assistant/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/w04m1/elte-rag-assistant/branch/dev/graph/badge.svg?token=8V3W1VBQBH)](https://codecov.io/gh/w04m1/elte-rag-assistant)

Retrieval-augmented FAQ assistant for ELTE policy and administration questions.

## Stack
- Backend: FastAPI + LangChain + FAISS + BM25 + optional rerankers (`off`, `cross_encoder`, `llm`)
- Frontend: Vite + React + TypeScript + Tailwind (chat + admin)
- Ingestion: Typesense document sync + Docling for PDF/DOCX + normalized JSON for news
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

### Chrome Demo Extension
```bash
cd extension
npm install
npm run build
```

Load the unpacked extension from `extension/dist` in Chrome (`chrome://extensions`).

- Injection scope: `https://inf.elte.hu/*` and `http://inf.elte.hu/*`
- Runtime API URL: configurable in extension options page
- Default API URL: `http://localhost:8001` (or `EXT_DEFAULT_API_BASE_URL` at build time)
- For local demo backend CORS can is set to `*` (`CORS_ALLOW_ORIGINS=*`)

## Docker
```bash
docker compose up --build
```
- NVIDIA GPU override (Windows/Linux hosts with NVIDIA container runtime):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:8001/docs](http://localhost:8001/docs)

## Admin Flow
1. Upload/delete source PDFs in **Admin → Embeddings and Files**.
2. Run **Documents Sync** to fetch official ELTE document links from Typesense and download `.pdf/.doc/.docx` files.
3. Run **Reindex Vector Store** to rebuild FAISS from local `.pdf/.docx` files + normalized news.
4. Run **News Index → Bootstrap/Sync** manually when you want to refresh news coverage.

Documents sync and reindex are intentionally separate operations.
News sync is also manual-only (no background periodic polling).

## Citation Note
Page-level citations depend on chunk metadata captured during ingestion. After ingestion logic changes, run a full reindex to refresh stored metadata.

## Index Snapshots
- Document indexes are now snapshot-based under `data/indexes/<snapshot-id>/`.
- Active index selection is profile-specific (`local_minilm`, `openai_small`, `openai_large`) and stored in `data/runtime/active_indexes.json`.
- Reindex creates a new immutable snapshot and updates the active pointer for the selected embedding profile.

## Usage Analytics
- Runtime query usage is logged to `data/runtime/usage_log.jsonl` (one JSON line per `/ask` call).
- Each `/ask` response includes `request_id`, which can be used to attach user feedback.
- Feedback endpoint:
  - `POST /feedback` with `{ "request_id": "...", "helpful": true|false }`
- Admin endpoints:
  - `GET /admin/usage?limit=200`
  - `GET /admin/usage/stats?window_days=7`

## Evaluation Command
Run the fixed-question evaluation against a live backend:

```bash
uv run python scripts/run_evaluation.py --api-base-url http://127.0.0.1:8001
```

Artifacts:
- `data/eval/latest_metrics.json`
- `docs/thesis/evaluation.md`

## Benchmark Commands
Cost estimate for missing corpus embeddings:

```bash
uv run python scripts/estimate_embedding_cost.py
```

Staged benchmark matrix (single-turn + multi-turn):

```bash
uv run python scripts/run_benchmarks.py --api-base-url http://127.0.0.1:8001
```

Solid-only full matrix benchmark (46 manually accepted gold rows):

```bash
uv run python scripts/run_benchmarks.py \
  --api-base-url http://127.0.0.1:8001 \
  --plan data/eval/benchmark_plan_solid_full_matrix.json \
  --single-turn data/eval/questions_solid_v2.json \
  --multi-turn data/eval/multi_turn_questions_solid_v2.json \
  --gold-set data/eval/gold_set_v2.json \
  --judge-model "" \
  --index-pkl data/indexes/<snapshot>__local_minilm__standard__docling_v1/index.pkl
```

## Reranker Decision Trail
On March 29, 2026 (benchmark artifact: `data/eval/benchmarks/benchmark_20260329_141148/benchmark_report.json`), we evaluated three reranker modes in Stage A: `off`, `cross_encoder`, and `llm`.

- Average Stage A family score by reranker mode:
  - `off`: `0.6155`
  - `cross_encoder`: `0.4697`
  - `llm`: `0.4190`
- Best `off` run (`enhanced_v2 + off + local_minilm`) vs best `llm` run (`enhanced_v2 + llm + local_minilm`):
  - Quality score delta: `+0.1300` in favor of `off`
  - Single-turn latency delta: `-1582.06 ms` (faster with `off`)
  - Multi-turn latency delta: `-1017.15 ms` (faster with `off`)
  - LLM reranker token overhead: `67699` reranker input tokens in the best `llm` run

Current runtime options keep all three reranker modes (`off`, `cross_encoder`, `llm`) for controlled comparison and benchmarking.

## Solid Gold Benchmark Decision Trail (April 6, 2026)
Artifact:
- `data/eval/benchmarks/benchmark_20260406_200454/benchmark_report.json`

Protocol:
- Gold-only set: `46` rows (`32` single-turn, `14` multi-turn)
- Full matrix: `3` embeddings × `2` pipelines × `3` rerankers (`18` configs)
- Deterministic gold scoring only (`--judge-model ""`)
- Comparability check: `true` (same corpus hash across embeddings)

Best quality configuration in this run:
- `openai_large + baseline_v1 + off`
- `grounded_correctness=0.6364`
- `faithfulness=0.5973`
- `citation_precision=0.8533`

Fastest configuration in this run:
- `local_minilm + baseline_v1 + off`
- single-turn latency `3660.78 ms`
- multi-turn latency `3527.94 ms`
