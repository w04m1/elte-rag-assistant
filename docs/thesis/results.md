# Results

## Milestone 3 Completion Status
Milestone 3 deadline was March 25, 2026. Final completion tasks were executed on March 28, 2026, including usage logging, quantitative evaluation artifacts, and full runtime validation.

## Implemented Outcomes
The delivered system now includes:
- Student-facing chat UI with citation-aware rendering and source links.
- Student feedback controls on assistant responses (`Helpful` / `Not helpful`) stored by request ID.
- Admin panel for runtime model/prompt control, document operations, sync/reindex jobs, and news indexing.
- Hybrid retrieval pipeline (FAISS MMR + BM25 + RRF), optional cross-encoder reranking, and structured response output.
- Follow-up-aware retrieval resolution:
  - Heuristic follow-up detection.
  - Conditional standalone query rewrite from recent turns.
  - Citation carry-over from the latest assistant turn (max 2), merged and deduplicated before rerank.
  - No full-history vector retrieval.
- Manual-only refresh operations for documents/news (no periodic background polling).
- End-to-end usage logging (`data/runtime/usage_log.jsonl`) with analytics endpoints.
- Reproducible evaluation runner producing JSON and Markdown artifacts.
- Profile-isolated index snapshot system for embedding experiments without destructive overwrite.
- Runtime-selectable pipeline/reranker/embedding profiles for controlled benchmarking.
- Staged benchmark harness (`scripts/run_benchmarks.py`) with single-turn and multi-turn evaluation support.
- Embedding-cost estimation utility (`scripts/estimate_embedding_cost.py`) for pre-run API spend planning.

## Verification Evidence
Validation commands completed successfully:
- `uv run pytest` -> 71 passed
- `npm test` -> 11 passed
- `npm run build` -> success
- `docker compose build` -> backend/frontend built
- `docker compose up -d` -> services started
- `curl http://127.0.0.1:8001/health` -> status ok
- `curl -I http://127.0.0.1:5173` -> HTTP 200
- `docker compose down` -> services stopped

## Quantitative Evaluation (Fixed 12-Question Set)
Source artifacts:
- `data/eval/latest_metrics.json`
- `docs/thesis/evaluation.md`

| Metric | Value |
| --- | ---: |
| Total queries | 12 |
| Successful queries | 12 |
| Citation presence rate | 100.00% |
| Non-empty answer rate | 100.00% |
| Average latency (ms) | 6945.00 |
| Confidence high/medium/low/unknown | 12 / 0 / 0 / 0 |
| Cited source mix (PDF / News) | 44 / 6 |

## Interpretation
The current system reliably returns non-empty and cited responses across the fixed question set, indicating stable end-to-end behavior for retrieval, generation, citation rendering, and admin observability. The primary improvement target is latency reduction, as average response time remains high for interactive use.

## Remaining Risks and Follow-Up
- The current evaluation set is intentionally small and should be expanded for broader coverage.
- High confidence on all evaluation prompts suggests calibration may need tightening for harder/ambiguous queries.
- Follow-up detection is heuristic-based; future work can replace it with a lightweight classifier for better precision/recall trade-offs.
- Admin authentication remains intentionally out of scope for the current milestone.

## Benchmark Decision Log (New)
- Baseline and enhanced pipeline logic are now both retained behind runtime flags.
- Production defaults remain conservative until benchmark gates are met.
- Winner selection criteria:
  - No regression in grounding/citation quality.
  - Latency improvement (or acceptable tolerance).
  - Documented cost envelope for chosen embedding/reranker profile.

## Reranker Ablation Outcome (March 29, 2026)
Benchmark source:
- `data/eval/benchmarks/benchmark_20260329_141148/benchmark_report.json`
- `data/eval/benchmarks/benchmark_20260329_141148/benchmark_summary.md`

Stage A evaluated `off`, `cross_encoder`, and `llm` rerankers.

Average Stage A family score by reranker mode:
- `off`: `0.6155`
- `cross_encoder`: `0.4697`
- `llm`: `0.4190`

Best-configuration comparison (`enhanced_v2 + local_minilm` family):
- Best `off`: score `0.6485`, single-turn latency `2817.93 ms`, multi-turn latency `4212.62 ms`
- Best `llm`: score `0.5185`, single-turn latency `4399.99 ms`, multi-turn latency `5229.77 ms`
- Delta (`off - llm`): `+0.1300` score, `-1582.06 ms` single-turn, `-1017.15 ms` multi-turn
- LLM reranker overhead (best `llm` run): `67699` reranker input tokens

Decision:
- LLM reranking was kept during experimentation to validate the hypothesis.
- Based on the above ablation, it was removed from runtime options as redundant for current student-policy corpus coverage.
