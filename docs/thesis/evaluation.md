# Evaluation Results

Generated at (UTC): 2026-03-28T14:43:51.054044+00:00
API base URL: `http://127.0.0.1:8001`
Question set: `data/eval/questions.json`

## Summary Metrics

| Metric | Value |
| --- | ---: |
| Total queries | 12 |
| Successful queries | 12 |
| Citation presence rate | 100.00% |
| Non-empty answer rate | 100.00% |
| Average latency (ms) | 6945.00 |

## Confidence Distribution

| Confidence | Count |
| --- | ---: |
| High | 12 |
| Medium | 0 |
| Low | 0 |
| Unknown | 0 |

## Source Mix (Cited Sources)

| Source Type | Count |
| --- | ---: |
| PDF | 44 |
| News | 6 |

## Notes

- Rates are computed over the full question set.
- Failed requests are included in denominator for stability and accountability.
- News/documents refresh is manual-triggered in admin workflows (no periodic background polling).

## Benchmark Matrix Protocol

For controlled rollout experiments, run:

```bash
uv run python scripts/run_benchmarks.py --api-base-url http://127.0.0.1:8001
```

Benchmark outputs are written to timestamped folders under:
- `data/eval/benchmarks/benchmark_<timestamp>/benchmark_report.json`
- `data/eval/benchmarks/benchmark_<timestamp>/benchmark_summary.md`

Each benchmark report includes:
- Stage A and Stage B run matrices
- Single-turn and multi-turn summary metrics
- Cost estimate payload (embedding + token-based inference estimates)

## Trial-and-Error Reranker Trace

On March 29, 2026, Stage A benchmark runs in:
- `data/eval/benchmarks/benchmark_20260329_141148/benchmark_report.json`

included three reranker strategies (`off`, `cross_encoder`, `llm`) to explicitly test whether LLM reranking improves grounded quality for student-policy chat queries.

Observed outcome (family-score aggregate):
- `off`: `0.6155`
- `cross_encoder`: `0.4697`
- `llm`: `0.4190`

Best `off` configuration outperformed best `llm` by `+0.1300` quality score while also reducing latency (`-1582.06 ms` single-turn, `-1017.15 ms` multi-turn).  
Therefore, LLM reranking was retained as experimental history but removed from runtime options in production defaults.
