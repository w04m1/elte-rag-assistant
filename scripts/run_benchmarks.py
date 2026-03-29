#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import httpx

from estimate_embedding_cost import build_estimate


EMBEDDING_PRICE_USD_PER_1M = {
    "openai_small": 0.02,
    "openai_large": 0.13,
}


@dataclass
class BenchmarkConfig:
    stage: str
    embedding_profile: str
    pipeline_mode: str
    reranker_mode: str

    @property
    def family_id(self) -> str:
        return f"{self.pipeline_mode}__{self.reranker_mode}"

    @property
    def run_id(self) -> str:
        return (
            f"{self.stage}__{self.embedding_profile}__"
            f"{self.pipeline_mode}__{self.reranker_mode}"
        )


def _load_single_turn_questions(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = raw.get("questions") if isinstance(raw, dict) else raw
    if not isinstance(questions, list):
        raise ValueError("Single-turn questions must be a JSON list or {'questions': [...]} payload.")
    return [str(q).strip() for q in questions if str(q).strip()]


def _load_multi_turn_scenarios(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = raw.get("scenarios") if isinstance(raw, dict) else raw
    if not isinstance(scenarios, list):
        raise ValueError("Multi-turn scenarios must be a JSON list or {'scenarios': [...]} payload.")
    normalized: list[dict[str, Any]] = []
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        history = item.get("history")
        if not isinstance(history, list):
            history = []
        normalized.append(
            {
                "id": str(item.get("id", f"scenario-{len(normalized)+1}")),
                "query": query,
                "history": history,
            }
        )
    return normalized


def _post_with_latency(
    client: httpx.Client,
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    status = "ok"
    error: str | None = None
    answer = ""
    confidence = "unknown"
    cited_sources: list[dict[str, Any]] = []

    payload: dict[str, Any] = {"query": query}
    if history:
        payload["history"] = history

    try:
        response = client.post("/ask", json=payload)
        response.raise_for_status()
        body = response.json()
        answer = str(body.get("answer", ""))
        confidence = str(body.get("confidence", "unknown")).strip().lower() or "unknown"
        raw_sources = body.get("cited_sources")
        if isinstance(raw_sources, list):
            cited_sources = [src for src in raw_sources if isinstance(src, dict)]
    except Exception as exc:
        status = "error"
        error = str(exc)

    latency_ms = round((perf_counter() - started) * 1000, 2)
    return {
        "status": status,
        "error": error,
        "query": query,
        "latency_ms": latency_ms,
        "answer_length_chars": len(answer.strip()),
        "confidence": confidence,
        "cited_sources_count": len(cited_sources),
    }


def _summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success = sum(1 for r in rows if r["status"] == "ok")
    non_empty = sum(1 for r in rows if r["answer_length_chars"] > 0)
    citation = sum(1 for r in rows if r["cited_sources_count"] > 0)
    latencies = [float(r["latency_ms"]) for r in rows]
    avg_latency = round(sum(latencies) / total, 2) if total else 0.0
    sorted_latencies = sorted(latencies)
    p95 = (
        round(sorted_latencies[int(round((len(sorted_latencies) - 1) * 0.95))], 2)
        if sorted_latencies
        else 0.0
    )
    score = (
        ((citation / total) + (non_empty / total)) / 2 if total else 0.0
    )
    return {
        "total": total,
        "success": success,
        "non_empty_answer_rate": round(non_empty / total, 4) if total else 0.0,
        "citation_presence_rate": round(citation / total, 4) if total else 0.0,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95,
        "grounding_score": round(score, 4),
    }


def _family_score(run_metrics: dict[str, Any]) -> float:
    single = run_metrics["single_turn"]["summary"]
    multi = run_metrics["multi_turn"]["summary"]
    quality = (single["grounding_score"] + multi["grounding_score"]) / 2
    latency_penalty = (single["avg_latency_ms"] + multi["avg_latency_ms"]) / 2 / 10_000
    return quality - latency_penalty


def _set_runtime_config(client: httpx.Client, config: BenchmarkConfig) -> dict[str, Any]:
    response = client.put(
        "/admin/settings",
        json={
            "embedding_profile": config.embedding_profile,
            "pipeline_mode": config.pipeline_mode,
            "reranker_mode": config.reranker_mode,
        },
    )
    response.raise_for_status()
    return response.json()


def _has_active_snapshot_for_profile(client: httpx.Client) -> bool:
    try:
        response = client.get("/admin/indexes/active")
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False
    return bool(payload.get("from_snapshot"))


def _trigger_and_wait_reindex(client: httpx.Client, timeout_seconds: int = 3600) -> dict[str, Any]:
    start = perf_counter()
    response = client.post("/admin/reindex")
    response.raise_for_status()
    while True:
        status_resp = client.get("/admin/reindex")
        status_resp.raise_for_status()
        payload = status_resp.json()
        status = str(payload.get("status", "")).strip().lower()
        if status in {"completed", "failed"}:
            return payload
        if (perf_counter() - start) > timeout_seconds:
            raise TimeoutError("Timed out while waiting for reindex job to finish.")
        sleep(2)


def _estimate_run_cost(
    *,
    config: BenchmarkConfig,
    rows: list[dict[str, Any]],
    retrieval_k: int,
    retrieval_fetch_k: int,
    generator_input_price_per_1m: float,
    generator_output_price_per_1m: float,
    reranker_input_price_per_1m: float,
    embedding_tokens_one_time: int,
) -> dict[str, Any]:
    approx_context_chars = retrieval_k * 768
    generator_input_tokens_est = 0
    generator_output_tokens_est = 0
    reranker_input_tokens_est = 0

    for row in rows:
        query_tokens = len(str(row.get("query", ""))) // 4
        generator_input_tokens_est += query_tokens + (approx_context_chars // 4)
        generator_output_tokens_est += int(row.get("answer_length_chars", 0)) // 4
        if config.reranker_mode == "llm":
            reranker_input_tokens_est += (
                retrieval_fetch_k * (500 // 4) + query_tokens
            )

    embedding_usd = 0.0
    embedding_price = EMBEDDING_PRICE_USD_PER_1M.get(config.embedding_profile)
    if embedding_price is not None:
        embedding_usd = (embedding_tokens_one_time / 1_000_000) * embedding_price

    generator_input_usd = (
        generator_input_tokens_est / 1_000_000 * generator_input_price_per_1m
    )
    generator_output_usd = (
        generator_output_tokens_est / 1_000_000 * generator_output_price_per_1m
    )
    reranker_input_usd = (
        reranker_input_tokens_est / 1_000_000 * reranker_input_price_per_1m
    )

    return {
        "embedding_tokens_one_time": embedding_tokens_one_time,
        "generator_input_tokens_est": generator_input_tokens_est,
        "generator_output_tokens_est": generator_output_tokens_est,
        "reranker_input_tokens_est": reranker_input_tokens_est,
        "embedding_usd": round(embedding_usd, 6),
        "generator_input_usd": round(generator_input_usd, 6),
        "generator_output_usd": round(generator_output_usd, 6),
        "reranker_input_usd": round(reranker_input_usd, 6),
        "total_estimated_usd": round(
            embedding_usd + generator_input_usd + generator_output_usd + reranker_input_usd,
            6,
        ),
        "estimate_method": "length-based token approximation (chars/4) + configured price table",
    }


def _evaluate_config(
    *,
    client: httpx.Client,
    config: BenchmarkConfig,
    single_turn_questions: list[str],
    multi_turn_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    single_rows = [_post_with_latency(client, query=question) for question in single_turn_questions]
    multi_rows = [
        _post_with_latency(
            client,
            query=scenario["query"],
            history=scenario.get("history"),
        )
        for scenario in multi_turn_scenarios
    ]
    return {
        "config": asdict(config),
        "single_turn": {
            "summary": _summarize_results(single_rows),
            "rows": single_rows,
        },
        "multi_turn": {
            "summary": _summarize_results(multi_rows),
            "rows": multi_rows,
        },
    }


def _build_stage_a_configs(plan: dict[str, Any]) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for embedding in plan["anchors"]:
        for pipeline_mode in plan["pipeline_modes"]:
            for reranker_mode in plan["reranker_modes"]:
                configs.append(
                    BenchmarkConfig(
                        stage="stage_a",
                        embedding_profile=embedding,
                        pipeline_mode=pipeline_mode,
                        reranker_mode=reranker_mode,
                    )
                )
    return configs


def _build_stage_b_configs(
    *,
    plan: dict[str, Any],
    winning_families: list[str],
) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for family in winning_families:
        pipeline_mode, reranker_mode = family.split("__", 1)
        for embedding in plan["embedding_profiles"]:
            configs.append(
                BenchmarkConfig(
                    stage="stage_b",
                    embedding_profile=embedding,
                    pipeline_mode=pipeline_mode,
                    reranker_mode=reranker_mode,
                )
            )
    return configs


def run() -> int:
    parser = argparse.ArgumentParser(description="Run staged benchmark matrix for ELTE RAG pipeline.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--plan", default="data/eval/benchmark_plan.json")
    parser.add_argument("--single-turn", default="data/eval/questions.json")
    parser.add_argument("--multi-turn", default="data/eval/multi_turn_questions.json")
    parser.add_argument("--output-dir", default="data/eval/benchmarks")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--index-pkl", default="data/vector_store/index.pkl")
    parser.add_argument("--generator-input-price-per-1m", type=float, default=0.0)
    parser.add_argument("--generator-output-price-per-1m", type=float, default=0.0)
    parser.add_argument("--reranker-input-price-per-1m", type=float, default=0.0)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    single_turn_questions = _load_single_turn_questions(Path(args.single_turn))
    multi_turn_scenarios = _load_multi_turn_scenarios(Path(args.multi_turn))
    embedding_estimate = build_estimate(Path(args.raw_dir), Path(args.index_pkl))

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_a_configs = _build_stage_a_configs(plan)
    stage_a_results: list[dict[str, Any]] = []
    stage_b_results: list[dict[str, Any]] = []
    profiles_reindexed: set[str] = set()

    with httpx.Client(base_url=args.api_base_url.rstrip("/"), timeout=120.0) as client:
        retrieval_k = 5
        retrieval_fetch_k = 30

        for config in stage_a_configs:
            _set_runtime_config(client, config)
            if config.embedding_profile not in profiles_reindexed:
                if _has_active_snapshot_for_profile(client):
                    reindex_status = {"status": "completed", "skipped": "active_snapshot_reused"}
                else:
                    reindex_status = _trigger_and_wait_reindex(client)
                    if str(reindex_status.get("status")) != "completed":
                        raise RuntimeError(
                            f"Reindex failed for profile {config.embedding_profile}: {reindex_status}"
                        )
                profiles_reindexed.add(config.embedding_profile)

            run_metrics = _evaluate_config(
                client=client,
                config=config,
                single_turn_questions=single_turn_questions,
                multi_turn_scenarios=multi_turn_scenarios,
            )
            one_time_embedding_tokens = embedding_estimate["missing_embedding_tokens"]["mid"]
            run_metrics["cost"] = _estimate_run_cost(
                config=config,
                rows=run_metrics["single_turn"]["rows"] + run_metrics["multi_turn"]["rows"],
                retrieval_k=retrieval_k,
                retrieval_fetch_k=retrieval_fetch_k,
                generator_input_price_per_1m=args.generator_input_price_per_1m,
                generator_output_price_per_1m=args.generator_output_price_per_1m,
                reranker_input_price_per_1m=args.reranker_input_price_per_1m,
                embedding_tokens_one_time=one_time_embedding_tokens,
            )
            run_metrics["family_score"] = _family_score(run_metrics)
            stage_a_results.append(run_metrics)

        family_scores: dict[str, float] = {}
        for run_metrics in stage_a_results:
            family = run_metrics["config"]["pipeline_mode"] + "__" + run_metrics["config"]["reranker_mode"]
            family_scores[family] = max(
                family_scores.get(family, float("-inf")),
                float(run_metrics["family_score"]),
            )
        winning_families = [
            family for family, _score in sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        ][: int(plan.get("stage_b_top_families", 2))]

        stage_b_configs = _build_stage_b_configs(plan=plan, winning_families=winning_families)
        stage_b_repeats = int(plan.get("stage_b_repeats", 3))

        for config in stage_b_configs:
            _set_runtime_config(client, config)
            if config.embedding_profile not in profiles_reindexed:
                if _has_active_snapshot_for_profile(client):
                    reindex_status = {"status": "completed", "skipped": "active_snapshot_reused"}
                else:
                    reindex_status = _trigger_and_wait_reindex(client)
                    if str(reindex_status.get("status")) != "completed":
                        raise RuntimeError(
                            f"Reindex failed for profile {config.embedding_profile}: {reindex_status}"
                        )
                profiles_reindexed.add(config.embedding_profile)

            repeat_runs: list[dict[str, Any]] = []
            for repeat_index in range(stage_b_repeats):
                repeat_metrics = _evaluate_config(
                    client=client,
                    config=config,
                    single_turn_questions=single_turn_questions,
                    multi_turn_scenarios=multi_turn_scenarios,
                )
                repeat_metrics["repeat_index"] = repeat_index + 1
                repeat_runs.append(repeat_metrics)

            avg_single_latency = sum(
                r["single_turn"]["summary"]["avg_latency_ms"] for r in repeat_runs
            ) / len(repeat_runs)
            avg_multi_latency = sum(
                r["multi_turn"]["summary"]["avg_latency_ms"] for r in repeat_runs
            ) / len(repeat_runs)
            aggregated = {
                "config": asdict(config),
                "repeats": stage_b_repeats,
                "avg_single_turn_latency_ms": round(avg_single_latency, 2),
                "avg_multi_turn_latency_ms": round(avg_multi_latency, 2),
                "single_turn_grounding_score_avg": round(
                    sum(r["single_turn"]["summary"]["grounding_score"] for r in repeat_runs)
                    / len(repeat_runs),
                    4,
                ),
                "multi_turn_grounding_score_avg": round(
                    sum(r["multi_turn"]["summary"]["grounding_score"] for r in repeat_runs)
                    / len(repeat_runs),
                    4,
                ),
                "repeat_runs": repeat_runs,
            }
            one_time_embedding_tokens = embedding_estimate["missing_embedding_tokens"]["mid"]
            aggregated["cost"] = _estimate_run_cost(
                config=config,
                rows=[
                    row
                    for repeat in repeat_runs
                    for row in (repeat["single_turn"]["rows"] + repeat["multi_turn"]["rows"])
                ],
                retrieval_k=retrieval_k,
                retrieval_fetch_k=retrieval_fetch_k,
                generator_input_price_per_1m=args.generator_input_price_per_1m,
                generator_output_price_per_1m=args.generator_output_price_per_1m,
                reranker_input_price_per_1m=args.reranker_input_price_per_1m,
                embedding_tokens_one_time=one_time_embedding_tokens,
            )
            stage_b_results.append(aggregated)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "api_base_url": args.api_base_url,
        "plan_path": args.plan,
        "single_turn_path": args.single_turn,
        "multi_turn_path": args.multi_turn,
        "pre_embedding_estimate": embedding_estimate,
        "stage_a_runs": stage_a_results,
        "stage_b_runs": stage_b_results,
    }

    report_path = run_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_lines = [
        "# Benchmark Summary",
        "",
        f"Generated at (UTC): {report['generated_at_utc']}",
        f"Stage A runs: {len(stage_a_results)}",
        f"Stage B runs: {len(stage_b_results)}",
        "",
        "## Stage A Top Families",
    ]
    stage_a_sorted = sorted(stage_a_results, key=lambda row: row["family_score"], reverse=True)
    for row in stage_a_sorted[:5]:
        cfg = row["config"]
        summary_lines.append(
            "- "
            f"{cfg['pipeline_mode']} + {cfg['reranker_mode']} + {cfg['embedding_profile']}: "
            f"score={row['family_score']:.4f}, "
            f"single_latency={row['single_turn']['summary']['avg_latency_ms']:.2f}ms, "
            f"multi_latency={row['multi_turn']['summary']['avg_latency_ms']:.2f}ms"
        )
    summary_lines.extend(
        [
            "",
            "## Stage B Configs",
        ]
    )
    for row in stage_b_results:
        cfg = row["config"]
        summary_lines.append(
            "- "
            f"{cfg['pipeline_mode']} + {cfg['reranker_mode']} + {cfg['embedding_profile']}: "
            f"single_latency={row['avg_single_turn_latency_ms']:.2f}ms, "
            f"multi_latency={row['avg_multi_turn_latency_ms']:.2f}ms"
        )

    summary_path = run_dir / "benchmark_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote benchmark report: {report_path}")
    print(f"Wrote benchmark summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
