#!/usr/bin/env python3
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

_ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def load_questions(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        questions = raw
    elif isinstance(raw, dict) and isinstance(raw.get("questions"), list):
        questions = raw["questions"]
    else:
        raise ValueError("Questions file must be a list or object with a 'questions' list.")

    normalized = [str(question).strip() for question in questions if str(question).strip()]
    if not normalized:
        raise ValueError("Questions list is empty.")
    return normalized


def normalize_confidence(value: Any) -> str:
    confidence = str(value or "").strip().lower()
    return confidence if confidence in _ALLOWED_CONFIDENCE else "unknown"


def count_source_types(cited_sources: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"pdf": 0, "news": 0}
    for source in cited_sources:
        source_type = str(source.get("source_type", "pdf")).strip().lower()
        if source_type == "news":
            counts["news"] += 1
        else:
            counts["pdf"] += 1
    return counts


def evaluate_questions(api_base_url: str, questions: list[str], timeout_seconds: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=api_base_url.rstrip("/"), timeout=timeout_seconds) as client:
        for question in questions:
            started_at = perf_counter()
            answer = ""
            model_used = ""
            confidence = "unknown"
            cited_sources: list[dict[str, Any]] = []
            status = "ok"
            error: str | None = None

            try:
                response = client.post("/ask", json={"query": question})
                response.raise_for_status()
                payload = response.json()
                answer = str(payload.get("answer", ""))
                model_used = str(payload.get("model_used", ""))
                confidence = normalize_confidence(payload.get("confidence"))
                raw_cited_sources = payload.get("cited_sources")
                if isinstance(raw_cited_sources, list):
                    cited_sources = [
                        source
                        for source in raw_cited_sources
                        if isinstance(source, dict)
                    ]
            except Exception as exc:
                status = "error"
                error = str(exc)

            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            source_types = count_source_types(cited_sources)
            results.append(
                {
                    "query": question,
                    "status": status,
                    "error": error,
                    "latency_ms": latency_ms,
                    "answer_length_chars": len(answer.strip()),
                    "confidence": confidence,
                    "model_used": model_used,
                    "cited_sources_count": len(cited_sources),
                    "source_types": source_types,
                }
            )
    return results


def build_metrics(results: list[dict[str, Any]], questions_path: str, api_base_url: str) -> dict[str, Any]:
    total_queries = len(results)
    successful_queries = sum(1 for row in results if row["status"] == "ok")
    non_empty_answer_count = sum(1 for row in results if row["answer_length_chars"] > 0)
    citation_presence_count = sum(1 for row in results if row["cited_sources_count"] > 0)

    latency_sum = sum(float(row["latency_ms"]) for row in results)
    avg_latency_ms = round(latency_sum / total_queries, 2) if total_queries else 0.0

    confidence_distribution = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    source_mix = {"pdf": 0, "news": 0}
    for row in results:
        confidence = normalize_confidence(row.get("confidence"))
        confidence_distribution[confidence] += 1
        source_mix["pdf"] += int(row["source_types"]["pdf"])
        source_mix["news"] += int(row["source_types"]["news"])

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "api_base_url": api_base_url,
        "questions_path": questions_path,
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "non_empty_answer_count": non_empty_answer_count,
        "citation_presence_count": citation_presence_count,
        "non_empty_answer_rate": round(non_empty_answer_count / total_queries, 4)
        if total_queries
        else 0.0,
        "citation_presence_rate": round(citation_presence_count / total_queries, 4)
        if total_queries
        else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "confidence_distribution": confidence_distribution,
        "source_mix_pdf_vs_news": source_mix,
        "results": results,
    }


def build_markdown(metrics: dict[str, Any]) -> str:
    confidence = metrics["confidence_distribution"]
    source_mix = metrics["source_mix_pdf_vs_news"]
    return "\n".join(
        [
            "# Evaluation Results",
            "",
            f"Generated at (UTC): {metrics['generated_at_utc']}",
            f"API base URL: `{metrics['api_base_url']}`",
            f"Question set: `{metrics['questions_path']}`",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Total queries | {metrics['total_queries']} |",
            f"| Successful queries | {metrics['successful_queries']} |",
            f"| Citation presence rate | {metrics['citation_presence_rate']:.2%} |",
            f"| Non-empty answer rate | {metrics['non_empty_answer_rate']:.2%} |",
            f"| Average latency (ms) | {metrics['avg_latency_ms']:.2f} |",
            "",
            "## Confidence Distribution",
            "",
            "| Confidence | Count |",
            "| --- | ---: |",
            f"| High | {confidence['high']} |",
            f"| Medium | {confidence['medium']} |",
            f"| Low | {confidence['low']} |",
            f"| Unknown | {confidence['unknown']} |",
            "",
            "## Source Mix (Cited Sources)",
            "",
            "| Source Type | Count |",
            "| --- | ---: |",
            f"| PDF | {source_mix['pdf']} |",
            f"| News | {source_mix['news']} |",
            "",
            "## Notes",
            "",
            "- Rates are computed over the full question set.",
            "- Failed requests are included in denominator for stability and accountability.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ELTE RAG assistant evaluation against a fixed question set.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001", help="Base URL of the running API")
    parser.add_argument("--questions", default="data/eval/questions.json", help="Path to evaluation questions JSON")
    parser.add_argument("--output-json", default="data/eval/latest_metrics.json", help="Metrics output JSON path")
    parser.add_argument("--output-md", default="docs/thesis/evaluation.md", help="Markdown report output path")
    parser.add_argument("--timeout-seconds", type=float, default=45.0, help="Per-request timeout")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_json_path = Path(args.output_json)
    output_md_path = Path(args.output_md)

    questions = load_questions(questions_path)
    results = evaluate_questions(args.api_base_url, questions, args.timeout_seconds)
    metrics = build_metrics(results, str(questions_path), args.api_base_url)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(build_markdown(metrics), encoding="utf-8")

    print(f"Evaluation complete: {metrics['successful_queries']}/{metrics['total_queries']} successful")
    print(f"Metrics JSON: {output_json_path}")
    print(f"Metrics Markdown: {output_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
