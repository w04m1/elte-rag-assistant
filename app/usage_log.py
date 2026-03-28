import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_USAGE_LOG_WRITE_LOCK = threading.Lock()
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}
_ALLOWED_STATUS = {"ok", "error"}


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if parsed >= 0 else default
    except (TypeError, ValueError):
        return default


def _normalize_source_types(payload: Any) -> dict[str, int]:
    source_types = payload if isinstance(payload, dict) else {}
    return {
        "pdf": _coerce_non_negative_int(source_types.get("pdf"), default=0),
        "news": _coerce_non_negative_int(source_types.get("news"), default=0),
    }


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_usage_entry(payload: dict[str, Any]) -> dict[str, Any]:
    raw_confidence = str(payload.get("confidence", "")).strip().lower()
    confidence = raw_confidence if raw_confidence in _ALLOWED_CONFIDENCE else "unknown"

    raw_status = str(payload.get("status", "")).strip().lower()
    status = raw_status if raw_status in _ALLOWED_STATUS else "error"

    timestamp = str(payload.get("timestamp_utc", "")).strip()
    if not timestamp:
        timestamp = datetime.now(UTC).isoformat()

    feedback = _coerce_optional_bool(payload.get("feedback"))
    feedback_timestamp = str(payload.get("feedback_timestamp_utc", "")).strip() or None
    if feedback is None:
        feedback_timestamp = None

    return {
        "request_id": str(payload.get("request_id", "")).strip(),
        "timestamp_utc": timestamp,
        "query_text": str(payload.get("query_text", "")).strip(),
        "answer_length_chars": _coerce_non_negative_int(
            payload.get("answer_length_chars"), default=0
        ),
        "confidence": confidence,
        "model_used": str(payload.get("model_used", "")).strip(),
        "reranker_model": str(payload.get("reranker_model", "")).strip(),
        "latency_ms": round(
            _coerce_non_negative_float(payload.get("latency_ms"), default=0.0), 2
        ),
        "cited_sources_count": _coerce_non_negative_int(
            payload.get("cited_sources_count"), default=0
        ),
        "source_types": _normalize_source_types(payload.get("source_types")),
        "feedback": feedback,
        "feedback_timestamp_utc": feedback_timestamp,
        "status": status,
    }


def append_usage_entry(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_payload = normalize_usage_entry(payload)
    line = json.dumps(normalized_payload, ensure_ascii=True)
    with _USAGE_LOG_WRITE_LOCK:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def set_usage_feedback(path: str | Path, request_id: str, helpful: bool) -> bool:
    normalized_request_id = request_id.strip()
    if not normalized_request_id:
        return False

    source = Path(path)
    if not source.exists():
        return False

    with _USAGE_LOG_WRITE_LOCK:
        lines = source.read_text(encoding="utf-8").splitlines()
        updated = False
        now_timestamp = datetime.now(UTC).isoformat()

        for index in range(len(lines) - 1, -1, -1):
            raw_line = lines[index].strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            candidate_id = str(payload.get("request_id", "")).strip()
            if candidate_id != normalized_request_id:
                continue

            payload["feedback"] = bool(helpful)
            payload["feedback_timestamp_utc"] = now_timestamp
            lines[index] = json.dumps(normalize_usage_entry(payload), ensure_ascii=True)
            updated = True
            break

        if updated:
            source.write_text(
                ("\n".join(lines) + "\n") if lines else "",
                encoding="utf-8",
            )

    return updated


def read_recent_usage_entries(path: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    source = Path(path)
    if not source.exists():
        return []

    entries: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for line in reversed(lines):
        raw_line = line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        entries.append(normalize_usage_entry(payload))
        if len(entries) >= limit:
            break

    return entries


def compute_usage_stats(path: str | Path, window_days: int = 7) -> dict[str, Any]:
    source = Path(path)
    bounded_window_days = max(1, int(window_days))
    now_utc = datetime.now(UTC)
    cutoff = now_utc - timedelta(days=bounded_window_days)

    confidence_distribution = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    total_queries = 0
    non_empty_answer_count = 0
    citation_present_count = 0
    latency_sum = 0.0
    source_mix = {"pdf": 0, "news": 0}
    helpful_feedback_count = 0
    unhelpful_feedback_count = 0

    if source.exists():
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                entry = normalize_usage_entry(payload)
                entry_timestamp = _parse_timestamp(entry["timestamp_utc"])
                if entry_timestamp is None or entry_timestamp < cutoff:
                    continue

                total_queries += 1
                latency_sum += _coerce_non_negative_float(entry.get("latency_ms"), 0.0)
                if _coerce_non_negative_int(entry.get("answer_length_chars"), 0) > 0:
                    non_empty_answer_count += 1
                if _coerce_non_negative_int(entry.get("cited_sources_count"), 0) > 0:
                    citation_present_count += 1

                confidence = str(entry.get("confidence", "unknown"))
                if confidence not in confidence_distribution:
                    confidence = "unknown"
                confidence_distribution[confidence] += 1

                source_types = _normalize_source_types(entry.get("source_types"))
                source_mix["pdf"] += source_types["pdf"]
                source_mix["news"] += source_types["news"]

                feedback = _coerce_optional_bool(entry.get("feedback"))
                if feedback is True:
                    helpful_feedback_count += 1
                elif feedback is False:
                    unhelpful_feedback_count += 1

    avg_latency_ms = round(latency_sum / total_queries, 2) if total_queries else 0.0
    citation_presence_rate = (
        round(citation_present_count / total_queries, 4) if total_queries else 0.0
    )
    non_empty_answer_rate = (
        round(non_empty_answer_count / total_queries, 4) if total_queries else 0.0
    )
    feedback_total = helpful_feedback_count + unhelpful_feedback_count
    helpful_feedback_rate = (
        round(helpful_feedback_count / feedback_total, 4) if feedback_total else 0.0
    )
    feedback_coverage_rate = (
        round(feedback_total / total_queries, 4) if total_queries else 0.0
    )

    return {
        "window_days": bounded_window_days,
        "generated_at_utc": now_utc.isoformat(),
        "total_queries": total_queries,
        "avg_latency_ms": avg_latency_ms,
        "citation_presence_rate": citation_presence_rate,
        "non_empty_answer_rate": non_empty_answer_rate,
        "confidence_distribution": confidence_distribution,
        "source_mix_pdf_vs_news": source_mix,
        "helpful_feedback_count": helpful_feedback_count,
        "unhelpful_feedback_count": unhelpful_feedback_count,
        "helpful_feedback_rate": helpful_feedback_rate,
        "feedback_coverage_rate": feedback_coverage_rate,
    }
