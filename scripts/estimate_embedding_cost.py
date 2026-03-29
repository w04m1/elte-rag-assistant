#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from statistics import median
from typing import Any

import pypdfium2

PRICE_TABLE_USD_PER_1M = {
    "openai_small_standard": 0.02,
    "openai_small_batch": 0.01,
    "openai_large_standard": 0.13,
    "openai_large_batch": 0.065,
}


def _load_index_docstore(index_pkl_path: Path):
    with index_pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, tuple):
        return payload[0]
    return payload


def _source_page_count(path: Path) -> int:
    try:
        return len(pypdfium2.PdfDocument(str(path)))
    except Exception:
        return 0


def build_estimate(raw_dir: Path, index_pkl: Path) -> dict[str, Any]:
    docstore = _load_index_docstore(index_pkl)
    source_chars: dict[str, int] = {}
    for doc in docstore._dict.values():
        source = str((doc.metadata or {}).get("source", "")).strip()
        if not source:
            continue
        source_chars[source] = source_chars.get(source, 0) + len(str(doc.page_content or ""))

    tokens_per_page: list[float] = []
    for source, chars in source_chars.items():
        source_path = raw_dir / source
        if not source_path.exists():
            continue
        pages = _source_page_count(source_path)
        if pages <= 0:
            continue
        tokens_per_page.append((chars / 4) / pages)

    if not tokens_per_page:
        raise ValueError("Unable to estimate tokens-per-page from current index sources.")

    tokens_per_page.sort()
    p25 = tokens_per_page[int(round((len(tokens_per_page) - 1) * 0.25))]
    p50 = median(tokens_per_page)
    p75 = tokens_per_page[int(round((len(tokens_per_page) - 1) * 0.75))]

    missing_files = []
    missing_pages = 0
    for file_path in sorted(raw_dir.glob("*.pdf")):
        if file_path.name in source_chars:
            continue
        pages = _source_page_count(file_path)
        missing_pages += pages
        missing_files.append(file_path.name)

    scenarios = {
        "low": int(missing_pages * p25),
        "mid": int(missing_pages * p50),
        "high": int(missing_pages * p75),
    }

    cost_estimates: dict[str, dict[str, float]] = {}
    for scenario_name, tokens in scenarios.items():
        cost_estimates[scenario_name] = {
            "openai_small_standard_usd": round(
                tokens / 1_000_000 * PRICE_TABLE_USD_PER_1M["openai_small_standard"], 6
            ),
            "openai_small_batch_usd": round(
                tokens / 1_000_000 * PRICE_TABLE_USD_PER_1M["openai_small_batch"], 6
            ),
            "openai_large_standard_usd": round(
                tokens / 1_000_000 * PRICE_TABLE_USD_PER_1M["openai_large_standard"], 6
            ),
            "openai_large_batch_usd": round(
                tokens / 1_000_000 * PRICE_TABLE_USD_PER_1M["openai_large_batch"], 6
            ),
        }

    return {
        "raw_pdf_count": len(list(raw_dir.glob("*.pdf"))),
        "indexed_source_count": len(source_chars),
        "missing_pdf_count": len(missing_files),
        "missing_page_count": missing_pages,
        "tokens_per_page": {
            "p25": round(p25, 2),
            "p50": round(p50, 2),
            "p75": round(p75, 2),
        },
        "missing_embedding_tokens": scenarios,
        "cost_estimates_usd": cost_estimates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate one-time embedding cost for missing source PDFs.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--index-pkl", default="data/vector_store/index.pkl")
    parser.add_argument("--output-json", default="data/eval/embedding_cost_estimate.json")
    args = parser.parse_args()

    estimate = build_estimate(
        raw_dir=Path(args.raw_dir),
        index_pkl=Path(args.index_pkl),
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(estimate, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Missing PDFs: {estimate['missing_pdf_count']}")
    print(f"Missing pages: {estimate['missing_page_count']}")
    print(f"Estimated missing tokens (mid): {estimate['missing_embedding_tokens']['mid']}")
    print(
        "Estimated OpenAI large standard cost (mid): "
        f"${estimate['cost_estimates_usd']['mid']['openai_large_standard_usd']:.6f}"
    )
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
