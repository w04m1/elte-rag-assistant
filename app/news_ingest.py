import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

DEFAULT_TYPESENSE_URL = "https://typesense.elte.hu/multi_search"
DEFAULT_TYPESENSE_QUERY_BY = (
    "title,summary_text,processed_text,stam_1_person_job,"
    "stam_2_title_org_unit,stam_3_person_field_of_science,stam_4_research_keywords"
)
DEFAULT_TYPESENSE_QUERY_BY_WEIGHTS = "3,2,1,2,2,1,1"
DEFAULT_TYPESENSE_FILTER = (
    "langcode:=en && protected_page:!=true && hide_from_search:!=true "
    "&& content_type:!=[student,programs] && login_required_to_access:!=true "
    "&& (faculties:=[Faculty of Informatics] || is_global:=true)"
)


class NewsSyncResult(dict):
    pass


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _html_to_text(value: str) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "html.parser")
    text = " ".join(chunk.strip() for chunk in soup.stripped_strings if chunk.strip())
    return _normalize_whitespace(text)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _news_url(document: dict[str, Any]) -> str | None:
    absolute = str(document.get("entity_url_domain") or "").strip()
    if absolute.startswith("http://") or absolute.startswith("https://"):
        return absolute

    relative = str(document.get("entity_url") or "").strip()
    if relative:
        return urljoin("https://www.inf.elte.hu", relative)

    return None


def _stable_news_id(document: dict[str, Any], url: str) -> str:
    raw_id = str(document.get("id") or "").strip()
    if raw_id:
        return raw_id

    raw_nid = document.get("nid")
    if isinstance(raw_nid, int):
        return f"nid:{raw_nid}"

    return f"url:{url}"


def _is_tagged_news(document: dict[str, Any]) -> bool:
    return bool(document.get("news_tag") or document.get("source_news_tag"))


def parse_typesense_hit(hit: dict[str, Any], *, scraped_at: datetime | None = None) -> dict[str, Any] | None:
    document = hit.get("document")
    if not isinstance(document, dict):
        return None

    if not _is_tagged_news(document):
        return None

    url = _news_url(document)
    if not url:
        return None

    title = str(document.get("title") or "").strip() or url
    summary = _normalize_whitespace(str(document.get("summary_text") or "").strip())
    body = _html_to_text(str(document.get("processed_text") or ""))
    if not body:
        body = summary
    if not body:
        return None

    created = _coerce_int(document.get("created"))
    published_at = (
        datetime.fromtimestamp(created, tz=UTC).isoformat() if created is not None else None
    )

    content_hash_source = f"{title}\n{summary}\n{body}"
    content_hash = hashlib.sha256(content_hash_source.encode("utf-8")).hexdigest()

    return {
        "id": _stable_news_id(document, url),
        "url": url,
        "title": title,
        "summary": summary,
        "body": body,
        "content_type": str(document.get("content_type") or "").strip() or "news",
        "created": created,
        "published_at": published_at,
        "content_hash": content_hash,
        "scraped_at": (scraped_at or datetime.now(UTC)).isoformat(),
    }


def _typesense_payload(page: int) -> dict[str, Any]:
    return {
        "searches": [
            {
                "query_by": DEFAULT_TYPESENSE_QUERY_BY,
                "query_by_weights": DEFAULT_TYPESENSE_QUERY_BY_WEIGHTS,
                "sort_by": "_text_match:desc,created:desc",
                "exclude_fields": "embedding",
                "exhaustive_search": True,
                "filter_by": DEFAULT_TYPESENSE_FILTER,
                "highlight_full_fields": DEFAULT_TYPESENSE_QUERY_BY,
                "collection": "content",
                "q": "*",
                "page": page,
            }
        ]
    }


def _typesense_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.inf.elte.hu",
        "Referer": "https://www.inf.elte.hu/",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        ),
    }


def fetch_typesense_hits(
    *,
    page: int,
    client: httpx.Client,
    endpoint: str,
    api_key: str,
) -> list[dict[str, Any]]:
    response = client.post(
        endpoint,
        params={"x-typesense-api-key": api_key},
        json=_typesense_payload(page),
        headers=_typesense_headers(),
    )
    response.raise_for_status()
    payload = response.json()

    try:
        hits = payload["results"][0]["hits"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected Typesense response shape") from exc

    if not isinstance(hits, list):
        raise ValueError("Unexpected Typesense response shape: hits is not a list")

    return [item for item in hits if isinstance(item, dict)]


def _load_news_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict):
        payload["items"] = {}
    return payload


def _write_news_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _file_name_for_news(news_id: str, fallback_url: str) -> str:
    base = news_id or fallback_url
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.json"


def _write_news_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sync_news_records(
    *,
    pages: int,
    records_dir: str | Path | None = None,
    state_path: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> NewsSyncResult:
    if pages <= 0:
        raise ValueError("pages must be greater than 0")

    endpoint = endpoint or settings.news_typesense_url or DEFAULT_TYPESENSE_URL
    api_key = (api_key or settings.news_typesense_api_key).strip()
    if not api_key:
        raise ValueError("NEWS_TYPESENSE_API_KEY is required for news sync")

    records_root = Path(records_dir or settings.news_records_path)
    records_root.mkdir(parents=True, exist_ok=True)
    state_file = Path(state_path or settings.news_state_path)
    state = _load_news_state(state_file)
    state_items: dict[str, Any] = state.setdefault("items", {})

    own_client = client is None
    client = client or httpx.Client(
        follow_redirects=True,
        timeout=settings.news_request_timeout_seconds,
    )

    now = datetime.now(UTC)
    candidates: dict[str, dict[str, Any]] = {}

    try:
        for page in range(1, pages + 1):
            hits = fetch_typesense_hits(
                page=page,
                client=client,
                endpoint=endpoint,
                api_key=api_key,
            )
            for hit in hits:
                parsed = parse_typesense_hit(hit, scraped_at=now)
                if parsed is None:
                    continue

                existing = candidates.get(parsed["id"])
                if existing is None:
                    candidates[parsed["id"]] = parsed
                    continue

                existing_created = _coerce_int(existing.get("created")) or -1
                parsed_created = _coerce_int(parsed.get("created")) or -1
                if parsed_created >= existing_created:
                    candidates[parsed["id"]] = parsed

        added_count = 0
        updated_count = 0
        unchanged_count = 0

        for news_id, record in sorted(candidates.items()):
            previous = state_items.get(news_id)
            previous_hash = None
            if isinstance(previous, dict):
                previous_hash = previous.get("content_hash")

            file_name = (
                previous.get("file_name")
                if isinstance(previous, dict) and previous.get("file_name")
                else _file_name_for_news(news_id, record["url"])
            )
            file_path = records_root / file_name

            if previous is None:
                added_count += 1
                _write_news_record(file_path, record)
            elif previous_hash != record["content_hash"]:
                updated_count += 1
                _write_news_record(file_path, record)
            else:
                unchanged_count += 1

            state_items[news_id] = {
                "id": news_id,
                "url": record["url"],
                "title": record["title"],
                "published_at": record.get("published_at"),
                "file_name": file_name,
                "content_hash": record["content_hash"],
                "updated_at": now.isoformat(),
            }

        state["last_sync_at"] = now.isoformat()
        _write_news_state(state_file, state)

        return NewsSyncResult(
            processed_count=len(candidates),
            added_count=added_count,
            updated_count=updated_count,
            unchanged_count=unchanged_count,
        )
    finally:
        if own_client:
            client.close()


def load_news_documents(news_dir: str | Path | None = None) -> list[Document]:
    records_root = Path(news_dir or settings.news_records_path)
    if not records_root.exists():
        return []

    documents: list[Document] = []
    for news_path in sorted(records_root.glob("*.json")):
        try:
            payload = json.loads(news_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping invalid news record %s: %s", news_path.name, exc)
            continue

        title = str(payload.get("title") or "").strip() or news_path.stem
        url = str(payload.get("url") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        body = str(payload.get("body") or "").strip()
        content = body or summary
        if not content:
            continue

        if summary and summary != body:
            page_content = f"{title}\n\n{summary}\n\n{body}"
        else:
            page_content = f"{title}\n\n{content}"

        if url:
            page_content = f"Source URL: {url}\n\n{page_content}"

        metadata: dict[str, Any] = {
            "source": url or str(payload.get("id") or news_path.name),
            "title": title,
            "type": "news",
            "source_type": "news",
        }
        published_at = payload.get("published_at")
        if published_at:
            metadata["published_at"] = str(published_at)

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def rebuild_news_vector_store(
    *,
    news_dir: str | Path | None = None,
    index_path: str | Path | None = None,
) -> int:
    documents = load_news_documents(news_dir)
    if not documents:
        logger.info("No news documents to index.")
        return 0

    embeddings = get_embeddings()
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(index_path or settings.news_faiss_index_path))
    return len(documents)


def run_news_pipeline(
    *,
    mode: Literal["bootstrap", "sync"],
    pages: int | None = None,
    records_dir: str | Path | None = None,
    state_path: str | Path | None = None,
    index_path: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> NewsSyncResult:
    if mode not in {"bootstrap", "sync"}:
        raise ValueError(f"Unknown news pipeline mode: {mode}")

    if pages is None:
        pages = (
            settings.news_bootstrap_pages
            if mode == "bootstrap"
            else settings.news_sync_pages
        )

    result = sync_news_records(
        pages=pages,
        records_dir=records_dir,
        state_path=state_path,
        endpoint=endpoint,
        api_key=api_key,
        client=client,
    )
    embedded_count = rebuild_news_vector_store(
        news_dir=records_dir,
        index_path=index_path,
    )

    result.update(
        {
            "mode": mode,
            "pages": pages,
            "embedded_count": embedded_count,
        }
    )
    return result
