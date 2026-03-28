import hashlib
import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings

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
DEFAULT_INCLUDE_FIELDS = (
    "id,title,created,entity_url,entity_url_domain,content_type,processed_text"
)

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}
MAX_PER_PAGE = 250


class DocumentSyncResult(dict):
    pass


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


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _normalize_url(url: str, *, base_url: str | None = None) -> str | None:
    value = str(url or "").strip()
    if not value:
        return None

    if base_url:
        value = urljoin(base_url, value)

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.hostname:
        return None

    netloc = parsed.hostname.lower()
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    # Keep query parameters, but canonicalize by stripping fragments.
    return urlunparse((parsed.scheme.lower(), netloc, parsed.path, "", parsed.query, ""))


def _is_allowed_host(host: str | None) -> bool:
    value = (host or "").strip().lower()
    return value == "elte.hu" or value.endswith(".elte.hu")


def _is_allowed_url(url: str) -> bool:
    return _is_allowed_host(urlparse(url).hostname)


def _extract_extension_from_url(url: str) -> str | None:
    path = unquote(urlparse(url).path or "")
    suffix = Path(path).suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else None


def _extension_from_content_type(content_type: str) -> str | None:
    value = (content_type or "").split(";", 1)[0].strip().lower()
    if not value:
        return None
    if value == "application/pdf":
        return ".pdf"
    if value == "application/msword":
        return ".doc"
    if "officedocument.wordprocessingml.document" in value:
        return ".docx"
    return None


def _is_html_content_type(content_type: str) -> bool:
    value = (content_type or "").split(";", 1)[0].strip().lower()
    return value in {"text/html", "application/xhtml+xml"}


def _resolve_extension(*, original_url: str, final_url: str, content_type: str) -> str | None:
    from_content_type = _extension_from_content_type(content_type)
    if from_content_type is not None:
        return from_content_type

    from_url = _extract_extension_from_url(final_url) or _extract_extension_from_url(original_url)
    if from_url is None:
        return None

    # Avoid indexing HTML pages that have file-like suffixes by mistake.
    if _is_html_content_type(content_type):
        return None

    return from_url


def _content_url(document: dict[str, Any]) -> str | None:
    absolute = _normalize_url(str(document.get("entity_url_domain") or ""))
    if absolute is not None:
        return absolute

    relative = str(document.get("entity_url") or "").strip()
    if relative:
        return _normalize_url(relative, base_url="https://www.inf.elte.hu")

    return None


def _source_ref(document: dict[str, Any], *, record_type: str) -> dict[str, Any]:
    return {
        "record_id": str(document.get("id") or ""),
        "record_type": record_type,
        "record_title": str(document.get("title") or "").strip(),
        "record_url": _content_url(document),
        "created": _coerce_int(document.get("created")),
    }


def _extract_global_document_link(document: dict[str, Any]) -> str | None:
    return _content_url(document)


def _extract_article_links(document: dict[str, Any]) -> set[str]:
    html = str(document.get("processed_text") or "").strip()
    if not html:
        return set()

    base_url = _content_url(document) or "https://www.inf.elte.hu"
    soup = BeautifulSoup(html, "html.parser")

    links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        normalized = _normalize_url(anchor.get("href", ""), base_url=base_url)
        if normalized is not None:
            links.add(normalized)
    return links


def _typesense_payload(*, page: int, per_page: int) -> dict[str, Any]:
    base_search = {
        "query_by": DEFAULT_TYPESENSE_QUERY_BY,
        "query_by_weights": DEFAULT_TYPESENSE_QUERY_BY_WEIGHTS,
        "sort_by": "created:desc",
        "exclude_fields": "embedding",
        "include_fields": DEFAULT_INCLUDE_FIELDS,
        "collection": "content",
        "q": "*",
        "page": page,
        "per_page": per_page,
        "highlight_fields": "none",
        "prefix": False,
    }
    return {
        "searches": [
            {
                **base_search,
                "filter_by": f"{DEFAULT_TYPESENSE_FILTER} && content_type:=global_document",
            },
            {
                **base_search,
                "filter_by": f"{DEFAULT_TYPESENSE_FILTER} && content_type:=article",
            },
        ]
    }


def fetch_typesense_page(
    *,
    page: int,
    per_page: int,
    client: httpx.Client,
    endpoint: str,
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.post(
        endpoint,
        params={"x-typesense-api-key": api_key},
        json=_typesense_payload(page=page, per_page=per_page),
        headers=_typesense_headers(),
    )
    response.raise_for_status()
    payload = response.json()

    results = payload.get("results")
    if not isinstance(results, list) or len(results) < 2:
        raise ValueError("Unexpected Typesense response shape")

    parsed: list[dict[str, Any]] = []
    for idx, result in enumerate(results[:2]):
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected Typesense result shape at index {idx}")

        error = str(result.get("error") or "").strip()
        if error:
            logger.warning("Typesense sub-search %d returned an error: %s", idx, error)
            parsed.append(
                {
                    "found": 0,
                    "hits": [],
                    "error": error,
                }
            )
            continue

        hits = result.get("hits")
        if not isinstance(hits, list):
            raise ValueError(f"Unexpected Typesense result shape at index {idx}: hits")
        parsed.append(
            {
                "found": _coerce_int(result.get("found")) or len(hits),
                "hits": [item for item in hits if isinstance(item, dict)],
                "error": None,
            }
        )

    return parsed[0], parsed[1]


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"items": {}}

    items = payload.get("items")
    if not isinstance(items, dict):
        payload["items"] = {}
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _merge_source_refs(
    existing: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for item in existing or []:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("record_id") or ""), str(item.get("record_url") or ""))
        merged[key] = item

    for item in incoming:
        key = (str(item.get("record_id") or ""), str(item.get("record_url") or ""))
        merged[key] = item

    return sorted(
        merged.values(),
        key=lambda entry: (
            str(entry.get("record_type") or ""),
            str(entry.get("record_id") or ""),
            str(entry.get("record_url") or ""),
        ),
    )


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._\- ]+", "_", value)
    cleaned = cleaned.strip(" .")
    return cleaned or "document"


def _filename_for_url(url: str, *, extension: str) -> str:
    parsed = urlparse(url)
    raw_name = unquote(Path(parsed.path).name or "").strip()
    if not raw_name:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        raw_name = f"document-{digest}{extension}"

    safe_name = _sanitize_filename(raw_name)
    stem = Path(safe_name).stem or "document"
    suffix = Path(safe_name).suffix.lower()
    if suffix != extension:
        safe_name = f"{stem}{extension}"

    return safe_name


def _ensure_unique_filename(
    *,
    preferred: str,
    final_url: str,
    file_owners: dict[str, str],
) -> str:
    owner = file_owners.get(preferred)
    if owner is None or owner == final_url:
        file_owners[preferred] = final_url
        return preferred

    stem = Path(preferred).stem
    suffix = Path(preferred).suffix
    index = 1
    while True:
        candidate = f"{stem}-{index}{suffix}"
        candidate_owner = file_owners.get(candidate)
        if candidate_owner is None or candidate_owner == final_url:
            file_owners[candidate] = final_url
            return candidate
        index += 1


def _resolve_candidate(
    *,
    candidate_url: str,
    client: httpx.Client,
) -> tuple[str, str, str]:
    with client.stream("GET", candidate_url, follow_redirects=True) as response:
        response.raise_for_status()
        final_url = _normalize_url(str(response.url))
        if final_url is None:
            raise ValueError(f"Unsupported final URL after redirects: {response.url}")
        content_type = str(response.headers.get("content-type") or "")

    extension = _resolve_extension(
        original_url=candidate_url,
        final_url=final_url,
        content_type=content_type,
    )
    if extension is None:
        raise ValueError("Unsupported file extension or content type")

    return final_url, extension, content_type


def _download_file(
    *,
    url: str,
    client: httpx.Client,
) -> tuple[bytes, str, str]:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    final_url = _normalize_url(str(response.url)) or url
    content_type = str(response.headers.get("content-type") or "")
    return response.content, final_url, content_type


def run_documents_sync(
    *,
    download_dir: str | Path | None = None,
    state_path: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    per_page: int | None = None,
    client: httpx.Client | None = None,
) -> DocumentSyncResult:
    endpoint = endpoint or settings.documents_typesense_url or settings.news_typesense_url or DEFAULT_TYPESENSE_URL
    api_key = (api_key or settings.documents_typesense_api_key or settings.news_typesense_api_key).strip()
    if not api_key:
        raise ValueError("DOCUMENTS_TYPESENSE_API_KEY (or NEWS_TYPESENSE_API_KEY) is required")

    page_size = per_page or settings.documents_sync_per_page
    page_size = max(1, min(MAX_PER_PAGE, int(page_size)))

    download_root = Path(download_dir or settings.raw_data_path)
    download_root.mkdir(parents=True, exist_ok=True)

    state_file = Path(state_path or settings.documents_sync_state_path)
    state = _load_state(state_file)
    state_items: dict[str, Any] = state.setdefault("items", {})

    own_client = client is None
    client = client or httpx.Client(
        follow_redirects=True,
        timeout=settings.documents_request_timeout_seconds,
    )

    now = datetime.now(UTC).isoformat()

    discovered_candidates: dict[str, list[dict[str, Any]]] = {}

    pages_fetched = 0
    blocked_domain_count = 0
    unsupported_extension_count = 0
    failed_count = 0

    try:
        # Fetch all pages for both search streams.
        page = 1
        max_pages: int | None = None
        while True:
            global_batch, article_batch = fetch_typesense_page(
                page=page,
                per_page=page_size,
                client=client,
                endpoint=endpoint,
                api_key=api_key,
            )
            pages_fetched += 1

            if (
                global_batch.get("error")
                and article_batch.get("error")
                and not global_batch["hits"]
                and not article_batch["hits"]
            ):
                raise ValueError(
                    "Typesense multi_search failed for both document streams: "
                    f"{global_batch['error']} | {article_batch['error']}"
                )

            if max_pages is None:
                max_pages = max(
                    1,
                    math.ceil(global_batch["found"] / page_size),
                    math.ceil(article_batch["found"] / page_size),
                )

            for hit in global_batch["hits"]:
                document = hit.get("document")
                if not isinstance(document, dict):
                    continue

                link = _extract_global_document_link(document)
                if link is None:
                    continue

                discovered_candidates.setdefault(link, []).append(
                    _source_ref(document, record_type="global_document")
                )

            for hit in article_batch["hits"]:
                document = hit.get("document")
                if not isinstance(document, dict):
                    continue

                source_ref = _source_ref(document, record_type="article")
                for link in _extract_article_links(document):
                    discovered_candidates.setdefault(link, []).append(source_ref)

            if page >= max_pages:
                break
            page += 1

        discovered_url_count = len(discovered_candidates)

        # Resolve redirects and canonicalize URLs.
        canonical_candidates: dict[str, dict[str, Any]] = {}
        duplicate_url_count = 0
        eligible_url_count = 0

        for candidate_url, source_refs in sorted(discovered_candidates.items()):
            if not _is_allowed_url(candidate_url):
                blocked_domain_count += 1
                continue

            try:
                final_url, extension, content_type = _resolve_candidate(
                    candidate_url=candidate_url,
                    client=client,
                )
            except Exception as exc:
                logger.warning("Skipping candidate %s: %s", candidate_url, exc)
                failed_count += 1
                continue

            if not _is_allowed_url(final_url):
                blocked_domain_count += 1
                continue

            if extension not in ALLOWED_EXTENSIONS:
                unsupported_extension_count += 1
                continue

            eligible_url_count += 1
            existing = canonical_candidates.get(final_url)
            if existing is None:
                canonical_candidates[final_url] = {
                    "url": final_url,
                    "extension": extension,
                    "content_type": content_type,
                    "source_refs": list(source_refs),
                }
                continue

            duplicate_url_count += 1
            existing["source_refs"] = _merge_source_refs(
                existing.get("source_refs"),
                list(source_refs),
            )

        canonical_url_count = len(canonical_candidates)

        file_owners: dict[str, str] = {}
        for existing_url, item in state_items.items():
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file_name") or "").strip()
            if file_name:
                file_owners[file_name] = existing_url

        downloaded_count = 0
        downloaded_pdf_count = 0
        downloaded_doc_count = 0
        downloaded_docx_count = 0
        skipped_count = 0

        for final_url, candidate in sorted(canonical_candidates.items()):
            extension = str(candidate["extension"])
            source_refs = _merge_source_refs([], list(candidate.get("source_refs") or []))

            previous = state_items.get(final_url)
            previous_file_name = ""
            if isinstance(previous, dict):
                previous_file_name = str(previous.get("file_name") or "").strip()
            previous_path = download_root / previous_file_name if previous_file_name else None

            if previous_path is not None and previous_path.exists():
                skipped_count += 1
                state_items[final_url] = {
                    **previous,
                    "url": final_url,
                    "extension": extension.lstrip("."),
                    "content_type": str(candidate.get("content_type") or previous.get("content_type") or ""),
                    "last_seen_at": now,
                    "source_refs": _merge_source_refs(
                        previous.get("source_refs") if isinstance(previous, dict) else None,
                        source_refs,
                    ),
                }
                continue

            try:
                content, downloaded_url, content_type = _download_file(url=final_url, client=client)
                canonical_downloaded_url = downloaded_url if _is_allowed_url(downloaded_url) else final_url

                # Keep extension from resolved metadata if possible.
                downloaded_extension = _resolve_extension(
                    original_url=final_url,
                    final_url=canonical_downloaded_url,
                    content_type=content_type,
                ) or extension

                if downloaded_extension not in ALLOWED_EXTENSIONS:
                    raise ValueError("Downloaded file has unsupported extension")

                preferred_name = previous_file_name or _filename_for_url(
                    canonical_downloaded_url,
                    extension=downloaded_extension,
                )
                file_name = _ensure_unique_filename(
                    preferred=preferred_name,
                    final_url=final_url,
                    file_owners=file_owners,
                )
                output_path = download_root / file_name
                output_path.write_bytes(content)

                content_hash = hashlib.sha256(content).hexdigest()
                downloaded_count += 1
                if downloaded_extension == ".pdf":
                    downloaded_pdf_count += 1
                elif downloaded_extension == ".doc":
                    downloaded_doc_count += 1
                elif downloaded_extension == ".docx":
                    downloaded_docx_count += 1

                state_items[final_url] = {
                    "url": final_url,
                    "file_name": file_name,
                    "extension": downloaded_extension.lstrip("."),
                    "content_hash": content_hash,
                    "content_type": content_type,
                    "downloaded_at": now,
                    "last_seen_at": now,
                    "source_refs": _merge_source_refs(
                        previous.get("source_refs") if isinstance(previous, dict) else None,
                        source_refs,
                    ),
                }
            except Exception as exc:
                logger.warning("Failed to download %s: %s", final_url, exc)
                failed_count += 1
                if isinstance(previous, dict):
                    state_items[final_url] = {
                        **previous,
                        "last_seen_at": now,
                        "source_refs": _merge_source_refs(
                            previous.get("source_refs"),
                            source_refs,
                        ),
                    }

        state["last_sync_at"] = now
        _write_state(state_file, state)

        return DocumentSyncResult(
            pages_fetched=pages_fetched,
            discovered_url_count=discovered_url_count,
            eligible_url_count=eligible_url_count,
            canonical_url_count=canonical_url_count,
            duplicate_url_count=duplicate_url_count,
            downloaded_count=downloaded_count,
            downloaded_pdf_count=downloaded_pdf_count,
            downloaded_doc_count=downloaded_doc_count,
            downloaded_docx_count=downloaded_docx_count,
            skipped_count=skipped_count,
            blocked_domain_count=blocked_domain_count,
            unsupported_extension_count=unsupported_extension_count,
            failed_count=failed_count,
        )
    finally:
        if own_client:
            client.close()
