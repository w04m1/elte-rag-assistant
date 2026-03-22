import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PAGES = [
    "https://www.inf.elte.hu/en",
    "https://www.inf.elte.hu/en/content/documents.t.295",
    "https://www.inf.elte.hu/en/content/news.t.53",
]
ALLOWED_DOMAINS = {"www.inf.elte.hu", "inf.elte.hu"}


class ScrapeResult(dict):
    pass


def _is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc in ALLOWED_DOMAINS


def _safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "downloaded.pdf"
    if name.lower().endswith(".pdf"):
        return name
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{digest}.html"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"targets": DEFAULT_TARGET_PAGES, "items": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_targeted_scrape(
    *,
    download_dir: str | Path,
    manifest_path: str | Path,
    client: httpx.Client | None = None,
    target_pages: list[str] | None = None,
) -> ScrapeResult:
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    download_root = Path(download_dir)
    download_root.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    existing_urls = {item["url"] for item in manifest.get("items", [])}
    existing_names = {item["file_name"] for item in manifest.get("items", [])}

    discovered_items: list[dict] = []
    downloaded_count = 0
    discovered_count = 0
    target_pages = target_pages or DEFAULT_TARGET_PAGES

    try:
        for page_url in target_pages:
            if not _is_allowed_url(page_url):
                logger.warning("Skipping non-official target page %s", page_url)
                continue

            response = client.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for anchor in soup.find_all("a", href=True):
                absolute_url = urljoin(page_url, anchor["href"])
                if not _is_allowed_url(absolute_url):
                    continue

                file_name = _safe_filename_from_url(absolute_url)
                if absolute_url in existing_urls or file_name in existing_names:
                    continue

                is_pdf = absolute_url.lower().endswith(".pdf")
                status = "discovered"
                saved_path = None

                if is_pdf:
                    file_response = client.get(absolute_url)
                    file_response.raise_for_status()
                    saved_path = str(download_root / file_name)
                    Path(saved_path).write_bytes(file_response.content)
                    downloaded_count += 1
                    status = "downloaded"

                discovered_count += 1
                item = {
                    "url": absolute_url,
                    "label": anchor.get_text(strip=True) or file_name,
                    "file_name": file_name,
                    "status": status,
                    "saved_path": saved_path,
                }
                discovered_items.append(item)
                existing_urls.add(absolute_url)
                existing_names.add(file_name)

        manifest["targets"] = target_pages
        manifest["items"] = manifest.get("items", []) + discovered_items
        _write_manifest(manifest_file, manifest)

        return ScrapeResult(
            targets=target_pages,
            discovered_count=discovered_count,
            downloaded_count=downloaded_count,
            items=discovered_items,
        )
    finally:
        if own_client:
            client.close()
