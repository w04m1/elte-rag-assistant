import json
from pathlib import Path

import httpx

from app.scraper import run_targeted_scrape


def test_targeted_scrape_downloads_pdf_and_deduplicates(tmp_path):
    transport = httpx.MockTransport(
        lambda request: (
            httpx.Response(
                200,
                text='<html><body><a href="/docs/rules.pdf">Rules</a></body></html>',
            )
            if str(request.url) == "https://www.inf.elte.hu/en"
            else httpx.Response(200, content=b"%PDF-1.4 demo pdf")
        )
    )
    client = httpx.Client(transport=transport, follow_redirects=True)

    result = run_targeted_scrape(
        download_dir=tmp_path / "downloads",
        manifest_path=tmp_path / "manifest.json",
        client=client,
        target_pages=["https://www.inf.elte.hu/en"],
    )
    second_result = run_targeted_scrape(
        download_dir=tmp_path / "downloads",
        manifest_path=tmp_path / "manifest.json",
        client=client,
        target_pages=["https://www.inf.elte.hu/en"],
    )

    assert result["downloaded_count"] == 1
    assert (tmp_path / "downloads" / "rules.pdf").exists()
    assert second_result["downloaded_count"] == 0
    assert second_result["discovered_count"] == 0


def test_targeted_scrape_indexes_news_articles(tmp_path):
    news_url = "https://www.inf.elte.hu/en/content/news.t.53"
    article_url = "https://www.inf.elte.hu/en/content/sample-news.t.1000"
    long_text = " ".join(["ELTE news update"] * 30)

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == news_url:
            return httpx.Response(
                200,
                text='<html><body><a href="/en/content/sample-news.t.1000">News</a></body></html>',
            )
        if url == article_url:
            return httpx.Response(
                200,
                text=(
                    "<html><head><title>Sample News</title></head><body>"
                    '<article><h1>Sample News</h1><time datetime="2026-03-21">March 21, 2026</time>'
                    f"<p>{long_text}</p></article></body></html>"
                ),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(_handler)
    client = httpx.Client(transport=transport, follow_redirects=True)

    result = run_targeted_scrape(
        download_dir=tmp_path / "downloads",
        news_output_dir=tmp_path / "news",
        manifest_path=tmp_path / "manifest.json",
        client=client,
        target_pages=[news_url],
    )
    second_result = run_targeted_scrape(
        download_dir=tmp_path / "downloads",
        news_output_dir=tmp_path / "news",
        manifest_path=tmp_path / "manifest.json",
        client=client,
        target_pages=[news_url],
    )

    assert result["downloaded_count"] == 0
    assert result["news_saved_count"] == 1
    assert result["discovered_count"] == 1

    news_files = list((tmp_path / "news").glob("*.json"))
    assert len(news_files) == 1
    payload = json.loads(news_files[0].read_text(encoding="utf-8"))
    assert payload["url"] == article_url
    assert payload["title"] == "Sample News"
    assert payload["body"]

    assert second_result["downloaded_count"] == 0
    assert second_result["news_saved_count"] == 0
    assert second_result["discovered_count"] == 0
