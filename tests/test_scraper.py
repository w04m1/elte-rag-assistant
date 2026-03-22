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
