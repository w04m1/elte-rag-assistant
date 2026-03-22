import json

import httpx

from app.news_ingest import parse_typesense_hit, sync_news_records


def _typesense_response(hits: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"results": [{"hits": hits}]})


def test_parse_typesense_hit_accepts_tagged_news_and_normalizes_html():
    hit = {
        "document": {
            "id": "entity:node-1:en",
            "title": "Demo News",
            "entity_url_domain": "https://inf.elte.hu/en/node/1",
            "summary_text": "Summary",
            "processed_text": "<p>Hello <strong>world</strong></p>",
            "created": 1772022012,
            "news_tag": True,
            "source_news_tag": False,
            "content_type": "article",
        }
    }

    parsed = parse_typesense_hit(hit)

    assert parsed is not None
    assert parsed["id"] == "entity:node-1:en"
    assert parsed["url"] == "https://inf.elte.hu/en/node/1"
    assert parsed["title"] == "Demo News"
    assert parsed["body"] == "Hello world"
    assert parsed["content_hash"]


def test_parse_typesense_hit_rejects_non_news_records():
    hit = {
        "document": {
            "id": "entity:node-2:en",
            "title": "Person Profile",
            "entity_url_domain": "https://inf.elte.hu/en/node/2",
            "processed_text": "<p>Bio</p>",
            "news_tag": False,
            "source_news_tag": False,
        }
    }

    assert parse_typesense_hit(hit) is None


def test_sync_news_records_deduplicates_existing_items(tmp_path):
    endpoint = "https://typesense.elte.hu/multi_search"

    def _handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(endpoint)
        payload = json.loads(request.content.decode("utf-8"))
        page = payload["searches"][0]["page"]

        if page == 1:
            return _typesense_response(
                [
                    {
                        "document": {
                            "id": "entity:node-101:en",
                            "title": "News A",
                            "entity_url": "/en/node/101",
                            "summary_text": "A",
                            "processed_text": "<p>News A body</p>",
                            "news_tag": True,
                            "created": 1772022012,
                        }
                    }
                ]
            )

        if page == 2:
            return _typesense_response(
                [
                    {
                        "document": {
                            "id": "entity:node-102:en",
                            "title": "News B",
                            "entity_url": "/en/node/102",
                            "summary_text": "B",
                            "processed_text": "<p>News B body</p>",
                            "source_news_tag": True,
                            "created": 1772022013,
                        }
                    }
                ]
            )

        return _typesense_response([])

    client = httpx.Client(transport=httpx.MockTransport(_handler))

    first = sync_news_records(
        pages=2,
        records_dir=tmp_path / "news",
        state_path=tmp_path / "state.json",
        endpoint=endpoint,
        api_key="demo-key",
        client=client,
    )
    second = sync_news_records(
        pages=2,
        records_dir=tmp_path / "news",
        state_path=tmp_path / "state.json",
        endpoint=endpoint,
        api_key="demo-key",
        client=client,
    )

    assert first["processed_count"] == 2
    assert first["added_count"] == 2
    assert first["updated_count"] == 0

    assert second["processed_count"] == 2
    assert second["added_count"] == 0
    assert second["updated_count"] == 0
    assert second["unchanged_count"] == 2


def test_sync_news_records_updates_changed_content(tmp_path):
    endpoint = "https://typesense.elte.hu/multi_search"
    payload_version = {"value": 1}

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        page = payload["searches"][0]["page"]
        if page != 1:
            return _typesense_response([])

        body = "<p>Body v1</p>" if payload_version["value"] == 1 else "<p>Body v2</p>"
        return _typesense_response(
            [
                {
                    "document": {
                        "id": "entity:node-201:en",
                        "title": "Mutable News",
                        "entity_url_domain": "https://inf.elte.hu/en/node/201",
                        "summary_text": "Summary",
                        "processed_text": body,
                        "news_tag": True,
                        "created": 1772022012,
                    }
                }
            ]
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))

    first = sync_news_records(
        pages=1,
        records_dir=tmp_path / "news",
        state_path=tmp_path / "state.json",
        endpoint=endpoint,
        api_key="demo-key",
        client=client,
    )

    payload_version["value"] = 2

    second = sync_news_records(
        pages=1,
        records_dir=tmp_path / "news",
        state_path=tmp_path / "state.json",
        endpoint=endpoint,
        api_key="demo-key",
        client=client,
    )

    assert first["added_count"] == 1
    assert second["updated_count"] == 1
    assert second["added_count"] == 0
