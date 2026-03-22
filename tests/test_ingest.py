import json

from app.ingest import (
    _extract_page_from_chunk,
    _load_news_documents,
    _title_from_filename,
)


class TestTitleFromFilename:
    def test_basic(self):
        assert _title_from_filename("thesis_rules.pdf") == "thesis rules"

    def test_hyphens(self):
        assert _title_from_filename("student-guide-2025.pdf") == "student guide 2025"

    def test_no_extension(self):
        assert _title_from_filename("readme") == "readme"


class _FakeProv:
    def __init__(self, page_no):
        self.page_no = page_no


class _FakeDocItem:
    def __init__(self, prov):
        self.prov = prov


class _FakeMeta:
    def __init__(self, doc_items):
        self.doc_items = doc_items


class _FakeChunk:
    def __init__(self, meta):
        self.meta = meta


class TestExtractPageFromChunk:
    def test_extracts_lowest_provenance_page(self):
        chunk = _FakeChunk(
            _FakeMeta(
                [
                    _FakeDocItem([_FakeProv(3), _FakeProv(4)]),
                    _FakeDocItem([_FakeProv(2)]),
                ]
            )
        )
        assert _extract_page_from_chunk(chunk) == 2

    def test_returns_none_when_no_provenance(self):
        chunk = _FakeChunk(_FakeMeta([]))
        assert _extract_page_from_chunk(chunk) is None


class TestLoadNewsDocuments:
    def test_loads_normalized_news_json(self, tmp_path):
        news_dir = tmp_path / "news"
        news_dir.mkdir()
        (news_dir / "item.json").write_text(
            json.dumps(
                {
                    "url": "https://www.inf.elte.hu/en/content/sample-news.t.1000",
                    "title": "Sample News",
                    "published_at": "2026-03-21",
                    "body": "This is a normalized article body for indexing.",
                }
            ),
            encoding="utf-8",
        )

        docs = _load_news_documents(news_dir)
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "news"
        assert docs[0].metadata["title"] == "Sample News"
        assert docs[0].metadata["source"] == "https://www.inf.elte.hu/en/content/sample-news.t.1000"
        assert "Source URL:" in docs[0].page_content
