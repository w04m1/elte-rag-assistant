from app.ingest import _title_from_filename


class TestTitleFromFilename:
    def test_basic(self):
        assert _title_from_filename("thesis_rules.pdf") == "thesis rules"

    def test_hyphens(self):
        assert _title_from_filename("student-guide-2025.pdf") == "student guide 2025"

    def test_no_extension(self):
        assert _title_from_filename("readme") == "readme"
