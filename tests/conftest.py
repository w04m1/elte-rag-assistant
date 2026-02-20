from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Students must submit their thesis by April 15th of the final semester.",
            metadata={"source": "thesis_rules.pdf", "title": "Thesis Rules", "page": 3},
        ),
        Document(
            page_content="The internship must be completed before the thesis defense.",
            metadata={
                "source": "internship_guide.pdf",
                "title": "Internship Guide",
                "page": 1,
            },
        ),
        Document(
            page_content="All students must pass a language proficiency exam before graduation.",
            metadata={
                "source": "graduation_reqs.pdf",
                "title": "Graduation Requirements",
                "page": 7,
            },
        ),
        Document(
            page_content="The final examination consists of a written and an oral part.",
            metadata={"source": "exam_rules.pdf", "title": "Exam Rules", "page": 2},
        ),
        Document(
            page_content="Students can enroll in elective courses through the Neptun system.",
            metadata={
                "source": "enrolment_guide.pdf",
                "title": "Enrolment Guide",
                "page": 5,
            },
        ),
    ]


@pytest.fixture()
def test_faiss_db(sample_documents):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        db = FAISS.from_documents(sample_documents, embeddings)
        return db
    except Exception:
        pytest.skip("Embedding model not available")


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.model_name = "mock-model"
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="This is a mock answer. [Thesis Rules, p. 3]")
    )
    return llm


@pytest.fixture()
def mock_reranker_llm():
    llm = MagicMock()
    llm.model_name = "mock-reranker"
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="[0.9, 0.5, 0.3, 0.1, 0.7]"))
    llm.__or__ = MagicMock(
        side_effect=lambda other: MagicMock(
            ainvoke=AsyncMock(return_value="[0.9, 0.5, 0.3, 0.1, 0.7]")
        )
    )
    return llm


@pytest.fixture()
def bm25_retriever(sample_documents):
    try:
        from langchain_community.retrievers import BM25Retriever

        return BM25Retriever.from_documents(sample_documents, k=5)
    except Exception:
        pytest.skip("rank-bm25 not available")
