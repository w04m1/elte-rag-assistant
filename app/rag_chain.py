import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field as PydanticField, field_validator

from app.config import settings

logger = logging.getLogger(__name__)


class CitedSource(BaseModel):
    """WIP: A single cited source from the context."""

    document: str
    page: int | None = None
    relevant_snippet: str

    @field_validator("page", mode="before")
    @classmethod
    def _coerce_page(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class RAGOutput(BaseModel):
    reasoning: str = PydanticField(
        description="Internal chain-of-thought explaining how the context was interpreted",
    )
    answer: str = PydanticField(
        description="User-facing answer with inline citations in [Document Title, p. X] format",
    )
    cited_chunk_ids: list[str] = PydanticField(
        default_factory=list,
        description="Structured list of cited chunk IDs like C1, C2 derived from the provided context",
    )
    confidence: Literal["high", "medium", "low"] = PydanticField(
        description="Confidence level based on how well the context answers the question",
    )


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful FAQ assistant for ELTE University.
Answer the student's question **based only on the provided context**.
If the context does not contain enough information, say "I don't have enough information to answer that question."

Rules:
- Be concise and accurate.
- Each context block has a stable citation ID such as [C1].
- When you cite claims inline, cite the chunk IDs like [C1] or [C1][C2].
- Do not make up information that is not in the context.
- Before answering, reason step-by-step about which parts of the context are relevant.
- Assess your confidence level (high, medium, or low) based on how well the context answers the question.
- Return cited_chunk_ids using only the chunk IDs that support the answer.
"""

def build_rag_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Context:\n[C1 | Thesis Rules, p. 3]\nStudents must submit their thesis by "
                "April 15th of the final semester. Late submissions will not be accepted "
                "without prior approval from the department head.\n\n---\n"
                "Question: When is the thesis submission deadline?",
            ),
            (
                "assistant",
                "The thesis submission deadline is April 15th of the final semester. "
                "Late submissions are not accepted without prior approval from the "
                "department head. [C1]",
            ),
            (
                "human",
                "Context:\n[C1 | Enrolment Guide, p. 5]\nStudents can enroll in elective "
                "courses through the Neptun system during the registration period."
                "\n\n---\n"
                "Question: What is the tuition fee for international students?",
            ),
            (
                "assistant",
                "I don't have enough information to answer that question. The available "
                "context covers course enrollment procedures but does not mention tuition "
                "fees for international students.",
            ),
            (
                "human",
                "Context:\n{context}\n\n---\nQuestion: {question}",
            ),
        ]
    )


RAG_PROMPT = build_rag_prompt(DEFAULT_SYSTEM_PROMPT)

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance scoring system. Given a query and a list of text "
            "chunks, score each chunk's relevance to the query on a scale of 0.0 to "
            "1.0. Return ONLY a JSON array of numbers representing the scores in the "
            "same order as the chunks. Example: [0.9, 0.3, 0.7]",
        ),
        ("human", "Query: {query}\n\nChunks:\n{chunks}"),
    ]
)


def _build_context_items(docs: list[Document]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source = meta.get("title", meta.get("source", "unknown"))
        page = meta.get("page", "?")
        items.append(
            {
                "citation_id": f"C{index}",
                "document": source,
                "page": page if page != "?" else None,
                "content": doc.page_content,
                "snippet": doc.page_content[:300],
                "source": meta.get("source", "unknown"),
            }
        )
    return items


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string with stable IDs."""
    parts: list[str] = []
    for item in _build_context_items(docs):
        page = item["page"] if item["page"] is not None else "?"
        parts.append(
            f"[{item['citation_id']} | {item['document']}, p. {page}]\n{item['content']}"
        )
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Build the sources list from retrieved documents."""
    return _build_context_items(docs)


def _replace_inline_chunk_citations(answer: str, citation_map: dict[str, dict[str, Any]]) -> str:
    def replace(match: re.Match[str]) -> str:
        citation_id = match.group(1).upper()
        item = citation_map.get(citation_id)
        if item is None:
            return match.group(0)
        if item["page"] is None:
            return f"[{item['document']}]"
        return f"[{item['document']}, p. {item['page']}]"

    return re.sub(r"\[([Cc]\d+)\]", replace, answer)


def _build_cited_sources(
    cited_chunk_ids: list[str],
    context_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {item["citation_id"]: item for item in context_items}
    cited_sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_id in cited_chunk_ids:
        citation_id = raw_id.upper()
        item = by_id.get(citation_id)
        if item is None or citation_id in seen:
            continue
        seen.add(citation_id)
        cited_sources.append(
            {
                "document": item["document"],
                "page": item["page"],
                "relevant_snippet": item["snippet"],
            }
        )

    if cited_sources:
        return cited_sources

    return [
        {
            "document": item["document"],
            "page": item["page"],
            "relevant_snippet": item["snippet"],
        }
        for item in context_items[: min(3, len(context_items))]
    ]


@dataclass
class RAGResult:
    """Structured response returned by the RAG chain."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    reasoning: str = ""
    confidence: str = ""
    cited_sources: list[dict[str, Any]] = field(default_factory=list)


def get_llm(model_override: str | None = None):
    """Return a LangChain chat model based on the configured provider."""
    if settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        model = model_override or settings.openrouter_model
        return ChatOpenAI(
            model=model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        model = model_override or settings.ollama_model
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


async def _rerank(
    query: str,
    docs: list[Document],
    top_k: int,
    reranker_model: str | None = None,
) -> list[Document]:
    """Rerank docs using an LLM-based relevance scorer."""
    if not settings.retrieval_use_reranker or not docs:
        return docs[:top_k]

    logger.info(
        "Reranking %d documents to top %d using %s",
        len(docs),
        top_k,
        reranker_model or settings.reranker_model,
    )

    reranker_llm = get_llm(model_override=reranker_model or settings.reranker_model)

    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}]: {doc.page_content[:500]}" for i, doc in enumerate(docs)
    )

    try:
        chain = RERANK_PROMPT | reranker_llm | StrOutputParser()
        result = await chain.ainvoke({"query": query, "chunks": chunks_text})

        # Strip markdown code fences if present
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        scores: list[float] = json.loads(result)

        if not isinstance(scores, list) or len(scores) != len(docs):
            logger.warning(
                "Reranker returned invalid scores (expected %d, got %s), "
                "skipping reranking",
                len(docs),
                len(scores) if isinstance(scores, list) else "non-list",
            )
            return docs[:top_k]

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _score in scored_docs[:top_k]]

        logger.info(
            "Reranking complete. Top scores: %s",
            [f"{s:.2f}" for _, s in scored_docs[:top_k]],
        )
        return reranked

    except Exception as exc:
        logger.warning(
            f"Reranking failed ({exc}), returning documents as-is",
        )
        return docs[:top_k]


def _reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    weights: list[float],
    k_rrf: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using weighted Reciprocal Rank Fusion.

    Each document gets a score of weight / (k_rrf + rank) from each list,
    summed across lists. Returns documents sorted by fused score descending.
    """
    fused_scores: dict[int, float] = {}  # id(doc) → score
    doc_map: dict[int, Document] = {}

    for doc_list, weight in zip(results_lists, weights):
        for rank, doc in enumerate(doc_list):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight / (
                k_rrf + rank + 1
            )

    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_map[did] for did in sorted_ids]


async def ask(
    query: str,
    db: FAISS,
    bm25_retriever=None,
    *,
    system_prompt: str | None = None,
    generator_model: str | None = None,
    reranker_model: str | None = None,
) -> RAGResult:
    """Run the full RAG pipeline and return a structured result."""
    k = settings.retrieval_k
    fetch_k = settings.retrieval_fetch_k

    # Retrieval
    if settings.retrieval_hybrid and bm25_retriever is not None:
        faiss_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 2},
        )
        weight = settings.retrieval_hybrid_weight
        faiss_docs = faiss_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        docs: list[Document] = _reciprocal_rank_fusion(
            [faiss_docs, bm25_docs],
            weights=[weight, 1 - weight],
        )
        docs = docs[:fetch_k]
    else:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 2},
        )
        docs = retriever.invoke(query)

    # Reranking
    docs = await _rerank(query, docs, top_k=k, reranker_model=reranker_model)

    # Generation
    llm = get_llm(model_override=generator_model)
    model_name = str(
        getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    )
    context_items = _build_context_items(docs)
    context = _format_docs(docs)
    sources = _extract_sources(docs)
    prompt = build_rag_prompt(system_prompt or DEFAULT_SYSTEM_PROMPT)
    citation_map = {item["citation_id"]: item for item in context_items}

    # Try structured output , fall back to plain string
    try:
        import warnings

        structured_llm = llm.with_structured_output(RAGOutput)
        chain = prompt | structured_llm
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            output: RAGOutput = await chain.ainvoke(
                {"context": context, "question": query}
            )
        cited_sources = _build_cited_sources(output.cited_chunk_ids, context_items)
        return RAGResult(
            answer=_replace_inline_chunk_citations(output.answer, citation_map),
            sources=sources,
            model_used=model_name,
            reasoning=output.reasoning,
            confidence=output.confidence,
            cited_sources=cited_sources,
        )
    except Exception as exc:
        logger.warning(
            f"Structured output failed ({exc}), falling back to plain string",
        )
        plain_chain = prompt | llm | StrOutputParser()
        answer: str = await plain_chain.ainvoke({"context": context, "question": query})
        fallback_citations = _build_cited_sources([], context_items)
        return RAGResult(
            answer=_replace_inline_chunk_citations(answer, citation_map),
            sources=sources,
            model_used=model_name,
            cited_sources=fallback_citations,
        )
