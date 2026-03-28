import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _title_from_filename(filename: str) -> str:
    """Get title PDF filename."""
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ")
    return name.strip()


def _extract_page_from_chunk(chunk: Any) -> int | None:
    """Extract page number from docling chunk metadata.

    Docling's HybridChunker stores provenance at chunk.meta.doc_items[*].prov[*].page_no.
    """
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None

    # Backward compatibility if page is attached directly.
    raw_page = getattr(meta, "page", None)
    if isinstance(raw_page, int):
        return raw_page
    if isinstance(raw_page, str) and raw_page.isdigit():
        return int(raw_page)

    doc_items = getattr(meta, "doc_items", None) or []
    pages: list[int] = []
    for item in doc_items:
        prov_items = getattr(item, "prov", None)
        if prov_items is None and isinstance(item, dict):
            prov_items = item.get("prov")
        if not prov_items:
            continue

        for prov in prov_items:
            page_no = getattr(prov, "page_no", None)
            if page_no is None and isinstance(prov, dict):
                page_no = prov.get("page_no")

            if isinstance(page_no, int):
                pages.append(page_no)
            elif isinstance(page_no, str) and page_no.isdigit():
                pages.append(int(page_no))

    return min(pages) if pages else None


def _load_news_documents(news_dir: str | Path | None = None) -> list[Document]:
    """Load normalized scraped news files and convert them to retrievable documents."""
    news_dir = Path(news_dir or settings.scrape_news_path)
    if not news_dir.exists():
        return []

    documents: list[Document] = []
    for news_path in sorted(news_dir.glob("*.json")):
        try:
            payload = json.loads(news_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping invalid news file %s: %s", news_path.name, exc)
            continue

        body = str(payload.get("body", "")).strip()
        if not body:
            continue

        title = str(payload.get("title", "")).strip() or _title_from_filename(news_path.name)
        source_url = str(payload.get("url", "")).strip()
        content = f"{title}\n\n{body}"
        if source_url:
            content = f"Source URL: {source_url}\n\n{content}"

        metadata: dict[str, Any] = {
            "source": source_url or news_path.name,
            "title": title,
            "type": "news",
        }
        published_at = payload.get("published_at")
        if published_at:
            metadata["published_at"] = str(published_at)

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def _list_ingestion_inputs(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    paths = list(root.glob("*.pdf")) + list(root.glob("*.docx"))
    return sorted(paths, key=lambda path: path.name.lower())


def create_vector_db(
    source_dir: str | None = None,
    output_dir: str | None = None,
) -> None:
    source_dir = source_dir or settings.raw_data_path
    output_dir = output_dir or settings.faiss_index_path

    source_paths = _list_ingestion_inputs(source_dir)
    if not source_paths:
        logger.warning("No supported ingestion inputs (.pdf/.docx) found in %s", source_dir)
        return

    logger.info("Found %d supported files (.pdf/.docx) in %s", len(source_paths), source_dir)

    documents: list[Document] = []

    if source_paths:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                ),
            },
        )
        chunker = HybridChunker(
            tokenizer=f"sentence-transformers/{settings.embedding_model_name}",
            max_tokens=settings.max_tokens,
            merge_peers=True,
        )

        for source_path in source_paths:
            logger.info("Converting %s...", source_path.name)
            result = converter.convert(str(source_path))
            title = _title_from_filename(source_path.name)

            chunks = list(chunker.chunk(result.document))
            for chunk in chunks:
                text = chunker.contextualize(chunk)

                meta: dict[str, Any] = {
                    "source": source_path.name,
                    "title": title,
                }
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    meta["headings"] = chunk.meta.headings

                page = _extract_page_from_chunk(chunk)
                if page is not None:
                    meta["page"] = page

                documents.append(Document(page_content=text, metadata=meta))

            logger.info(
                "  → %d chunks from %s (%d pages)",
                len(chunks),
                source_path.name,
                len(result.document.pages),
            )

    logger.info("Total chunks/documents indexed: %d", len(documents))

    embeddings = get_embeddings()

    logger.info("Creating vector store...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(output_dir)
    logger.info(f"Vector store saved to {output_dir} ({len(documents)} vectors).")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Ingest PDF/DOCX documents into FAISS vector store"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source directory with .pdf/.docx files (default: from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output FAISS index directory (default: from config)",
    )
    args = parser.parse_args()
    create_vector_db(source_dir=args.source, output_dir=args.output)


if __name__ == "__main__":
    main()
