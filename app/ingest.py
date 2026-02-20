import argparse
import logging
import os
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
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


def create_vector_db(
    source_dir: str | None = None,
    output_dir: str | None = None,
) -> None:
    source_dir = source_dir or settings.raw_data_path
    output_dir = output_dir or settings.faiss_index_path

    pdf_paths = sorted(Path(source_dir).glob("*.pdf"))

    if not pdf_paths:
        logger.warning(f"No PDF documents found in {source_dir}")
        return

    logger.info(f"Found {len(pdf_paths)} PDF files in {source_dir}")

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

    documents: list[Document] = []

    for pdf_path in pdf_paths:
        logger.info(f"Converting {pdf_path.name}...")
        result = converter.convert(str(pdf_path))
        title = _title_from_filename(pdf_path.name)

        chunks = list(chunker.chunk(result.document))
        for chunk in chunks:
            text = chunker.contextualize(chunk)

            meta: dict = {
                "source": pdf_path.name,
                "title": title,
            }
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                meta["headings"] = chunk.meta.headings
            if hasattr(chunk.meta, "page") and chunk.meta.page is not None:
                meta["page"] = chunk.meta.page

            documents.append(Document(page_content=text, metadata=meta))

        logger.info(
            f"  → {len(chunks)} chunks from {pdf_path.name} ({len(result.document.pages)} pages)"
        )

    logger.info(f"Total chunks: {len(documents)}")

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
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS vector store")
    parser.add_argument(
        "--source",
        default=None,
        help="Source directory with PDFs (default: from config)",
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
