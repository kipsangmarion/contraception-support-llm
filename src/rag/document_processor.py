"""
Document processing utilities for converting PDFs to text and chunking.
Handles WHO FP Handbook and BCS+ Toolkit documents.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import pdfplumber
from loguru import logger


class DocumentProcessor:
    """Processes PDF documents into text chunks with metadata."""

    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize document processor.

        Args:
            use_pdfplumber: If True, use pdfplumber (better text extraction).
                          If False, use PyPDF2 (faster but less accurate).
        """
        self.use_pdfplumber = use_pdfplumber

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF file page by page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries containing page text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting text from {pdf_path.name}")

        pages = []

        if self.use_pdfplumber:
            pages = self._extract_with_pdfplumber(pdf_path)
        else:
            pages = self._extract_with_pypdf2(pdf_path)

        logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
        return pages

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, any]]:
        """Extract text using pdfplumber (better quality)."""
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages.append({
                        'page_number': page_num,
                        'text': text.strip(),
                        'source': pdf_path.name,
                        'source_path': str(pdf_path)
                    })

        return pages

    def _extract_with_pypdf2(self, pdf_path: Path) -> List[Dict[str, any]]:
        """Extract text using PyPDF2 (faster)."""
        pages = []

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                if text:
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text.strip(),
                        'source': pdf_path.name,
                        'source_path': str(pdf_path)
                    })

        return pages

    def chunk_text(
        self,
        pages: List[Dict[str, any]],
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        min_chunk_size: int = 150,
        max_chunk_size: int = 300
    ) -> List[Dict[str, any]]:
        """
        Split text into chunks with overlap.

        Args:
            pages: List of page dictionaries from extract_text_from_pdf
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            min_chunk_size: Minimum chunk size to keep
            max_chunk_size: Maximum chunk size

        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Chunking text with chunk_size={chunk_size}, overlap={chunk_overlap}")

        all_chunks = []
        chunk_id = 0

        for page in pages:
            page_text = page['text']
            page_number = page['page_number']
            source = page['source']

            # Split by paragraphs first (double newlines or single newlines)
            paragraphs = self._split_into_paragraphs(page_text)

            current_chunk = ""

            for paragraph in paragraphs:
                # If adding this paragraph exceeds max_chunk_size, save current chunk
                if len(current_chunk) + len(paragraph) > max_chunk_size:
                    if len(current_chunk) >= min_chunk_size:
                        all_chunks.append({
                            'id': f"chunk_{chunk_id:05d}",
                            'text': current_chunk.strip(),
                            'page_number': page_number,
                            'source': source,
                            'source_path': page['source_path'],
                            'chunk_index': chunk_id
                        })
                        chunk_id += 1

                    # Start new chunk with overlap
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        current_chunk = current_chunk[-chunk_overlap:] + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph

            # Add remaining text as final chunk
            if len(current_chunk) >= min_chunk_size:
                all_chunks.append({
                    'id': f"chunk_{chunk_id:05d}",
                    'text': current_chunk.strip(),
                    'page_number': page_number,
                    'source': source,
                    'source_path': page['source_path'],
                    'chunk_index': chunk_id
                })
                chunk_id += 1

        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines first
        paragraphs = text.split('\n\n')

        # Further split long paragraphs by single newlines
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is too long, split by sentences or newlines
            if len(para) > 400:
                sub_paras = para.split('\n')
                result.extend([p.strip() for p in sub_paras if p.strip()])
            else:
                result.append(para)

        return result

    def process_directory(
        self,
        directory: str,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        min_chunk_size: int = 150,
        max_chunk_size: int = 300
    ) -> List[Dict[str, any]]:
        """
        Process all PDF files in a directory.

        Args:
            directory: Path to directory containing PDFs
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size

        Returns:
            List of all chunks from all PDFs
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        all_chunks = []

        for pdf_file in pdf_files:
            try:
                pages = self.extract_text_from_pdf(pdf_file)
                chunks = self.chunk_text(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


def get_document_statistics(chunks: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Calculate statistics about processed documents.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {}

    sources = set(chunk['source'] for chunk in chunks)
    total_chars = sum(len(chunk['text']) for chunk in chunks)
    avg_chunk_length = total_chars / len(chunks) if chunks else 0

    chunk_lengths = [len(chunk['text']) for chunk in chunks]

    stats = {
        'total_chunks': len(chunks),
        'total_sources': len(sources),
        'sources': list(sources),
        'total_characters': total_chars,
        'avg_chunk_length': round(avg_chunk_length, 2),
        'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
        'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0
    }

    return stats
