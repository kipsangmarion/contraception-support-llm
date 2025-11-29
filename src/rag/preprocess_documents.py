"""
Main script to preprocess WHO and BCS+ documents.
Extracts text, creates chunks, generates embeddings, and builds FAISS index.
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.document_processor import DocumentProcessor, get_document_statistics
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import FAISSVectorStore
from src.utils.reproducibility import load_config, set_random_seeds
from src.utils.logger import setup_logger

logger = setup_logger()


def main(args):
    """Main preprocessing pipeline."""

    logger.info("="*60)
    logger.info("Starting Document Preprocessing Pipeline")
    logger.info("="*60)

    # Set random seeds
    seeds = set_random_seeds()
    logger.info(f"Random seeds set: {seeds}")

    # Load configuration
    config = load_config(args.config)
    rag_config = config.get('rag', {})
    paths_config = config.get('paths', {})

    # Initialize document processor
    logger.info("Initializing document processor...")
    processor = DocumentProcessor(use_pdfplumber=True)

    # Process WHO documents
    all_chunks = []

    if args.process_who:
        who_path = Path(paths_config['data']['who'])
        logger.info(f"Processing WHO documents from {who_path}")

        if who_path.exists() and list(who_path.glob("*.pdf")):
            who_chunks = processor.process_directory(
                str(who_path),
                chunk_size=rag_config['chunking']['chunk_size'],
                chunk_overlap=rag_config['chunking']['chunk_overlap'],
                min_chunk_size=rag_config['chunking']['min_chunk_size'],
                max_chunk_size=rag_config['chunking']['max_chunk_size']
            )
            all_chunks.extend(who_chunks)
        else:
            logger.warning(f"No PDF files found in {who_path}")

    # Process BCS+ documents
    if args.process_bcs:
        bcs_path = Path(paths_config['data']['bcs'])
        logger.info(f"Processing BCS+ documents from {bcs_path}")

        if bcs_path.exists() and list(bcs_path.glob("*.pdf")):
            bcs_chunks = processor.process_directory(
                str(bcs_path),
                chunk_size=rag_config['chunking']['chunk_size'],
                chunk_overlap=rag_config['chunking']['chunk_overlap'],
                min_chunk_size=rag_config['chunking']['min_chunk_size'],
                max_chunk_size=rag_config['chunking']['max_chunk_size']
            )
            all_chunks.extend(who_chunks)
        else:
            logger.warning(f"No PDF files found in {bcs_path}")

    if not all_chunks:
        logger.error("No chunks created! Please add PDF files to data/who/ or data/bcs/")
        return

    # Get statistics
    stats = get_document_statistics(all_chunks)
    logger.info("Document Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Save chunks (without embeddings)
    chunks_output_path = Path(paths_config['index']['chunks'])
    chunks_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving chunks to {chunks_output_path}")
    with open(chunks_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_config = config['models']['embedding']
    embedder = EmbeddingGenerator(
        model_name=embedding_config['model_name'],
        provider=embedding_config['provider']
    )

    # Embed chunks
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedder.generate_embeddings(chunk_texts, show_progress=True)

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    # Build FAISS index
    logger.info("Building FAISS index...")
    vector_store = FAISSVectorStore(
        dimension=embedder.get_embedding_dimension(),
        index_type="Flat"  # Use Flat for small-medium datasets
    )
    vector_store.add_embeddings(embeddings, all_chunks)

    # Save index
    index_output_path = Path(paths_config['index']['faiss'])
    logger.info(f"Saving FAISS index to {index_output_path}")
    vector_store.save(
        index_path=str(index_output_path),
        chunks_path=str(chunks_output_path)
    )

    # Save statistics
    stats_path = Path(paths_config['data']['processed']) / 'statistics.json'
    logger.info(f"Saving statistics to {stats_path}")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("="*60)
    logger.info("Preprocessing Complete!")
    logger.info("="*60)
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    logger.info(f"Index saved to: {index_output_path}")
    logger.info(f"Chunks saved to: {chunks_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WHO and BCS+ documents")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--process-who",
        action="store_true",
        default=True,
        help="Process WHO documents"
    )
    parser.add_argument(
        "--process-bcs",
        action="store_true",
        default=True,
        help="Process BCS+ documents"
    )

    args = parser.parse_args()
    main(args)
