#!/usr/bin/env python3
"""
Build features for emails in the database.

Computes metadata, historical patterns, and embeddings for all emails
in the database.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --batch-size 50 --limit 1000
    python scripts/build_features.py --embeddings-only
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.features import (
    MetadataExtractor,
    HistoricalPatternExtractor,
    EmbeddingExtractor,
    FeatureStore
)
from src.utils import Config, setup_logger


def main():
    """Build features for emails."""
    parser = argparse.ArgumentParser(
        description='Build features for emails in database'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of emails to process per batch (default: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of emails to process (default: all)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database file (default: data/emails.db)'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip embedding generation (faster, but less accurate)'
    )
    parser.add_argument(
        '--embeddings-only',
        action='store_true',
        help='Only compute embeddings for emails with metadata features'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('build_features', level='INFO')

    logger.info("=== Feature Building ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize database
    db_path = args.db_path or f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}")
    db = Database(db_path)

    # Check database has emails
    stats = db.get_stats()
    if stats['emails'] == 0:
        logger.error("No emails in database. Run scripts/export_history.py first")
        sys.exit(1)

    logger.info(f"Database contains {stats['emails']} emails\n")

    # Initialize extractors
    logger.info("Initializing feature extractors...")
    metadata_extractor = MetadataExtractor()
    embedding_extractor = None if args.skip_embeddings else EmbeddingExtractor()

    if embedding_extractor:
        logger.info(f"  - Embedding model: {embedding_extractor.model_name}")
        logger.info(f"  - Embedding dimensions: {embedding_extractor.embedding_dim}")

    logger.info("")

    # Get emails to process
    with db.get_session() as session:
        store = FeatureStore(session)

        if args.embeddings_only:
            # Only process emails that already have metadata features
            logger.info("Finding emails with metadata features but no embeddings...")
            from src.database.schema import EmailFeatures
            emails_with_features = session.query(EmailFeatures).filter(
                EmailFeatures.subject_embedding.is_(None)
            ).limit(args.limit if args.limit else 10000).all()

            message_ids = [f.message_id for f in emails_with_features]
            from src.database.repository import EmailRepository
            repo = EmailRepository(session)
            emails_to_process = repo.get_by_ids(message_ids)

            logger.info(f"Found {len(emails_to_process)} emails needing embeddings\n")

        else:
            # Get emails without features
            logger.info("Finding emails without features...")
            emails_to_process = store.get_emails_without_features(limit=args.limit)
            logger.info(f"Found {len(emails_to_process)} emails to process\n")

    if not emails_to_process:
        logger.info("No emails to process. All done!")
        sys.exit(0)

    # Process in batches
    total_processed = 0
    batch_size = args.batch_size

    for i in range(0, len(emails_to_process), batch_size):
        batch = emails_to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(emails_to_process) + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} emails)...")

        try:
            with db.get_session() as session:
                store = FeatureStore(session)
                historical_extractor = HistoricalPatternExtractor(session)

                features_list = []

                for email in batch:
                    # Combine features from all extractors
                    features = {}

                    if not args.embeddings_only:
                        # Extract metadata features
                        metadata_features = metadata_extractor.extract(email)
                        features.update(metadata_features)

                        # Extract historical features
                        historical_features = historical_extractor.extract(email)
                        features.update(historical_features)

                    # Extract embeddings (if enabled)
                    if embedding_extractor:
                        embedding_features = embedding_extractor.extract(email)
                        features.update(embedding_features)

                    features_list.append((email.message_id, features))

                # Save all features
                store.save_features_batch(features_list)
                total_processed += len(batch)

            logger.info(f"  ✓ Batch {batch_num} complete (total: {total_processed}/{len(emails_to_process)})\n")

        except KeyboardInterrupt:
            logger.warning("\nProcessing interrupted by user")
            break
        except Exception as e:
            logger.error(f"  ✗ Batch {batch_num} failed: {e}\n")
            continue

    # Show final statistics
    logger.info("=== Feature Building Complete ===\n")
    logger.info(f"Processed {total_processed} emails\n")

    with db.get_session() as session:
        store = FeatureStore(session)
        feature_stats = store.get_feature_stats()

    logger.info("Feature statistics:")
    logger.info(f"  - Total emails: {feature_stats['total_emails']}")
    logger.info(f"  - Features computed: {feature_stats['features_count']}")
    logger.info(f"  - Coverage: {feature_stats['coverage_pct']:.1f}%")
    logger.info(f"  - Newsletters detected: {feature_stats['newsletter_count']} ({feature_stats['newsletter_pct']:.1f}%)")
    logger.info(f"  - Avg sender open rate: {feature_stats['avg_sender_open_rate']:.1%}")
    logger.info(f"  - Avg domain open rate: {feature_stats['avg_domain_open_rate']:.1%}")

    logger.info("\nNext steps:")
    logger.info("  1. Verify features: sqlite3 data/emails.db 'SELECT COUNT(*) FROM email_features;'")
    logger.info("  2. Proceed to Phase 4: ML model training")


if __name__ == '__main__':
    main()
