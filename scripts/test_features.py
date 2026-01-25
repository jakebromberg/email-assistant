#!/usr/bin/env python3
"""
Test feature extraction.

Verifies that all feature extractors work correctly.

Usage:
    python scripts/test_features.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.database.schema import Email
from src.features import (
    MetadataExtractor,
    HistoricalPatternExtractor,
    EmbeddingExtractor,
    FeatureStore
)
from src.utils import Config, setup_logger


def main():
    """Test feature extraction."""
    # Set up logging
    logger = setup_logger('test_features', level='INFO')

    logger.info("=== Feature Extraction Test ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Create test database
    test_db_path = f"{config.DATA_DIR}/test_features.db"
    logger.info(f"Creating test database: {test_db_path}")
    db = Database(test_db_path)
    db.create_tables()

    # Create test email
    logger.info("1. Creating test email...")
    test_email = Email(
        message_id="test_features_123",
        thread_id="thread123",
        from_address="newsletter@techweekly.com",
        from_name="Tech Weekly",
        to_address="me@example.com",
        subject="This Week in AI: GPT-5 and Beyond",
        date=datetime.now(),
        snippet="Latest developments in artificial intelligence...",
        body_plain="Welcome to Tech Weekly! This week we cover...",
        body_html="<html>...</html>",
        labels=["INBOX", "UNREAD"],
        was_read=False,
        was_archived=False,
        headers={"List-Unsubscribe": "<mailto:unsub@techweekly.com>"},
        collected_at=datetime.utcnow()
    )

    with db.get_session() as session:
        session.add(test_email)

    logger.info("   ✓ Test email created\n")

    # Test metadata extraction
    logger.info("2. Testing metadata extraction...")
    try:
        metadata_extractor = MetadataExtractor()
        metadata_features = metadata_extractor.extract(test_email)

        logger.info("   ✓ Metadata features:")
        logger.info(f"     - Sender domain: {metadata_features['sender_domain']}")
        logger.info(f"     - Is newsletter: {metadata_features['is_newsletter']}")
        logger.info(f"     - Day of week: {metadata_features['day_of_week']}")
        logger.info(f"     - Subject length: {metadata_features['subject_length']}")
        logger.info("")
    except Exception as e:
        logger.error(f"   ✗ Metadata extraction failed: {e}\n")
        sys.exit(1)

    # Test historical extraction
    logger.info("3. Testing historical pattern extraction...")
    try:
        with db.get_session() as session:
            historical_extractor = HistoricalPatternExtractor(session)
            historical_features = historical_extractor.extract(test_email)

        logger.info("   ✓ Historical features:")
        logger.info(f"     - Sender email count: {historical_features['sender_email_count']}")
        logger.info(f"     - Sender open rate: {historical_features['sender_open_rate']:.1%}")
        logger.info(f"     - Domain open rate: {historical_features['domain_open_rate']:.1%}")
        logger.info("")
    except Exception as e:
        logger.error(f"   ✗ Historical extraction failed: {e}\n")
        sys.exit(1)

    # Test embedding extraction
    logger.info("4. Testing embedding extraction...")
    logger.info("   (This will download the model on first run)")
    try:
        embedding_extractor = EmbeddingExtractor()
        embedding_features = embedding_extractor.extract(test_email)

        logger.info("   ✓ Embedding features:")
        logger.info(f"     - Model: {embedding_extractor.model_name}")
        logger.info(f"     - Embedding dimensions: {embedding_extractor.embedding_dim}")
        logger.info(f"     - Subject embedding shape: {len(embedding_features['subject_embedding'])}")
        logger.info(f"     - Body embedding shape: {len(embedding_features['body_embedding'])}")

        # Test similarity
        similarity = embedding_extractor.compute_similarity(
            embedding_features['subject_embedding'],
            embedding_features['body_embedding']
        )
        logger.info(f"     - Subject-body similarity: {similarity:.3f}")
        logger.info("")
    except Exception as e:
        logger.error(f"   ✗ Embedding extraction failed: {e}\n")
        logger.error("   Make sure PyTorch and sentence-transformers are installed")
        sys.exit(1)

    # Test feature store
    logger.info("5. Testing feature store...")
    try:
        with db.get_session() as session:
            store = FeatureStore(session)

            # Combine all features
            all_features = {}
            all_features.update(metadata_features)
            all_features.update(historical_features)
            all_features.update(embedding_features)

            # Save features
            store.save_features("test_features_123", all_features)

        logger.info("   ✓ Features saved to database")

        # Retrieve features
        with db.get_session() as session:
            store = FeatureStore(session)
            retrieved = store.get_features("test_features_123")

        if retrieved:
            logger.info("   ✓ Features retrieved successfully")
            logger.info(f"     - Sender domain: {retrieved.sender_domain}")
            logger.info(f"     - Is newsletter: {retrieved.is_newsletter}")
            logger.info(f"     - Sender open rate: {retrieved.sender_open_rate:.1%}")
            logger.info("")
        else:
            logger.error("   ✗ Failed to retrieve features\n")
            sys.exit(1)

    except Exception as e:
        logger.error(f"   ✗ Feature store failed: {e}\n")
        sys.exit(1)

    # Get feature stats
    logger.info("6. Getting feature statistics...")
    try:
        with db.get_session() as session:
            store = FeatureStore(session)
            stats = store.get_feature_stats()

        logger.info("   ✓ Feature statistics:")
        logger.info(f"     - Total emails: {stats['total_emails']}")
        logger.info(f"     - Features computed: {stats['features_count']}")
        logger.info(f"     - Coverage: {stats['coverage_pct']:.1f}%")
        logger.info("")
    except Exception as e:
        logger.error(f"   ✗ Statistics failed: {e}\n")
        sys.exit(1)

    # Clean up
    logger.info("7. Cleaning up test database...")
    try:
        Path(test_db_path).unlink()
        logger.info("   ✓ Test database deleted\n")
    except Exception as e:
        logger.warning(f"   ! Could not delete test database: {e}\n")

    # Success
    logger.info("=== All Tests Passed ===")
    logger.info("\nFeature extraction is working correctly!")
    logger.info("You can now run: python scripts/build_features.py")


if __name__ == '__main__':
    main()
