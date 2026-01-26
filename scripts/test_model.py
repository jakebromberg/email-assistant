#!/usr/bin/env python3
"""
Test trained ML model on sample emails.

Loads a trained model and scores sample emails from the database to
verify predictions look reasonable.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --model models/model_v20240101.txt
    python scripts/test_model.py --limit 20
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database, EmailRepository
from src.features import FeatureStore
from src.ml import EmailScorer, EmailCategorizer
from src.utils import Config, setup_logger


def main():
    """Test trained model."""
    parser = argparse.ArgumentParser(
        description='Test trained ML model on sample emails'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (default: latest in models/)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database file (default: data/emails.db)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of emails to test (default: 10)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('test_model', level='INFO')

    logger.info("=== Model Testing ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Find model if not specified
    if args.model is None:
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("No models directory found. Train a model first.")
            sys.exit(1)

        model_files = list(models_dir.glob("model_v*.txt"))
        if not model_files:
            logger.error("No trained models found. Run scripts/train_model.py first")
            sys.exit(1)

        # Use most recent
        args.model = str(sorted(model_files)[-1])
        logger.info(f"Using latest model: {args.model}")

    # Initialize database
    db_path = args.db_path or f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}\n")
    db = Database(db_path)

    # Initialize scorer and categorizer
    logger.info("Loading model...")
    try:
        scorer = EmailScorer(args.model)
        categorizer = EmailCategorizer()
        logger.info("Model loaded successfully\n")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Get sample emails with features and score them
    logger.info(f"Fetching {args.limit} sample emails...")
    with db.get_session() as session:
        store = FeatureStore(session)
        repo = EmailRepository(session)

        # Get emails with features
        all_features = store.get_all_features(limit=args.limit * 2)

        if not all_features:
            logger.error("No emails with features found. Run scripts/build_features.py first")
            sys.exit(1)

        # Get corresponding emails
        message_ids = [f.message_id for f in all_features[:args.limit]]
        emails = repo.get_by_ids(message_ids)

        # Create (email, features) pairs
        emails_features = [
            (e, f) for e, f in zip(emails, all_features[:args.limit])
            if e is not None
        ]

        if not emails_features:
            logger.error("No emails found")
            sys.exit(1)

        logger.info(f"Testing on {len(emails_features)} emails\n")

        # Score and display results (inside session to access attributes)
        logger.info("=" * 80)
        for i, (email, features) in enumerate(emails_features, 1):
            # Score email
            score = scorer.score_email(email, features)

            # Get decision
            decision = scorer.make_decision(score)

            # Categorize
            category = categorizer.categorize(email, features.is_newsletter)

            # Display
            logger.info(f"\nEmail {i}/{len(emails_features)}")
            logger.info(f"From: {email.from_name or email.from_address}")
            logger.info(f"Subject: {email.subject[:60]}...")
            logger.info(f"Date: {email.date.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Was read: {'Yes' if email.was_read else 'No'}")
            logger.info(f"Was archived: {'Yes' if email.was_archived else 'No'}")
            logger.info("")
            logger.info(f"Model prediction:")
            logger.info(f"  - Score: {score:.3f}")
            logger.info(f"  - Decision: {decision['action']} ({decision['confidence']} confidence)")
            logger.info(f"  - Category: {category}")
            logger.info(f"  - Reasoning: {decision['reasoning']}")

            # Check if prediction matches reality
            if email.was_read and score >= 0.5:
                logger.info(f"  ✓ Correct: Predicted would read, actually read")
            elif not email.was_read and score < 0.5:
                logger.info(f"  ✓ Correct: Predicted wouldn't read, actually didn't read")
            elif email.was_read and score < 0.5:
                logger.info(f"  ✗ Incorrect: Predicted wouldn't read, but actually read")
            else:
                logger.info(f"  ✗ Incorrect: Predicted would read, but actually didn't")

            logger.info("=" * 80)

        # Summary statistics (still inside session)
        logger.info("\n=== Summary ===\n")

        scores = [scorer.score_email(e, f) for e, f in emails_features]
        actual_read = [e.was_read for e, _ in emails_features]

        logger.info(f"Score distribution:")
        logger.info(f"  - Mean: {sum(scores) / len(scores):.3f}")
        logger.info(f"  - Min: {min(scores):.3f}")
        logger.info(f"  - Max: {max(scores):.3f}")
        logger.info("")

        logger.info(f"Decisions:")
        high_keep = sum(1 for s in scores if s >= 0.7)
        high_archive = sum(1 for s in scores if s <= 0.3)
        low_conf = len(scores) - high_keep - high_archive

        logger.info(f"  - High confidence keep: {high_keep} ({high_keep / len(scores) * 100:.1f}%)")
        logger.info(f"  - High confidence archive: {high_archive} ({high_archive / len(scores) * 100:.1f}%)")
        logger.info(f"  - Low confidence: {low_conf} ({low_conf / len(scores) * 100:.1f}%)")
        logger.info("")

        # Accuracy on this sample
        correct = sum(
            1 for s, actual in zip(scores, actual_read)
            if (s >= 0.5 and actual) or (s < 0.5 and not actual)
        )
        logger.info(f"Accuracy on sample: {correct}/{len(scores)} ({correct / len(scores) * 100:.1f}%)")

    logger.info("\nModel testing complete!")


if __name__ == '__main__':
    main()
