#!/usr/bin/env python3
"""
Collect implicit feedback from Gmail.

Checks for user actions like moving emails back to inbox,
label changes, etc.

Usage:
    python scripts/collect_feedback.py
    python scripts/collect_feedback.py --days 7
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.gmail import GmailAuthenticator, GmailClient
from src.triage import FeedbackCollector
from src.utils import Config, setup_logger


def main():
    """Collect implicit feedback."""
    parser = argparse.ArgumentParser(
        description='Collect implicit feedback from Gmail'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Days back to check (default: 7)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('collect_feedback', level='INFO')

    logger.info("=== Feedback Collection ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize Gmail client
    logger.info("Initializing Gmail client...")
    try:
        auth = GmailAuthenticator(
            credentials_path=config.GMAIL_CREDENTIALS_PATH,
            token_path=config.GMAIL_TOKEN_PATH
        )
        client = GmailClient(auth)
        logger.info("Gmail client initialized\n")
    except Exception as e:
        logger.error(f"Failed to initialize Gmail: {e}")
        sys.exit(1)

    # Initialize database
    db_path = f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}\n")
    db = Database(db_path)

    # Initialize feedback collector
    collector = FeedbackCollector(client, db)

    # Collect feedback
    logger.info(f"Collecting feedback for last {args.days} days...")
    try:
        feedback = collector.collect_feedback(days_back=args.days)

        logger.info("\n=== Feedback Summary ===")
        logger.info(f"Bot actions checked: {feedback['bot_actions']}")
        logger.info(f"False positives detected: {feedback['false_positives']}")
        logger.info(f"Label changes detected: {feedback['label_changes']}")

        if feedback['false_positives'] > 0:
            logger.warning(
                f"\n⚠ {feedback['false_positives']} false positive(s) detected!"
            )
            logger.warning("Consider retraining the model or adjusting thresholds")

        logger.info("\nFeedback saved to database")
        logger.info("Use this data for model retraining: python scripts/retrain_model.py")

    except Exception as e:
        logger.error(f"Failed to collect feedback: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
