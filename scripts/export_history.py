#!/usr/bin/env python3
"""
Export historical emails from Gmail to local database.

This script exports emails from the last N months to the SQLite database
for training the ML model.

Usage:
    python scripts/export_history.py --months 6
    python scripts/export_history.py --months 3 --max 5000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors import EmailCollector
from src.database import Database
from src.gmail import GmailAuthenticator, GmailClient
from src.utils import Config, setup_logger


def main():
    """Export historical emails to database."""
    parser = argparse.ArgumentParser(
        description='Export historical emails from Gmail to database'
    )
    parser.add_argument(
        '--months',
        type=int,
        default=6,
        help='Number of months to go back (default: 6)'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum number of emails to export (default: unlimited)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of message IDs to process per iteration. The client will internally batch these with rate-aware sizing (default: 100)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database file (default: data/emails.db)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('export_history', level='INFO')

    logger.info("=== Historical Email Export ===\n")

    # Load configuration
    logger.info("Loading configuration...")
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
        logger.error(f"Failed to initialize Gmail client: {e}")
        logger.error("Run scripts/authenticate.py first")
        sys.exit(1)

    # Initialize database
    db_path = args.db_path or f"{config.DATA_DIR}/emails.db"
    logger.info(f"Initializing database: {db_path}")
    db = Database(db_path)
    db.create_tables()
    logger.info("Database initialized\n")

    # Check existing data
    stats = db.get_stats()
    logger.info("Current database stats:")
    logger.info(f"  - Emails: {stats['emails']}")
    logger.info(f"  - Labels: {stats['email_labels']}")
    logger.info(f"  - Actions: {stats['actions']}")
    logger.info(f"  - Feedback: {stats['feedback_reviews']}\n")

    if stats['emails'] > 0:
        response = input("Database already contains emails. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Export cancelled")
            sys.exit(0)

    # Initialize collector
    logger.info("Starting export...")
    collector = EmailCollector(client, db)

    # Export emails
    try:
        count = collector.export_historical(
            months=args.months,
            max_emails=args.max,
            batch_size=args.batch_size
        )

        logger.info("\n=== Export Complete ===")
        logger.info(f"Exported {count} emails")

        # Show updated stats
        stats = db.get_stats()
        logger.info("\nFinal database stats:")
        logger.info(f"  - Total emails: {stats['emails']}")
        logger.info(f"  - Total labels: {stats['email_labels']}")
        logger.info(f"  - Total actions: {stats['actions']}")

        # Vacuum database to optimize
        logger.info("\nOptimizing database...")
        db.vacuum()
        logger.info("Database optimized")

        logger.info("\nNext steps:")
        logger.info("  1. Verify data quality: sqlite3 data/emails.db 'SELECT COUNT(*) FROM emails;'")
        logger.info("  2. Proceed to Phase 3: Feature engineering")

    except KeyboardInterrupt:
        logger.warning("\nExport interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nExport failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
