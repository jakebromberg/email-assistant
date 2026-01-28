#!/usr/bin/env python3
"""
Daily email triage script.

Automatically triages new emails using trained model, applies labels,
and archives low-priority emails.

Usage:
    python scripts/triage_inbox.py --dry-run
    python scripts/triage_inbox.py --days 1
    python scripts/triage_inbox.py --model models/model_v20240101.txt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.gmail import GmailAuthenticator, GmailClient, GmailOperations
from src.triage import TriagePipeline
from src.utils import Config, setup_logger


def main():
    """Run daily email triage."""
    parser = argparse.ArgumentParser(
        description='Daily email triage script'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (no actual changes)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Days back to process (default: 1)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (default: latest in models/)'
    )
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Skip embeddings (faster but less accurate)'
    )
    parser.add_argument(
        '--max-emails',
        type=int,
        default=None,
        help='Maximum emails to process (default: unlimited)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('triage_inbox', level='INFO')

    if args.dry_run:
        logger.info("=== DRY RUN MODE - No changes will be made ===\n")
    else:
        logger.info("=== Daily Email Triage ===\n")

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
        ops = GmailOperations(client)
        logger.info("Gmail client initialized\n")
    except Exception as e:
        logger.error(f"Failed to initialize Gmail: {e}")
        sys.exit(1)

    # Initialize database
    db_path = f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}")
    db = Database(db_path)

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

        args.model = str(sorted(model_files)[-1])
        logger.info(f"Using latest model: {args.model}")

    logger.info("")

    # Initialize pipeline
    logger.info("Initializing triage pipeline...")
    try:
        pipeline = TriagePipeline(
            gmail_client=client,
            gmail_ops=ops,
            database=db,
            model_path=args.model,
            use_embeddings=not args.no_embeddings
        )
        logger.info("Pipeline initialized\n")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run triage
    logger.info(f"Running triage (last {args.days} day(s))...")
    try:
        results = pipeline.run_triage(
            days_back=args.days,
            dry_run=args.dry_run,
            max_emails=args.max_emails
        )

        logger.info("\n=== Triage Summary ===")
        logger.info(f"Total emails processed: {results['total']}")
        logger.info(f"High confidence keep: {results['high_keep']} ({results['high_keep'] / results['total'] * 100:.1f}%)")
        logger.info(f"High confidence archive: {results['high_archive']} ({results['high_archive'] / results['total'] * 100:.1f}%)")
        logger.info(f"Low confidence: {results['low_confidence']} ({results['low_confidence'] / results['total'] * 100:.1f}%)")

        if results['errors'] > 0:
            logger.warning(f"Errors: {results['errors']}")

        # Show sample decisions
        if results['decisions']:
            logger.info("\nSample decisions:")
            for decision in results['decisions'][:5]:
                logger.info(f"  - {decision['subject'][:50]}")
                logger.info(f"    Score: {decision['score']:.3f}, Action: {decision['action']}, Category: {decision['category']}")

        if args.dry_run:
            logger.info("\nDRY RUN complete - no actual changes made")
            logger.info("Run without --dry-run to apply decisions")
        else:
            logger.info("\nTriage complete!")
            logger.info("Review decisions with: python scripts/review_decisions.py")

    except KeyboardInterrupt:
        logger.warning("\nTriage interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTriage failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
