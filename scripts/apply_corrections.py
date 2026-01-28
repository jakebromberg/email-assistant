#!/usr/bin/env python3
"""
Apply feedback corrections to Gmail.

Reads corrections from review_decisions.py feedback and applies them:
- Moves incorrectly archived emails back to inbox
- Fixes incorrect labels

Usage:
    python scripts/apply_corrections.py
    python scripts/apply_corrections.py --dry-run
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmail import GmailAuthenticator, GmailClient, GmailOperations
from src.database import Database, EmailRepository
from src.database.schema import FeedbackReview, EmailAction
from src.utils import Config, setup_logger


def main():
    """Apply feedback corrections to Gmail."""
    parser = argparse.ArgumentParser(
        description='Apply feedback corrections to Gmail'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('apply_corrections', level='INFO')

    logger.info("=== Apply Feedback Corrections ===\n")

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made\n")

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
    logger.info(f"Opening database: {db_path}\n")
    db = Database(db_path)

    # Track statistics
    stats = {
        'total_corrections': 0,
        'moved_to_inbox': 0,
        'labels_fixed': 0,
        'already_applied': 0,
        'errors': 0,
    }

    with db.get_session() as session:
        repo = EmailRepository(session)

        # Get feedback with corrections (decision or label was wrong)
        corrections = session.query(FeedbackReview).filter(
            (FeedbackReview.decision_correct == False) |
            (FeedbackReview.label_correct == False)
        ).all()

        if not corrections:
            logger.info("No corrections to apply")
            return

        logger.info(f"Found {len(corrections)} feedback correction(s) to process\n")

        for feedback in corrections:
            email = repo.get_by_id(feedback.message_id)
            if not email:
                logger.warning(f"Email {feedback.message_id} not found in database, skipping")
                continue

            stats['total_corrections'] += 1
            logger.info(f"Processing: {email.subject[:50]}...")

            # Check if correction was already applied
            existing_correction = session.query(EmailAction).filter(
                EmailAction.message_id == feedback.message_id,
                EmailAction.source == 'correction',
                EmailAction.timestamp >= feedback.review_date
            ).first()

            if existing_correction:
                logger.info("  → Already applied, skipping")
                stats['already_applied'] += 1
                continue

            try:
                # Apply decision correction (move to inbox)
                if feedback.correct_decision == 'keep':
                    logger.info(f"  → Moving to inbox")
                    result = ops.move_to_inbox([feedback.message_id], dry_run=args.dry_run)

                    if result.success:
                        stats['moved_to_inbox'] += 1
                        if not args.dry_run:
                            repo.record_action(
                                message_id=feedback.message_id,
                                action_type='move_to_inbox',
                                source='correction',
                                action_data={'reason': 'feedback_correction'}
                            )

                # Apply label correction
                if feedback.correct_label:
                    # Parse labels (may be comma-separated)
                    correct_labels = [l.strip() for l in feedback.correct_label.split(',')]

                    # Get current bot labels to remove
                    current_bot_labels = [l for l in (email.labels or []) if l.startswith('Bot/')]

                    # Remove old bot labels (except Bot/AutoArchived if we're keeping)
                    labels_to_remove = [l for l in current_bot_labels if l not in correct_labels]

                    if labels_to_remove:
                        logger.info(f"  → Removing labels: {', '.join(labels_to_remove)}")
                        result = ops.remove_labels(
                            [feedback.message_id],
                            labels_to_remove,
                            dry_run=args.dry_run
                        )

                    # Add correct labels
                    labels_to_add = [l for l in correct_labels if l not in current_bot_labels]

                    if labels_to_add:
                        logger.info(f"  → Adding labels: {', '.join(labels_to_add)}")
                        result = ops.add_labels(
                            [feedback.message_id],
                            labels_to_add,
                            dry_run=args.dry_run
                        )

                    if labels_to_remove or labels_to_add:
                        stats['labels_fixed'] += 1
                        if not args.dry_run:
                            repo.record_action(
                                message_id=feedback.message_id,
                                action_type='label_correction',
                                source='correction',
                                action_data={
                                    'removed': labels_to_remove,
                                    'added': labels_to_add,
                                    'reason': 'feedback_correction'
                                }
                            )

                if not args.dry_run:
                    session.commit()

            except Exception as e:
                logger.error(f"  → Error: {e}")
                stats['errors'] += 1
                continue

    # Summary
    logger.info("\n=== Correction Summary ===")
    logger.info(f"Total corrections processed: {stats['total_corrections']}")
    logger.info(f"Moved to inbox: {stats['moved_to_inbox']}")
    logger.info(f"Labels fixed: {stats['labels_fixed']}")
    logger.info(f"Already applied: {stats['already_applied']}")

    if stats['errors'] > 0:
        logger.warning(f"Errors: {stats['errors']}")

    if args.dry_run:
        logger.info("\nDRY RUN - No changes were made")
        logger.info("Run without --dry-run to apply corrections")
    else:
        logger.info("\nCorrections applied successfully")


if __name__ == '__main__':
    main()
