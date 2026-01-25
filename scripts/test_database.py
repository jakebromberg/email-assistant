#!/usr/bin/env python3
"""
Test database setup and operations.

Verifies that the database schema is created correctly and
basic operations work.

Usage:
    python scripts/test_database.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database, EmailRepository
from src.database.schema import Email, EmailAction, FeedbackReview
from src.utils import Config, setup_logger


def main():
    """Test database setup."""
    # Set up logging
    logger = setup_logger('test_database', level='INFO')

    logger.info("=== Database Test ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize database
    test_db_path = f"{config.DATA_DIR}/test_emails.db"
    logger.info(f"Creating test database: {test_db_path}")

    db = Database(test_db_path)

    # Test: Create tables
    logger.info("1. Creating tables...")
    try:
        db.create_tables()
        logger.info("   ✓ Tables created successfully\n")
    except Exception as e:
        logger.error(f"   ✗ Failed to create tables: {e}")
        sys.exit(1)

    # Test: Insert test email
    logger.info("2. Inserting test email...")
    try:
        with db.get_session() as session:
            test_email = Email(
                message_id="test123",
                thread_id="thread123",
                from_address="test@example.com",
                from_name="Test Sender",
                to_address="me@example.com",
                subject="Test Email",
                date=datetime.now(),
                snippet="This is a test email",
                body_plain="Test body",
                body_html="<p>Test body</p>",
                labels=["INBOX", "UNREAD"],
                was_read=False,
                was_archived=False,
                collected_at=datetime.utcnow()
            )
            session.add(test_email)

        logger.info("   ✓ Test email inserted\n")
    except Exception as e:
        logger.error(f"   ✗ Failed to insert email: {e}")
        sys.exit(1)

    # Test: Query email
    logger.info("3. Querying email...")
    try:
        with db.get_session() as session:
            repo = EmailRepository(session)
            email = repo.get_by_id("test123")

            if email and email.subject == "Test Email":
                logger.info(f"   ✓ Found email: {email.subject}")
                logger.info(f"     From: {email.from_address}")
                logger.info(f"     Date: {email.date}")
                logger.info(f"     Labels: {email.labels}\n")
            else:
                logger.error("   ✗ Failed to query email")
                sys.exit(1)
    except Exception as e:
        logger.error(f"   ✗ Query failed: {e}")
        sys.exit(1)

    # Test: Record action
    logger.info("4. Recording action...")
    try:
        with db.get_session() as session:
            repo = EmailRepository(session)
            action = repo.record_action(
                message_id="test123",
                action_type="archive",
                source="bot",
                action_data={"reason": "test", "score": 0.95}
            )

        logger.info("   ✓ Action recorded\n")
    except Exception as e:
        logger.error(f"   ✗ Failed to record action: {e}")
        sys.exit(1)

    # Test: Save feedback
    logger.info("5. Saving feedback...")
    try:
        with db.get_session() as session:
            repo = EmailRepository(session)
            feedback = repo.save_feedback(
                message_id="test123",
                decision_correct=True,
                label_correct=True,
                user_comment="Test feedback"
            )

        logger.info("   ✓ Feedback saved\n")
    except Exception as e:
        logger.error(f"   ✗ Failed to save feedback: {e}")
        sys.exit(1)

    # Test: Get statistics
    logger.info("6. Getting database stats...")
    try:
        stats = db.get_stats()
        logger.info(f"   ✓ Database stats:")
        logger.info(f"     - Emails: {stats['emails']}")
        logger.info(f"     - Labels: {stats['email_labels']}")
        logger.info(f"     - Actions: {stats['actions']}")
        logger.info(f"     - Feedback: {stats['feedback_reviews']}\n")
    except Exception as e:
        logger.error(f"   ✗ Failed to get stats: {e}")
        sys.exit(1)

    # Test: Sender stats
    logger.info("7. Getting sender stats...")
    try:
        with db.get_session() as session:
            repo = EmailRepository(session)
            sender_stats = repo.get_sender_stats("test@example.com")

        logger.info(f"   ✓ Sender stats:")
        logger.info(f"     - Total emails: {sender_stats['total_emails']}")
        logger.info(f"     - Read count: {sender_stats['read_count']}")
        logger.info(f"     - Open rate: {sender_stats['open_rate']:.1%}\n")
    except Exception as e:
        logger.error(f"   ✗ Failed to get sender stats: {e}")
        sys.exit(1)

    # Clean up
    logger.info("8. Cleaning up test database...")
    try:
        Path(test_db_path).unlink()
        logger.info("   ✓ Test database deleted\n")
    except Exception as e:
        logger.warning(f"   ! Could not delete test database: {e}\n")

    # Success
    logger.info("=== All Tests Passed ===")
    logger.info("\nDatabase setup is working correctly!")
    logger.info("You can now run: python scripts/export_history.py")


if __name__ == '__main__':
    main()
