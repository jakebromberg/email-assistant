#!/usr/bin/env python3
"""
Test Gmail API connection and basic operations.

Tests:
- Authentication
- Listing messages
- Fetching message details
- Getting labels

Usage:
    python scripts/test_connection.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmail import GmailAuthenticator, GmailClient, GmailOperations
from src.utils import Config, setup_logger


def main():
    """Test Gmail API connection."""
    # Set up logging
    logger = setup_logger('test_connection', level='INFO')

    logger.info("=== Gmail API Connection Test ===\n")

    # Load configuration
    logger.info("1. Loading configuration...")
    config = Config.load()
    config.validate()
    logger.info("   Configuration loaded successfully\n")

    # Initialize authenticator
    logger.info("2. Initializing authenticator...")
    auth = GmailAuthenticator(
        credentials_path=config.GMAIL_CREDENTIALS_PATH,
        token_path=config.GMAIL_TOKEN_PATH,
        scopes=[GmailAuthenticator.SCOPE_MODIFY]
    )
    logger.info("   Authenticator initialized\n")

    # Authenticate
    logger.info("3. Authenticating with Gmail API...")
    try:
        creds = auth.authenticate()
        logger.info(f"   Authenticated successfully (valid={creds.valid})\n")
    except Exception as e:
        logger.error(f"   Authentication failed: {e}")
        logger.error("   Run scripts/authenticate.py first")
        sys.exit(1)

    # Initialize client
    logger.info("4. Initializing Gmail client...")
    client = GmailClient(auth)
    logger.info("   Gmail client initialized\n")

    # Test: Get labels
    logger.info("5. Fetching Gmail labels...")
    try:
        labels = client.get_all_labels()
        logger.info(f"   Found {len(labels)} labels:")
        for label in labels[:10]:  # Show first 10
            logger.info(f"      - {label['name']} ({label['id']})")
        if len(labels) > 10:
            logger.info(f"      ... and {len(labels) - 10} more")
        logger.info("")
    except Exception as e:
        logger.error(f"   Failed to fetch labels: {e}\n")

    # Test: List recent messages
    logger.info("6. Listing recent messages (last 7 days)...")
    try:
        week_ago = datetime.now() - timedelta(days=7)
        message_ids, next_token = client.list_messages(
            max_results=10,
            after_date=week_ago
        )
        logger.info(f"   Found {len(message_ids)} messages")
        logger.info(f"   Has more: {next_token is not None}\n")

        if message_ids:
            # Test: Fetch message details
            logger.info("7. Fetching details for first message...")
            try:
                email = client.get_message(message_ids[0])
                logger.info(f"   Message ID: {email.message_id}")
                logger.info(f"   From: {email.from_name} <{email.from_address}>")
                logger.info(f"   Subject: {email.subject}")
                logger.info(f"   Date: {email.date}")
                logger.info(f"   Unread: {email.is_unread}")
                logger.info(f"   In inbox: {email.is_in_inbox}")
                logger.info(f"   Labels: {', '.join(email.labels)}")
                logger.info(f"   Snippet: {email.snippet[:100]}...")
                logger.info("")
            except Exception as e:
                logger.error(f"   Failed to fetch message: {e}\n")

            # Test: Batch fetch
            logger.info("8. Testing batch fetch (up to 5 messages)...")
            try:
                batch_ids = message_ids[:5]
                emails = client.get_messages_batch(batch_ids)
                logger.info(f"   Fetched {len(emails)} messages in batch:")
                for email in emails:
                    logger.info(
                        f"      - {email.from_address}: {email.subject[:50]}"
                    )
                logger.info("")
            except Exception as e:
                logger.error(f"   Failed to batch fetch: {e}\n")

            # Test: Operations (dry-run only)
            logger.info("9. Testing operations (DRY RUN)...")
            try:
                ops = GmailOperations(client)

                # Test archive
                result = ops.archive([message_ids[0]], dry_run=True)
                logger.info(f"   Archive test: {result.message}")

                # Test mark read
                result = ops.mark_read([message_ids[0]], dry_run=True)
                logger.info(f"   Mark read test: {result.message}")

                # Test add label
                result = ops.add_labels(
                    [message_ids[0]],
                    ['Test/Label'],
                    dry_run=True
                )
                logger.info(f"   Add label test: {result.message}")
                logger.info("")
            except Exception as e:
                logger.error(f"   Failed to test operations: {e}\n")

        else:
            logger.info("   No messages found in last 7 days\n")

    except Exception as e:
        logger.error(f"   Failed to list messages: {e}\n")

    # Success
    logger.info("=== All Tests Completed Successfully ===")
    logger.info("\nYou can now:")
    logger.info("  - Use the Gmail client in your code")
    logger.info("  - Proceed to Phase 2 (data collection)")


if __name__ == '__main__':
    main()
