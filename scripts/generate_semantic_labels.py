#!/usr/bin/env python3
"""
Generate semantic labels for emails using Claude API.

Sends email subjects and snippets to Claude for intelligent categorization,
then stores the labels for training a local classifier.

Usage:
    python scripts/generate_semantic_labels.py
    python scripts/generate_semantic_labels.py --limit 500
    python scripts/generate_semantic_labels.py --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from src.database import Database, EmailRepository
from src.database.schema import Email, EmailFeatures, SemanticLabel
from src.utils import Config, setup_logger


SYSTEM_PROMPT = """You are an email categorization assistant. Your job is to analyze email subjects and snippets and assign a concise, descriptive category label.

Guidelines:
- Return ONLY the category label, nothing else
- Use 1-3 words in Title-Case separated by hyphens (e.g., "Product-Update", "Security-Alert")
- Be specific but not overly narrow (e.g., "Newsletter-Tech" not "Newsletter-About-AI-From-Company-X")
- Common categories include: Product-Update, Security-Alert, Order-Confirmation, Shipping-Update, Newsletter-Tech, Newsletter-Finance, Newsletter-News, Event-Invitation, Account-Notice, Marketing-Promotion, Social-Notification, Calendar-Invite, Support-Ticket, Job-Alert, Travel-Booking, Subscription-Renewal, Community-Digest, Weekly-Summary, Welcome-Email, Password-Reset

If the email doesn't fit common categories, create an appropriate descriptive label."""

USER_PROMPT_TEMPLATE = """Categorize this email:

From: {sender}
Subject: {subject}
Preview: {snippet}

Reply with ONLY the category label (1-3 words, Title-Case, hyphenated)."""


def generate_label_for_email(
    client: anthropic.Anthropic,
    email: Email,
    model: str = "claude-sonnet-4-20250514"
) -> str | None:
    """
    Generate a semantic label for an email using Claude.

    Args:
        client: Anthropic client
        email: Email to categorize
        model: Model to use

    Returns:
        Generated label or None on error
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        sender=email.from_name or email.from_address,
        subject=email.subject or "(no subject)",
        snippet=(email.snippet or "")[:200]
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=50,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        label = response.content[0].text.strip()
        # Clean up the label
        label = label.strip('"\'')
        # Ensure proper formatting
        label = '-'.join(word.capitalize() for word in label.replace('_', '-').split('-'))

        return label

    except anthropic.APIError as e:
        return None


def main():
    """Generate semantic labels for emails."""
    parser = argparse.ArgumentParser(
        description='Generate semantic labels using Claude API'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=500,
        help='Maximum number of emails to process (default: 500)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-20250514',
        help='Claude model to use (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without calling API'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip emails that already have semantic labels (default: True)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('generate_semantic_labels', level='INFO')

    logger.info("=== Generate Semantic Labels ===\n")

    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Check for Anthropic API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key and not args.dry_run:
        logger.error("ANTHROPIC_API_KEY not found in environment or .env file")
        logger.error("Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Initialize Anthropic client
    if not args.dry_run:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Using model: {args.model}")
    else:
        client = None

    # Initialize database
    db_path = f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}\n")
    db = Database(db_path)

    # Ensure semantic_labels table exists
    from src.database.schema import Base
    Base.metadata.create_all(db.engine)

    # Track statistics
    stats = {
        'processed': 0,
        'labeled': 0,
        'skipped': 0,
        'errors': 0,
    }

    with db.get_session() as session:
        repo = EmailRepository(session)

        # Get emails to process
        query = session.query(Email).join(
            EmailFeatures, Email.message_id == EmailFeatures.message_id
        )

        if args.skip_existing:
            # Exclude emails that already have semantic labels
            existing_ids = session.query(SemanticLabel.message_id).scalar_subquery()
            query = query.filter(~Email.message_id.in_(existing_ids))

        # Order by date descending to get recent emails first
        query = query.order_by(Email.date.desc()).limit(args.limit)

        emails = query.all()

        if not emails:
            logger.info("No emails to process (all may already have labels)")
            return

        logger.info(f"Processing {len(emails)} emails...\n")

        # Process each email
        for i, email in enumerate(emails, 1):
            stats['processed'] += 1

            # Progress indicator
            if i % 10 == 0 or i == 1:
                logger.info(f"Progress: {i}/{len(emails)}")

            if args.dry_run:
                logger.info(f"  [{i}] Subject: {email.subject[:50]}...")
                logger.info(f"       From: {email.from_address}")
                stats['labeled'] += 1
                continue

            # Generate label
            label = generate_label_for_email(client, email, args.model)

            if label:
                # Save to database
                semantic_label = SemanticLabel(
                    message_id=email.message_id,
                    label=label,
                    source='claude',
                    model=args.model
                )
                session.add(semantic_label)
                session.commit()

                stats['labeled'] += 1
                logger.debug(f"  {email.subject[:40]}... -> {label}")
            else:
                stats['errors'] += 1
                logger.warning(f"  Failed to label: {email.subject[:50]}...")

            # Rate limiting - be nice to the API
            time.sleep(0.1)

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Labeled: {stats['labeled']}")
    logger.info(f"Errors: {stats['errors']}")

    if args.dry_run:
        logger.info("\nDRY RUN - No API calls were made")
        logger.info("Run without --dry-run to send these emails to Claude for labeling")
        logger.info("Claude will return labels like 'Product-Update', 'Job-Alert', etc.")
    else:
        logger.info("\nLabels saved to database")
        logger.info("Run scripts/train_semantic_classifier.py to train local model")


if __name__ == '__main__':
    main()
