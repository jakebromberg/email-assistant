#!/usr/bin/env python3
"""
Interactive CLI tool to review and provide feedback on bot decisions.

Presents bot decisions to user for y/n/c/s feedback with natural language
comments support.

Usage:
    python scripts/review_decisions.py
    python scripts/review_decisions.py --days 1
    python scripts/review_decisions.py --filter archived
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database, EmailRepository
from src.database.schema import EmailAction
from src.ml import EmailCategorizer
from src.utils import Config, setup_logger


def main():
    """Interactive feedback review."""
    parser = argparse.ArgumentParser(
        description='Review bot decisions and provide feedback'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Days back to review (default: 1)'
    )
    parser.add_argument(
        '--filter',
        choices=['all', 'archived', 'kept', 'low-confidence'],
        default='all',
        help='Filter decisions to review (default: all)'
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize database
    db_path = f"{config.DATA_DIR}/emails.db"
    db = Database(db_path)

    # Get bot actions to review
    start_date = datetime.now() - timedelta(days=args.days)

    with db.get_session() as session:
        # Query bot actions
        query = session.query(EmailAction).filter(
            EmailAction.source == 'bot',
            EmailAction.timestamp >= start_date
        ).order_by(EmailAction.timestamp.desc())

        # Apply filter
        if args.filter == 'archived':
            query = query.filter(EmailAction.action_type == 'archive')
        elif args.filter == 'kept':
            query = query.filter(EmailAction.action_type == 'keep')
        elif args.filter == 'low-confidence':
            # Check action_data for low confidence
            pass  # Would need JSON query

        actions = query.all()

    if not actions:
        print(f"\nNo bot decisions found in last {args.days} day(s)")
        return

    print(f"\n=== Bot Decision Review ===")
    print(f"Found {len(actions)} decisions to review\n")

    # Review each decision
    reviewed = 0
    correct_count = 0
    corrected_count = 0

    categorizer = EmailCategorizer()

    with db.get_session() as session:
        repo = EmailRepository(session)

        for i, action in enumerate(actions, 1):
            # Get email
            email = repo.get_by_id(action.message_id)
            if not email:
                continue

            # Display email info
            print("=" * 80)
            print(f"\nEmail {i}/{len(actions)}")
            print(f"From: {email.from_name or email.from_address}")
            print(f"Subject: {email.subject}")
            print(f"Date: {email.date.strftime('%Y-%m-%d %H:%M')}")
            print(f"Snippet: {email.snippet[:100]}...")
            print()

            # Display bot decision
            action_type = action.action_type.capitalize()
            score = action.action_data.get('score', 'N/A')
            confidence = action.action_data.get('confidence', 'N/A')
            labels = action.action_data.get('labels', [])

            print(f"Bot Decision: {action_type}")
            print(f"Score: {score}")
            print(f"Confidence: {confidence}")
            print(f"Labels: {', '.join(labels) if labels else 'None'}")
            print()

            # Get feedback
            while True:
                response = input("Decision correct? (y/n/s/c for comment/q to quit): ").lower().strip()

                if response == 'q':
                    print("\nReview ended by user")
                    break

                elif response == 's':
                    # Skip
                    print("Skipped\n")
                    break

                elif response == 'y':
                    # Correct
                    repo.save_feedback(
                        message_id=email.message_id,
                        decision_correct=True,
                        label_correct=True
                    )
                    correct_count += 1
                    reviewed += 1
                    print("✓ Recorded as correct\n")
                    break

                elif response == 'n':
                    # Incorrect - get details
                    print("\nWhat was wrong?")
                    print("1. Should archive (bot kept)")
                    print("2. Should keep (bot archived)")
                    print("3. Wrong label")

                    choice = input("Choice (1/2/3): ").strip()

                    if choice == '1':
                        repo.save_feedback(
                            message_id=email.message_id,
                            decision_correct=False,
                            correct_decision='archive'
                        )
                        corrected_count += 1
                        reviewed += 1
                        print("✓ Feedback recorded\n")
                        break

                    elif choice == '2':
                        repo.save_feedback(
                            message_id=email.message_id,
                            decision_correct=False,
                            correct_decision='keep'
                        )
                        corrected_count += 1
                        reviewed += 1
                        print("✓ Feedback recorded\n")
                        break

                    elif choice == '3':
                        # Show available categories
                        categories = categorizer.get_all_categories()
                        print("\nAvailable categories:")
                        for idx, cat in enumerate(categories, 1):
                            print(f"{idx}. {cat}")

                        cat_choice = input("\nSelect correct category (number or name): ").strip()

                        if cat_choice.isdigit() and 1 <= int(cat_choice) <= len(categories):
                            correct_label = categories[int(cat_choice) - 1]
                        else:
                            correct_label = cat_choice

                        repo.save_feedback(
                            message_id=email.message_id,
                            label_correct=False,
                            correct_label=correct_label
                        )
                        corrected_count += 1
                        reviewed += 1
                        print("✓ Feedback recorded\n")
                        break

                elif response == 'c':
                    # Add comment
                    comment = input("Enter feedback: ").strip()

                    if comment:
                        repo.save_feedback(
                            message_id=email.message_id,
                            user_comment=comment
                        )
                        reviewed += 1
                        print("✓ Comment saved\n")
                        break
                    else:
                        print("No comment entered\n")
                        continue

                else:
                    print("Invalid choice. Use y/n/s/c/q")
                    continue

            if response == 'q':
                break

    # Summary
    print("\n=== Review Summary ===")
    print(f"Total reviewed: {reviewed}")
    print(f"Correct: {correct_count} ({correct_count / reviewed * 100:.1f}%)" if reviewed > 0 else "Correct: 0")
    print(f"Corrected: {corrected_count} ({corrected_count / reviewed * 100:.1f}%)" if reviewed > 0 else "Corrected: 0")
    print("\nFeedback saved to database for model retraining")


if __name__ == '__main__':
    main()
