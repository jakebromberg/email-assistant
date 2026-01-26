#!/usr/bin/env python3
"""
Interactive CLI tool to review and provide feedback on bot decisions.

Presents bot decisions to user for single-key feedback.

Keyboard shortcuts:
    y - Mark decision as correct
    n - Mark decision as incorrect (prompts for details)
    s - Skip this email
    c - Add a comment
    q - Quit review

Usage:
    python scripts/review_decisions.py
    python scripts/review_decisions.py --days 1
    python scripts/review_decisions.py --filter archived
"""

import sys
import tty
import termios
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database, EmailRepository
from src.database.schema import EmailAction
from src.ml import EmailCategorizer
from src.utils import Config, setup_logger


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'      # Magenta
    BLUE = '\033[94m'        # Blue
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'         # Bold
    UNDERLINE = '\033[4m'    # Underline
    END = '\033[0m'          # Reset


def colored(text, color):
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.END}"


def format_score(score):
    """Format score with color based on value."""
    if isinstance(score, (int, float)):
        if score >= 0.7:
            return colored(f"{score:.3f}", Colors.GREEN)
        elif score <= 0.3:
            return colored(f"{score:.3f}", Colors.RED)
        else:
            return colored(f"{score:.3f}", Colors.YELLOW)
    return score


def format_confidence(confidence):
    """Format confidence with color."""
    if confidence == 'high':
        return colored(confidence, Colors.GREEN)
    elif confidence == 'low':
        return colored(confidence, Colors.YELLOW)
    return confidence


def format_action(action_type):
    """Format action type with color."""
    if action_type.lower() == 'archive':
        return colored(action_type, Colors.RED)
    elif action_type.lower() == 'keep':
        return colored(action_type, Colors.GREEN)
    return action_type


def getch():
    """Get a single character from user without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


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

    # Get bot actions to review (extract data before session closes)
    action_data_list = []

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

        # Extract data while session is active
        for action in actions:
            action_data_list.append({
                'message_id': action.message_id,
                'action_type': action.action_type,
                'timestamp': action.timestamp,
                'action_data': action.action_data,
            })

    if not action_data_list:
        print(colored(f"\nNo bot decisions found in last {args.days} day(s)", Colors.YELLOW))
        return

    print(colored(f"\n{'=' * 30}", Colors.CYAN))
    print(colored("   Bot Decision Review", Colors.CYAN + Colors.BOLD))
    print(colored(f"{'=' * 30}\n", Colors.CYAN))
    print(f"Found {colored(str(len(action_data_list)), Colors.BOLD)} decisions to review\n")

    # Review each decision
    reviewed = 0
    correct_count = 0
    corrected_count = 0

    categorizer = EmailCategorizer()

    with db.get_session() as session:
        repo = EmailRepository(session)

        for i, action_data in enumerate(action_data_list, 1):
            # Get email
            email = repo.get_by_id(action_data['message_id'])
            if not email:
                continue

            # Display email info
            print(colored("=" * 80, Colors.BLUE))
            print(colored(f"\nEmail {i}/{len(action_data_list)}", Colors.BOLD))
            print(f"{colored('From:', Colors.CYAN)} {email.from_name or email.from_address}")
            print(f"{colored('Subject:', Colors.CYAN)} {email.subject}")
            print(f"{colored('Date:', Colors.CYAN)} {email.date.strftime('%Y-%m-%d %H:%M')}")
            print(f"{colored('Snippet:', Colors.CYAN)} {email.snippet[:100] if email.snippet else 'N/A'}...")
            print()

            # Display bot decision
            action_type = action_data['action_type'].capitalize()
            score = action_data['action_data'].get('score', 'N/A')
            confidence = action_data['action_data'].get('confidence', 'N/A')
            labels = action_data['action_data'].get('labels', [])

            print(f"{colored('Bot Decision:', Colors.BOLD)} {format_action(action_type)}")
            print(f"{colored('Score:', Colors.BOLD)} {format_score(score)}")
            print(f"{colored('Confidence:', Colors.BOLD)} {format_confidence(confidence)}")
            print(f"{colored('Labels:', Colors.BOLD)} {', '.join(labels) if labels else colored('None', Colors.YELLOW)}")
            print()

            # Get feedback
            while True:
                print(colored("Decision correct? ", Colors.BOLD) + "(y/n/s/c/q): ", end='', flush=True)
                response = getch().lower()
                print(response)  # Echo the character

                if response == 'q':
                    print(colored("\nReview ended by user", Colors.YELLOW))
                    break

                elif response == 's':
                    # Skip
                    print(colored("Skipped", Colors.YELLOW) + "\n")
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
                    print(colored("✓ Recorded as correct", Colors.GREEN) + "\n")
                    break

                elif response == 'n':
                    # Incorrect - get details
                    print(colored("\nWhat was wrong?", Colors.BOLD))
                    print(colored("1.", Colors.CYAN) + " Should archive (bot kept)")
                    print(colored("2.", Colors.CYAN) + " Should keep (bot archived)")
                    print(colored("3.", Colors.CYAN) + " Wrong label")

                    choice = input(colored("\nChoice (1/2/3): ", Colors.BOLD)).strip()

                    if choice == '1':
                        repo.save_feedback(
                            message_id=email.message_id,
                            decision_correct=False,
                            correct_decision='archive'
                        )
                        corrected_count += 1
                        reviewed += 1
                        print(colored("✓ Feedback recorded", Colors.GREEN) + "\n")
                        break

                    elif choice == '2':
                        repo.save_feedback(
                            message_id=email.message_id,
                            decision_correct=False,
                            correct_decision='keep'
                        )
                        corrected_count += 1
                        reviewed += 1
                        print(colored("✓ Feedback recorded", Colors.GREEN) + "\n")
                        break

                    elif choice == '3':
                        # Show available categories
                        categories = categorizer.get_all_categories()
                        print(colored("\nAvailable categories:", Colors.BOLD))
                        for idx, cat in enumerate(categories, 1):
                            print(f"{colored(str(idx) + '.', Colors.CYAN)} {cat}")

                        cat_choice = input(colored("\nSelect correct category (number or name): ", Colors.BOLD)).strip()

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
                        print(colored("✓ Feedback recorded", Colors.GREEN) + "\n")
                        break

                elif response == 'c':
                    # Add comment
                    comment = input(colored("Enter feedback: ", Colors.BOLD)).strip()

                    if comment:
                        repo.save_feedback(
                            message_id=email.message_id,
                            user_comment=comment
                        )
                        reviewed += 1
                        print(colored("✓ Comment saved", Colors.GREEN) + "\n")
                        break
                    else:
                        print(colored("No comment entered", Colors.YELLOW) + "\n")
                        continue

                else:
                    print(colored("Invalid choice. Use y/n/s/c/q", Colors.RED))
                    continue

            if response == 'q':
                break

    # Summary
    print(colored("\n=== Review Summary ===", Colors.CYAN + Colors.BOLD))
    print(f"{colored('Total reviewed:', Colors.BOLD)} {colored(str(reviewed), Colors.CYAN)}")

    if reviewed > 0:
        correct_pct = correct_count / reviewed * 100
        corrected_pct = corrected_count / reviewed * 100

        print(f"{colored('Correct:', Colors.BOLD)} {colored(str(correct_count), Colors.GREEN)} ({correct_pct:.1f}%)")
        print(f"{colored('Corrected:', Colors.BOLD)} {colored(str(corrected_count), Colors.YELLOW)} ({corrected_pct:.1f}%)")
    else:
        print(f"{colored('Correct:', Colors.BOLD)} 0")
        print(f"{colored('Corrected:', Colors.BOLD)} 0")

    print(colored("\n✓ Feedback saved to database for model retraining", Colors.GREEN))


if __name__ == '__main__':
    main()
