#!/usr/bin/env python3
"""
Interactive CLI tool to review and provide feedback on bot decisions.

Presents bot decisions to user for single-key feedback.

Keyboard shortcuts:
    y - Mark decision as correct
    n - Mark decision as incorrect (prompts for details)
    s - Skip this email
    c - Add a comment
    u - Undo last feedback
    q - Quit review

Usage:
    python scripts/review_decisions.py
    python scripts/review_decisions.py --days 1
    python scripts/review_decisions.py --filter archived
"""

import argparse
import os
import shlex
import sys
import termios
import tty
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database, EmailRepository
from src.database.schema import EmailAction, FeedbackReview
from src.ml import EmailCategorizer
from src.utils import Config


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


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


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
        # Get message_ids that already have feedback
        reviewed_ids = session.query(FeedbackReview.message_id).scalar_subquery()

        # Query bot actions, excluding already reviewed emails
        query = session.query(EmailAction).filter(
            EmailAction.source == 'bot',
            EmailAction.timestamp >= start_date,
            ~EmailAction.message_id.in_(reviewed_ids)
        )

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

    # Sort by confidence: most uncertain (closest to 0.5) first
    def confidence_sort_key(action_data):
        score = action_data['action_data'].get('score')
        if score is None or not isinstance(score, (int, float)):
            return 999  # Put invalid scores at the end
        # Distance from 0.5 (smaller distance = less confident)
        return abs(score - 0.5)

    action_data_list.sort(key=confidence_sort_key)

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
    feedback_history = []  # Track feedback for undo functionality

    with db.get_session() as session:
        repo = EmailRepository(session)
        categorizer = EmailCategorizer(session)

        email_index = 0
        while email_index < len(action_data_list):
            action_data = action_data_list[email_index]
            i = email_index + 1
            # Get email
            email = repo.get_by_id(action_data['message_id'])
            if not email:
                continue

            # Clear screen and display email info
            clear_screen()

            # Progress header
            print(colored(f"Bot Decision Review - Email {i}/{len(action_data_list)}", Colors.CYAN + Colors.BOLD))
            progress_stats = f"Reviewed: {reviewed} | Correct: {correct_count} | Corrected: {corrected_count}"
            print(colored(progress_stats, Colors.YELLOW))
            print(colored("=" * 80, Colors.BLUE))
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
                print(colored("Decision correct? ", Colors.BOLD) + "(y/n/s/c/u/q): ", end='', flush=True)
                response = getch().lower()
                print(response)  # Echo the character

                if response == 'u':
                    # Undo last feedback
                    if not feedback_history:
                        print(colored("Nothing to undo", Colors.YELLOW))
                        import time
                        time.sleep(1)  # Brief pause to show message
                        break  # Refresh current screen

                    last_feedback = feedback_history.pop()

                    # Delete the feedback from database
                    session.query(FeedbackReview).filter(
                        FeedbackReview.message_id == last_feedback['message_id']
                    ).delete()
                    session.commit()

                    # Update counters
                    if last_feedback['action'] == 'correct':
                        correct_count -= 1
                        reviewed -= 1
                    elif last_feedback['action'] in ['incorrect', 'comment']:
                        corrected_count -= 1
                        reviewed -= 1

                    # Go back to that email
                    email_index = last_feedback['email_index']
                    break  # Refresh to show the undone email

                elif response == 'q':
                    print(colored("\nReview ended by user", Colors.YELLOW))
                    break

                elif response == 's':
                    # Skip
                    print(colored("Skipped", Colors.YELLOW))
                    email_index += 1
                    break

                elif response == 'y':
                    # Correct
                    repo.save_feedback(
                        message_id=email.message_id,
                        decision_correct=True,
                        label_correct=True
                    )
                    session.commit()
                    feedback_history.append({
                        'email_index': email_index,
                        'message_id': email.message_id,
                        'action': 'correct'
                    })
                    correct_count += 1
                    reviewed += 1
                    print(colored("✓ Recorded as correct", Colors.GREEN))
                    email_index += 1
                    break

                elif response == 'n':
                    # Incorrect - get details
                    print(colored("\nWhat was wrong?", Colors.BOLD))
                    print(colored("1.", Colors.CYAN) + " Wrong decision (archive vs keep)")
                    print(colored("2.", Colors.CYAN) + " Wrong label")
                    print(colored("Enter one or more (e.g., '1' or '1 2' or '1,2'):", Colors.YELLOW))

                    choice_input = input(colored("\nChoice: ", Colors.BOLD)).strip()

                    # Parse multiple choices (space or comma separated)
                    choices = set()
                    for part in choice_input.replace(',', ' ').split():
                        if part in ['1', '2']:
                            choices.add(part)

                    if not choices:
                        print(colored("Invalid choice. Please enter 1, 2, or both.", Colors.RED))
                        continue

                    # Prepare feedback parameters
                    feedback_params = {'message_id': email.message_id}
                    feedback_messages = []

                    # Handle wrong decision (choice 1)
                    if '1' in choices:
                        bot_decision = action_data['action_type']
                        correct_decision = 'keep' if bot_decision == 'archive' else 'archive'
                        feedback_params['decision_correct'] = False
                        feedback_params['correct_decision'] = correct_decision
                        feedback_messages.append(f"should {correct_decision}")

                    # Handle wrong label (choice 2)
                    if '2' in choices:
                        categories = categorizer.get_all_categories(session)
                        print(colored("\nAvailable categories:", Colors.BOLD))
                        for idx, cat in enumerate(categories, 1):
                            print(f"{colored(str(idx) + '.', Colors.CYAN)} {cat}")

                        print(colored("\nSelect by number or create new labels:", Colors.YELLOW))
                        print(colored("  Examples: '1' or '1 3' or 'Bot/MyNewLabel' or '1 \"Bot/My Label\"'", Colors.YELLOW))
                        cat_input = input(colored("Categories: ", Colors.BOLD)).strip()

                        # Parse multiple category choices (supporting quoted strings for multi-word labels)
                        selected_labels = []
                        try:
                            parts = shlex.split(cat_input.replace(',', ' '))
                        except ValueError:
                            # If shlex fails, fall back to simple split
                            parts = cat_input.replace(',', ' ').split()

                        for part in parts:
                            if part.isdigit():
                                idx = int(part)
                                if 1 <= idx <= len(categories):
                                    selected_labels.append(categories[idx - 1])
                                else:
                                    print(colored(f"Warning: Category number {idx} out of range, skipping", Colors.YELLOW))
                            else:
                                # Allow entering new category name directly
                                selected_labels.append(part)
                                print(colored(f"  → Creating new label: {part}", Colors.CYAN))

                        if not selected_labels:
                            print(colored("No valid categories selected, skipping label correction.", Colors.YELLOW))
                        else:
                            # Store as comma-separated string if multiple, or single string
                            correct_label = ','.join(selected_labels) if len(selected_labels) > 1 else selected_labels[0]
                            feedback_params['label_correct'] = False
                            feedback_params['correct_label'] = correct_label
                            label_display = ', '.join(selected_labels)
                            feedback_messages.append(f"labels: {label_display}")

                    # Save all feedback at once
                    repo.save_feedback(**feedback_params)
                    session.commit()
                    feedback_history.append({
                        'email_index': email_index,
                        'message_id': email.message_id,
                        'action': 'incorrect'
                    })
                    corrected_count += 1
                    reviewed += 1

                    # Display confirmation message
                    message = "✓ Feedback recorded (" + ", ".join(feedback_messages) + ")"
                    print(colored(message, Colors.GREEN))
                    email_index += 1
                    break

                elif response == 'c':
                    # Add comment
                    comment = input(colored("Enter feedback: ", Colors.BOLD)).strip()

                    if comment:
                        repo.save_feedback(
                            message_id=email.message_id,
                            user_comment=comment
                        )
                        session.commit()
                        feedback_history.append({
                            'email_index': email_index,
                            'message_id': email.message_id,
                            'action': 'comment'
                        })
                        reviewed += 1
                        print(colored("✓ Comment saved", Colors.GREEN))
                        email_index += 1
                        break
                    else:
                        print(colored("No comment entered", Colors.YELLOW))
                        continue

                else:
                    print(colored("Invalid choice. Use y/n/s/c/u/q", Colors.RED))
                    continue

            if response == 'q':
                break

        # Final confirmation
        if reviewed > 0:
            clear_screen()
            print(colored("=" * 80, Colors.BLUE))
            print(colored("Review Complete!", Colors.GREEN + Colors.BOLD))
            print(colored("=" * 80, Colors.BLUE))
            print(f"\nYou reviewed {colored(str(reviewed), Colors.BOLD)} decision(s).")
            print(colored("\nPress Enter to finalize and save all feedback...", Colors.BOLD), end='', flush=True)
            input()  # Wait for Enter
            print(colored("✓ All feedback finalized", Colors.GREEN))

    # Summary
    clear_screen()
    print(colored("=== Review Summary ===", Colors.CYAN + Colors.BOLD))
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
