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
import html
import os
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


def get_key():
    """Get a keypress, handling arrow keys and special keys."""
    ch = getch()
    if ch == '\x1b':  # Escape sequence
        ch2 = getch()
        if ch2 == '[':
            ch3 = getch()
            if ch3 == 'A':
                return 'UP'
            elif ch3 == 'B':
                return 'DOWN'
            elif ch3 == 'C':
                return 'RIGHT'
            elif ch3 == 'D':
                return 'LEFT'
        return 'ESC'
    elif ch == '\r' or ch == '\n':
        return 'ENTER'
    elif ch == ' ':
        return 'SPACE'
    elif ch == '\x7f' or ch == '\x08':  # Backspace
        return 'BACKSPACE'
    elif ch == '\x03':  # Ctrl+C
        return 'CTRL_C'
    else:
        return ch


class InteractiveLabelSelector:
    """
    Interactive label selector with arrow navigation and checkboxes.

    Features:
    - Arrow up/down to navigate
    - Space to toggle checkbox
    - Type to add new labels
    - Enter on Confirm to accept
    """

    def __init__(self, current_labels: list[str], available_labels: list[str]):
        """
        Initialize selector.

        Args:
            current_labels: Labels currently on the email (pre-checked)
            available_labels: All available label options
        """
        # Build items list: available labels + current labels not in available
        self.items = []
        seen = set()

        # Add available labels first
        for label in available_labels:
            checked = label in current_labels
            self.items.append({'label': label, 'checked': checked, 'is_input': False})
            seen.add(label)

        # Add any current labels not in available list
        for label in current_labels:
            if label not in seen:
                self.items.append({'label': label, 'checked': True, 'is_input': False})

        # Add empty input field
        self.items.append({'label': '', 'checked': False, 'is_input': True})

        # Add confirm option
        self.confirm_index = len(self.items)

        self.cursor = 0
        self.input_buffer = ''
        self.editing_input = False

    def _render(self):
        """Render the selector UI."""
        # Move cursor up to overwrite previous render
        total_lines = len(self.items) + 3  # items + header + confirm + instructions
        print(f"\033[{total_lines}A", end='')  # Move up
        print("\033[J", end='')  # Clear from cursor to end

        print(colored("Select labels (Space=toggle, Enter=confirm):", Colors.BOLD))
        print()

        for i, item in enumerate(self.items):
            is_selected = (i == self.cursor)
            prefix = colored("→ ", Colors.CYAN) if is_selected else "  "

            if item['is_input']:
                # Input field
                checkbox = "[ ]"
                if item['label']:
                    label_text = colored(item['label'], Colors.GREEN)
                    if is_selected and self.editing_input:
                        label_text += colored("_", Colors.YELLOW)  # Cursor
                else:
                    if is_selected:
                        label_text = colored("(type to add new label)", Colors.YELLOW)
                        if self.editing_input:
                            label_text = colored(self.input_buffer + "_", Colors.GREEN)
                    else:
                        label_text = colored("(add new...)", Colors.YELLOW)
                print(f"{prefix}{checkbox} {label_text}")
            else:
                # Regular label
                checkbox = colored("[✓]", Colors.GREEN) if item['checked'] else "[ ]"
                label_text = item['label']
                if is_selected:
                    label_text = colored(label_text, Colors.CYAN + Colors.BOLD)
                print(f"{prefix}{checkbox} {label_text}")

        # Confirm option
        print()
        is_confirm_selected = (self.cursor == self.confirm_index)
        if is_confirm_selected:
            print(colored("→ ", Colors.CYAN) + colored("[ Confirm ]", Colors.GREEN + Colors.BOLD))
        else:
            print("  [ Confirm ]")

    def _initial_render(self):
        """Initial render with space for updates."""
        print(colored("Select labels (Space=toggle, Enter=confirm):", Colors.BOLD))
        print()
        for i, item in enumerate(self.items):
            print()  # Placeholder lines
        print()
        print()  # Confirm line
        # Now render actual content
        self._render()

    def run(self) -> list[str] | None:
        """
        Run the interactive selector.

        Returns:
            List of selected labels, or None if cancelled
        """
        self._initial_render()

        while True:
            key = get_key()

            if key == 'CTRL_C' or key == 'ESC':
                return None

            elif key == 'UP':
                self.editing_input = False
                self.cursor = max(0, self.cursor - 1)

            elif key == 'DOWN':
                self.editing_input = False
                self.cursor = min(self.confirm_index, self.cursor + 1)

            elif key == 'SPACE':
                if self.cursor < len(self.items):
                    item = self.items[self.cursor]
                    if item['is_input']:
                        # Start editing input
                        self.editing_input = True
                    else:
                        # Toggle checkbox
                        item['checked'] = not item['checked']

            elif key == 'ENTER':
                if self.cursor == self.confirm_index:
                    # Confirm selection
                    selected = [item['label'] for item in self.items
                                if item['checked'] and item['label']]
                    return selected
                elif self.cursor < len(self.items):
                    item = self.items[self.cursor]
                    if item['is_input'] and self.editing_input and self.input_buffer:
                        # Finalize new label
                        item['label'] = self.input_buffer
                        item['checked'] = True
                        item['is_input'] = False
                        self.input_buffer = ''
                        # Add new empty input field
                        self.items.append({'label': '', 'checked': False, 'is_input': True})
                        self.confirm_index = len(self.items)
                        self.cursor += 1  # Move to new input field
                        self.editing_input = False

            elif key == 'BACKSPACE':
                if self.editing_input and self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]

            elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                # Typing a character
                if self.cursor < len(self.items):
                    item = self.items[self.cursor]
                    if item['is_input']:
                        self.editing_input = True
                        self.input_buffer += key

            self._render()


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
            snippet = html.unescape(email.snippet[:100]) if email.snippet else 'N/A'
            print(f"{colored('Snippet:', Colors.CYAN)} {snippet}...")
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
                        # Get current labels on the email and available categories
                        current_labels = [lbl for lbl in (email.labels or []) if lbl.startswith('Bot/')]
                        available_categories = categorizer.get_all_categories(session)

                        print()  # Blank line before selector
                        selector = InteractiveLabelSelector(current_labels, available_categories)
                        selected_labels = selector.run()

                        if selected_labels is None:
                            # User cancelled
                            print(colored("Label selection cancelled.", Colors.YELLOW))
                        elif not selected_labels:
                            print(colored("No labels selected, skipping label correction.", Colors.YELLOW))
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
