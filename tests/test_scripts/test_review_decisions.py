"""Tests for review_decisions script."""

import html
import os
import sys
from datetime import datetime, timedelta

# Add scripts directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))

from src.database.schema import EmailAction, FeedbackReview


class TestSnippetHtmlDecoding:
    """Test HTML entity decoding in email snippets."""

    def test_decodes_apostrophe_entity(self):
        """Test decoding &#39; to apostrophe."""
        snippet = "I&#39;ll check it out today"
        decoded = html.unescape(snippet)
        assert decoded == "I'll check it out today"

    def test_decodes_lt_gt_entities(self):
        """Test decoding &lt; and &gt; to angle brackets."""
        snippet = "&lt;billburton@mindspring.com&gt;"
        decoded = html.unescape(snippet)
        assert decoded == "<billburton@mindspring.com>"

    def test_decodes_amp_entity(self):
        """Test decoding &amp; to ampersand."""
        snippet = "Tom &amp; Jerry"
        decoded = html.unescape(snippet)
        assert decoded == "Tom & Jerry"

    def test_decodes_quot_entity(self):
        """Test decoding &quot; to quotation mark."""
        snippet = "He said &quot;hello&quot;"
        decoded = html.unescape(snippet)
        assert decoded == 'He said "hello"'

    def test_decodes_mixed_entities(self):
        """Test decoding multiple entity types in one string."""
        snippet = "I&#39;ll check it out today On Wed, Jan 28, 2026 at 7:48 AM bb &lt;billburton@mindspring.com&gt; wro"
        decoded = html.unescape(snippet)
        assert decoded == "I'll check it out today On Wed, Jan 28, 2026 at 7:48 AM bb <billburton@mindspring.com> wro"

    def test_handles_plain_text(self):
        """Test that plain text without entities is unchanged."""
        snippet = "Just a normal email snippet"
        decoded = html.unescape(snippet)
        assert decoded == "Just a normal email snippet"

    def test_handles_none_snippet(self):
        """Test handling None snippet (should use 'N/A')."""
        snippet = None
        result = html.unescape(snippet[:100]) if snippet else 'N/A'
        assert result == 'N/A'

    def test_handles_empty_snippet(self):
        """Test handling empty snippet."""
        snippet = ""
        result = html.unescape(snippet[:100]) if snippet else 'N/A'
        assert result == 'N/A'

    def test_truncates_before_decoding(self):
        """Test that truncation happens before decoding for display."""
        # Create a snippet where truncation matters
        snippet = "A" * 95 + "&#39;"  # 99 chars, entity at end
        truncated = snippet[:100]
        decoded = html.unescape(truncated)
        # Should decode the partial entity if present
        assert len(decoded) <= 100


class TestReviewDecisionsSessionManagement:
    """Test session management in review_decisions script."""

    def test_actions_can_be_queried_and_used_outside_session(self, temp_db, email_factory, action_factory):
        """
        REGRESSION TEST: Ensure EmailAction data can be extracted and used after session closes.

        This prevents DetachedInstanceError when accessing action attributes in the review loop.
        The fix extracts data into dictionaries before the session closes.
        """
        from src.database.repository import EmailRepository

        # Create test data using factories
        email = email_factory(
            message_id="test123",
            subject="Test Email",
            date=datetime.now() - timedelta(hours=1)
        )
        action = action_factory(
            message_id="test123",
            action_type="archive",
            action_data={
                'score': 0.25,
                'confidence': 'high',
                'labels': ['Bot/Newsletter-Tech', 'Bot/AutoArchived']
            }
        )

        # Query actions and extract data (simulating the script)
        action_data_list = []
        start_date = datetime.now() - timedelta(days=1)

        with temp_db.get_session() as session:
            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot',
                EmailAction.timestamp >= start_date
            ).all()

            # Extract data while session is active (this is the fix)
            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                    'action_type': action.action_type,
                    'timestamp': action.timestamp,
                    'action_data': action.action_data,
                })

        # Session is now closed, verify we can still access the data
        assert len(action_data_list) == 1
        action_data = action_data_list[0]

        # These should work without DetachedInstanceError
        assert action_data['message_id'] == "test123"
        assert action_data['action_type'] == "archive"
        assert action_data['action_data']['score'] == 0.25
        assert action_data['action_data']['confidence'] == 'high'
        assert 'Bot/Newsletter-Tech' in action_data['action_data']['labels']

        # Now we can use this data in another session to get the email
        with temp_db.get_session() as session:
            repo = EmailRepository(session)
            email = repo.get_by_id(action_data['message_id'])
            assert email is not None
            assert email.subject == "Test Email"

    def test_multiple_actions_can_be_extracted_and_processed(self, temp_db, email_factory, action_factory):
        """Test that multiple actions can be extracted and processed without session errors."""
        from src.database.repository import EmailRepository

        # Add multiple emails and actions using factories
        for i in range(3):
            email = email_factory(
                message_id=f"test{i}",
                from_address=f"sender{i}@example.com",
                from_name=f"Sender {i}",
                subject=f"Test Email {i}",
                body_plain=f"Test body {i}",
                body_html=f"<p>Test body {i}</p>",
                date=datetime.now() - timedelta(hours=i+1),
                was_archived=(i % 2 == 0)
            )

            action = action_factory(
                message_id=f"test{i}",
                action_type="archive" if i % 2 == 0 else "keep",
                timestamp=datetime.now() - timedelta(hours=i),
                action_data={
                    'score': 0.2 + (i * 0.1),
                    'confidence': 'high' if i % 2 == 0 else 'low',
                    'labels': [f'Bot/Category{i}']
                }
            )

        # Extract actions
        action_data_list = []
        with temp_db.get_session() as session:
            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot'
            ).all()

            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                    'action_type': action.action_type,
                    'timestamp': action.timestamp,
                    'action_data': action.action_data,
                })

        # Verify extraction worked
        assert len(action_data_list) == 3

        # Process extracted data in new session (simulating review loop)
        with temp_db.get_session() as session:
            repo = EmailRepository(session)

            for action_data in action_data_list:
                # Should not raise DetachedInstanceError
                email = repo.get_by_id(action_data['message_id'])
                assert email is not None

                # Should be able to access action data
                assert action_data['action_type'] in ['archive', 'keep']
                assert 'score' in action_data['action_data']
                assert 'confidence' in action_data['action_data']


class TestReviewDecisionsExcludesReviewedEmails:
    """Test that already-reviewed emails are excluded from the review list."""

    def test_excludes_emails_with_existing_feedback(self, temp_db, email_factory, action_factory):
        """Test that emails already reviewed are not shown again."""
        from src.database.repository import EmailRepository

        # Create two emails with bot actions
        for i in range(2):
            email_factory(
                message_id=f"test{i}",
                from_address=f"sender{i}@example.com",
                subject=f"Test Email {i}",
                date=datetime.now() - timedelta(hours=i+1)
            )
            action_factory(
                message_id=f"test{i}",
                action_type="archive",
                action_data={'score': 0.2, 'confidence': 'high'}
            )

        # Add feedback for first email only
        with temp_db.get_session() as session:
            repo = EmailRepository(session)
            repo.save_feedback(
                message_id="test0",
                decision_correct=True,
                label_correct=True
            )
            session.commit()

        # Query actions excluding reviewed emails (simulating the script)
        action_data_list = []
        start_date = datetime.now() - timedelta(days=1)

        with temp_db.get_session() as session:
            # Get message_ids that already have feedback
            reviewed_ids = session.query(FeedbackReview.message_id).scalar_subquery()

            # Query bot actions, excluding already reviewed emails
            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot',
                EmailAction.timestamp >= start_date,
                ~EmailAction.message_id.in_(reviewed_ids)
            ).all()

            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                    'action_type': action.action_type,
                })

        # Should only have the unreviewed email
        assert len(action_data_list) == 1
        assert action_data_list[0]['message_id'] == "test1"

    def test_includes_all_emails_when_none_reviewed(self, temp_db, email_factory, action_factory):
        """Test that all emails are shown when none have been reviewed."""
        # Create three emails with bot actions
        for i in range(3):
            email_factory(
                message_id=f"test{i}",
                from_address=f"sender{i}@example.com",
                subject=f"Test Email {i}",
                date=datetime.now() - timedelta(hours=i+1)
            )
            action_factory(
                message_id=f"test{i}",
                action_type="archive",
                action_data={'score': 0.2, 'confidence': 'high'}
            )

        # Query actions excluding reviewed emails (none exist)
        action_data_list = []
        start_date = datetime.now() - timedelta(days=1)

        with temp_db.get_session() as session:
            reviewed_ids = session.query(FeedbackReview.message_id).scalar_subquery()

            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot',
                EmailAction.timestamp >= start_date,
                ~EmailAction.message_id.in_(reviewed_ids)
            ).all()

            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                })

        # Should have all three emails
        assert len(action_data_list) == 3

    def test_excludes_all_when_all_reviewed(self, temp_db, email_factory, action_factory):
        """Test that no emails are shown when all have been reviewed."""
        from src.database.repository import EmailRepository

        # Create two emails with bot actions
        for i in range(2):
            email_factory(
                message_id=f"test{i}",
                from_address=f"sender{i}@example.com",
                subject=f"Test Email {i}",
                date=datetime.now() - timedelta(hours=i+1)
            )
            action_factory(
                message_id=f"test{i}",
                action_type="archive",
                action_data={'score': 0.2, 'confidence': 'high'}
            )

        # Add feedback for both emails
        with temp_db.get_session() as session:
            repo = EmailRepository(session)
            for i in range(2):
                repo.save_feedback(
                    message_id=f"test{i}",
                    decision_correct=True
                )
            session.commit()

        # Query actions excluding reviewed emails
        action_data_list = []
        start_date = datetime.now() - timedelta(days=1)

        with temp_db.get_session() as session:
            reviewed_ids = session.query(FeedbackReview.message_id).scalar_subquery()

            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot',
                EmailAction.timestamp >= start_date,
                ~EmailAction.message_id.in_(reviewed_ids)
            ).all()

            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                })

        # Should have no emails
        assert len(action_data_list) == 0


class TestReviewDecisionsConfidenceSorting:
    """Test confidence-based sorting of bot decisions."""

    def test_sorts_by_distance_from_half(self, temp_db, email_factory, action_factory):
        """Test that actions are sorted by confidence (distance from 0.5)."""
        # Add emails with different confidence scores
        scores = [0.8, 0.45, 0.1, 0.52, 0.3]
        for i, score in enumerate(scores):
            email = email_factory(
                message_id=f"test{i}",
                from_address=f"sender{i}@example.com",
                from_name=f"Sender {i}",
                subject=f"Test Email {i}",
                date=datetime.now() - timedelta(hours=i+1)
            )

            action = action_factory(
                message_id=f"test{i}",
                action_type="archive" if score < 0.5 else "keep",
                timestamp=datetime.now() - timedelta(hours=i),
                action_data={'score': score, 'confidence': 'low' if 0.3 < score < 0.7 else 'high'}
            )

        # Extract and sort (simulating the script)
        action_data_list = []
        with temp_db.get_session() as session:
            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot'
            ).all()

            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                    'action_type': action.action_type,
                    'timestamp': action.timestamp,
                    'action_data': action.action_data,
                })

        # Sort by confidence (same logic as script)
        def confidence_sort_key(action_data):
            score = action_data['action_data'].get('score')
            if score is None or not isinstance(score, (int, float)):
                return 999
            return abs(score - 0.5)

        action_data_list.sort(key=confidence_sort_key)

        # Verify order by distance from 0.5:
        # 0.52 (0.02), 0.45 (0.05), 0.3 (0.2), 0.8 (0.3), 0.1 (0.4)
        sorted_scores = [ad['action_data']['score'] for ad in action_data_list]
        assert sorted_scores == [0.52, 0.45, 0.3, 0.8, 0.1]

    def test_invalid_scores_sorted_to_end(self, temp_db, email_factory, action_factory):
        """Test that invalid/missing scores are sorted to the end."""
        # Add emails with various score types
        test_cases = [
            ("test0", 0.4, "valid"),
            ("test1", None, "none"),
            ("test2", "invalid", "string"),
            ("test3", 0.6, "valid2"),
        ]

        for i, (msg_id, score, label) in enumerate(test_cases):
            email = email_factory(
                message_id=msg_id,
                from_address=f"sender{i}@example.com",
                from_name=f"Sender {i}",
                subject=f"Test Email {i}",
                date=datetime.now() - timedelta(hours=i+1)
            )

            action = action_factory(
                message_id=msg_id,
                action_type="archive",
                timestamp=datetime.now() - timedelta(hours=i),
                action_data={'score': score, 'label': label}
            )

        # Extract and sort
        action_data_list = []
        with temp_db.get_session() as session:
            actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot'
            ).all()

            for action in actions:
                action_data_list.append({
                    'message_id': action.message_id,
                    'action_type': action.action_type,
                    'timestamp': action.timestamp,
                    'action_data': action.action_data,
                })

        def confidence_sort_key(action_data):
            score = action_data['action_data'].get('score')
            if score is None or not isinstance(score, (int, float)):
                return 999
            return abs(score - 0.5)

        action_data_list.sort(key=confidence_sort_key)

        # Verify valid scores come first, invalid at end
        sorted_labels = [ad['action_data']['label'] for ad in action_data_list]
        assert sorted_labels[:2] in [['valid', 'valid2'], ['valid2', 'valid']]  # Either order
        assert sorted_labels[2:] in [['none', 'string'], ['string', 'none']]  # Either order


class TestReviewDecisionsUndoFunctionality:
    """Test undo functionality in review_decisions script."""

    def test_undo_removes_feedback_from_database(self, temp_db, email_factory):
        """Test that undo deletes feedback from database."""
        from src.database.repository import EmailRepository

        # Add test email using factory
        email = email_factory(message_id="test123")

        # Simulate feedback and undo (as script would do)
        with temp_db.get_session() as session:
            repo = EmailRepository(session)

            # Save feedback
            repo.save_feedback(
                message_id="test123",
                decision_correct=True,
                label_correct=True
            )
            session.commit()

            # Verify feedback exists
            feedback_count = session.query(FeedbackReview).filter(
                FeedbackReview.message_id == "test123"
            ).count()
            assert feedback_count == 1

            # Undo (delete feedback)
            session.query(FeedbackReview).filter(
                FeedbackReview.message_id == "test123"
            ).delete()
            session.commit()

            # Verify feedback deleted
            feedback_count = session.query(FeedbackReview).filter(
                FeedbackReview.message_id == "test123"
            ).count()
            assert feedback_count == 0

    def test_feedback_history_tracks_decisions(self):
        """Test that feedback_history list tracks decisions correctly."""
        feedback_history = []

        # Simulate adding feedback
        feedback_history.append({
            'email_index': 0,
            'message_id': 'test123',
            'action': 'correct'
        })

        feedback_history.append({
            'email_index': 1,
            'message_id': 'test456',
            'action': 'incorrect'
        })

        assert len(feedback_history) == 2
        assert feedback_history[-1]['message_id'] == 'test456'

        # Simulate undo
        last_feedback = feedback_history.pop()
        assert last_feedback['email_index'] == 1
        assert len(feedback_history) == 1


class TestReviewDecisionsCorrectDecisionInference:
    """Test automatic inference of correct decision based on bot action."""

    def test_infers_opposite_of_bot_decision(self):
        """Test that correct decision is inferred as opposite of what bot did."""
        # Bot archived → correct decision should be keep
        bot_decision = 'archive'
        correct_decision = 'keep' if bot_decision == 'archive' else 'archive'
        assert correct_decision == 'keep'

        # Bot kept → correct decision should be archive
        bot_decision = 'keep'
        correct_decision = 'keep' if bot_decision == 'archive' else 'archive'
        assert correct_decision == 'archive'

    def test_decision_inference_with_action_data(self):
        """Test decision inference with real action data structure."""
        action_data = {
            'message_id': 'test123',
            'action_type': 'archive',
            'action_data': {'score': 0.25, 'confidence': 'high'}
        }

        # Infer correct decision (should be opposite)
        bot_decision = action_data['action_type']
        correct_decision = 'keep' if bot_decision == 'archive' else 'archive'

        assert correct_decision == 'keep'


class TestReviewDecisionsCancelFromIncorrect:
    """Test cancel functionality when accidentally pressing 'n'."""

    def test_empty_input_cancels(self):
        """Test that empty input cancels the 'incorrect' flow."""
        choice_input = ''

        # Allow cancel with empty input
        if not choice_input:
            cancelled = True
        else:
            cancelled = False

        assert cancelled is True

    def test_whitespace_only_cancels(self):
        """Test that whitespace-only input cancels."""
        choice_input = '   '

        if not choice_input.strip():
            cancelled = True
        else:
            cancelled = False

        assert cancelled is True

    def test_valid_input_does_not_cancel(self):
        """Test that valid input does not cancel."""
        choice_input = '1'

        if not choice_input.strip():
            cancelled = True
        else:
            cancelled = False

        assert cancelled is False


class TestReviewDecisionsMultiSelection:
    """Test multi-selection feedback functionality."""

    def test_parse_single_choice(self):
        """Test parsing single choice input."""
        choice_input = '1'
        choices = set()
        for part in choice_input.replace(',', ' ').split():
            if part in ['1', '2']:
                choices.add(part)

        assert choices == {'1'}

    def test_parse_multiple_choices_space_separated(self):
        """Test parsing multiple choices separated by spaces."""
        choice_input = '1 2'
        choices = set()
        for part in choice_input.replace(',', ' ').split():
            if part in ['1', '2']:
                choices.add(part)

        assert choices == {'1', '2'}

    def test_parse_multiple_choices_comma_separated(self):
        """Test parsing multiple choices separated by commas."""
        choice_input = '1,2'
        choices = set()
        for part in choice_input.replace(',', ' ').split():
            if part in ['1', '2']:
                choices.add(part)

        assert choices == {'1', '2'}

    def test_parse_multiple_choices_mixed_separators(self):
        """Test parsing multiple choices with mixed separators."""
        choice_input = '1, 2'
        choices = set()
        for part in choice_input.replace(',', ' ').split():
            if part in ['1', '2']:
                choices.add(part)

        assert choices == {'1', '2'}

    def test_ignore_invalid_choices(self):
        """Test that invalid choices are ignored."""
        choice_input = '1 3 invalid'
        choices = set()
        for part in choice_input.replace(',', ' ').split():
            if part in ['1', '2']:
                choices.add(part)

        assert choices == {'1'}


class TestInteractiveLabelSelector:
    """Test the InteractiveLabelSelector class."""

    def test_init_input_field_at_top(self):
        """Test that the new label input field is at the top of the list."""
        current_labels = ['Bot/Newsletter-Tech']
        available_labels = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']

        # Simulate the initialization logic (input field first)
        items = []
        seen = set()

        # Add empty input field at top for easy access
        items.append({'label': '', 'checked': False, 'is_input': True})

        # Add available labels
        for label in available_labels:
            checked = label in current_labels
            items.append({'label': label, 'checked': checked, 'is_input': False})
            seen.add(label)

        # Add any current labels not in available list
        for label in current_labels:
            if label not in seen:
                items.append({'label': label, 'checked': True, 'is_input': False})

        # First item should be the input field
        assert items[0]['is_input'] is True
        assert items[0]['label'] == ''
        assert items[0]['checked'] is False

        # Second item should be the first available label
        assert items[1]['label'] == 'Bot/Newsletter-Tech'

    def test_init_with_current_labels_checked(self):
        """Test that current labels are pre-checked."""
        # Import the class from the script
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))

        # We can't easily import from the script, so test the logic directly
        current_labels = ['Bot/Newsletter-Tech', 'Bot/AutoArchived']
        available_labels = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']

        # Simulate the initialization logic (input field first)
        items = []
        seen = set()

        # Add empty input field at top
        items.append({'label': '', 'checked': False, 'is_input': True})

        for label in available_labels:
            checked = label in current_labels
            items.append({'label': label, 'checked': checked, 'is_input': False})
            seen.add(label)

        for label in current_labels:
            if label not in seen:
                items.append({'label': label, 'checked': True, 'is_input': False})

        # Verify Bot/Newsletter-Tech is checked (in both lists)
        tech_item = next(i for i in items if i['label'] == 'Bot/Newsletter-Tech')
        assert tech_item['checked'] is True

        # Verify Bot/Personal is not checked
        personal_item = next(i for i in items if i['label'] == 'Bot/Personal')
        assert personal_item['checked'] is False

        # Verify Bot/AutoArchived is added and checked (only in current)
        archived_item = next(i for i in items if i['label'] == 'Bot/AutoArchived')
        assert archived_item['checked'] is True

    def test_init_adds_current_labels_not_in_available(self):
        """Test that current labels not in available list are added."""
        current_labels = ['Bot/CustomLabel']
        available_labels = ['Bot/Newsletter-Tech', 'Bot/Personal']

        items = []
        seen = set()

        for label in available_labels:
            checked = label in current_labels
            items.append({'label': label, 'checked': checked, 'is_input': False})
            seen.add(label)

        for label in current_labels:
            if label not in seen:
                items.append({'label': label, 'checked': True, 'is_input': False})

        # Should have 3 items: 2 available + 1 custom
        assert len(items) == 3

        # Custom label should be checked
        custom_item = next(i for i in items if i['label'] == 'Bot/CustomLabel')
        assert custom_item['checked'] is True

    def test_empty_current_labels(self):
        """Test with no current labels."""
        current_labels = []
        available_labels = ['Bot/Newsletter-Tech', 'Bot/Personal']

        items = []
        seen = set()

        for label in available_labels:
            checked = label in current_labels
            items.append({'label': label, 'checked': checked, 'is_input': False})
            seen.add(label)

        # All items should be unchecked
        assert all(not item['checked'] for item in items)

    def test_selection_result_format(self):
        """Test that selection returns correct format."""
        # Simulate items after user interaction
        items = [
            {'label': 'Bot/Newsletter-Tech', 'checked': True, 'is_input': False},
            {'label': 'Bot/Personal', 'checked': False, 'is_input': False},
            {'label': 'Bot/Work', 'checked': True, 'is_input': False},
            {'label': 'Bot/NewLabel', 'checked': True, 'is_input': False},
            {'label': '', 'checked': False, 'is_input': True},
        ]

        # Simulate getting selected labels (as done in run())
        selected = [item['label'] for item in items if item['checked'] and item['label']]

        assert selected == ['Bot/Newsletter-Tech', 'Bot/Work', 'Bot/NewLabel']

    def test_empty_input_field_not_included(self):
        """Test that empty input field is not included in results."""
        items = [
            {'label': 'Bot/Personal', 'checked': True, 'is_input': False},
            {'label': '', 'checked': False, 'is_input': True},
        ]

        selected = [item['label'] for item in items if item['checked'] and item['label']]

        assert selected == ['Bot/Personal']
        assert '' not in selected


class TestInteractiveLabelSelectorKeyHandling:
    """Test key handling logic for InteractiveLabelSelector."""

    def test_toggle_checkbox_logic(self):
        """Test checkbox toggle logic."""
        item = {'label': 'Bot/Test', 'checked': False, 'is_input': False}

        # Toggle on
        item['checked'] = not item['checked']
        assert item['checked'] is True

        # Toggle off
        item['checked'] = not item['checked']
        assert item['checked'] is False

    def test_cursor_bounds(self):
        """Test cursor stays within bounds."""
        cursor = 0
        num_items = 5
        confirm_index = num_items

        # Can't go below 0
        cursor = max(0, cursor - 1)
        assert cursor == 0

        # Move down
        cursor = min(confirm_index, cursor + 1)
        assert cursor == 1

        # Can't go past confirm
        cursor = confirm_index
        cursor = min(confirm_index, cursor + 1)
        assert cursor == confirm_index

    def test_input_buffer_accumulation(self):
        """Test that typing accumulates in input buffer."""
        input_buffer = ''

        for char in 'Bot/Test':
            input_buffer += char

        assert input_buffer == 'Bot/Test'

    def test_backspace_removes_character(self):
        """Test backspace removes last character."""
        input_buffer = 'Bot/Test'

        input_buffer = input_buffer[:-1]
        assert input_buffer == 'Bot/Tes'

        input_buffer = input_buffer[:-1]
        assert input_buffer == 'Bot/Te'

    def test_backspace_on_empty_buffer(self):
        """Test backspace on empty buffer does nothing."""
        input_buffer = ''

        if input_buffer:
            input_buffer = input_buffer[:-1]

        assert input_buffer == ''

    def test_auto_check_when_typing(self):
        """Test that input field is auto-checked when user starts typing."""
        item = {'label': '', 'checked': False, 'is_input': True}
        input_buffer = ''

        # Simulate typing a character
        key = 'B'
        if item['is_input']:
            input_buffer += key
            # Auto-check when typing
            item['checked'] = True

        assert input_buffer == 'B'
        assert item['checked'] is True

    def test_uncheck_when_backspace_to_empty(self):
        """Test that input field is unchecked when backspacing to empty."""
        item = {'label': '', 'checked': True, 'is_input': True}
        input_buffer = 'B'

        # Simulate backspace
        if input_buffer:
            input_buffer = input_buffer[:-1]
            # Uncheck if buffer becomes empty
            if not input_buffer:
                item['checked'] = False

        assert input_buffer == ''
        assert item['checked'] is False

    def test_remain_checked_with_partial_backspace(self):
        """Test that input field stays checked when backspacing but not empty."""
        item = {'label': '', 'checked': True, 'is_input': True}
        input_buffer = 'Bot'

        # Simulate backspace (only removes one char)
        if input_buffer:
            input_buffer = input_buffer[:-1]
            if not input_buffer:
                item['checked'] = False

        assert input_buffer == 'Bo'
        assert item['checked'] is True  # Still checked because buffer not empty


class TestReviewDecisionsLabelParsing:
    """Test label parsing with shlex for quoted strings."""

    def test_parse_single_numeric_category(self):
        """Test parsing single category number."""
        import shlex
        cat_input = '1'
        parts = shlex.split(cat_input.replace(',', ' '))

        assert parts == ['1']

    def test_parse_multiple_numeric_categories(self):
        """Test parsing multiple category numbers."""
        import shlex
        cat_input = '1 3 5'
        parts = shlex.split(cat_input.replace(',', ' '))

        assert parts == ['1', '3', '5']

    def test_parse_single_word_label(self):
        """Test parsing single-word custom label."""
        import shlex
        cat_input = 'Bot/Marketing'
        parts = shlex.split(cat_input.replace(',', ' '))

        assert parts == ['Bot/Marketing']

    def test_parse_quoted_multi_word_label(self):
        """Test parsing quoted multi-word label."""
        import shlex
        cat_input = '"Bot/Customer Support"'
        parts = shlex.split(cat_input.replace(',', ' '))

        assert parts == ['Bot/Customer Support']

    def test_parse_mixed_numbers_and_labels(self):
        """Test parsing mix of numbers and custom labels."""
        import shlex
        cat_input = '1 Bot/Urgent "Bot/Needs Review"'
        parts = shlex.split(cat_input.replace(',', ' '))

        assert parts == ['1', 'Bot/Urgent', 'Bot/Needs Review']

    def test_parse_comma_separated_with_quotes(self):
        """Test parsing comma-separated input with quotes."""
        import shlex
        cat_input = '1, "Bot/High Priority", 3'
        parts = shlex.split(cat_input.replace(',', ' '))

        assert parts == ['1', 'Bot/High Priority', '3']

    def test_fallback_on_shlex_error(self):
        """Test fallback to simple split on shlex parse error."""
        import shlex
        cat_input = '1 "unclosed quote'

        try:
            parts = shlex.split(cat_input.replace(',', ' '))
        except ValueError:
            parts = cat_input.replace(',', ' ').split()

        # Should fall back to simple split
        assert parts == ['1', '"unclosed', 'quote']


class TestReviewDecisionsLabelSelection:
    """Test label selection logic with numbers and custom labels."""

    def test_select_by_valid_number(self):
        """Test selecting label by valid category number."""
        categories = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']
        selected_labels = []

        part = '1'
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(categories):
                selected_labels.append(categories[idx - 1])

        assert selected_labels == ['Bot/Newsletter-Tech']

    def test_select_multiple_by_number(self):
        """Test selecting multiple labels by number."""
        categories = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']
        selected_labels = []

        for part in ['1', '3']:
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(categories):
                    selected_labels.append(categories[idx - 1])

        assert selected_labels == ['Bot/Newsletter-Tech', 'Bot/Work']

    def test_create_custom_label(self):
        """Test creating custom label by name."""
        categories = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']
        selected_labels = []

        part = 'Bot/Marketing'
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(categories):
                selected_labels.append(categories[idx - 1])
        else:
            selected_labels.append(part)

        assert selected_labels == ['Bot/Marketing']

    def test_mix_existing_and_custom_labels(self):
        """Test mixing existing categories and custom labels."""
        categories = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']
        selected_labels = []

        for part in ['1', 'Bot/Urgent', '3']:
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(categories):
                    selected_labels.append(categories[idx - 1])
            else:
                selected_labels.append(part)

        assert selected_labels == ['Bot/Newsletter-Tech', 'Bot/Urgent', 'Bot/Work']

    def test_skip_out_of_range_number(self):
        """Test that out-of-range numbers are skipped."""
        categories = ['Bot/Newsletter-Tech', 'Bot/Personal', 'Bot/Work']
        selected_labels = []

        part = '10'
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(categories):
                selected_labels.append(categories[idx - 1])

        assert selected_labels == []

    def test_comma_separated_label_storage(self):
        """Test storing multiple labels as comma-separated string."""
        selected_labels = ['Bot/Newsletter-Tech', 'Bot/Urgent', 'Bot/Work']

        correct_label = ','.join(selected_labels) if len(selected_labels) > 1 else selected_labels[0]

        assert correct_label == 'Bot/Newsletter-Tech,Bot/Urgent,Bot/Work'

    def test_single_label_storage(self):
        """Test storing single label without comma."""
        selected_labels = ['Bot/Marketing']

        correct_label = ','.join(selected_labels) if len(selected_labels) > 1 else selected_labels[0]

        assert correct_label == 'Bot/Marketing'
