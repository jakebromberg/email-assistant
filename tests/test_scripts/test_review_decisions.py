"""Tests for review_decisions script."""

import os
import sys
from datetime import datetime, timedelta

# Add scripts directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))

from src.database.schema import EmailAction, FeedbackReview


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
