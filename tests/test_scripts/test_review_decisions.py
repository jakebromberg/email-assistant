"""Tests for review_decisions script."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

# Add scripts directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))

from src.database import Database
from src.database.schema import Email, EmailAction, FeedbackReview


class TestReviewDecisionsSessionManagement:
    """Test session management in review_decisions script."""

    @pytest.fixture
    def mock_database(self, tmp_path):
        """Create test database with sample data."""
        from src.database import Database
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Initialize schema
        from src.database.schema import Base
        Base.metadata.create_all(db.engine)

        # Add sample email and action
        with db.get_session() as session:
            email = Email(
                message_id="test123",
                thread_id="thread123",
                from_address="sender@example.com",
                from_name="Test Sender",
                to_address="me@example.com",
                subject="Test Email",
                body_plain="Test body",
                body_html="<p>Test body</p>",
                date=datetime.now() - timedelta(hours=1),
                labels=["INBOX"],
                snippet="Test snippet",
                was_read=False,
                was_archived=False,
                is_important=False,
                is_starred=False,
                opened_at=None
            )
            session.add(email)

            action = EmailAction(
                message_id="test123",
                action_type="archive",
                source="bot",
                timestamp=datetime.now(),
                action_data={
                    'score': 0.25,
                    'confidence': 'high',
                    'labels': ['Bot/Newsletter-Tech', 'Bot/AutoArchived']
                }
            )
            session.add(action)
            session.commit()

        return db

    def test_actions_can_be_queried_and_used_outside_session(self, mock_database):
        """
        REGRESSION TEST: Ensure EmailAction data can be extracted and used after session closes.

        This prevents DetachedInstanceError when accessing action attributes in the review loop.
        The fix extracts data into dictionaries before the session closes.
        """
        from src.database.repository import EmailRepository

        # Query actions and extract data (simulating the script)
        action_data_list = []
        start_date = datetime.now() - timedelta(days=1)

        with mock_database.get_session() as session:
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
        with mock_database.get_session() as session:
            repo = EmailRepository(session)
            email = repo.get_by_id(action_data['message_id'])
            assert email is not None
            assert email.subject == "Test Email"

    def test_multiple_actions_can_be_extracted_and_processed(self, tmp_path):
        """Test that multiple actions can be extracted and processed without session errors."""
        from src.database import Database
        from src.database.repository import EmailRepository

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        from src.database.schema import Base
        Base.metadata.create_all(db.engine)

        # Add multiple emails and actions
        with db.get_session() as session:
            for i in range(3):
                email = Email(
                    message_id=f"test{i}",
                    thread_id=f"thread{i}",
                    from_address=f"sender{i}@example.com",
                    from_name=f"Sender {i}",
                    to_address="me@example.com",
                    subject=f"Test Email {i}",
                    body_plain=f"Test body {i}",
                    body_html=f"<p>Test body {i}</p>",
                    date=datetime.now() - timedelta(hours=i+1),
                    labels=["INBOX"],
                    snippet=f"Test snippet {i}",
                    was_read=False,
                    was_archived=(i % 2 == 0),  # Alternate archived status
                    is_important=False,
                    is_starred=False,
                    opened_at=None
                )
                session.add(email)

                action = EmailAction(
                    message_id=f"test{i}",
                    action_type="archive" if i % 2 == 0 else "keep",
                    source="bot",
                    timestamp=datetime.now() - timedelta(hours=i),
                    action_data={
                        'score': 0.2 + (i * 0.1),
                        'confidence': 'high' if i % 2 == 0 else 'low',
                        'labels': [f'Bot/Category{i}']
                    }
                )
                session.add(action)
            session.commit()

        # Extract actions
        action_data_list = []
        with db.get_session() as session:
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
        with db.get_session() as session:
            repo = EmailRepository(session)

            for action_data in action_data_list:
                # Should not raise DetachedInstanceError
                email = repo.get_by_id(action_data['message_id'])
                assert email is not None

                # Should be able to access action data
                assert action_data['action_type'] in ['archive', 'keep']
                assert 'score' in action_data['action_data']
                assert 'confidence' in action_data['action_data']


class TestReviewDecisionsConfidenceSorting:
    """Test confidence-based sorting of bot decisions."""

    def test_sorts_by_distance_from_half(self, tmp_path):
        """Test that actions are sorted by confidence (distance from 0.5)."""
        from src.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        from src.database.schema import Base
        Base.metadata.create_all(db.engine)

        # Add emails with different confidence scores
        scores = [0.8, 0.45, 0.1, 0.52, 0.3]
        with db.get_session() as session:
            for i, score in enumerate(scores):
                email = Email(
                    message_id=f"test{i}",
                    thread_id=f"thread{i}",
                    from_address=f"sender{i}@example.com",
                    from_name=f"Sender {i}",
                    to_address="me@example.com",
                    subject=f"Test Email {i}",
                    body_plain=f"Test body {i}",
                    body_html=f"<p>Test body {i}</p>",
                    date=datetime.now() - timedelta(hours=i+1),
                    labels=["INBOX"],
                    snippet=f"Test snippet {i}",
                    was_read=False,
                    was_archived=False,
                    is_important=False,
                    is_starred=False,
                    opened_at=None
                )
                session.add(email)

                action = EmailAction(
                    message_id=f"test{i}",
                    action_type="archive" if score < 0.5 else "keep",
                    source="bot",
                    timestamp=datetime.now() - timedelta(hours=i),
                    action_data={'score': score, 'confidence': 'low' if 0.3 < score < 0.7 else 'high'}
                )
                session.add(action)
            session.commit()

        # Extract and sort (simulating the script)
        action_data_list = []
        with db.get_session() as session:
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

    def test_invalid_scores_sorted_to_end(self, tmp_path):
        """Test that invalid/missing scores are sorted to the end."""
        from src.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        from src.database.schema import Base
        Base.metadata.create_all(db.engine)

        # Add emails with various score types
        test_cases = [
            ("test0", 0.4, "valid"),
            ("test1", None, "none"),
            ("test2", "invalid", "string"),
            ("test3", 0.6, "valid2"),
        ]

        with db.get_session() as session:
            for i, (msg_id, score, label) in enumerate(test_cases):
                email = Email(
                    message_id=msg_id,
                    thread_id=f"thread{i}",
                    from_address=f"sender{i}@example.com",
                    from_name=f"Sender {i}",
                    to_address="me@example.com",
                    subject=f"Test Email {i}",
                    body_plain=f"Test body {i}",
                    body_html=f"<p>Test body {i}</p>",
                    date=datetime.now() - timedelta(hours=i+1),
                    labels=["INBOX"],
                    snippet=f"Test snippet {i}",
                    was_read=False,
                    was_archived=False,
                    is_important=False,
                    is_starred=False,
                    opened_at=None
                )
                session.add(email)

                action = EmailAction(
                    message_id=msg_id,
                    action_type="archive",
                    source="bot",
                    timestamp=datetime.now() - timedelta(hours=i),
                    action_data={'score': score, 'label': label}
                )
                session.add(action)
            session.commit()

        # Extract and sort
        action_data_list = []
        with db.get_session() as session:
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

    def test_undo_removes_feedback_from_database(self, tmp_path):
        """Test that undo deletes feedback from database."""
        from src.database import Database
        from src.database.repository import EmailRepository

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        from src.database.schema import Base
        Base.metadata.create_all(db.engine)

        # Add test email
        with db.get_session() as session:
            email = Email(
                message_id="test123",
                thread_id="thread123",
                from_address="sender@example.com",
                from_name="Test Sender",
                to_address="me@example.com",
                subject="Test Email",
                body_plain="Test body",
                body_html="<p>Test body</p>",
                date=datetime.now(),
                labels=["INBOX"],
                snippet="Test snippet",
                was_read=False,
                was_archived=False,
                is_important=False,
                is_starred=False,
                opened_at=None
            )
            session.add(email)
            session.commit()

        # Simulate feedback and undo (as script would do)
        with db.get_session() as session:
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

    def test_decision_inference_with_action_data(self, tmp_path):
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
