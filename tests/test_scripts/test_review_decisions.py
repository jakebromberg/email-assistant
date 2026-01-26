"""Tests for review_decisions script."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.database import Database
from src.database.schema import Email, EmailAction


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
