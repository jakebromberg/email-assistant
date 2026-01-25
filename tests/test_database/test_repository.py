"""Tests for email repository."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.database.schema import Email, EmailAction, FeedbackReview
from src.database.repository import EmailRepository
from src.gmail.models import Email as GmailEmail


class TestEmailRepository:
    """Test EmailRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        session.query.return_value.filter.return_value.first.return_value = None
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance."""
        return EmailRepository(mock_session)

    @pytest.fixture
    def gmail_email(self):
        """Create sample Gmail email."""
        return GmailEmail(
            message_id="test123",
            thread_id="thread123",
            from_address="sender@example.com",
            from_name="Test Sender",
            to_address="me@example.com",
            subject="Test Subject",
            date=datetime.now(),
            snippet="Test snippet",
            labels=["INBOX", "UNREAD"],
            body_plain="Test body",
            body_html="<p>Test body</p>",
            headers={}
        )

    def test_save_email_new(self, repository, gmail_email, mock_session):
        """Test saving a new email."""
        # Mock no existing email
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Save email
        result = repository.save_email(gmail_email)

        # Verify email was added
        mock_session.add.assert_called_once()
        assert isinstance(result, Email)
        assert result.message_id == "test123"
        assert result.from_address == "sender@example.com"

    def test_record_action(self, repository, mock_session):
        """Test recording an action."""
        action = repository.record_action(
            message_id="test123",
            action_type="archive",
            source="bot",
            action_data={"score": 0.95}
        )

        mock_session.add.assert_called_once()
        assert isinstance(action, EmailAction)
        assert action.message_id == "test123"
        assert action.action_type == "archive"
        assert action.source == "bot"

    def test_save_feedback(self, repository, mock_session):
        """Test saving feedback."""
        feedback = repository.save_feedback(
            message_id="test123",
            decision_correct=False,
            correct_decision="keep",
            user_comment="Important email"
        )

        mock_session.add.assert_called_once()
        assert isinstance(feedback, FeedbackReview)
        assert feedback.message_id == "test123"
        assert feedback.decision_correct == False
        assert feedback.user_comment == "Important email"

    # TODO: Add more tests for:
    # - get_by_id, get_by_ids
    # - get_by_sender
    # - get_sender_stats
    # - Update existing email
