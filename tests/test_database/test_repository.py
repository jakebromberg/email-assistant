"""Tests for email repository."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.database.repository import EmailRepository
from src.database.schema import Email, EmailAction, FeedbackReview


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
    def gmail_email(self, gmail_email_factory):
        """Create sample Gmail email."""
        return gmail_email_factory(message_id="test123")

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
        assert feedback.decision_correct is False
        assert feedback.user_comment == "Important email"


class TestEmailRepositoryWithRealDB:
    """Test EmailRepository with actual database fixtures."""

    def test_save_email_update_existing(self, email_repo, email_factory, temp_db_session, gmail_email_factory):
        """Test updating an existing email."""
        # Create initial email in database
        email = email_factory(
            message_id="update_test",
            subject="Original Subject",
            from_address="original@example.com",
        )
        temp_db_session.commit()

        # Create a new Gmail email with same ID but different data
        updated_gmail = gmail_email_factory(
            message_id="update_test",
            from_address="updated@example.com",
            from_name="Updated Sender",
            subject="Updated Subject",
        )

        result = email_repo.save_email(updated_gmail)

        assert result.message_id == "update_test"
        assert result.subject == "Updated Subject"
        assert result.from_address == "updated@example.com"

    def test_get_by_id(self, email_repo, email_factory):
        """Test getting email by ID."""
        email = email_factory(message_id="get_test")

        result = email_repo.get_by_id("get_test")

        assert result is not None
        assert result.message_id == "get_test"

    def test_get_by_id_not_found(self, email_repo):
        """Test getting nonexistent email returns None."""
        result = email_repo.get_by_id("nonexistent")
        assert result is None

    def test_get_by_ids(self, email_repo, email_factory):
        """Test getting multiple emails by IDs."""
        emails = [email_factory(message_id=f"batch_{i}") for i in range(3)]

        result = email_repo.get_by_ids(["batch_0", "batch_1", "batch_2"])

        assert len(result) == 3
        message_ids = {e.message_id for e in result}
        assert message_ids == {"batch_0", "batch_1", "batch_2"}

    def test_get_all(self, email_repo, email_factory, temp_db_session):
        """Test getting all emails."""
        for i in range(5):
            email_factory(message_id=f"all_{i}")
        temp_db_session.commit()

        result = email_repo.get_all()

        assert len(result) >= 5

    def test_get_all_with_pagination(self, email_repo, email_factory, temp_db_session):
        """Test getting all emails with pagination."""
        for i in range(10):
            email_factory(message_id=f"page_{i}")
        temp_db_session.commit()

        result = email_repo.get_all(limit=3, offset=0)
        assert len(result) == 3

    def test_get_by_sender(self, email_repo, email_factory, temp_db_session):
        """Test getting emails by sender."""
        email_factory(from_address="target@example.com")
        email_factory(from_address="target@example.com")
        email_factory(from_address="other@example.com")
        temp_db_session.commit()

        result = email_repo.get_by_sender("target@example.com")

        assert len(result) == 2
        assert all(e.from_address == "target@example.com" for e in result)

    def test_get_by_sender_with_limit(self, email_repo, email_factory, temp_db_session):
        """Test getting emails by sender with limit."""
        for _ in range(5):
            email_factory(from_address="prolific@example.com")
        temp_db_session.commit()

        result = email_repo.get_by_sender("prolific@example.com", limit=2)

        assert len(result) == 2

    def test_get_unread(self, email_repo, email_factory, temp_db_session):
        """Test getting unread emails."""
        email_factory(was_read=False)
        email_factory(was_read=False)
        email_factory(was_read=True)
        temp_db_session.commit()

        result = email_repo.get_unread()

        assert len(result) >= 2
        assert all(not e.was_read for e in result)

    def test_get_unread_with_limit(self, email_repo, email_factory, temp_db_session):
        """Test getting unread emails with limit."""
        for _ in range(5):
            email_factory(was_read=False)
        temp_db_session.commit()

        result = email_repo.get_unread(limit=2)

        assert len(result) == 2

    def test_get_in_date_range(self, email_repo, email_factory, temp_db_session):
        """Test getting emails in date range."""
        from datetime import timedelta

        now = datetime.now()
        email_factory(date=now - timedelta(days=1))
        email_factory(date=now - timedelta(days=2))
        email_factory(date=now - timedelta(days=10))  # Outside range
        temp_db_session.commit()

        start = now - timedelta(days=5)
        end = now
        result = email_repo.get_in_date_range(start, end)

        assert len(result) >= 2

    def test_get_in_date_range_with_limit(self, email_repo, email_factory, temp_db_session):
        """Test getting emails in date range with limit."""
        from datetime import timedelta

        now = datetime.now()
        for i in range(5):
            email_factory(date=now - timedelta(days=i))
        temp_db_session.commit()

        start = now - timedelta(days=10)
        result = email_repo.get_in_date_range(start, limit=2)

        assert len(result) == 2

    def test_count(self, email_repo, email_factory, temp_db_session):
        """Test counting emails."""
        initial_count = email_repo.count()

        for _ in range(3):
            email_factory()
        temp_db_session.commit()

        assert email_repo.count() == initial_count + 3

    def test_get_actions_for_email(self, email_repo, email_factory, action_factory, temp_db_session):
        """Test getting actions for an email."""
        email = email_factory()
        action_factory(message_id=email.message_id, action_type='archive')
        action_factory(message_id=email.message_id, action_type='label_add')
        temp_db_session.commit()

        result = email_repo.get_actions_for_email(email.message_id)

        assert len(result) == 2
        action_types = {a.action_type for a in result}
        assert action_types == {'archive', 'label_add'}

    def test_get_sender_stats(self, email_repo, email_factory, temp_db_session):
        """Test getting sender statistics."""
        # Create emails with different read/archived states
        email_factory(from_address="stats@example.com", was_read=True, was_archived=False)
        email_factory(from_address="stats@example.com", was_read=True, was_archived=True)
        email_factory(from_address="stats@example.com", was_read=False, was_archived=True)
        temp_db_session.commit()

        stats = email_repo.get_sender_stats("stats@example.com")

        assert stats['total_emails'] == 3
        assert stats['read_count'] == 2
        assert stats['open_rate'] == 2 / 3
        assert stats['archived_count'] == 2
        assert stats['archive_rate'] == 2 / 3

    def test_get_sender_stats_empty(self, email_repo):
        """Test getting stats for unknown sender."""
        stats = email_repo.get_sender_stats("unknown@example.com")

        assert stats['total_emails'] == 0
        assert stats['read_count'] == 0
        assert stats['open_rate'] == 0.0


class TestSenderLabelMapping:
    """Test sender label mapping operations."""

    def test_save_sender_label_mapping_by_address(self, email_repo, temp_db_session):
        """Test saving sender label mapping by address."""
        mapping = email_repo.save_sender_label_mapping(
            label="Bot/Newsletter-Tech",
            sender_address="newsletter@tech.com"
        )
        temp_db_session.commit()

        assert mapping.label == "Bot/Newsletter-Tech"
        assert mapping.sender_address == "newsletter@tech.com"

    def test_save_sender_label_mapping_by_domain(self, email_repo, temp_db_session):
        """Test saving sender label mapping by domain."""
        mapping = email_repo.save_sender_label_mapping(
            label="Bot/Company",
            sender_domain="company.com"
        )
        temp_db_session.commit()

        assert mapping.label == "Bot/Company"
        assert mapping.sender_domain == "company.com"
        assert mapping.sender_address is None

    def test_save_sender_label_mapping_update(self, email_repo, temp_db_session):
        """Test updating existing sender label mapping."""
        email_repo.save_sender_label_mapping(
            label="Bot/OldLabel",
            sender_address="sender@example.com"
        )
        temp_db_session.commit()

        updated = email_repo.save_sender_label_mapping(
            label="Bot/NewLabel",
            sender_address="sender@example.com"
        )
        temp_db_session.commit()

        assert updated.label == "Bot/NewLabel"

    def test_save_sender_label_mapping_requires_address_or_domain(self, email_repo):
        """Test that mapping requires address or domain."""
        with pytest.raises(ValueError, match="Must provide sender_address or sender_domain"):
            email_repo.save_sender_label_mapping(label="Bot/Test")

    def test_get_sender_label_by_address(self, email_repo, temp_db_session):
        """Test getting sender label by address."""
        email_repo.save_sender_label_mapping(
            label="Bot/Specific",
            sender_address="specific@example.com"
        )
        temp_db_session.commit()

        label = email_repo.get_sender_label("specific@example.com")

        assert label == "Bot/Specific"

    def test_get_sender_label_by_domain(self, email_repo, temp_db_session):
        """Test getting sender label by domain fallback."""
        email_repo.save_sender_label_mapping(
            label="Bot/Domain",
            sender_domain="domain.com"
        )
        temp_db_session.commit()

        # No exact address match, should fall back to domain
        label = email_repo.get_sender_label("anyone@domain.com")

        assert label == "Bot/Domain"

    def test_get_sender_label_address_takes_priority(self, email_repo, temp_db_session):
        """Test that address mapping takes priority over domain."""
        email_repo.save_sender_label_mapping(
            label="Bot/Domain",
            sender_domain="example.com"
        )
        email_repo.save_sender_label_mapping(
            label="Bot/Specific",
            sender_address="vip@example.com"
        )
        temp_db_session.commit()

        # VIP gets specific label
        assert email_repo.get_sender_label("vip@example.com") == "Bot/Specific"
        # Others get domain label
        assert email_repo.get_sender_label("anyone@example.com") == "Bot/Domain"

    def test_get_sender_label_not_found(self, email_repo):
        """Test getting sender label when not mapped."""
        label = email_repo.get_sender_label("unknown@nowhere.com")
        assert label is None

    def test_get_all_custom_labels(self, email_repo, temp_db_session):
        """Test getting all custom labels."""
        email_repo.save_sender_label_mapping(label="Bot/A", sender_address="a@example.com")
        email_repo.save_sender_label_mapping(label="Bot/B", sender_address="b@example.com")
        email_repo.save_sender_label_mapping(label="Bot/A", sender_address="a2@example.com")  # Duplicate label
        temp_db_session.commit()

        labels = email_repo.get_all_custom_labels()

        assert set(labels) == {"Bot/A", "Bot/B"}
