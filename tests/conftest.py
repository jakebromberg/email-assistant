"""Shared test fixtures for all test modules."""

from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session

from src.database import Database
from src.database.repository import EmailRepository
from src.database.schema import Base, Email, EmailAction, EmailFeatures
from src.features import FeatureStore

# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db(tmp_path: Path) -> Generator[Database, None, None]:
    """
    Create a temporary test database with schema initialized.

    Usage:
        def test_something(temp_db):
            with temp_db.get_session() as session:
                # Use session
    """
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))

    # Initialize schema
    Base.metadata.create_all(db.engine)

    yield db

    # Cleanup
    db.engine.dispose()


@pytest.fixture
def temp_db_session(temp_db: Database) -> Generator[Session, None, None]:
    """
    Create a database session for testing.

    Usage:
        def test_something(temp_db_session):
            email = Email(...)
            temp_db_session.add(email)
            temp_db_session.commit()
    """
    with temp_db.get_session() as session:
        yield session


@pytest.fixture
def email_repo(temp_db_session: Session) -> EmailRepository:
    """
    Create an EmailRepository for testing.

    Usage:
        def test_something(email_repo):
            email = email_repo.get_by_id("msg123")
    """
    return EmailRepository(temp_db_session)


@pytest.fixture
def feature_store(temp_db_session: Session) -> FeatureStore:
    """
    Create a FeatureStore for testing.

    Usage:
        def test_something(feature_store):
            features = feature_store.get_features("msg123")
    """
    return FeatureStore(temp_db_session)


# ============================================================================
# Factory Fixtures
# ============================================================================

@pytest.fixture
def email_factory(temp_db_session: Session):
    """
    Factory for creating test emails.

    Usage:
        def test_something(email_factory):
            email = email_factory(subject="Test", from_address="test@example.com")
            # email is already added to database
    """
    def _create_email(
        message_id: str = None,
        thread_id: str = None,
        from_address: str = "sender@example.com",
        from_name: str = "Test Sender",
        to_address: str = "me@example.com",
        subject: str = "Test Email",
        body_plain: str = "Test body",
        body_html: str = "<p>Test body</p>",
        date: datetime = None,
        labels: list[str] = None,
        snippet: str = "Test snippet",
        was_read: bool = False,
        was_archived: bool = False,
        is_important: bool = False,
        is_starred: bool = False,
        opened_at: datetime = None,
        **kwargs
    ) -> Email:
        """Create a test email with sensible defaults."""
        # Generate unique IDs if not provided
        if message_id is None:
            import uuid
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
        if thread_id is None:
            thread_id = f"thread_{message_id}"

        # Default values
        if date is None:
            date = datetime.now()
        if labels is None:
            labels = ["INBOX", "UNREAD"] if not was_read else ["INBOX"]

        email = Email(
            message_id=message_id,
            thread_id=thread_id,
            from_address=from_address,
            from_name=from_name,
            to_address=to_address,
            subject=subject,
            body_plain=body_plain,
            body_html=body_html,
            date=date,
            labels=labels,
            snippet=snippet,
            was_read=was_read,
            was_archived=was_archived,
            is_important=is_important,
            is_starred=is_starred,
            opened_at=opened_at,
            **kwargs
        )

        temp_db_session.add(email)
        temp_db_session.commit()
        temp_db_session.refresh(email)

        return email

    return _create_email


@pytest.fixture
def action_factory(temp_db_session: Session):
    """
    Factory for creating test email actions.

    Usage:
        def test_something(action_factory, email_factory):
            email = email_factory()
            action = action_factory(message_id=email.message_id, action_type='archive')
    """
    def _create_action(
        message_id: str,
        action_type: str = 'keep',
        source: str = 'bot',
        action_data: dict = None,
        timestamp: datetime = None,
        **kwargs
    ) -> EmailAction:
        """Create a test email action with sensible defaults."""
        if timestamp is None:
            timestamp = datetime.now()
        if action_data is None:
            action_data = {
                'score': 0.5,
                'confidence': 'low',
                'reasoning': 'Test action'
            }

        action = EmailAction(
            message_id=message_id,
            action_type=action_type,
            source=source,
            action_data=action_data,
            timestamp=timestamp,
            **kwargs
        )

        temp_db_session.add(action)
        temp_db_session.commit()
        temp_db_session.refresh(action)

        return action

    return _create_action


@pytest.fixture
def features_factory(temp_db_session: Session):
    """
    Factory for creating test email features.

    Usage:
        def test_something(features_factory, email_factory):
            email = email_factory()
            features = features_factory(message_id=email.message_id, is_newsletter=True)
    """
    def _create_features(
        message_id: str,
        sender_domain: str = "example.com",
        sender_email_count: int = 1,
        sender_open_rate: float = 0.5,
        sender_days_since_last: int = None,
        is_newsletter: bool = False,
        subject_length: int = 20,
        body_length: int = 100,
        has_attachments: bool = False,
        thread_length: int = 1,
        day_of_week: int = 1,
        hour_of_day: int = 12,
        subject_embedding: list[float] = None,
        body_embedding: list[float] = None,
        **kwargs
    ) -> EmailFeatures:
        """Create test email features with sensible defaults."""
        features = EmailFeatures(
            message_id=message_id,
            sender_domain=sender_domain,
            sender_email_count=sender_email_count,
            sender_open_rate=sender_open_rate,
            sender_days_since_last=sender_days_since_last,
            is_newsletter=is_newsletter,
            subject_length=subject_length,
            body_length=body_length,
            has_attachments=has_attachments,
            thread_length=thread_length,
            day_of_week=day_of_week,
            hour_of_day=hour_of_day,
            subject_embedding=subject_embedding,
            body_embedding=body_embedding,
            **kwargs
        )

        temp_db_session.add(features)
        temp_db_session.commit()
        temp_db_session.refresh(features)

        return features

    return _create_features


# ============================================================================
# Gmail Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_gmail_auth():
    """
    Create a mock Gmail authenticator.

    Usage:
        def test_something(mock_gmail_auth):
            client = GmailClient(mock_gmail_auth)
    """
    auth = Mock()
    auth.get_credentials = Mock(return_value=Mock())
    return auth


@pytest.fixture
def mock_gmail_client():
    """
    Create a mock Gmail client.

    Usage:
        def test_something(mock_gmail_client):
            mock_gmail_client.list_messages.return_value = (["msg1"], None)
    """
    client = Mock()
    client.list_messages = Mock(return_value=([], None))
    client.list_all_messages = Mock(return_value=[])
    client.get_messages_batch = Mock(return_value=[])
    client.get_message = Mock(return_value=None)
    return client


@pytest.fixture
def mock_gmail_ops(mock_gmail_client):
    """
    Create a mock Gmail operations.

    Usage:
        def test_something(mock_gmail_ops):
            result = mock_gmail_ops.archive(['msg1'], dry_run=True)
    """
    from src.gmail.operations import GmailOperations, OperationResult

    ops = Mock(spec=GmailOperations)

    # Default successful operation result
    def _mock_operation(message_ids, **kwargs):
        return OperationResult(
            success=True,
            message_ids=message_ids,
            message=f"Operation successful for {len(message_ids)} messages"
        )

    ops.archive = Mock(side_effect=_mock_operation)
    ops.mark_read = Mock(side_effect=_mock_operation)
    ops.mark_unread = Mock(side_effect=_mock_operation)
    ops.move_to_inbox = Mock(side_effect=_mock_operation)
    ops.add_labels = Mock(side_effect=_mock_operation)
    ops.remove_labels = Mock(side_effect=_mock_operation)

    return ops


# ============================================================================
# Helper Fixtures
# ============================================================================

@pytest.fixture
def sample_gmail_message():
    """
    Create a sample Gmail API message structure.

    Usage:
        def test_something(sample_gmail_message):
            email = Email.from_gmail_message(sample_gmail_message)
    """
    return {
        'id': 'msg123',
        'threadId': 'thread123',
        'labelIds': ['INBOX', 'UNREAD'],
        'snippet': 'This is a test email snippet',
        'internalDate': '1704038400000',  # 2024-01-01 00:00:00 UTC
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'Test Sender <sender@example.com>'},
                {'name': 'To', 'value': 'me@example.com'},
                {'name': 'Subject', 'value': 'Test Email'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 00:00:00 +0000'},
            ],
            'mimeType': 'multipart/alternative',
            'parts': [
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': 'VGVzdCBib2R5'  # base64 for "Test body"
                    }
                },
                {
                    'mimeType': 'text/html',
                    'body': {
                        'data': 'PHA+VGVzdCBib2R5PC9wPg=='  # base64 for "<p>Test body</p>"
                    }
                }
            ]
        }
    }
