"""Test the shared fixtures in conftest.py."""

from datetime import datetime

from src.database.schema import Email


def test_temp_db_fixture(temp_db):
    """Test temp_db fixture creates a working database."""
    assert temp_db is not None
    assert temp_db.engine is not None

    with temp_db.get_session() as session:
        # Should be able to query
        emails = session.query(Email).all()
        assert emails == []


def test_temp_db_session_fixture(temp_db_session):
    """Test temp_db_session fixture provides a working session."""
    assert temp_db_session is not None

    # Should be able to add and commit
    email = Email(
        message_id="test123",
        thread_id="thread123",
        from_address="test@example.com",
        to_address="me@example.com",
        subject="Test",
        date=datetime.now(),
        labels=["INBOX"]
    )
    temp_db_session.add(email)
    temp_db_session.commit()

    # Should be able to query
    result = temp_db_session.query(Email).filter_by(message_id="test123").first()
    assert result is not None
    assert result.subject == "Test"


def test_email_factory_fixture(email_factory):
    """Test email_factory creates emails with defaults."""
    email = email_factory()

    assert email.message_id is not None
    assert email.from_address == "sender@example.com"
    assert email.subject == "Test Email"
    assert email.was_read is False


def test_email_factory_custom_values(email_factory):
    """Test email_factory accepts custom values."""
    email = email_factory(
        message_id="custom123",
        subject="Custom Subject",
        was_read=True
    )

    assert email.message_id == "custom123"
    assert email.subject == "Custom Subject"
    assert email.was_read is True


def test_action_factory_fixture(action_factory, email_factory):
    """Test action_factory creates actions."""
    email = email_factory()
    action = action_factory(message_id=email.message_id, action_type='archive')

    assert action.message_id == email.message_id
    assert action.action_type == 'archive'
    assert action.source == 'bot'
    assert 'score' in action.action_data


def test_features_factory_fixture(features_factory, email_factory):
    """Test features_factory creates features."""
    email = email_factory()
    features = features_factory(
        message_id=email.message_id,
        is_newsletter=True
    )

    assert features.message_id == email.message_id
    assert features.is_newsletter is True
    assert features.sender_domain == "example.com"


def test_email_repo_fixture(email_repo, email_factory):
    """Test email_repo fixture provides working repository."""
    email = email_factory(message_id="test123")

    saved = email_repo.get_by_id("test123")
    assert saved is not None
    assert saved.message_id == "test123"


def test_feature_store_fixture(feature_store, email_factory, features_factory):
    """Test feature_store fixture provides working store."""
    email = email_factory(message_id="test123")
    features = features_factory(message_id=email.message_id)

    retrieved = feature_store.get_features("test123")
    assert retrieved is not None
    assert retrieved.message_id == "test123"


def test_mock_gmail_client_fixture(mock_gmail_client):
    """Test mock_gmail_client has expected methods."""
    assert hasattr(mock_gmail_client, 'list_messages')
    assert hasattr(mock_gmail_client, 'get_messages_batch')

    # Should return empty list by default
    result = mock_gmail_client.list_all_messages()
    assert result == []


def test_mock_gmail_ops_fixture(mock_gmail_ops):
    """Test mock_gmail_ops has expected methods."""
    assert hasattr(mock_gmail_ops, 'archive')
    assert hasattr(mock_gmail_ops, 'mark_read')

    # Should return successful result
    result = mock_gmail_ops.archive(['msg1'], dry_run=True)
    assert result.success is True
    assert result.message_ids == ['msg1']


def test_sample_gmail_message_fixture(sample_gmail_message):
    """Test sample_gmail_message has expected structure."""
    assert sample_gmail_message['id'] == 'msg123'
    assert sample_gmail_message['threadId'] == 'thread123'
    assert 'payload' in sample_gmail_message
    assert 'headers' in sample_gmail_message['payload']
