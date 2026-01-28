"""Triage-specific test fixtures."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_triage_pipeline_dependencies(
    mock_gmail_client,
    mock_gmail_ops,
    temp_db,
    mock_email_scorer
):
    """Create all dependencies for TriagePipeline testing."""
    return {
        'gmail_client': mock_gmail_client,
        'gmail_ops': mock_gmail_ops,
        'database': temp_db,
        'model_path': 'dummy_model.txt',
        'use_embeddings': False
    }


@pytest.fixture
def mock_feature_extractor():
    """Create a mock feature extractor."""
    from src.features.extractors import MetadataExtractor

    extractor = Mock(spec=MetadataExtractor)
    extractor.extract_features = Mock(return_value=Mock(
        sender_domain='example.com',
        is_newsletter=False,
        subject_length=20,
        body_length=100,
        has_attachments=False,
        thread_length=1,
        day_of_week=1,
        hour_of_day=12
    ))

    return extractor


@pytest.fixture
def mock_historical_extractor():
    """Create a mock historical pattern extractor."""
    from src.features.extractors import HistoricalPatternExtractor

    extractor = Mock(spec=HistoricalPatternExtractor)
    extractor.extract_features = Mock(return_value=Mock(
        sender_email_count=5,
        sender_open_rate=0.6,
        sender_days_since_last=7,
        domain_open_rate=0.5
    ))

    return extractor


@pytest.fixture
def mock_embedding_extractor():
    """Create a mock embedding extractor."""
    extractor = Mock()
    # Return 384-dim embedding (standard size for all-MiniLM-L6-v2)
    extractor.extract = Mock(return_value=[0.1] * 384)
    extractor.extract_subject = Mock(return_value=[0.1] * 384)
    extractor.extract_body = Mock(return_value=[0.1] * 384)

    return extractor


@pytest.fixture
def mock_embedding_extractor_disabled():
    """Create a disabled embedding extractor (returns None)."""
    extractor = Mock()
    extractor.extract = Mock(return_value=None)
    extractor.extract_subject = Mock(return_value=None)
    extractor.extract_body = Mock(return_value=None)

    return extractor


@pytest.fixture
def sample_gmail_email_for_triage():
    """Create a sample Gmail email object for triage testing."""
    email = Mock()
    email.message_id = "triage_test_123"
    email.thread_id = "thread_triage_123"
    email.from_address = "sender@example.com"
    email.from_name = "Test Sender"
    email.to_address = "me@example.com"
    email.subject = "Test Email for Triage"
    email.body_plain = "This is a test email body for triage."
    email.body_html = "<p>This is a test email body for triage.</p>"
    email.date = datetime(2024, 1, 15, 14, 30)
    email.labels = ["INBOX", "UNREAD"]
    email.snippet = "Test email snippet"
    email.headers = {}
    email.was_read = False
    email.was_archived = False
    email.is_important = False
    email.is_starred = False
    email.opened_at = None

    return email


@pytest.fixture
def sample_gmail_emails_batch():
    """Create a batch of Gmail email objects for triage testing."""
    emails = []
    for i in range(3):
        email = Mock()
        email.message_id = f"batch_test_{i}"
        email.thread_id = f"thread_batch_{i}"
        email.from_address = f"sender{i}@example.com"
        email.from_name = f"Sender {i}"
        email.to_address = "me@example.com"
        email.subject = f"Test Email {i}"
        email.body_plain = f"Test body {i}"
        email.body_html = f"<p>Test body {i}</p>"
        email.date = datetime(2024, 1, 15 + i, 14, 30)
        email.labels = ["INBOX", "UNREAD"]
        email.snippet = f"Test snippet {i}"
        email.headers = {}
        email.was_read = False
        email.was_archived = False
        email.is_important = False
        email.is_starred = False
        email.opened_at = None
        emails.append(email)

    return emails


@pytest.fixture
def mock_triage_result_keep():
    """Create a mock triage result for a 'keep' decision."""
    return {
        'message_id': 'test_keep_123',
        'action': 'keep',
        'confidence': 'high',
        'score': 0.85,
        'reasoning': 'High importance, keep in inbox',
        'labels': ['Bot/Personal']
    }


@pytest.fixture
def mock_triage_result_archive():
    """Create a mock triage result for an 'archive' decision."""
    return {
        'message_id': 'test_archive_123',
        'action': 'archive',
        'confidence': 'high',
        'score': 0.15,
        'reasoning': 'Newsletter, archive',
        'labels': ['Bot/Newsletter-Tech', 'Bot/AutoArchived']
    }


@pytest.fixture
def mock_triage_result_low_confidence():
    """Create a mock triage result with low confidence."""
    return {
        'message_id': 'test_low_conf_123',
        'action': 'keep',
        'confidence': 'low',
        'score': 0.45,
        'reasoning': 'Uncertain, keep for review',
        'labels': ['Bot/LowConfidence']
    }


@pytest.fixture
def mock_triage_pipeline(
    mock_gmail_client,
    mock_gmail_ops,
    temp_db,
    mock_email_scorer
):
    """Create a mock TriagePipeline for testing."""
    from src.triage.pipeline import TriagePipeline

    with patch('src.triage.pipeline.EmailScorer') as mock_scorer_class:
        mock_scorer_class.return_value = mock_email_scorer

        pipeline = TriagePipeline(
            gmail_client=mock_gmail_client,
            gmail_ops=mock_gmail_ops,
            database=temp_db,
            model_path="dummy_model.txt",
            use_embeddings=False
        )

        return pipeline
