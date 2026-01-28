"""Utility functions for testing."""

import base64
from datetime import datetime, timedelta
from typing import Any


def create_gmail_message(
    message_id: str = "msg123",
    thread_id: str = "thread123",
    from_email: str = "sender@example.com",
    from_name: str = "Test Sender",
    subject: str = "Test Subject",
    body_plain: str = "Test body",
    body_html: str = "<p>Test body</p>",
    labels: list[str] = None,
    date: datetime = None,
    headers: list[dict[str, str]] = None,
    has_attachments: bool = False
) -> dict[str, Any]:
    """
    Create a Gmail API message structure for testing.

    Args:
        message_id: Message ID
        thread_id: Thread ID
        from_email: Sender email address
        from_name: Sender display name
        subject: Email subject
        body_plain: Plain text body
        body_html: HTML body
        labels: Gmail labels
        date: Email date
        headers: Additional headers
        has_attachments: Whether to include an attachment

    Returns:
        Gmail API message dict
    """
    if labels is None:
        labels = ["INBOX", "UNREAD"]
    if date is None:
        date = datetime.now()
    if headers is None:
        headers = []

    # Convert date to Gmail timestamp (milliseconds since epoch)
    timestamp = str(int(date.timestamp() * 1000))

    # Encode body content
    plain_encoded = base64.urlsafe_b64encode(body_plain.encode()).decode()
    html_encoded = base64.urlsafe_b64encode(body_html.encode()).decode()

    # Build headers
    all_headers = [
        {'name': 'From', 'value': f'{from_name} <{from_email}>'},
        {'name': 'Subject', 'value': subject},
        {'name': 'Date', 'value': date.strftime('%a, %d %b %Y %H:%M:%S +0000')},
    ] + headers

    # Build parts
    parts = [
        {
            'mimeType': 'text/plain',
            'body': {'data': plain_encoded}
        },
        {
            'mimeType': 'text/html',
            'body': {'data': html_encoded}
        }
    ]

    # Add attachment if requested
    if has_attachments:
        parts.append({
            'filename': 'document.pdf',
            'mimeType': 'application/pdf',
            'body': {
                'attachmentId': 'att123',
                'size': 12345
            }
        })

    return {
        'id': message_id,
        'threadId': thread_id,
        'labelIds': labels,
        'snippet': body_plain[:100],
        'internalDate': timestamp,
        'payload': {
            'headers': all_headers,
            'mimeType': 'multipart/alternative',
            'parts': parts
        }
    }


def create_gmail_newsletter_message(
    message_id: str = "newsletter123",
    from_email: str = "newsletter@example.com",
    subject: str = "Weekly Newsletter"
) -> dict[str, Any]:
    """
    Create a Gmail API message structure for a newsletter.

    Args:
        message_id: Message ID
        from_email: Newsletter sender email
        subject: Newsletter subject

    Returns:
        Gmail API message dict with List-Unsubscribe header
    """
    return create_gmail_message(
        message_id=message_id,
        from_email=from_email,
        from_name="Newsletter",
        subject=subject,
        body_plain="This is a newsletter",
        body_html="<html><body>This is a newsletter</body></html>",
        headers=[
            {'name': 'List-Unsubscribe', 'value': '<mailto:unsubscribe@example.com>'}
        ]
    )


def assert_email_equals(actual, expected, fields=None):
    """
    Assert two email objects are equal.

    Args:
        actual: Actual email object
        expected: Expected email object (or dict of expected values)
        fields: List of fields to compare (default: common fields)
    """
    if fields is None:
        fields = [
            'message_id', 'from_address', 'to_address', 'subject',
            'body_plain', 'was_read', 'was_archived'
        ]

    for field in fields:
        expected_value = expected.get(field) if isinstance(expected, dict) else getattr(expected, field)
        actual_value = getattr(actual, field)
        assert actual_value == expected_value, f"Field {field} mismatch: {actual_value} != {expected_value}"


def assert_email_features_complete(features):
    """
    Assert that email features object has all required fields.

    Args:
        features: EmailFeatures object to check
    """
    required_metadata = [
        'sender_domain', 'is_newsletter', 'subject_length', 'body_length',
        'has_attachments', 'thread_length', 'day_of_week', 'hour_of_day'
    ]

    required_historical = [
        'sender_email_count', 'sender_open_rate'
    ]

    for field in required_metadata:
        assert hasattr(features, field), f"Missing metadata feature: {field}"
        assert getattr(features, field) is not None, f"Metadata feature {field} is None"

    for field in required_historical:
        assert hasattr(features, field), f"Missing historical feature: {field}"
        # Note: sender_days_since_last can be None for first email


def create_date_range(days_back: int = 7, start_date: datetime = None) -> list[datetime]:
    """
    Create a list of dates going back N days.

    Args:
        days_back: Number of days to go back
        start_date: Starting date (default: now)

    Returns:
        List of datetime objects
    """
    if start_date is None:
        start_date = datetime.now()
    return [start_date - timedelta(days=i) for i in range(days_back)]


def mock_gmail_batch_response(emails: list[dict[str, Any]]) -> list[Any]:
    """
    Create a mock Gmail batch response.

    Args:
        emails: List of Gmail message dicts

    Returns:
        List of mock email objects
    """
    from src.gmail.models import Email

    return [Email.from_gmail_message(email) for email in emails]


def create_mock_features(
    message_id: str,
    is_newsletter: bool = False,
    sender_open_rate: float = 0.5,
    sender_email_count: int = 5
) -> dict[str, Any]:
    """
    Create a mock features dictionary for testing.

    Args:
        message_id: Message ID
        is_newsletter: Whether email is a newsletter
        sender_open_rate: Sender's historical open rate
        sender_email_count: Number of emails from sender

    Returns:
        Dictionary of feature values
    """
    return {
        'message_id': message_id,
        'sender_domain': 'example.com',
        'is_newsletter': is_newsletter,
        'subject_length': 20,
        'body_length': 100,
        'has_attachments': False,
        'thread_length': 1,
        'day_of_week': 1,
        'hour_of_day': 12,
        'sender_email_count': sender_email_count,
        'sender_open_rate': sender_open_rate,
        'sender_days_since_last': 7,
        'domain_open_rate': 0.5
    }


def assert_operation_result_success(result, expected_message_ids: list[str] = None):
    """
    Assert that an OperationResult indicates success.

    Args:
        result: OperationResult object
        expected_message_ids: Expected list of message IDs (optional)
    """
    assert result.success, f"Operation failed: {result.message}"

    if expected_message_ids is not None:
        assert result.message_ids == expected_message_ids, \
            f"Message IDs mismatch: {result.message_ids} != {expected_message_ids}"


def assert_operation_result_failure(result, expected_message: str = None):
    """
    Assert that an OperationResult indicates failure.

    Args:
        result: OperationResult object
        expected_message: Expected error message substring (optional)
    """
    assert not result.success, "Operation succeeded when failure was expected"

    if expected_message is not None:
        assert expected_message in result.message, \
            f"Expected message '{expected_message}' not in '{result.message}'"


def create_mock_triage_decision(
    action: str = 'keep',
    confidence: str = 'low',
    score: float = 0.5,
    labels: list[str] = None
) -> dict[str, Any]:
    """
    Create a mock triage decision for testing.

    Args:
        action: Decision action ('keep' or 'archive')
        confidence: Confidence level ('high' or 'low')
        score: Prediction score (0.0-1.0)
        labels: List of labels to apply

    Returns:
        Decision dictionary
    """
    if labels is None:
        labels = []

    return {
        'action': action,
        'confidence': confidence,
        'score': score,
        'reasoning': f'Test {action} decision',
        'labels': labels
    }


def assert_triage_decision_valid(decision: dict[str, Any]):
    """
    Assert that a triage decision has all required fields.

    Args:
        decision: Decision dictionary to validate
    """
    required_fields = ['action', 'confidence', 'score', 'reasoning', 'labels']

    for field in required_fields:
        assert field in decision, f"Missing required field: {field}"

    assert decision['action'] in ['keep', 'archive'], \
        f"Invalid action: {decision['action']}"

    assert decision['confidence'] in ['high', 'low'], \
        f"Invalid confidence: {decision['confidence']}"

    assert 0.0 <= decision['score'] <= 1.0, \
        f"Invalid score: {decision['score']}"

    assert isinstance(decision['labels'], list), \
        f"Labels must be a list, got {type(decision['labels'])}"


def compare_emails_ignore_dates(email1, email2) -> bool:
    """
    Compare two email objects ignoring timestamp fields.

    Useful for comparing emails where exact timestamps don't matter.

    Args:
        email1: First email object
        email2: Second email object

    Returns:
        True if emails match (ignoring dates), False otherwise
    """
    fields_to_compare = [
        'message_id', 'thread_id', 'from_address', 'from_name',
        'to_address', 'subject', 'body_plain', 'body_html',
        'labels', 'snippet', 'was_read', 'was_archived'
    ]

    for field in fields_to_compare:
        val1 = getattr(email1, field, None)
        val2 = getattr(email2, field, None)
        if val1 != val2:
            return False

    return True
