"""Gmail-specific test fixtures."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_gmail_message_with_attachments(sample_gmail_message):
    """Gmail message with attachments."""
    message = sample_gmail_message.copy()
    message['payload']['parts'].append({
        'filename': 'document.pdf',
        'mimeType': 'application/pdf',
        'body': {
            'attachmentId': 'att123',
            'size': 12345
        }
    })
    return message


@pytest.fixture
def mock_gmail_message_multipart_nested(sample_gmail_message):
    """Gmail message with deeply nested multipart structure."""
    message = sample_gmail_message.copy()
    message['payload'] = {
        'mimeType': 'multipart/mixed',
        'parts': [
            {
                'mimeType': 'multipart/alternative',
                'parts': [
                    {
                        'mimeType': 'text/plain',
                        'body': {'data': 'VGVzdCBib2R5'}
                    },
                    {
                        'mimeType': 'text/html',
                        'body': {'data': 'PHA+VGVzdCBib2R5PC9wPg=='}
                    }
                ]
            }
        ]
    }
    return message


@pytest.fixture
def mock_gmail_message_plain_only():
    """Gmail message with plain text only (no HTML)."""
    return {
        'id': 'plain123',
        'threadId': 'thread_plain',
        'labelIds': ['INBOX'],
        'snippet': 'Plain text email',
        'internalDate': '1704038400000',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'sender@example.com'},
                {'name': 'Subject', 'value': 'Plain Text Email'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 00:00:00 +0000'},
            ],
            'mimeType': 'text/plain',
            'body': {
                'data': 'VGhpcyBpcyBhIHBsYWluIHRleHQgZW1haWw='  # "This is a plain text email"
            }
        }
    }


@pytest.fixture
def mock_gmail_message_html_only():
    """Gmail message with HTML only (no plain text)."""
    return {
        'id': 'html123',
        'threadId': 'thread_html',
        'labelIds': ['INBOX'],
        'snippet': 'HTML email',
        'internalDate': '1704038400000',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'sender@example.com'},
                {'name': 'Subject', 'value': 'HTML Email'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 00:00:00 +0000'},
            ],
            'mimeType': 'text/html',
            'body': {
                'data': 'PGh0bWw+PGJvZHk+VGhpcyBpcyBhbiBIVE1MIGVtYWlsPC9ib2R5PjwvaHRtbD4='
            }
        }
    }


@pytest.fixture
def mock_gmail_message_with_list_unsubscribe():
    """Gmail message with List-Unsubscribe header (newsletter)."""
    message = {
        'id': 'newsletter123',
        'threadId': 'thread_newsletter',
        'labelIds': ['INBOX'],
        'snippet': 'Newsletter email',
        'internalDate': '1704038400000',
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'newsletter@example.com'},
                {'name': 'Subject', 'value': 'Weekly Newsletter'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 00:00:00 +0000'},
                {'name': 'List-Unsubscribe', 'value': '<mailto:unsubscribe@example.com>'},
            ],
            'mimeType': 'text/html',
            'body': {
                'data': 'PGh0bWw+PGJvZHk+TmV3c2xldHRlciBjb250ZW50PC9ib2R5PjwvaHRtbD4='
            }
        }
    }
    return message


@pytest.fixture
def mock_operation_result():
    """Create a mock successful operation result."""
    from src.gmail.operations import OperationResult

    return OperationResult(
        success=True,
        message_ids=['msg1', 'msg2'],
        message="Operation successful"
    )


@pytest.fixture
def mock_operation_result_failed():
    """Create a mock failed operation result."""
    from src.gmail.operations import OperationResult

    return OperationResult(
        success=False,
        message_ids=[],
        message="Operation failed: API error"
    )
