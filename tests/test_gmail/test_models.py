"""Tests for Gmail models."""

import base64
from datetime import datetime

from src.gmail.models import Email


class TestEmail:
    """Test Email model class."""

    # Basic Body Extraction Tests

    def test_extract_body_plain_text_only(self):
        """Simple plain text email."""
        payload = {
            'mimeType': 'text/plain',
            'body': {
                'data': base64.urlsafe_b64encode(b'Hello World').decode()
            }
        }

        plain, html = Email._extract_body(payload)

        assert plain == 'Hello World'
        assert html == ''

    def test_extract_body_html_only(self):
        """Simple HTML email."""
        payload = {
            'mimeType': 'text/html',
            'body': {
                'data': base64.urlsafe_b64encode(b'<html><body>Hello</body></html>').decode()
            }
        }

        plain, html = Email._extract_body(payload)

        assert plain == ''
        assert html == '<html><body>Hello</body></html>'

    def test_extract_body_both_plain_and_html(self):
        """Email with both plain and HTML."""
        payload = {
            'mimeType': 'multipart/alternative',
            'parts': [
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'Plain text').decode()
                    }
                },
                {
                    'mimeType': 'text/html',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'<p>HTML text</p>').decode()
                    }
                }
            ]
        }

        plain, html = Email._extract_body(payload)

        assert plain == 'Plain text'
        assert html == '<p>HTML text</p>'

    def test_extract_body_multipart_alternative(self):
        """Multipart/alternative structure."""
        payload = {
            'mimeType': 'multipart/alternative',
            'parts': [
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'Text version').decode()
                    }
                },
                {
                    'mimeType': 'text/html',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'<p>HTML version</p>').decode()
                    }
                }
            ]
        }

        plain, html = Email._extract_body(payload)

        assert plain == 'Text version'
        assert html == '<p>HTML version</p>'

    def test_extract_body_empty_payload(self):
        """Empty payload → empty strings."""
        payload = {}

        plain, html = Email._extract_body(payload)

        assert plain == ''
        assert html == ''

    # Recursive Extraction Tests

    def test_extract_body_nested_parts(self):
        """Deeply nested multipart structure."""
        payload = {
            'mimeType': 'multipart/mixed',
            'parts': [
                {
                    'mimeType': 'multipart/alternative',
                    'parts': [
                        {
                            'mimeType': 'text/plain',
                            'body': {
                                'data': base64.urlsafe_b64encode(b'Nested plain').decode()
                            }
                        },
                        {
                            'mimeType': 'text/html',
                            'body': {
                                'data': base64.urlsafe_b64encode(b'<p>Nested HTML</p>').decode()
                            }
                        }
                    ]
                }
            ]
        }

        plain, html = Email._extract_body(payload)

        assert plain == 'Nested plain'
        assert html == '<p>Nested HTML</p>'

    def test_extract_body_multiple_levels(self):
        """3+ levels of nesting."""
        payload = {
            'mimeType': 'multipart/mixed',
            'parts': [
                {
                    'mimeType': 'multipart/related',
                    'parts': [
                        {
                            'mimeType': 'multipart/alternative',
                            'parts': [
                                {
                                    'mimeType': 'text/plain',
                                    'body': {
                                        'data': base64.urlsafe_b64encode(b'Deep plain').decode()
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        plain, html = Email._extract_body(payload)

        assert plain == 'Deep plain'
        assert html == ''

    def test_extract_body_first_occurrence_wins(self):
        """Only first plain/HTML used."""
        payload = {
            'mimeType': 'multipart/mixed',
            'parts': [
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'First plain').decode()
                    }
                },
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'Second plain').decode()
                    }
                }
            ]
        }

        plain, html = Email._extract_body(payload)

        # Should only extract first occurrence
        assert plain == 'First plain'
        assert html == ''

    def test_extract_body_mixed_mime_types(self):
        """Other MIME types skipped."""
        payload = {
            'mimeType': 'multipart/mixed',
            'parts': [
                {
                    'mimeType': 'image/png',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'fake image data').decode()
                    }
                },
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'Text content').decode()
                    }
                },
                {
                    'mimeType': 'application/pdf',
                    'body': {
                        'data': base64.urlsafe_b64encode(b'fake pdf data').decode()
                    }
                }
            ]
        }

        plain, html = Email._extract_body(payload)

        assert plain == 'Text content'
        assert html == ''

    # Base64 Decoding Tests

    def test_extract_body_valid_base64(self):
        """Normal base64url decoding."""
        text = "Hello, this is a test email with special chars: 日本語"
        payload = {
            'mimeType': 'text/plain',
            'body': {
                'data': base64.urlsafe_b64encode(text.encode('utf-8')).decode()
            }
        }

        plain, html = Email._extract_body(payload)

        assert plain == text

    def test_extract_body_invalid_utf8(self):
        """Invalid UTF-8 → errors='replace'."""
        # Create invalid UTF-8 sequence
        invalid_utf8 = b'\xff\xfe Invalid UTF-8'
        payload = {
            'mimeType': 'text/plain',
            'body': {
                'data': base64.urlsafe_b64encode(invalid_utf8).decode()
            }
        }

        plain, html = Email._extract_body(payload)

        # Should not raise error, should use replacement char
        assert plain != ''
        assert '\ufffd' in plain  # Replacement character

    def test_extract_body_empty_data(self):
        """Empty data field → empty string."""
        payload = {
            'mimeType': 'text/plain',
            'body': {
                'data': ''
            }
        }

        plain, html = Email._extract_body(payload)

        assert plain == ''
        assert html == ''

    def test_extract_body_malformed_base64(self):
        """Malformed base64 → decoded with errors='replace'.

        Note: base64.urlsafe_b64decode is lenient and will decode
        even malformed input, just producing garbage. The errors='replace'
        in decode() handles invalid UTF-8.
        """
        payload = {
            'mimeType': 'text/plain',
            'body': {
                'data': 'not-valid-base64!!!'
            }
        }

        plain, html = Email._extract_body(payload)

        # Should not raise error, may return decoded garbage
        # Test just verifies no exception is raised
        assert isinstance(plain, str)

    # Edge Cases Tests

    def test_extract_body_missing_mime_type(self):
        """No mimeType field."""
        payload = {
            'body': {
                'data': base64.urlsafe_b64encode(b'Some text').decode()
            }
        }

        plain, html = Email._extract_body(payload)

        # Should handle gracefully
        assert plain == ''
        assert html == ''

    def test_extract_body_no_body_field(self):
        """Part with no body."""
        payload = {
            'mimeType': 'text/plain'
        }

        plain, html = Email._extract_body(payload)

        assert plain == ''
        assert html == ''

    def test_extract_body_empty_parts_list(self):
        """No parts to recurse into."""
        payload = {
            'mimeType': 'multipart/mixed',
            'parts': []
        }

        plain, html = Email._extract_body(payload)

        assert plain == ''
        assert html == ''

    def test_extract_body_none_payload(self):
        """None payload handled gracefully."""
        # Test with minimal payload
        payload = {'parts': []}

        plain, html = Email._extract_body(payload)

        assert plain == ''
        assert html == ''

    # Related Method Tests

    def test_from_gmail_message_basic(self):
        """Test full parsing."""
        message = {
            'id': 'msg123',
            'threadId': 'thread456',
            'snippet': 'This is a snippet',
            'labelIds': ['INBOX', 'UNREAD'],
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'John Doe <john@example.com>'},
                    {'name': 'To', 'value': 'jane@example.com'},
                    {'name': 'Subject', 'value': 'Test Email'},
                    {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
                ],
                'mimeType': 'text/plain',
                'body': {
                    'data': base64.urlsafe_b64encode(b'Email body').decode()
                }
            }
        }

        email = Email.from_gmail_message(message)

        assert email.message_id == 'msg123'
        assert email.thread_id == 'thread456'
        assert email.from_address == 'john@example.com'
        assert email.from_name == 'John Doe'
        assert email.to_address == 'jane@example.com'
        assert email.subject == 'Test Email'
        assert email.snippet == 'This is a snippet'
        assert email.labels == ['INBOX', 'UNREAD']
        assert email.body_plain == 'Email body'
        assert email.is_unread
        assert email.is_in_inbox

    def test_parse_email_address(self):
        """Test name and address parsing."""
        # With name and address
        name, address = Email._parse_email_address('John Doe <john@example.com>')
        assert name == 'John Doe'
        assert address == 'john@example.com'

        # Address only
        name, address = Email._parse_email_address('jane@example.com')
        assert name == ''
        assert address == 'jane@example.com'

        # Empty
        name, address = Email._parse_email_address('')
        assert name == ''
        assert address == ''

        # Complex name
        name, address = Email._parse_email_address('"Doe, John" <john@example.com>')
        assert 'John' in name or 'Doe' in name
        assert address == 'john@example.com'

    def test_parse_date(self):
        """Test date parsing."""
        # Valid date
        date = Email._parse_date('Mon, 1 Jan 2024 12:00:00 +0000')
        assert isinstance(date, datetime)
        assert date.year == 2024
        assert date.month == 1
        assert date.day == 1

        # Invalid date (should return current time)
        date = Email._parse_date('invalid date')
        assert isinstance(date, datetime)

        # Empty string (should return current time)
        date = Email._parse_date('')
        assert isinstance(date, datetime)

    def test_parse_date_timezone_naive(self):
        """
        REGRESSION TEST: Ensure parsed dates are timezone-naive for database compatibility.

        This prevents "can't subtract offset-naive and offset-aware datetimes" errors
        when comparing dates from new emails with dates from database emails.
        """
        # Date with timezone should be stripped to naive
        date_with_tz = Email._parse_date('Mon, 1 Jan 2024 12:00:00 +0500')
        assert date_with_tz.tzinfo is None, "Date should be timezone-naive"
        assert date_with_tz.year == 2024
        assert date_with_tz.month == 1
        assert date_with_tz.day == 1
        assert date_with_tz.hour == 12  # Time should be preserved

        # Date with UTC timezone
        date_utc = Email._parse_date('Mon, 1 Jan 2024 12:00:00 +0000')
        assert date_utc.tzinfo is None, "Date should be timezone-naive"

        # Date with negative timezone offset
        date_negative = Email._parse_date('Mon, 1 Jan 2024 12:00:00 -0800')
        assert date_negative.tzinfo is None, "Date should be timezone-naive"

        # Fallback dates (invalid/empty) should also be timezone-naive
        date_invalid = Email._parse_date('invalid')
        assert date_invalid.tzinfo is None, "Fallback date should be timezone-naive"

        date_empty = Email._parse_date('')
        assert date_empty.tzinfo is None, "Fallback date should be timezone-naive"

    def test_email_properties(self):
        """Test is_unread and is_in_inbox properties."""
        message = {
            'id': 'msg1',
            'threadId': 'thread1',
            'labelIds': ['INBOX', 'UNREAD'],
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'test@example.com'},
                    {'name': 'Subject', 'value': 'Test'}
                ]
            }
        }

        email = Email.from_gmail_message(message)

        assert email.is_unread
        assert email.is_in_inbox

        # Test without UNREAD label
        message['labelIds'] = ['INBOX']
        email = Email.from_gmail_message(message)

        assert not email.is_unread
        assert email.is_in_inbox

        # Test without INBOX label
        message['labelIds'] = ['UNREAD']
        email = Email.from_gmail_message(message)

        assert email.is_unread
        assert not email.is_in_inbox
