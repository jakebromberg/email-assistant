"""Tests for Gmail client."""

from unittest.mock import Mock, patch

import pytest

from src.gmail.auth import GmailAuthenticator
from src.gmail.client import GmailClient


class TestGmailClient:
    """Test GmailClient class."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock authenticator."""
        auth = Mock(spec=GmailAuthenticator)
        auth.authenticate.return_value = Mock(valid=True)
        return auth

    @pytest.fixture
    def client(self, mock_auth):
        """Create Gmail client with mock auth."""
        with patch('src.gmail.client.build'):
            client = GmailClient(mock_auth, enable_rate_limiting=False)
            client.service = Mock()
            return client

    @pytest.fixture
    def client_with_rate_limiting(self, mock_auth):
        """Create Gmail client with rate limiting enabled."""
        with patch('src.gmail.client.build'):
            client = GmailClient(mock_auth, enable_rate_limiting=True)
            client.service = Mock()
            return client

    def test_list_messages(self, client):
        """Test listing messages."""
        # Mock API response
        mock_response = {
            'messages': [
                {'id': 'msg1'},
                {'id': 'msg2'}
            ],
            'nextPageToken': 'token123'
        }

        client.service.users().messages().list().execute.return_value = mock_response

        # Call method
        message_ids, next_token = client.list_messages(max_results=10)

        # Assert results
        assert message_ids == ['msg1', 'msg2']
        assert next_token == 'token123'

    def test_list_messages_with_query(self, client):
        """Test listing messages with query."""
        mock_response = {
            'messages': [{'id': 'msg1'}],
        }

        client.service.users().messages().list().execute.return_value = mock_response

        # Call with query
        message_ids, _ = client.list_messages(query="is:unread")

        # Verify query was passed
        assert len(message_ids) == 1

    def test_get_messages_batch_simple(self, client):
        """Test batch fetching messages."""
        # Mock batch request
        mock_batch = Mock()
        client.service.new_batch_http_request.return_value = mock_batch

        # Mock responses
        mock_responses = [
            {
                'id': 'msg1',
                'threadId': 'thread1',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'sender@example.com'},
                        {'name': 'To', 'value': 'recipient@example.com'},
                        {'name': 'Subject', 'value': 'Test Email'},
                        {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'},
                    ],
                    'body': {'data': 'dGVzdA=='}  # 'test' in base64
                },
                'labelIds': ['INBOX']
            }
        ]

        def mock_execute():
            # Simulate successful batch execution
            # In real scenario, callbacks would be called
            pass

        mock_batch.execute.side_effect = mock_execute

        # Create a real batch callback that we can trigger
        callbacks_triggered = []

        def mock_add(request, callback):
            # Store callback and trigger it immediately with mock response
            callbacks_triggered.append(callback)
            callback(None, mock_responses[0], None)

        mock_batch.add.side_effect = mock_add

        # Call method
        emails = client.get_messages_batch(['msg1'])

        # Assert
        assert len(emails) == 1
        assert emails[0].message_id == 'msg1'

    def test_get_messages_batch_with_rate_limiting(self, client_with_rate_limiting):
        """Test batch fetching with rate limiting enabled."""
        client = client_with_rate_limiting

        # Mock batch request
        mock_batch = Mock()
        client.service.new_batch_http_request.return_value = mock_batch

        def mock_execute():
            pass

        mock_batch.execute.side_effect = mock_execute

        mock_response = {
            'id': 'msg1',
            'threadId': 'thread1',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'To', 'value': 'recipient@example.com'},
                    {'name': 'Subject', 'value': 'Test'},
                    {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'},
                ],
                'body': {'data': 'dGVzdA=='}
            },
            'labelIds': ['INBOX']
        }

        def mock_add(request, callback):
            callback(None, mock_response, None)

        mock_batch.add.side_effect = mock_add

        # Call method
        emails = client.get_messages_batch(['msg1'], batch_size=10)

        # Rate limiter should be active
        assert client.rate_limiter is not None
        assert len(emails) == 1

    def test_get_messages_batch_error_handling(self, client):
        """Test batch fetching handles errors gracefully."""
        # Mock batch request
        mock_batch = Mock()
        client.service.new_batch_http_request.return_value = mock_batch

        def mock_execute():
            pass

        mock_batch.execute.side_effect = mock_execute

        # Simulate all messages failing
        def mock_add(request, callback):
            mock_error = Mock()
            mock_error.resp = Mock()
            mock_error.resp.status = 404
            callback(None, None, mock_error)

        mock_batch.add.side_effect = mock_add

        # Call method - should not crash
        emails = client.get_messages_batch(['msg1'])

        # Should return empty list when all fail
        assert emails == []

    def test_get_messages_batch_different_formats(self, client):
        """Test batch fetching with different message formats."""
        mock_batch = Mock()
        client.service.new_batch_http_request.return_value = mock_batch

        def mock_execute():
            pass

        mock_batch.execute.side_effect = mock_execute

        requests_made = []

        def mock_add(request, callback):
            # Track what format was requested
            requests_made.append(request)
            mock_response = {
                'id': 'msg1',
                'threadId': 'thread1',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'sender@example.com'},
                        {'name': 'To', 'value': 'recipient@example.com'},
                        {'name': 'Subject', 'value': 'Test'},
                        {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'},
                    ],
                    'body': {'data': 'dGVzdA=='}
                },
                'labelIds': ['INBOX']
            }
            callback(None, mock_response, None)

        mock_batch.add.side_effect = mock_add

        # Test with 'metadata' format
        emails = client.get_messages_batch(['msg1'], message_format='metadata')

        assert len(emails) == 1
        # Verify format was passed (would need to inspect the request mock)
