"""Tests for Gmail client."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.gmail.client import GmailClient
from src.gmail.auth import GmailAuthenticator


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
            client = GmailClient(mock_auth)
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

    # TODO: Add more tests for:
    # - get_message
    # - get_messages_batch
    # - Error handling
    # - Retry logic
