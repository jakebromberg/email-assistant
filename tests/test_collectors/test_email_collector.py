"""Tests for email collector."""

from unittest.mock import Mock

import pytest

from src.collectors.email_collector import EmailCollector


class TestEmailCollectorInit:
    """Test EmailCollector initialization."""

    def test_init(self, mock_gmail_client, temp_db):
        """Test collector initialization."""
        collector = EmailCollector(mock_gmail_client, temp_db)

        assert collector.client is mock_gmail_client
        assert collector.database is temp_db


class TestExportHistorical:
    """Test export_historical method."""

    def test_export_historical_no_messages(self, mock_gmail_client, temp_db):
        """Test export when no messages found."""
        mock_gmail_client.list_all_messages = Mock(return_value=[])

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.export_historical(months=1)

        assert result == 0

    def test_export_historical_with_messages(self, mock_gmail_client, temp_db, gmail_email_factory):
        """Test export with messages."""
        mock_gmail_client.list_all_messages = Mock(return_value=["msg1", "msg2"])
        mock_emails = [
            gmail_email_factory(message_id="msg1", subject="Subject 1"),
            gmail_email_factory(message_id="msg2", subject="Subject 2"),
        ]
        mock_gmail_client.get_messages_batch = Mock(return_value=mock_emails)

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.export_historical(months=1, batch_size=10)

        assert result == 2
        mock_gmail_client.list_all_messages.assert_called_once()
        mock_gmail_client.get_messages_batch.assert_called()

    def test_export_historical_batch_error(self, mock_gmail_client, temp_db):
        """Test export handles batch errors gracefully."""
        mock_gmail_client.list_all_messages = Mock(return_value=["msg1", "msg2"])
        mock_gmail_client.get_messages_batch = Mock(side_effect=Exception("API Error"))

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.export_historical(months=1)

        assert result == 0

    def test_export_historical_empty_batch_response(self, mock_gmail_client, temp_db):
        """Test export handles empty batch response."""
        mock_gmail_client.list_all_messages = Mock(return_value=["msg1", "msg2"])
        mock_gmail_client.get_messages_batch = Mock(return_value=[])

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.export_historical(months=1)

        assert result == 0


class TestExportRecent:
    """Test export_recent method."""

    def test_export_recent_no_messages(self, mock_gmail_client, temp_db):
        """Test export recent when no messages found."""
        mock_gmail_client.list_all_messages = Mock(return_value=[])

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.export_recent(days=1)

        assert result == 0

    def test_export_recent_with_messages(self, mock_gmail_client, temp_db, gmail_email_factory):
        """Test export recent with messages."""
        mock_gmail_client.list_all_messages = Mock(return_value=["msg1"])
        mock_emails = [gmail_email_factory(message_id="msg1", subject="Recent Email")]
        mock_gmail_client.get_messages_batch = Mock(return_value=mock_emails)

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.export_recent(days=1)

        assert result == 1

    def test_export_recent_error(self, mock_gmail_client, temp_db):
        """Test export recent raises on error."""
        mock_gmail_client.list_all_messages = Mock(return_value=["msg1"])
        mock_gmail_client.get_messages_batch = Mock(side_effect=Exception("API Error"))

        collector = EmailCollector(mock_gmail_client, temp_db)

        with pytest.raises(Exception, match="API Error"):
            collector.export_recent(days=1)


class TestUpdateEmail:
    """Test update_email method."""

    def test_update_email_success(self, mock_gmail_client, temp_db, gmail_email_factory):
        """Test updating a single email."""
        mock_email = gmail_email_factory(message_id="msg1", subject="Updated Email")
        mock_gmail_client.get_message = Mock(return_value=mock_email)

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.update_email("msg1")

        assert result is True
        mock_gmail_client.get_message.assert_called_once_with("msg1")

    def test_update_email_error(self, mock_gmail_client, temp_db):
        """Test update email handles errors."""
        mock_gmail_client.get_message = Mock(side_effect=Exception("API Error"))

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.update_email("msg1")

        assert result is False


class TestUpdateEmailsBatch:
    """Test update_emails_batch method."""

    def test_update_emails_batch_empty(self, mock_gmail_client, temp_db):
        """Test update with empty list."""
        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.update_emails_batch([])

        assert result == 0

    def test_update_emails_batch_success(self, mock_gmail_client, temp_db, gmail_email_factory):
        """Test batch update success."""
        mock_emails = [gmail_email_factory(message_id="msg1", subject="Email 1")]
        mock_gmail_client.get_messages_batch = Mock(return_value=mock_emails)

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.update_emails_batch(["msg1"])

        assert result == 1

    def test_update_emails_batch_error(self, mock_gmail_client, temp_db):
        """Test batch update handles errors."""
        mock_gmail_client.get_messages_batch = Mock(side_effect=Exception("API Error"))

        collector = EmailCollector(mock_gmail_client, temp_db)
        result = collector.update_emails_batch(["msg1", "msg2"])

        assert result == 0


class TestGetStats:
    """Test get_stats method."""

    def test_get_stats(self, mock_gmail_client, temp_db):
        """Test getting collection stats."""
        collector = EmailCollector(mock_gmail_client, temp_db)
        stats = collector.get_stats()

        assert isinstance(stats, dict)
        assert 'emails' in stats
