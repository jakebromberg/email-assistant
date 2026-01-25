"""Tests for Gmail operations."""

import pytest
from unittest.mock import Mock

from src.gmail.operations import GmailOperations, OperationResult


class TestGmailOperations:
    """Test GmailOperations class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Gmail client."""
        client = Mock()
        client.service = Mock()
        return client

    @pytest.fixture
    def operations(self, mock_client):
        """Create operations instance."""
        return GmailOperations(mock_client)

    def test_archive_dry_run(self, operations):
        """Test archive in dry-run mode."""
        result = operations.archive(['msg1', 'msg2'], dry_run=True)

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']
        assert 'DRY RUN' in result.message

    def test_mark_read_dry_run(self, operations):
        """Test mark read in dry-run mode."""
        result = operations.mark_read(['msg1'], dry_run=True)

        assert result.success
        assert result.message_ids == ['msg1']
        assert 'DRY RUN' in result.message

    def test_batch_modify_empty_list(self, operations):
        """Test batch modify with empty list."""
        result = operations.batch_modify(message_ids=[])

        assert result.success
        assert len(result.message_ids) == 0

    # TODO: Add more tests for:
    # - Actual operations (not dry-run)
    # - Label creation
    # - Error handling
