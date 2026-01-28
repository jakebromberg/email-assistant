"""Tests for Gmail operations."""

from unittest.mock import Mock

import pytest

from src.gmail.operations import GmailOperations


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

    # Dry-Run Mode Tests

    def test_mark_unread_dry_run(self, operations):
        """Test mark unread in dry-run mode."""
        result = operations.mark_unread(['msg1', 'msg2'], dry_run=True)

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']
        assert 'DRY RUN' in result.message
        assert 'unread' in result.message.lower()

    def test_move_to_inbox_dry_run(self, operations):
        """Test move to inbox in dry-run mode."""
        result = operations.move_to_inbox(['msg1'], dry_run=True)

        assert result.success
        assert result.message_ids == ['msg1']
        assert 'DRY RUN' in result.message
        assert 'inbox' in result.message.lower()

    def test_add_labels_dry_run(self, operations):
        """Test add labels in dry-run mode."""
        result = operations.add_labels(
            ['msg1', 'msg2'],
            ['Bot/Newsletter-Tech', 'Bot/AutoArchived'],
            dry_run=True
        )

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']
        assert 'DRY RUN' in result.message

    def test_remove_labels_dry_run(self, operations):
        """Test remove labels in dry-run mode."""
        result = operations.remove_labels(
            ['msg1'],
            ['Bot/LowConfidence'],
            dry_run=True
        )

        assert result.success
        assert result.message_ids == ['msg1']
        assert 'DRY RUN' in result.message

    def test_batch_modify_dry_run_with_labels(self, operations):
        """Test batch modify dry-run with add/remove labels."""
        result = operations.batch_modify(
            message_ids=['msg1', 'msg2'],
            add_label_ids=['Label_1'],
            remove_label_ids=['Label_2'],
            dry_run=True
        )

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']
        assert 'DRY RUN' in result.message

    def test_dry_run_does_not_call_api(self, operations, mock_client):
        """CRITICAL: Verify no API calls made in dry-run mode."""
        # Call all operations in dry-run mode
        operations.mark_read(['msg1'], dry_run=True)
        operations.mark_unread(['msg1'], dry_run=True)
        operations.archive(['msg1'], dry_run=True)
        operations.move_to_inbox(['msg1'], dry_run=True)
        operations.add_labels(['msg1'], ['TestLabel'], dry_run=True)
        operations.remove_labels(['msg1'], ['TestLabel'], dry_run=True)
        operations.batch_modify(['msg1'], dry_run=True)

        # Verify service was never called
        mock_client.service.users().messages().batchModify.assert_not_called()

    # Actual Operation Tests

    def test_mark_read_actual(self, operations, mock_client):
        """Test actual mark_read operation."""
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.mark_read(['msg1', 'msg2'], dry_run=False)

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']

        # Verify API was called with correct parameters
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert call_args[1]['userId'] == 'me'
        assert call_args[1]['body']['ids'] == ['msg1', 'msg2']
        assert call_args[1]['body']['removeLabelIds'] == ['UNREAD']

    def test_mark_unread_actual(self, operations, mock_client):
        """Test actual mark_unread operation."""
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.mark_unread(['msg1'], dry_run=False)

        assert result.success
        assert result.message_ids == ['msg1']

        # Verify API was called with correct parameters
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert call_args[1]['body']['addLabelIds'] == ['UNREAD']

    def test_archive_actual(self, operations, mock_client):
        """Test actual archive operation."""
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.archive(['msg1', 'msg2'], dry_run=False)

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']

        # Verify API was called with correct parameters
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert call_args[1]['body']['removeLabelIds'] == ['INBOX']

    def test_move_to_inbox_actual(self, operations, mock_client):
        """Test actual move_to_inbox operation."""
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.move_to_inbox(['msg1'], dry_run=False)

        assert result.success
        assert result.message_ids == ['msg1']

        # Verify API was called with correct parameters
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert call_args[1]['body']['addLabelIds'] == ['INBOX']

    def test_add_labels_actual(self, operations, mock_client):
        """Test actual add_labels operation with label creation."""
        # Mock get_all_labels to return existing labels
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'},
            {'name': 'ExistingLabel', 'id': 'Label_123'}
        ]

        # Mock label creation
        mock_client.service.users().labels().create().execute.return_value = {
            'id': 'Label_456',
            'name': 'Bot/Newsletter-Tech'
        }

        # Mock batch modify
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.add_labels(
            ['msg1'],
            ['Bot/Newsletter-Tech'],
            dry_run=False
        )

        assert result.success
        assert result.message_ids == ['msg1']

        # Verify label was created (called with correct parameters)
        create_calls = [call for call in mock_client.service.users().labels().create.call_args_list
                       if call[1].get('body') is not None]
        assert len(create_calls) > 0
        assert create_calls[0][1]['body']['name'] == 'Bot/Newsletter-Tech'

        # Verify batch modify was called with created label ID
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert 'Label_456' in call_args[1]['body']['addLabelIds']

    def test_remove_labels_actual(self, operations, mock_client):
        """Test actual remove_labels operation."""
        # Mock get_all_labels to return existing labels
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'},
            {'name': 'Bot/LowConfidence', 'id': 'Label_789'}
        ]

        # Mock batch modify
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.remove_labels(
            ['msg1', 'msg2'],
            ['Bot/LowConfidence'],
            dry_run=False
        )

        assert result.success
        assert result.message_ids == ['msg1', 'msg2']

        # Verify batch modify was called with correct label ID
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert 'Label_789' in call_args[1]['body']['removeLabelIds']

    # Label Management Tests

    def test_create_label(self, operations, mock_client):
        """Test label creation."""
        mock_client.service.users().labels().create().execute.return_value = {
            'id': 'Label_NewLabel',
            'name': 'Bot/Newsletter-Tech'
        }

        label_id = operations.create_label('Bot/Newsletter-Tech')

        assert label_id == 'Label_NewLabel'

        # Verify create was called with correct parameters
        call_args = mock_client.service.users().labels().create.call_args
        assert call_args[1]['userId'] == 'me'
        assert call_args[1]['body']['name'] == 'Bot/Newsletter-Tech'
        assert call_args[1]['body']['labelListVisibility'] == 'labelShow'

    def test_get_or_create_label_existing(self, operations, mock_client):
        """Test getting existing label."""
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'},
            {'name': 'Bot/Newsletter-Tech', 'id': 'Label_Existing'}
        ]

        label_id = operations.get_or_create_label('Bot/Newsletter-Tech')

        assert label_id == 'Label_Existing'

        # Verify create was NOT called
        mock_client.service.users().labels().create.assert_not_called()

    def test_get_or_create_label_new(self, operations, mock_client):
        """Test creating new label."""
        # First call: get existing labels (doesn't exist)
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'}
        ]

        # Mock label creation
        mock_client.service.users().labels().create().execute.return_value = {
            'id': 'Label_New',
            'name': 'Bot/NewLabel'
        }

        label_id = operations.get_or_create_label('Bot/NewLabel')

        assert label_id == 'Label_New'

        # Verify create was called (check call with body parameter exists)
        create_calls = [call for call in mock_client.service.users().labels().create.call_args_list
                       if call[1].get('body') is not None]
        assert len(create_calls) > 0
        assert create_calls[0][1]['body']['name'] == 'Bot/NewLabel'

    def test_get_label_map_caching(self, operations, mock_client):
        """Test label map cache."""
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'},
            {'name': 'Label1', 'id': 'Label_1'}
        ]

        # Call twice
        label_map1 = operations._get_label_map()
        label_map2 = operations._get_label_map()

        # Should only call API once due to caching
        assert mock_client.get_all_labels.call_count == 1
        assert label_map1 == label_map2
        assert 'INBOX' in label_map1
        assert 'Label1' in label_map1

    def test_get_label_id_missing(self, operations, mock_client):
        """Test getting non-existent label."""
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'}
        ]

        label_id = operations.get_label_id('NonExistentLabel')

        assert label_id is None

    # Edge Cases Tests

    def test_batch_modify_mixed_labels(self, operations, mock_client):
        """Test add and remove in same call."""
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.batch_modify(
            message_ids=['msg1'],
            add_label_ids=['Label_Add'],
            remove_label_ids=['Label_Remove'],
            dry_run=False
        )

        assert result.success

        # Verify both add and remove in API call
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert call_args[1]['body']['addLabelIds'] == ['Label_Add']
        assert call_args[1]['body']['removeLabelIds'] == ['Label_Remove']

    def test_remove_labels_nonexistent(self, operations, mock_client):
        """Test removing labels that don't exist."""
        # Mock get_all_labels to return labels (none match)
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'}
        ]

        result = operations.remove_labels(
            ['msg1'],
            ['NonExistentLabel'],
            dry_run=False
        )

        assert result.success
        assert "don't exist" in result.message

        # Verify batch modify was NOT called
        mock_client.service.users().messages().batchModify.assert_not_called()

    def test_batch_modify_error_handling(self, operations, mock_client):
        """Test HttpError handling."""
        from googleapiclient.errors import HttpError

        # Mock API to raise error
        mock_response = Mock()
        mock_response.status = 400
        mock_client.service.users().messages().batchModify().execute.side_effect = \
            HttpError(mock_response, b'Bad Request')

        result = operations.batch_modify(
            message_ids=['msg1'],
            add_label_ids=['Label_1'],
            dry_run=False
        )

        assert not result.success
        assert 'Failed' in result.message

    def test_empty_message_ids_all_operations(self, operations, mock_client):
        """Test empty list for all operations."""
        # Mock get_all_labels for label operations
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'},
            {'name': 'TestLabel', 'id': 'Label_Test'}
        ]

        # All should succeed with empty list
        assert operations.mark_read([]).success
        assert operations.mark_unread([]).success
        assert operations.archive([]).success
        assert operations.move_to_inbox([]).success
        assert operations.add_labels([], ['TestLabel']).success
        assert operations.remove_labels([], ['TestLabel']).success

    def test_label_cache_invalidation(self, operations, mock_client):
        """Test cache invalidation after create."""
        # Initial cache
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'}
        ]

        # Get initial cache
        label_map1 = operations._get_label_map()
        assert len(label_map1) == 1

        # Create a new label (should invalidate cache)
        mock_client.service.users().labels().create().execute.return_value = {
            'id': 'Label_New',
            'name': 'NewLabel'
        }

        operations.create_label('NewLabel')

        # Update mock to return new label
        mock_client.get_all_labels.return_value = [
            {'name': 'INBOX', 'id': 'INBOX'},
            {'name': 'NewLabel', 'id': 'Label_New'}
        ]

        # Get cache again (should fetch fresh)
        label_map2 = operations._get_label_map()
        assert len(label_map2) == 2
        assert 'NewLabel' in label_map2

    def test_batch_modify_no_labels(self, operations, mock_client):
        """Test batch_modify with no add/remove."""
        mock_client.service.users().messages().batchModify().execute.return_value = {}

        result = operations.batch_modify(
            message_ids=['msg1'],
            add_label_ids=None,
            remove_label_ids=None,
            dry_run=False
        )

        assert result.success

        # Verify API was called but without label modifications
        call_args = mock_client.service.users().messages().batchModify.call_args
        assert 'addLabelIds' not in call_args[1]['body']
        assert 'removeLabelIds' not in call_args[1]['body']
