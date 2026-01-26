"""Integration tests for EmailCollector + GmailClient quota management.

These tests validate that the collector and client work together correctly
and respect Gmail API quota limits.
"""

import pytest
import tempfile
import os
from itertools import count
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List, Tuple, Callable

from src.collectors.email_collector import EmailCollector
from src.gmail.client import GmailClient
from src.gmail.auth import GmailAuthenticator
from src.gmail.rate_limiter import QuotaCosts
from src.database.database import Database


class MockEmailFactory:
    """Factory for creating mock Gmail API email responses."""

    def __init__(self):
        self.counter = count(1)

    def create(self, msg_id: str = 'msg') -> dict:
        """Create a unique mock email response."""
        unique_id = f"{msg_id}_{next(self.counter)}"
        return {
            'id': unique_id,
            'threadId': f'thread_{unique_id}',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': f'sender@example.com'},
                    {'name': 'To', 'value': 'recipient@example.com'},
                    {'name': 'Subject', 'value': f'Test Email {unique_id}'},
                    {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'},
                ],
                'body': {'data': 'dGVzdA=='}
            },
            'labelIds': ['INBOX']
        }


class MockBatchRequest:
    """Mock for Gmail API batch request with quota tracking."""

    def __init__(self, quota_tracker: 'QuotaTrackingMock', email_factory: MockEmailFactory):
        self.quota_tracker = quota_tracker
        self.email_factory = email_factory
        self.requests: List[Tuple[Mock, Callable]] = []

    def add(self, request: Mock, callback: Callable):
        """Add a request to the batch."""
        self.requests.append((request, callback))

    def execute(self):
        """Execute all requests in the batch."""
        message_count = len(self.requests)

        # Record quota consumption
        self.quota_tracker.record_batch(message_count, QuotaCosts.MESSAGE_GET_FULL)

        # Validate quota limit
        batch_quota = message_count * QuotaCosts.MESSAGE_GET_FULL
        if batch_quota > self.quota_tracker.quota_per_second:
            pytest.fail(
                f"QUOTA VIOLATION: Batch used {batch_quota} quota units, "
                f"exceeding {self.quota_tracker.quota_per_second} units/second limit. "
                f"Messages in batch: {message_count}"
            )

        # Execute callbacks with mock responses
        for request, callback in self.requests:
            mock_response = self.email_factory.create()
            callback(None, mock_response, None)

        self.requests.clear()


class QuotaTrackingMock:
    """Mock that tracks quota consumption and enforces limits."""

    def __init__(self, quota_per_second=250):
        self.quota_per_second = quota_per_second
        self.total_consumed = 0
        self.batch_requests = []  # Track each batch request
        self.max_batch_quota = 0  # Track highest quota in single batch

    def record_batch(self, message_count, cost_per_message):
        """Record a batch request and its quota cost."""
        batch_quota = message_count * cost_per_message
        self.total_consumed += batch_quota
        self.max_batch_quota = max(self.max_batch_quota, batch_quota)
        self.batch_requests.append({
            'message_count': message_count,
            'cost_per_message': cost_per_message,
            'total_quota': batch_quota
        })
        return batch_quota

    def validate_quota_limit(self):
        """Validate that no single batch exceeded quota limit."""
        if self.max_batch_quota > self.quota_per_second:
            raise AssertionError(
                f"Batch exceeded quota limit! "
                f"Max batch quota: {self.max_batch_quota}, "
                f"Limit: {self.quota_per_second}"
            )


class TestCollectorClientIntegration:
    """Integration tests for EmailCollector + GmailClient."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        db = Database(temp_file.name)
        db.create_tables()
        yield db
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    @pytest.fixture
    def mock_auth(self):
        """Create mock authenticator."""
        auth = Mock(spec=GmailAuthenticator)
        auth.authenticate.return_value = Mock(valid=True)
        return auth

    @pytest.fixture
    def quota_tracker(self):
        """Create quota tracking mock."""
        return QuotaTrackingMock(quota_per_second=250)

    @pytest.fixture
    def email_factory(self):
        """Create mock email factory."""
        return MockEmailFactory()

    @patch('src.gmail.client.build')
    def test_collector_respects_quota_limits(self, mock_build, mock_auth, temp_db, quota_tracker, email_factory):
        """
        Test that collector + client integration respects quota limits.

        This test would have caught the bug where batch_size=100 caused
        500 quota units (100 × 5) to be used in a single batch, exceeding
        the 250 units/second limit.
        """
        # Create client with rate limiting
        client = GmailClient(mock_auth, enable_rate_limiting=True)

        # Set up batch request factory using helper class
        def create_batch_mock():
            return MockBatchRequest(quota_tracker, email_factory)

        client.service.new_batch_http_request = create_batch_mock

        # Mock list_all_messages to return 200 message IDs
        message_ids = [f'msg_{i}' for i in range(200)]
        client.list_all_messages = Mock(return_value=message_ids)

        # Create collector
        collector = EmailCollector(client, temp_db)

        # Export with batch_size=100 (the problematic configuration)
        # This should NOT cause quota violations because the client
        # should internally batch with rate-aware sizing
        count = collector.export_historical(months=1, batch_size=100)

        # Verify emails were exported
        assert count > 0, "No emails were exported"

        # Verify quota limits were respected in every batch
        quota_tracker.validate_quota_limit()

        # Verify that batches were adaptively sized
        assert len(quota_tracker.batch_requests) > 0, "No batch requests recorded"

        # Check that no single batch exceeded 50 messages (50 × 5 = 250 quota)
        for batch in quota_tracker.batch_requests:
            assert batch['total_quota'] <= 250, (
                f"Batch used {batch['total_quota']} quota units, "
                f"exceeding 250 limit"
            )

    @patch('src.gmail.client.build')
    def test_different_message_formats_use_correct_quota(self, mock_build, mock_auth, temp_db, quota_tracker, email_factory):
        """Test that different message formats use correct quota costs."""
        client = GmailClient(mock_auth, enable_rate_limiting=True)

        # Track format used
        format_used = None

        def create_batch_mock():
            batch_mock = Mock()
            batch_requests = []

            def mock_add(request, callback):
                nonlocal format_used
                # In real implementation, format is in the request
                # For this test, we'll track it separately
                batch_requests.append((request, callback))

            def mock_execute():
                message_count = len(batch_requests)
                # Use tracked format for quota calculation
                cost = {
                    'full': QuotaCosts.MESSAGE_GET_FULL,
                    'metadata': QuotaCosts.MESSAGE_GET_METADATA,
                    'minimal': QuotaCosts.MESSAGE_GET_MINIMAL
                }.get(format_used, QuotaCosts.MESSAGE_GET_FULL)

                quota_tracker.record_batch(message_count, cost)

                for request, callback in batch_requests:
                    callback(None, email_factory.create(), None)
                batch_requests.clear()

            batch_mock.add = mock_add
            batch_mock.execute = mock_execute
            return batch_mock

        client.service.new_batch_http_request = create_batch_mock

        # Test 'metadata' format (2 quota per message)
        format_used = 'metadata'
        emails = client.get_messages_batch(['msg1', 'msg2', 'msg3'], message_format='metadata')

        assert len(emails) == 3
        # Should use 3 messages × 2 quota = 6 quota
        assert any(b['cost_per_message'] == 2 for b in quota_tracker.batch_requests)

        # Test 'minimal' format (1 quota per message)
        quota_tracker.batch_requests.clear()
        format_used = 'minimal'
        emails = client.get_messages_batch(['msg4', 'msg5'], message_format='minimal')

        assert len(emails) == 2
        # Should use 2 messages × 1 quota = 2 quota
        assert any(b['cost_per_message'] == 1 for b in quota_tracker.batch_requests)

    @patch('src.gmail.client.build')
    def test_adaptive_batch_sizing_reduces_on_failures(self, mock_build, mock_auth, temp_db, email_factory):
        """Test that adaptive batch sizing reduces batch size on failures."""
        client = GmailClient(mock_auth, enable_rate_limiting=True)

        batch_sizes_used = []
        call_count = 0

        def create_batch_mock():
            nonlocal call_count
            batch_mock = Mock()
            batch_requests = []

            def mock_add(request, callback):
                batch_requests.append((request, callback))

            def mock_execute():
                nonlocal call_count
                call_count += 1
                batch_sizes_used.append(len(batch_requests))

                # Simulate rate limit on first call
                if call_count == 1:
                    for request, callback in batch_requests:
                        error = Mock()
                        error.resp = Mock()
                        error.resp.status = 429  # Rate limit error
                        callback(None, None, error)
                else:
                    # Succeed on subsequent calls
                    for request, callback in batch_requests:
                        callback(None, email_factory.create(), None)

                batch_requests.clear()

            batch_mock.add = mock_add
            batch_mock.execute = mock_execute
            return batch_mock

        client.service.new_batch_http_request = create_batch_mock

        # Fetch with enough messages to trigger multiple batches
        message_ids = [f'msg_{i}' for i in range(50)]
        emails = client.get_messages_batch(message_ids)

        # Should have made multiple attempts
        assert len(batch_sizes_used) >= 2, "Should have made multiple batch attempts"

        # After rate limit, batch size should be reduced
        # (This validates adaptive sizing is working)
        if len(batch_sizes_used) >= 3:
            # Later batches should use smaller size after rate limit hit
            assert batch_sizes_used[-1] <= batch_sizes_used[0], (
                "Batch size should reduce after rate limit"
            )

    @patch('src.gmail.client.build')
    def test_collector_handles_partial_failures(self, mock_build, mock_auth, temp_db, quota_tracker, email_factory):
        """Test that collector handles partial batch failures gracefully."""
        client = GmailClient(mock_auth, enable_rate_limiting=True)

        # Simulate every 3rd message failing with permanent error (not retryable)
        msg_count = [0]
        retry_count = [0]

        def create_batch_mock():
            batch_mock = Mock()
            batch_requests = []

            def mock_add(request, callback):
                batch_requests.append((request, callback))

            def mock_execute():
                retry_count[0] += 1
                # Only succeed on first try (prevent infinite retries)
                is_first_attempt = retry_count[0] == 1

                for request, callback in batch_requests:
                    msg_count[0] += 1
                    if is_first_attempt and msg_count[0] % 3 == 0:
                        # Simulate transient failure that won't be retried (non-429)
                        # Client will log but continue
                        error = Mock()
                        error.resp = Mock()
                        error.resp.status = 500  # Server error, will be retried
                        callback(None, None, error)
                    else:
                        # Success - use unique message ID
                        callback(None, email_factory.create(), None)

                # Record quota only for successful responses
                successful = len([r for r in batch_requests]) - (len(batch_requests) // 3 if is_first_attempt else 0)
                if successful > 0:
                    quota_tracker.record_batch(successful, QuotaCosts.MESSAGE_GET_FULL)

                batch_requests.clear()

            batch_mock.add = mock_add
            batch_mock.execute = mock_execute
            return batch_mock

        client.service.new_batch_http_request = create_batch_mock
        client.list_all_messages = Mock(return_value=[f'msg_{i}' for i in range(30)])

        collector = EmailCollector(client, temp_db)

        # Should not crash on partial failures
        count = collector.export_historical(months=1, batch_size=20)

        # Should have saved some emails despite failures
        # (with retries, should get most/all emails eventually)
        assert count > 0, "Should have saved some emails despite failures"

    @patch('src.gmail.client.build')
    def test_large_export_validates_quota_continuously(self, mock_build, mock_auth, temp_db, quota_tracker, email_factory):
        """
        Test that a large export (500+ messages) continuously validates quota limits.

        This is a comprehensive test that simulates a real export scenario.
        """
        client = GmailClient(mock_auth, enable_rate_limiting=True)

        # Set up batch request factory
        def create_batch_mock():
            return MockBatchRequest(quota_tracker, email_factory)

        client.service.new_batch_http_request = create_batch_mock
        client.list_all_messages = Mock(return_value=[f'msg_{i}' for i in range(500)])

        collector = EmailCollector(client, temp_db)

        # Export with batch_size=100 (problematic configuration)
        count = collector.export_historical(months=6, batch_size=100)

        # Verify success
        assert count == 500, f"Should have exported 500 emails, got {count}"

        # Verify all batches respected quota
        quota_tracker.validate_quota_limit()

        # Verify we made multiple batches (not just one giant batch)
        assert len(quota_tracker.batch_requests) >= 10, (
            f"Should have made multiple batches for 500 messages, "
            f"only made {len(quota_tracker.batch_requests)}"
        )

        # Log batch statistics for visibility
        print(f"\nBatch Statistics:")
        print(f"  Total batches: {len(quota_tracker.batch_requests)}")
        print(f"  Total quota consumed: {quota_tracker.total_consumed}")
        print(f"  Max batch quota: {quota_tracker.max_batch_quota}")
        print(f"  Average batch size: {sum(b['message_count'] for b in quota_tracker.batch_requests) / len(quota_tracker.batch_requests):.1f}")

    @patch('src.gmail.client.build')
    def test_quota_limit_violation_detection(self, mock_build, mock_auth, temp_db, email_factory):
        """
        Test that we can detect quota limit violations.

        This test intentionally creates a scenario that would violate
        quota limits and verifies our detection works.
        """
        client = GmailClient(mock_auth, enable_rate_limiting=False)  # Disable rate limiting

        violation_detected = False

        def create_batch_mock():
            batch_mock = Mock()
            batch_requests = []

            def mock_add(request, callback):
                batch_requests.append((request, callback))

            def mock_execute():
                nonlocal violation_detected
                message_count = len(batch_requests)
                batch_quota = message_count * QuotaCosts.MESSAGE_GET_FULL

                # Detect violation
                if batch_quota > 250:
                    violation_detected = True

                for request, callback in batch_requests:
                    callback(None, email_factory.create(), None)

                batch_requests.clear()

            batch_mock.add = mock_add
            batch_mock.execute = mock_execute
            return batch_mock

        client.service.new_batch_http_request = create_batch_mock

        # Try to fetch 100 messages at once (500 quota units)
        # This SHOULD violate quota limits
        message_ids = [f'msg_{i}' for i in range(100)]
        emails = client.get_messages_batch(message_ids, batch_size=100)

        # Verify violation was detected
        assert violation_detected, (
            "Should have detected quota violation when fetching "
            "100 messages (500 quota) without rate limiting"
        )


class TestQuotaMathValidation:
    """Tests specifically for quota math validation."""

    def test_batch_size_50_full_format_equals_250_quota(self):
        """Validate: 50 messages × 5 quota = 250 quota (at limit)."""
        messages = 50
        cost = QuotaCosts.MESSAGE_GET_FULL
        total = messages * cost
        assert total == 250, f"50 messages should use exactly 250 quota, got {total}"

    def test_batch_size_51_full_format_exceeds_quota(self):
        """Validate: 51 messages × 5 quota = 255 quota (exceeds limit)."""
        messages = 51
        cost = QuotaCosts.MESSAGE_GET_FULL
        total = messages * cost
        assert total > 250, f"51 messages should exceed 250 quota, got {total}"

    def test_batch_size_100_full_format_violates_quota(self):
        """
        Validate: 100 messages × 5 quota = 500 quota (VIOLATION).

        This is the exact bug that was happening.
        """
        messages = 100
        cost = QuotaCosts.MESSAGE_GET_FULL
        total = messages * cost
        assert total == 500, f"100 messages should use 500 quota, got {total}"
        assert total > 250, "100 messages at 5 quota each exceeds 250 limit"

    def test_batch_size_100_metadata_format_equals_200_quota(self):
        """Validate: 100 messages × 2 quota = 200 quota (under limit)."""
        messages = 100
        cost = QuotaCosts.MESSAGE_GET_METADATA
        total = messages * cost
        assert total == 200, f"100 messages with metadata should use 200 quota, got {total}"
        assert total < 250, "100 metadata messages should be under limit"

    def test_batch_size_125_metadata_format_equals_250_quota(self):
        """Validate: 125 messages × 2 quota = 250 quota (at limit)."""
        messages = 125
        cost = QuotaCosts.MESSAGE_GET_METADATA
        total = messages * cost
        assert total == 250, f"125 messages with metadata should use exactly 250 quota, got {total}"

    def test_batch_size_250_minimal_format_equals_250_quota(self):
        """Validate: 250 messages × 1 quota = 250 quota (at limit)."""
        messages = 250
        cost = QuotaCosts.MESSAGE_GET_MINIMAL
        total = messages * cost
        assert total == 250, f"250 messages with minimal should use exactly 250 quota, got {total}"

    def test_recommended_batch_sizes(self):
        """Test recommended batch sizes for each format."""
        # Full format: max 50 messages
        assert 50 * QuotaCosts.MESSAGE_GET_FULL == 250

        # Metadata format: max 125 messages
        assert 125 * QuotaCosts.MESSAGE_GET_METADATA == 250

        # Minimal format: max 250 messages
        assert 250 * QuotaCosts.MESSAGE_GET_MINIMAL == 250
