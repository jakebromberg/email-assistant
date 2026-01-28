"""Tests for Gmail rate limiting."""

import time

import pytest

from src.gmail.rate_limiter import AdaptiveBatchSizer, QuotaCosts, QuotaTracker, RateLimiter


class TestQuotaCosts:
    """Test quota cost constants."""

    def test_quota_costs_defined(self):
        """Test that quota costs are properly defined."""
        assert QuotaCosts.MESSAGE_GET_FULL == 5
        assert QuotaCosts.MESSAGE_GET_METADATA == 2
        assert QuotaCosts.MESSAGE_GET_MINIMAL == 1
        assert QuotaCosts.MESSAGE_LIST == 5
        assert QuotaCosts.LABEL_GET == 1
        assert QuotaCosts.MODIFY == 5


class TestQuotaTracker:
    """Test QuotaTracker class."""

    def test_initialization(self):
        """Test tracker initializes with correct quota."""
        tracker = QuotaTracker(quota_per_second=250)
        assert tracker.quota_per_second == 250
        assert tracker.tokens == 250.0
        assert tracker.total_consumed == 0

    def test_can_make_request_sufficient_quota(self):
        """Test can make request when quota available."""
        tracker = QuotaTracker(quota_per_second=250)
        assert tracker.can_make_request(50) is True
        assert tracker.can_make_request(250) is True

    def test_can_make_request_insufficient_quota(self):
        """Test can't make request when quota insufficient."""
        tracker = QuotaTracker(quota_per_second=250)
        tracker.tokens = 40
        assert tracker.can_make_request(50) is False

    def test_consume_quota(self):
        """Test consuming quota reduces tokens."""
        tracker = QuotaTracker(quota_per_second=250)
        initial_tokens = tracker.tokens

        tracker.consume(50)

        assert tracker.tokens == initial_tokens - 50
        assert tracker.total_consumed == 50

    def test_refill_over_time(self):
        """Test tokens refill over time."""
        tracker = QuotaTracker(quota_per_second=250)
        tracker.tokens = 100

        # Simulate 0.2 seconds passing
        tracker.last_update = time.time() - 0.2
        tracker._refill()

        # Should have added ~50 tokens (250 * 0.2)
        assert tracker.tokens > 100
        assert tracker.tokens == pytest.approx(150, abs=1)  # Allow 1 token tolerance

    def test_refill_cap_at_max(self):
        """Test refill doesn't exceed max quota."""
        tracker = QuotaTracker(quota_per_second=250)
        tracker.tokens = 200

        # Simulate 1 second passing (would add 250 tokens)
        tracker.last_update = time.time() - 1.0
        tracker._refill()

        # Should cap at 250
        assert tracker.tokens == 250

    def test_wait_for_quota(self):
        """Test waiting for quota availability."""
        tracker = QuotaTracker(quota_per_second=250)
        tracker.tokens = 10

        start = time.time()
        wait_time = tracker.wait_for_quota(50)
        elapsed = time.time() - start

        # Should have waited ~0.16 seconds ((50-10)/250)
        assert wait_time > 0
        assert elapsed >= wait_time * 0.9  # Allow 10% tolerance

    def test_wait_for_quota_when_available(self):
        """Test no wait when quota already available."""
        tracker = QuotaTracker(quota_per_second=250)

        wait_time = tracker.wait_for_quota(50)

        assert wait_time == 0.0

    def test_get_stats(self):
        """Test getting quota statistics."""
        tracker = QuotaTracker(quota_per_second=250)
        tracker.consume(100)

        stats = tracker.get_stats()

        assert stats['total_consumed'] == 100
        assert stats['quota_per_second'] == 250
        assert 'current_tokens' in stats
        assert 'total_waited' in stats


class TestAdaptiveBatchSizer:
    """Test AdaptiveBatchSizer class."""

    def test_initialization(self):
        """Test sizer initializes with correct values."""
        sizer = AdaptiveBatchSizer(initial_size=50, min_size=10, max_size=100)

        assert sizer.current_size == 50
        assert sizer.min_size == 10
        assert sizer.max_size == 100
        assert sizer.consecutive_successes == 0
        assert sizer.consecutive_failures == 0

    def test_get_size(self):
        """Test getting current batch size."""
        sizer = AdaptiveBatchSizer(initial_size=50)
        assert sizer.get_size() == 50

    def test_on_success_increases_size(self):
        """Test size increases after consecutive successes."""
        sizer = AdaptiveBatchSizer(initial_size=50, max_size=100)

        # Record 5 successes (threshold)
        for _ in range(5):
            size = sizer.on_success()

        # Should increase after 5th success
        assert size == 55
        assert sizer.current_size == 55

    def test_on_success_caps_at_max(self):
        """Test size doesn't exceed maximum."""
        sizer = AdaptiveBatchSizer(initial_size=95, max_size=100)

        # Record 5 successes
        for _ in range(5):
            size = sizer.on_success()

        # Should cap at max
        assert size == 100
        assert sizer.current_size == 100

    def test_on_success_resets_failures(self):
        """Test success resets failure counter."""
        sizer = AdaptiveBatchSizer(initial_size=50)
        sizer.consecutive_failures = 3

        sizer.on_success()

        assert sizer.consecutive_failures == 0

    def test_on_rate_limit_decreases_size(self):
        """Test size decreases on rate limit."""
        sizer = AdaptiveBatchSizer(initial_size=50, min_size=10)

        size = sizer.on_rate_limit()

        # Should halve the size
        assert size == 25
        assert sizer.current_size == 25

    def test_on_rate_limit_caps_at_min(self):
        """Test size doesn't go below minimum."""
        sizer = AdaptiveBatchSizer(initial_size=15, min_size=10)

        size = sizer.on_rate_limit()

        # Should cap at min
        assert size == 10
        assert sizer.current_size == 10

    def test_on_rate_limit_resets_successes(self):
        """Test rate limit resets success counter."""
        sizer = AdaptiveBatchSizer(initial_size=50)
        sizer.consecutive_successes = 3

        sizer.on_rate_limit()

        assert sizer.consecutive_successes == 0

    def test_get_stats(self):
        """Test getting batch sizing statistics."""
        sizer = AdaptiveBatchSizer(initial_size=50)
        sizer.on_success()
        sizer.on_success()
        sizer.on_rate_limit()

        stats = sizer.get_stats()

        assert stats['current_size'] == 25
        assert stats['total_operations'] == 3
        assert stats['total_successes'] == 2
        assert stats['total_failures'] == 1
        assert stats['success_rate'] == pytest.approx(66.67, rel=0.01)


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(
            quota_per_second=250,
            initial_batch_size=50,
            enable_adaptive_sizing=True
        )

        assert limiter.quota_tracker is not None
        assert limiter.batch_sizer is not None
        assert limiter.enable_adaptive_sizing is True

    def test_calculate_batch_cost(self):
        """Test calculating quota cost for batch."""
        limiter = RateLimiter()

        cost = limiter.calculate_batch_cost(batch_size=50, cost_per_item=5)

        assert cost == 250

    def test_wait_for_batch_sufficient_quota(self):
        """Test no wait when quota sufficient."""
        limiter = RateLimiter(quota_per_second=250)

        wait_time = limiter.wait_for_batch(batch_size=10, cost_per_item=5)

        assert wait_time == 0.0

    def test_wait_for_batch_insufficient_quota(self):
        """Test waits when quota insufficient."""
        limiter = RateLimiter(quota_per_second=250)
        limiter.quota_tracker.tokens = 10

        start = time.time()
        wait_time = limiter.wait_for_batch(batch_size=50, cost_per_item=5)
        elapsed = time.time() - start

        # Should wait for (250 - 10) / 250 = ~0.96 seconds
        assert wait_time > 0
        assert elapsed >= wait_time * 0.9  # 10% tolerance

    def test_consume_batch(self):
        """Test consuming quota for batch."""
        limiter = RateLimiter(quota_per_second=250)
        initial_tokens = limiter.quota_tracker.tokens

        limiter.consume_batch(batch_size=10, cost_per_item=5)

        assert limiter.quota_tracker.tokens == initial_tokens - 50
        assert limiter.quota_tracker.total_consumed == 50

    def test_on_batch_success_with_adaptive_sizing(self):
        """Test batch success with adaptive sizing enabled."""
        limiter = RateLimiter(
            enable_adaptive_sizing=True,
            initial_batch_size=30  # Start lower so we can see increase
        )
        initial_size = limiter.get_recommended_batch_size()

        # Record 5 successes to reach threshold, then 6th triggers increase
        for _ in range(6):
            new_size = limiter.on_batch_success()

        # Should have increased after 6th call
        assert new_size > initial_size
        assert new_size == 35  # Should have increased by 5

    def test_on_batch_success_without_adaptive_sizing(self):
        """Test batch success with adaptive sizing disabled."""
        limiter = RateLimiter(enable_adaptive_sizing=False)

        new_size = limiter.on_batch_success()

        assert new_size is None

    def test_on_rate_limit_with_retry_after(self):
        """Test rate limit handling with Retry-After header."""
        limiter = RateLimiter(enable_adaptive_sizing=True)
        initial_size = limiter.get_recommended_batch_size()

        start = time.time()
        new_size = limiter.on_rate_limit(retry_after=1)
        elapsed = time.time() - start

        # Should have waited ~1 second
        assert elapsed >= 0.9
        # Should have reduced batch size
        assert new_size < initial_size

    def test_on_rate_limit_without_retry_after(self):
        """Test rate limit handling without Retry-After."""
        limiter = RateLimiter(enable_adaptive_sizing=True)
        initial_size = limiter.get_recommended_batch_size()

        start = time.time()
        new_size = limiter.on_rate_limit(retry_after=None)
        elapsed = time.time() - start

        # Should use default 2 second backoff
        assert elapsed >= 1.9
        # Should have reduced batch size
        assert new_size < initial_size

    def test_get_recommended_batch_size(self):
        """Test getting recommended batch size."""
        limiter = RateLimiter(initial_batch_size=50)

        size = limiter.get_recommended_batch_size()

        assert size == 50

    def test_get_stats(self):
        """Test getting comprehensive stats."""
        limiter = RateLimiter()
        limiter.consume_batch(10, 5)
        limiter.on_batch_success()

        stats = limiter.get_stats()

        assert 'quota' in stats
        assert 'batch_sizing' in stats
        assert stats['quota']['total_consumed'] == 50
        assert stats['batch_sizing']['total_successes'] == 1

    def test_log_stats(self):
        """Test logging statistics doesn't raise errors."""
        limiter = RateLimiter()
        limiter.consume_batch(10, 5)

        # Should not raise any errors
        limiter.log_stats()

        # Verify stats are accessible
        stats = limiter.get_stats()
        assert stats['quota']['total_consumed'] == 50


class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""

    def test_full_workflow(self):
        """Test complete rate limiting workflow."""
        limiter = RateLimiter(
            quota_per_second=250,
            initial_batch_size=50,
            enable_adaptive_sizing=True
        )

        # Simulate processing multiple batches
        total_messages = 200
        batch_size = limiter.get_recommended_batch_size()
        cost_per_message = QuotaCosts.MESSAGE_GET_FULL

        processed = 0
        while processed < total_messages:
            current_batch = min(batch_size, total_messages - processed)

            # Wait for quota
            limiter.wait_for_batch(current_batch, cost_per_message)

            # Consume quota
            limiter.consume_batch(current_batch, cost_per_message)

            # Record success
            new_size = limiter.on_batch_success()
            if new_size:
                batch_size = new_size

            processed += current_batch

        assert processed == total_messages

        # Check stats
        stats = limiter.get_stats()
        assert stats['quota']['total_consumed'] == total_messages * cost_per_message

    def test_adaptive_behavior_under_rate_limits(self):
        """Test adaptive sizing reduces batch size under rate limits."""
        limiter = RateLimiter(
            quota_per_second=250,
            initial_batch_size=50,
            enable_adaptive_sizing=True
        )

        initial_size = limiter.get_recommended_batch_size()

        # Simulate rate limit
        limiter.on_rate_limit(retry_after=1)

        new_size = limiter.get_recommended_batch_size()

        # Should have reduced
        assert new_size < initial_size
        assert new_size >= 10  # Should respect minimum

    def test_adaptive_behavior_increases_on_success(self):
        """Test adaptive sizing increases on sustained success."""
        limiter = RateLimiter(
            quota_per_second=250,
            initial_batch_size=30,
            enable_adaptive_sizing=True
        )

        initial_size = limiter.get_recommended_batch_size()

        # Simulate 5+ successful batches
        for _ in range(6):
            limiter.on_batch_success()

        new_size = limiter.get_recommended_batch_size()

        # Should have increased
        assert new_size > initial_size
