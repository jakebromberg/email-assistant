"""Smart rate limiting for Gmail API quota management."""

import time
from typing import Optional
from dataclasses import dataclass
from ..utils import get_logger


logger = get_logger(__name__)


@dataclass
class QuotaCosts:
    """Gmail API quota costs per operation."""
    MESSAGE_GET_FULL = 5       # messages.get with format='full'
    MESSAGE_GET_METADATA = 2   # messages.get with format='metadata'
    MESSAGE_GET_MINIMAL = 1    # messages.get with format='minimal'
    MESSAGE_LIST = 5           # messages.list per page
    LABEL_GET = 1              # labels.get
    MODIFY = 5                 # messages.modify (archive, label, etc.)


class QuotaTracker:
    """
    Token bucket algorithm for tracking Gmail API quota usage.

    Gmail API limit: 250 quota units per user per second.
    This tracker ensures we never exceed the quota by maintaining
    a bucket of available quota units that refills over time.
    """

    def __init__(self, quota_per_second: int = 250):
        """
        Initialize quota tracker.

        Args:
            quota_per_second: Gmail API quota limit (default: 250)
        """
        self.quota_per_second = quota_per_second
        self.tokens = float(quota_per_second)
        self.last_update = time.time()
        self.total_consumed = 0
        self.total_waited = 0.0

    def can_make_request(self, cost: int) -> bool:
        """
        Check if we have enough quota to make a request.

        Args:
            cost: Quota cost of the request

        Returns:
            True if we have enough quota
        """
        self._refill()
        return self.tokens >= cost

    def wait_for_quota(self, cost: int) -> float:
        """
        Wait until we have enough quota to make a request.

        Args:
            cost: Quota cost of the request

        Returns:
            Time waited in seconds
        """
        self._refill()

        if self.tokens >= cost:
            return 0.0

        # Calculate how long to wait
        deficit = cost - self.tokens
        wait_time = deficit / self.quota_per_second

        logger.debug(
            f"Insufficient quota (need {cost}, have {self.tokens:.1f}), "
            f"waiting {wait_time:.2f}s"
        )

        time.sleep(wait_time)
        self.total_waited += wait_time
        self._refill()

        return wait_time

    def consume(self, cost: int):
        """
        Consume quota units for a request.

        Args:
            cost: Quota units to consume
        """
        self._refill()
        self.tokens -= cost
        self.total_consumed += cost

        logger.debug(
            f"Consumed {cost} quota units, {self.tokens:.1f} remaining"
        )

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens for elapsed time
        refill_amount = elapsed * self.quota_per_second
        self.tokens = min(
            self.quota_per_second,
            self.tokens + refill_amount
        )

        self.last_update = now

    def get_stats(self) -> dict:
        """Get quota usage statistics."""
        return {
            'total_consumed': self.total_consumed,
            'total_waited': self.total_waited,
            'current_tokens': self.tokens,
            'quota_per_second': self.quota_per_second
        }


class AdaptiveBatchSizer:
    """
    Adaptively adjust batch size based on success/failure rates.

    Increases batch size when operations succeed consistently,
    decreases when rate limits are hit.
    """

    def __init__(
        self,
        initial_size: int = 50,
        min_size: int = 10,
        max_size: int = 50
    ):
        """
        Initialize adaptive batch sizer.

        Args:
            initial_size: Starting batch size
            min_size: Minimum allowed batch size
            max_size: Maximum allowed batch size
        """
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size

        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.total_operations = 0
        self.total_successes = 0
        self.total_failures = 0

    def on_success(self) -> int:
        """
        Record a successful operation.

        Returns:
            Updated batch size
        """
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.total_successes += 1
        self.total_operations += 1

        # Gradually increase batch size if stable
        if self.consecutive_successes >= 5:
            old_size = self.current_size
            self.current_size = min(self.current_size + 5, self.max_size)

            if self.current_size > old_size:
                logger.info(
                    f"Increasing batch size: {old_size} → {self.current_size} "
                    f"(stable for {self.consecutive_successes} batches)"
                )

            self.consecutive_successes = 0

        return self.current_size

    def on_rate_limit(self) -> int:
        """
        Record a rate limit error.

        Returns:
            Updated batch size
        """
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.total_failures += 1
        self.total_operations += 1

        # Reduce batch size when hitting limits
        old_size = self.current_size
        self.current_size = max(self.current_size // 2, self.min_size)

        logger.warning(
            f"Rate limit hit, reducing batch size: {old_size} → {self.current_size}"
        )

        return self.current_size

    def get_size(self) -> int:
        """Get current batch size."""
        return self.current_size

    def get_stats(self) -> dict:
        """Get batch sizing statistics."""
        success_rate = (
            self.total_successes / self.total_operations * 100
            if self.total_operations > 0 else 0
        )

        return {
            'current_size': self.current_size,
            'total_operations': self.total_operations,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'consecutive_successes': self.consecutive_successes,
            'consecutive_failures': self.consecutive_failures
        }


class RateLimiter:
    """
    Comprehensive rate limiter combining quota tracking and adaptive batch sizing.
    """

    def __init__(
        self,
        quota_per_second: int = 250,
        initial_batch_size: int = 50,
        enable_adaptive_sizing: bool = True
    ):
        """
        Initialize rate limiter.

        Args:
            quota_per_second: Gmail API quota limit
            initial_batch_size: Starting batch size
            enable_adaptive_sizing: Whether to adapt batch size dynamically
        """
        self.quota_tracker = QuotaTracker(quota_per_second)
        self.batch_sizer = AdaptiveBatchSizer(initial_size=initial_batch_size)
        self.enable_adaptive_sizing = enable_adaptive_sizing

        logger.info(
            f"Rate limiter initialized: {quota_per_second} quota/sec, "
            f"batch size {initial_batch_size}"
        )

    def calculate_batch_cost(self, batch_size: int, cost_per_item: int) -> int:
        """
        Calculate quota cost for a batch operation.

        Args:
            batch_size: Number of items in batch
            cost_per_item: Quota cost per item

        Returns:
            Total quota cost
        """
        return batch_size * cost_per_item

    def wait_for_batch(self, batch_size: int, cost_per_item: int) -> float:
        """
        Wait until we have enough quota for a batch operation.

        Args:
            batch_size: Number of items in batch
            cost_per_item: Quota cost per item

        Returns:
            Time waited in seconds
        """
        total_cost = self.calculate_batch_cost(batch_size, cost_per_item)
        return self.quota_tracker.wait_for_quota(total_cost)

    def consume_batch(self, batch_size: int, cost_per_item: int):
        """
        Consume quota for a batch operation.

        Args:
            batch_size: Number of items in batch
            cost_per_item: Quota cost per item
        """
        total_cost = self.calculate_batch_cost(batch_size, cost_per_item)
        self.quota_tracker.consume(total_cost)

    def on_batch_success(self) -> Optional[int]:
        """
        Record successful batch operation.

        Returns:
            New recommended batch size if adaptive sizing enabled
        """
        if self.enable_adaptive_sizing:
            return self.batch_sizer.on_success()
        return None

    def on_rate_limit(self, retry_after: Optional[int] = None) -> Optional[int]:
        """
        Handle rate limit error.

        Args:
            retry_after: Seconds to wait from Retry-After header

        Returns:
            New recommended batch size if adaptive sizing enabled
        """
        # Parse Retry-After header if present
        if retry_after:
            logger.warning(f"Rate limit hit, Retry-After: {retry_after}s")
            time.sleep(retry_after)
        else:
            # Default backoff
            logger.warning("Rate limit hit, using default backoff")
            time.sleep(2.0)

        # Reduce batch size
        if self.enable_adaptive_sizing:
            return self.batch_sizer.on_rate_limit()
        return None

    def get_recommended_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.batch_sizer.get_size()

    def get_stats(self) -> dict:
        """Get comprehensive rate limiting statistics."""
        return {
            'quota': self.quota_tracker.get_stats(),
            'batch_sizing': self.batch_sizer.get_stats()
        }

    def log_stats(self):
        """Log rate limiting statistics."""
        stats = self.get_stats()
        quota = stats['quota']
        batch = stats['batch_sizing']

        logger.info(
            f"Rate limiter stats - "
            f"Quota consumed: {quota['total_consumed']:.0f}, "
            f"Time waited: {quota['total_waited']:.1f}s, "
            f"Batch size: {batch['current_size']}, "
            f"Success rate: {batch['success_rate']:.1f}%"
        )
