"""Gmail API client and operations."""

from .auth import GmailAuthenticator
from .client import GmailClient
from .models import Email
from .operations import GmailOperations, OperationResult
from .rate_limiter import AdaptiveBatchSizer, QuotaCosts, QuotaTracker, RateLimiter

__all__ = [
    "GmailAuthenticator",
    "Email",
    "GmailClient",
    "GmailOperations",
    "OperationResult",
    "RateLimiter",
    "QuotaTracker",
    "AdaptiveBatchSizer",
    "QuotaCosts",
]
