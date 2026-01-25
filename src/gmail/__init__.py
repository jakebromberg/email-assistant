"""Gmail API client and operations."""

from .auth import GmailAuthenticator
from .models import Email
from .client import GmailClient
from .operations import GmailOperations, OperationResult

__all__ = [
    "GmailAuthenticator",
    "Email",
    "GmailClient",
    "GmailOperations",
    "OperationResult",
]
