"""Database models and access layer."""

from .database import Database
from .schema import Email, EmailLabel, EmailAction, FeedbackReview, EmailFeatures
from .repository import EmailRepository

__all__ = [
    "Database",
    "Email",
    "EmailLabel",
    "EmailAction",
    "FeedbackReview",
    "EmailFeatures",
    "EmailRepository",
]
