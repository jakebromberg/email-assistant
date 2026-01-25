"""Metadata feature extraction from emails."""

import hashlib
import re
from typing import Dict, Any
from datetime import datetime

from ..database.schema import Email
from ..utils import get_logger


logger = get_logger(__name__)


class MetadataExtractor:
    """
    Extract metadata features from emails.

    Extracts features like sender domain, time features, email structure,
    and newsletter detection.

    Example:
        >>> extractor = MetadataExtractor()
        >>> features = extractor.extract(email)
        >>> print(features['sender_domain'])
    """

    # Newsletter indicators
    NEWSLETTER_HEADERS = [
        'list-unsubscribe',
        'list-id',
        'list-post',
        'precedence: bulk',
        'x-campaign-id',
        'x-mailchimp',
        'x-mailer-recptid'
    ]

    NEWSLETTER_PATTERNS = [
        r'unsubscribe',
        r'newsletter',
        r'mailing.?list',
        r'subscription',
        r'opt.?out'
    ]

    def __init__(self):
        """Initialize metadata extractor."""
        pass

    def extract(self, email: Email) -> Dict[str, Any]:
        """
        Extract metadata features from an email.

        Args:
            email: Email database model

        Returns:
            Dictionary of metadata features

        Example:
            >>> features = extractor.extract(email)
            >>> print(f"Domain: {features['sender_domain']}")
        """
        features = {}

        # Sender features
        features['sender_domain'] = self._extract_domain(email.from_address)
        features['sender_address_hash'] = self._hash_address(email.from_address)

        # Newsletter detection
        features['is_newsletter'] = self._is_newsletter(email)

        # Time features
        features['day_of_week'] = email.date.weekday()  # 0=Monday, 6=Sunday
        features['hour_of_day'] = email.date.hour

        # Structure features
        features['subject_length'] = len(email.subject) if email.subject else 0
        features['body_length'] = len(email.body_plain) if email.body_plain else 0
        features['has_attachments'] = False  # TODO: detect from headers
        features['thread_length'] = 1  # TODO: count thread

        logger.debug(f"Extracted metadata features for {email.message_id}")

        return features

    def extract_batch(self, emails: list[Email]) -> list[Dict[str, Any]]:
        """
        Extract metadata features from multiple emails.

        Args:
            emails: List of Email database models

        Returns:
            List of feature dictionaries

        Example:
            >>> features_list = extractor.extract_batch(emails)
        """
        return [self.extract(email) for email in emails]

    def _extract_domain(self, email_address: str) -> str:
        """
        Extract domain from email address.

        Args:
            email_address: Email address

        Returns:
            Domain part of email address

        Example:
            >>> extractor._extract_domain("user@example.com")
            'example.com'
        """
        if not email_address or '@' not in email_address:
            return 'unknown'

        return email_address.split('@')[-1].lower()

    def _hash_address(self, email_address: str) -> str:
        """
        Create hash of email address for privacy.

        Args:
            email_address: Email address

        Returns:
            SHA256 hash of email address

        Example:
            >>> hash_val = extractor._hash_address("user@example.com")
        """
        if not email_address:
            return ''

        return hashlib.sha256(email_address.lower().encode()).hexdigest()[:16]

    def _is_newsletter(self, email: Email) -> bool:
        """
        Detect if email is a newsletter.

        Checks for common newsletter indicators in headers and content.

        Args:
            email: Email database model

        Returns:
            True if email appears to be a newsletter

        Example:
            >>> is_nl = extractor._is_newsletter(email)
        """
        # Check headers (stored in labels for common ones)
        headers_lower = {k.lower(): v.lower() for k, v in email.headers.items()}

        for indicator in self.NEWSLETTER_HEADERS:
            if indicator in headers_lower:
                return True

        # Check subject and body for newsletter patterns
        text = f"{email.subject or ''} {email.body_plain or ''}".lower()

        for pattern in self.NEWSLETTER_PATTERNS:
            if re.search(pattern, text):
                return True

        return False

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        Get list of feature names produced by this extractor.

        Returns:
            List of feature names

        Example:
            >>> names = MetadataExtractor.get_feature_names()
        """
        return [
            'sender_domain',
            'sender_address_hash',
            'is_newsletter',
            'day_of_week',
            'hour_of_day',
            'subject_length',
            'body_length',
            'has_attachments',
            'thread_length',
        ]
