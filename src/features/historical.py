"""Historical pattern feature extraction."""

from typing import Dict, Any, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from ..database.schema import Email
from ..database.repository import EmailRepository
from ..utils import get_logger


logger = get_logger(__name__)


class HistoricalPatternExtractor:
    """
    Extract historical pattern features based on past behavior.

    Computes features like sender open rate, email frequency, and
    domain-level patterns.

    Attributes:
        session: Database session for queries

    Example:
        >>> with db.get_session() as session:
        ...     extractor = HistoricalPatternExtractor(session)
        ...     features = extractor.extract(email)
        ...     print(f"Open rate: {features['sender_open_rate']:.1%}")
    """

    def __init__(self, session: Session):
        """
        Initialize historical pattern extractor.

        Args:
            session: SQLAlchemy database session

        Example:
            >>> with db.get_session() as session:
            ...     extractor = HistoricalPatternExtractor(session)
        """
        self.session = session
        self.repo = EmailRepository(session)

    def extract(self, email: Email) -> Dict[str, Any]:
        """
        Extract historical pattern features for an email.

        Args:
            email: Email database model

        Returns:
            Dictionary of historical features

        Example:
            >>> features = extractor.extract(email)
            >>> print(f"Sender emails: {features['sender_email_count']}")
        """
        features = {}

        # Sender statistics
        sender_stats = self.repo.get_sender_stats(email.from_address)
        features['sender_email_count'] = sender_stats['total_emails']
        features['sender_open_rate'] = sender_stats['open_rate']

        # Days since last email from sender
        features['sender_days_since_last'] = self._days_since_last_email(
            email.from_address,
            before_date=email.date
        )

        # Domain-level statistics
        domain = self._extract_domain(email.from_address)
        domain_stats = self._get_domain_stats(domain)
        features['domain_open_rate'] = domain_stats['open_rate']

        logger.debug(f"Extracted historical features for {email.message_id}")

        return features

    def extract_batch(self, emails: list[Email]) -> list[Dict[str, Any]]:
        """
        Extract historical features from multiple emails.

        Args:
            emails: List of Email database models

        Returns:
            List of feature dictionaries

        Example:
            >>> features_list = extractor.extract_batch(emails)
        """
        return [self.extract(email) for email in emails]

    def _days_since_last_email(
        self,
        sender_address: str,
        before_date: datetime
    ) -> Optional[float]:
        """
        Calculate days since last email from sender.

        Args:
            sender_address: Sender email address
            before_date: Calculate before this date

        Returns:
            Days since last email, or None if no previous emails

        Example:
            >>> days = extractor._days_since_last_email("sender@example.com", email.date)
        """
        # Query for most recent email before this one
        previous = self.session.query(Email).filter(
            Email.from_address == sender_address,
            Email.date < before_date
        ).order_by(Email.date.desc()).first()

        if not previous:
            return None

        delta = before_date - previous.date
        return delta.total_seconds() / 86400  # Convert to days

    def _get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a sender domain.

        Args:
            domain: Email domain

        Returns:
            Dictionary with domain statistics

        Example:
            >>> stats = extractor._get_domain_stats("example.com")
            >>> print(f"Domain open rate: {stats['open_rate']:.1%}")
        """
        # Query all emails from this domain
        domain_emails = self.session.query(Email).filter(
            Email.from_address.like(f'%@{domain}')
        ).all()

        if not domain_emails:
            return {
                'total_emails': 0,
                'open_rate': 0.0,
            }

        total = len(domain_emails)
        opened = sum(1 for e in domain_emails if e.was_read)

        return {
            'total_emails': total,
            'open_rate': opened / total if total > 0 else 0.0,
        }

    def _extract_domain(self, email_address: str) -> str:
        """Extract domain from email address."""
        if not email_address or '@' not in email_address:
            return 'unknown'
        return email_address.split('@')[-1].lower()

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        Get list of feature names produced by this extractor.

        Returns:
            List of feature names

        Example:
            >>> names = HistoricalPatternExtractor.get_feature_names()
        """
        return [
            'sender_email_count',
            'sender_open_rate',
            'sender_days_since_last',
            'domain_open_rate',
        ]
