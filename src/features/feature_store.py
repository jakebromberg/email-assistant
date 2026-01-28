"""Feature storage and retrieval."""

from typing import Dict, Any, List, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from ..database.schema import Email, EmailFeatures
from ..utils import get_logger


logger = get_logger(__name__)


class FeatureStore:
    """
    Store and retrieve computed features.

    Manages storage of features in the database and provides
    methods for batch operations and queries.

    Attributes:
        session: Database session

    Example:
        >>> with db.get_session() as session:
        ...     store = FeatureStore(session)
        ...     store.save_features("msg123", features)
    """

    def __init__(self, session: Session):
        """
        Initialize feature store.

        Args:
            session: SQLAlchemy database session

        Example:
            >>> with db.get_session() as session:
            ...     store = FeatureStore(session)
        """
        self.session = session

    def save_features(
        self,
        message_id: str,
        features: Dict[str, Any]
    ) -> EmailFeatures:
        """
        Save features for an email.

        Args:
            message_id: Email message ID
            features: Dictionary of features

        Returns:
            EmailFeatures model

        Example:
            >>> features = {
            ...     'sender_domain': 'example.com',
            ...     'is_newsletter': True,
            ...     'sender_open_rate': 0.75,
            ... }
            >>> store.save_features("msg123", features)
        """
        # Check if features already exist
        existing = self.get_features(message_id)

        if existing:
            # Update existing features
            for key, value in features.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.computed_at = datetime.utcnow()

            logger.debug(f"Updated features for {message_id}")
            return existing

        else:
            # Create new features
            email_features = EmailFeatures(
                message_id=message_id,
                computed_at=datetime.utcnow(),
                **features
            )

            self.session.add(email_features)
            logger.debug(f"Saved new features for {message_id}")
            return email_features

    def save_features_batch(
        self,
        features_list: List[tuple[str, Dict[str, Any]]]
    ) -> List[EmailFeatures]:
        """
        Save features for multiple emails.

        Args:
            features_list: List of (message_id, features) tuples

        Returns:
            List of EmailFeatures models

        Example:
            >>> features_list = [
            ...     ("msg1", {"sender_domain": "example.com", ...}),
            ...     ("msg2", {"sender_domain": "test.com", ...}),
            ... ]
            >>> store.save_features_batch(features_list)
        """
        saved = []
        for message_id, features in features_list:
            email_features = self.save_features(message_id, features)
            saved.append(email_features)

        self.session.flush()
        logger.info(f"Saved features for {len(saved)} emails")
        return saved

    def get_features(self, message_id: str) -> Optional[EmailFeatures]:
        """
        Get features for an email.

        Args:
            message_id: Email message ID

        Returns:
            EmailFeatures if exists, None otherwise

        Example:
            >>> features = store.get_features("msg123")
            >>> if features:
            ...     print(f"Open rate: {features.sender_open_rate:.1%}")
        """
        return self.session.query(EmailFeatures).filter(
            EmailFeatures.message_id == message_id
        ).first()

    def get_features_batch(self, message_ids: List[str]) -> List[Optional[EmailFeatures]]:
        """
        Get features for multiple emails.

        Args:
            message_ids: List of email message IDs

        Returns:
            List of EmailFeatures (None for emails without features)

        Example:
            >>> features_list = store.get_features_batch(["msg1", "msg2"])
        """
        features_dict = {
            f.message_id: f
            for f in self.session.query(EmailFeatures).filter(
                EmailFeatures.message_id.in_(message_ids)
            ).all()
        }

        return [features_dict.get(msg_id) for msg_id in message_ids]

    def has_features(self, message_id: str) -> bool:
        """
        Check if features exist for an email.

        Args:
            message_id: Email message ID

        Returns:
            True if features exist

        Example:
            >>> if not store.has_features("msg123"):
            ...     # Compute features
            ...     pass
        """
        return self.session.query(EmailFeatures).filter(
            EmailFeatures.message_id == message_id
        ).count() > 0

    def get_all_features(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[EmailFeatures]:
        """
        Get all computed features.

        Args:
            limit: Maximum number to return
            offset: Number to skip

        Returns:
            List of EmailFeatures

        Example:
            >>> all_features = store.get_all_features(limit=1000)
        """
        query = self.session.query(EmailFeatures).order_by(
            EmailFeatures.computed_at.desc()
        )

        if limit:
            query = query.limit(limit).offset(offset)

        return query.all()

    def count_features(self) -> int:
        """
        Count emails with computed features.

        Returns:
            Number of emails with features

        Example:
            >>> count = store.count_features()
            >>> print(f"Features computed for {count} emails")
        """
        return self.session.query(EmailFeatures).count()

    def delete_features(self, message_id: str) -> bool:
        """
        Delete features for an email.

        Args:
            message_id: Email message ID

        Returns:
            True if deleted, False if not found

        Example:
            >>> store.delete_features("msg123")
        """
        features = self.get_features(message_id)
        if features:
            self.session.delete(features)
            logger.debug(f"Deleted features for {message_id}")
            return True
        return False

    def get_emails_without_features(
        self,
        limit: Optional[int] = None
    ) -> List[Email]:
        """
        Get emails that don't have computed features.

        Args:
            limit: Maximum number to return

        Returns:
            List of Email objects without features

        Example:
            >>> emails = store.get_emails_without_features(limit=100)
            >>> print(f"Need to compute features for {len(emails)} emails")
        """
        # Query emails that don't have a corresponding EmailFeatures record
        query = self.session.query(Email).outerjoin(
            EmailFeatures,
            Email.message_id == EmailFeatures.message_id
        ).filter(
            EmailFeatures.message_id.is_(None)
        ).order_by(Email.date.desc())

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_feature_stats(self) -> Dict[str, Any]:
        """
        Get statistics about computed features.

        Returns:
            Dictionary with feature statistics

        Example:
            >>> stats = store.get_feature_stats()
            >>> print(f"Coverage: {stats['coverage_pct']:.1f}%")
        """
        from sqlalchemy import func

        total_emails = self.session.query(func.count(Email.message_id)).scalar()
        features_count = self.count_features()

        # Get average open rates
        avg_sender_open_rate = self.session.query(
            func.avg(EmailFeatures.sender_open_rate)
        ).scalar() or 0.0

        avg_domain_open_rate = self.session.query(
            func.avg(EmailFeatures.domain_open_rate)
        ).scalar() or 0.0

        # Newsletter count
        newsletter_count = self.session.query(EmailFeatures).filter(
            EmailFeatures.is_newsletter.is_(True)
        ).count()

        return {
            'total_emails': total_emails,
            'features_count': features_count,
            'coverage_pct': (features_count / total_emails * 100) if total_emails > 0 else 0,
            'avg_sender_open_rate': avg_sender_open_rate,
            'avg_domain_open_rate': avg_domain_open_rate,
            'newsletter_count': newsletter_count,
            'newsletter_pct': (newsletter_count / features_count * 100) if features_count > 0 else 0,
        }
