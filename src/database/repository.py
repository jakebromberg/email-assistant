"""Repository layer for email data access."""

from datetime import datetime
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..gmail.models import Email as GmailEmail
from ..utils import get_logger
from .schema import Email, EmailAction, FeedbackReview

logger = get_logger(__name__)


class EmailRepository:
    """
    Repository for email data access operations.

    Provides high-level methods for storing and retrieving emails,
    labels, actions, and feedback.

    Attributes:
        session: SQLAlchemy session

    Example:
        >>> from src.database import Database
        >>> db = Database()
        >>> with db.get_session() as session:
        ...     repo = EmailRepository(session)
        ...     email = repo.get_by_id("msg123")
    """

    def __init__(self, session: Session):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy session

        Example:
            >>> with db.get_session() as session:
            ...     repo = EmailRepository(session)
        """
        self.session = session

    def save_email(self, gmail_email: GmailEmail) -> Email:
        """
        Save or update an email from Gmail API.

        Args:
            gmail_email: Email object from Gmail client

        Returns:
            Saved Email database model

        Example:
            >>> gmail_email = client.get_message("msg123")
            >>> db_email = repo.save_email(gmail_email)
        """
        # Check if email already exists
        existing = self.get_by_id(gmail_email.message_id)

        if existing:
            # Update existing email
            existing.thread_id = gmail_email.thread_id
            existing.from_address = gmail_email.from_address
            existing.from_name = gmail_email.from_name
            existing.to_address = gmail_email.to_address
            existing.subject = gmail_email.subject
            existing.date = gmail_email.date
            existing.snippet = gmail_email.snippet
            existing.body_plain = gmail_email.body_plain
            existing.body_html = gmail_email.body_html
            existing.labels = gmail_email.labels
            existing.was_read = not gmail_email.is_unread
            existing.was_archived = not gmail_email.is_in_inbox
            existing.is_important = gmail_email.is_important
            existing.is_starred = gmail_email.is_starred

            logger.debug(f"Updated email: {gmail_email.message_id}")
            return existing

        else:
            # Create new email
            email = Email(
                message_id=gmail_email.message_id,
                thread_id=gmail_email.thread_id,
                from_address=gmail_email.from_address,
                from_name=gmail_email.from_name,
                to_address=gmail_email.to_address,
                subject=gmail_email.subject,
                date=gmail_email.date,
                snippet=gmail_email.snippet,
                body_plain=gmail_email.body_plain,
                body_html=gmail_email.body_html,
                labels=gmail_email.labels,
                was_read=not gmail_email.is_unread,
                was_archived=not gmail_email.is_in_inbox,
                is_important=gmail_email.is_important,
                is_starred=gmail_email.is_starred,
                collected_at=datetime.utcnow()
            )

            self.session.add(email)
            logger.debug(f"Saved new email: {gmail_email.message_id}")
            return email

    def save_emails_batch(self, gmail_emails: list[GmailEmail]) -> list[Email]:
        """
        Save multiple emails in batch.

        Args:
            gmail_emails: List of Gmail Email objects

        Returns:
            List of saved Email database models

        Example:
            >>> gmail_emails = client.get_messages_batch(message_ids)
            >>> db_emails = repo.save_emails_batch(gmail_emails)
            >>> print(f"Saved {len(db_emails)} emails")
        """
        saved = []
        for gmail_email in gmail_emails:
            email = self.save_email(gmail_email)
            saved.append(email)

        self.session.flush()
        logger.info(f"Saved batch of {len(saved)} emails")
        return saved

    def get_by_id(self, message_id: str) -> Email | None:
        """
        Get email by message ID.

        Args:
            message_id: Gmail message ID

        Returns:
            Email if found, None otherwise

        Example:
            >>> email = repo.get_by_id("msg123")
            >>> if email:
            ...     print(email.subject)
        """
        return self.session.query(Email).filter(Email.message_id == message_id).first()

    def get_by_ids(self, message_ids: list[str]) -> list[Email]:
        """
        Get multiple emails by message IDs.

        Args:
            message_ids: List of Gmail message IDs

        Returns:
            List of Email objects

        Example:
            >>> emails = repo.get_by_ids(["msg1", "msg2", "msg3"])
        """
        return self.session.query(Email).filter(Email.message_id.in_(message_ids)).all()

    def get_all(self, limit: int | None = None, offset: int = 0) -> list[Email]:
        """
        Get all emails with optional pagination.

        Args:
            limit: Maximum number of emails to return
            offset: Number of emails to skip

        Returns:
            List of Email objects

        Example:
            >>> # Get first 100 emails
            >>> emails = repo.get_all(limit=100, offset=0)
            >>> # Get next 100 emails
            >>> emails = repo.get_all(limit=100, offset=100)
        """
        query = self.session.query(Email).order_by(desc(Email.date))

        if limit:
            query = query.limit(limit).offset(offset)

        return query.all()

    def get_by_sender(self, sender_address: str, limit: int | None = None) -> list[Email]:
        """
        Get emails from a specific sender.

        Args:
            sender_address: Sender email address
            limit: Maximum number of emails to return

        Returns:
            List of Email objects

        Example:
            >>> emails = repo.get_by_sender("newsletter@example.com", limit=10)
        """
        query = self.session.query(Email).filter(
            Email.from_address == sender_address
        ).order_by(desc(Email.date))

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_unread(self, limit: int | None = None) -> list[Email]:
        """
        Get unread emails.

        Args:
            limit: Maximum number of emails to return

        Returns:
            List of Email objects

        Example:
            >>> unread = repo.get_unread(limit=50)
        """
        query = self.session.query(Email).filter(
            Email.was_read.is_(False)
        ).order_by(desc(Email.date))

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_in_date_range(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
        limit: int | None = None
    ) -> list[Email]:
        """
        Get emails in a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive), defaults to now
            limit: Maximum number of emails to return

        Returns:
            List of Email objects

        Example:
            >>> from datetime import datetime, timedelta
            >>> last_week = datetime.now() - timedelta(days=7)
            >>> emails = repo.get_in_date_range(last_week)
        """
        if end_date is None:
            end_date = datetime.utcnow()

        query = self.session.query(Email).filter(
            Email.date >= start_date,
            Email.date <= end_date
        ).order_by(desc(Email.date))

        if limit:
            query = query.limit(limit)

        return query.all()

    def count(self) -> int:
        """
        Get total number of emails.

        Returns:
            Total email count

        Example:
            >>> total = repo.count()
            >>> print(f"Total emails: {total}")
        """
        return self.session.query(Email).count()

    def record_action(
        self,
        message_id: str,
        action_type: str,
        source: str = 'user',
        action_data: dict[str, Any] | None = None
    ) -> EmailAction:
        """
        Record an action performed on an email.

        Args:
            message_id: Email message ID
            action_type: Type of action (archive, unarchive, label_add, etc.)
            source: Who performed action ('user' or 'bot')
            action_data: Additional action details

        Returns:
            Created EmailAction

        Example:
            >>> repo.record_action(
            ...     "msg123",
            ...     "archive",
            ...     source="bot",
            ...     action_data={"reason": "high_confidence_archive", "score": 0.95}
            ... )
        """
        action = EmailAction(
            message_id=message_id,
            action_type=action_type,
            source=source,
            action_data=action_data or {},
            timestamp=datetime.utcnow()
        )

        self.session.add(action)
        logger.debug(f"Recorded action: {action_type} on {message_id} by {source}")
        return action

    def get_actions_for_email(self, message_id: str) -> list[EmailAction]:
        """
        Get all actions for an email.

        Args:
            message_id: Email message ID

        Returns:
            List of EmailAction objects

        Example:
            >>> actions = repo.get_actions_for_email("msg123")
            >>> for action in actions:
            ...     print(f"{action.timestamp}: {action.action_type}")
        """
        return self.session.query(EmailAction).filter(
            EmailAction.message_id == message_id
        ).order_by(EmailAction.timestamp).all()

    def save_feedback(
        self,
        message_id: str,
        decision_correct: bool | None = None,
        label_correct: bool | None = None,
        correct_decision: str | None = None,
        correct_label: str | None = None,
        user_comment: str | None = None
    ) -> FeedbackReview:
        """
        Save feedback from interactive review.

        Args:
            message_id: Email message ID
            decision_correct: Whether bot's decision was correct
            label_correct: Whether bot's label was correct
            correct_decision: What decision should have been
            correct_label: What label should have been
            user_comment: Natural language feedback

        Returns:
            Created FeedbackReview

        Example:
            >>> repo.save_feedback(
            ...     "msg123",
            ...     decision_correct=False,
            ...     correct_decision="keep",
            ...     user_comment="This sender always sends important updates"
            ... )
        """
        feedback = FeedbackReview(
            message_id=message_id,
            review_date=datetime.utcnow(),
            decision_correct=decision_correct,
            label_correct=label_correct,
            correct_decision=correct_decision,
            correct_label=correct_label,
            user_comment=user_comment,
            used_in_training=False
        )

        self.session.add(feedback)
        logger.debug(f"Saved feedback for {message_id}")
        return feedback

    def get_sender_stats(self, sender_address: str) -> dict[str, Any]:
        """
        Get statistics for a sender.

        Args:
            sender_address: Sender email address

        Returns:
            Dictionary with sender statistics

        Example:
            >>> stats = repo.get_sender_stats("newsletter@example.com")
            >>> print(f"Open rate: {stats['open_rate']:.1%}")
        """
        emails = self.get_by_sender(sender_address)

        total = len(emails)
        if total == 0:
            return {
                'total_emails': 0,
                'read_count': 0,
                'open_rate': 0.0,
                'archived_count': 0,
                'archive_rate': 0.0,
            }

        read_count = sum(1 for e in emails if e.was_read)
        archived_count = sum(1 for e in emails if e.was_archived)

        return {
            'total_emails': total,
            'read_count': read_count,
            'open_rate': read_count / total,
            'archived_count': archived_count,
            'archive_rate': archived_count / total,
        }
