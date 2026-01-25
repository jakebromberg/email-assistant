"""SQLAlchemy database schema for email storage."""

from datetime import datetime
from typing import Optional, List
import json

from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class Email(Base):
    """
    Email message with metadata and content.

    Stores complete email data including headers, body, labels, and
    behavioral signals (read status, archived status).

    Attributes:
        message_id: Gmail message ID (primary key)
        thread_id: Gmail thread ID
        from_address: Sender email address
        from_name: Sender display name
        to_address: Recipient email address
        subject: Email subject line
        date: Email date/time
        snippet: Short preview text
        body_plain: Plain text body
        body_html: HTML body
        labels: JSON array of label names
        was_read: Whether email was read when collected
        was_archived: Whether email was archived when collected
        is_important: Whether email was marked important
        is_starred: Whether email was starred
        opened_at: Timestamp when email was first opened (if known)
        collected_at: When this email was added to database
    """

    __tablename__ = 'emails'

    message_id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    from_address = Column(String, index=True, nullable=False)
    from_name = Column(String)
    to_address = Column(String)
    subject = Column(String)
    date = Column(DateTime, index=True, nullable=False)
    snippet = Column(Text)
    body_plain = Column(Text)
    body_html = Column(Text)

    # Labels as JSON array
    labels = Column(JSON, default=list)

    # Behavioral signals
    was_read = Column(Boolean, default=False, index=True)
    was_archived = Column(Boolean, default=False, index=True)
    is_important = Column(Boolean, default=False)
    is_starred = Column(Boolean, default=False)

    # Timestamps
    opened_at = Column(DateTime, nullable=True)
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    email_labels = relationship("EmailLabel", back_populates="email", cascade="all, delete-orphan")
    actions = relationship("EmailAction", back_populates="email", cascade="all, delete-orphan")
    feedback_reviews = relationship("FeedbackReview", back_populates="email", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<Email(id={self.message_id[:10]}..., "
            f"from={self.from_address}, "
            f"subject='{self.subject[:30]}...')>"
        )

    @property
    def user_labels(self) -> List[str]:
        """Get user-applied labels (excluding system labels)."""
        system_labels = {'INBOX', 'UNREAD', 'IMPORTANT', 'STARRED', 'SENT', 'DRAFT', 'TRASH', 'SPAM'}
        return [label for label in self.labels if label not in system_labels and not label.startswith('Bot/')]

    @property
    def bot_labels(self) -> List[str]:
        """Get bot-applied labels."""
        return [label for label in self.labels if label.startswith('Bot/')]


class EmailLabel(Base):
    """
    Many-to-many relationship between emails and labels.

    Tracks when labels were applied and by whom (user or bot).

    Attributes:
        id: Primary key
        message_id: Foreign key to emails table
        label_name: Label name
        applied_at: When label was applied
        applied_by: Who applied the label ('user' or 'bot')
    """

    __tablename__ = 'email_labels'

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey('emails.message_id'), nullable=False, index=True)
    label_name = Column(String, nullable=False, index=True)
    applied_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    applied_by = Column(String, default='user')  # 'user' or 'bot'

    # Relationships
    email = relationship("Email", back_populates="email_labels")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<EmailLabel(message_id={self.message_id[:10]}..., label={self.label_name})>"


class EmailAction(Base):
    """
    Audit trail of all actions performed on emails.

    Records both user actions (manual moves, label changes) and bot actions
    (auto-archive, label application) for learning.

    Attributes:
        id: Primary key
        message_id: Foreign key to emails table
        action_type: Type of action (archive, unarchive, label_add, label_remove, mark_read, mark_unread)
        action_data: JSON with additional action details
        timestamp: When action occurred
        source: Who performed action ('user' or 'bot')
    """

    __tablename__ = 'email_actions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey('emails.message_id'), nullable=False, index=True)
    action_type = Column(String, nullable=False, index=True)
    action_data = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    source = Column(String, default='user', index=True)  # 'user' or 'bot'

    # Relationships
    email = relationship("Email", back_populates="actions")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<EmailAction(message_id={self.message_id[:10]}..., "
            f"action={self.action_type}, source={self.source})>"
        )


class FeedbackReview(Base):
    """
    Interactive feedback from CLI review tool.

    Stores user corrections and natural language comments about bot decisions.

    Attributes:
        id: Primary key
        message_id: Foreign key to emails table
        review_date: When review occurred
        decision_correct: Whether bot's decision was correct (True/False/None)
        label_correct: Whether bot's label was correct (True/False/None)
        correct_decision: What decision should have been ('keep' or 'archive')
        correct_label: What label should have been applied
        user_comment: Natural language feedback from user
        used_in_training: Whether this feedback has been used in model retraining
    """

    __tablename__ = 'feedback_reviews'

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey('emails.message_id'), nullable=False, index=True)
    review_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Correctness flags
    decision_correct = Column(Boolean, nullable=True)
    label_correct = Column(Boolean, nullable=True)

    # Corrections
    correct_decision = Column(String, nullable=True)  # 'keep' or 'archive'
    correct_label = Column(String, nullable=True)

    # Natural language feedback
    user_comment = Column(Text, nullable=True)

    # Training tracking
    used_in_training = Column(Boolean, default=False, index=True)

    # Relationships
    email = relationship("Email", back_populates="feedback_reviews")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<FeedbackReview(message_id={self.message_id[:10]}..., "
            f"decision_correct={self.decision_correct})>"
        )
