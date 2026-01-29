"""SQLAlchemy database schema for email storage."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
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
    def user_labels(self) -> list[str]:
        """Get user-applied labels (excluding system labels)."""
        system_labels = {'INBOX', 'UNREAD', 'IMPORTANT', 'STARRED', 'SENT', 'DRAFT', 'TRASH', 'SPAM'}
        return [label for label in self.labels if label not in system_labels and not label.startswith('Bot/')]

    @property
    def bot_labels(self) -> list[str]:
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


class SenderLabelMapping(Base):
    """
    Learned sender-to-label mappings from user feedback.

    When a user corrects a label during review, we store the mapping
    so future emails from the same sender get the correct label.

    Attributes:
        id: Primary key
        sender_address: Full sender email address
        sender_domain: Sender's domain (for domain-level mappings)
        label: The correct label for this sender
        source: How this mapping was learned ('feedback', 'manual')
        created_at: When mapping was created
        updated_at: When mapping was last updated
    """

    __tablename__ = 'sender_label_mappings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sender_address = Column(String, index=True, nullable=True)
    sender_domain = Column(String, index=True, nullable=True)
    label = Column(String, nullable=False)
    source = Column(String, default='feedback')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        """Return string representation."""
        sender = self.sender_address or self.sender_domain
        return f"<SenderLabelMapping({sender} -> {self.label})>"


class SemanticLabel(Base):
    """
    Semantic labels generated by Claude API.

    Used as training data for the local semantic classifier.

    Attributes:
        id: Primary key
        message_id: Foreign key to emails table
        label: Semantic label (e.g., "Product-Update", "Security-Alert")
        source: How label was generated ('claude', 'user', 'classifier')
        model: Model used to generate label (if source='claude')
        confidence: Confidence score (if source='classifier')
        created_at: When label was generated
    """

    __tablename__ = 'semantic_labels'

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey('emails.message_id'), nullable=False, index=True)
    label = Column(String, nullable=False, index=True)
    source = Column(String, default='claude')  # 'claude', 'user', 'classifier'
    model = Column(String, nullable=True)  # e.g., 'claude-sonnet-4-20250514'
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<SemanticLabel(message_id={self.message_id[:10]}..., label={self.label})>"


class EmailFeatures(Base):
    """
    Computed features for an email.

    Stores metadata features, historical patterns, and topic embeddings
    for ML model training and inference.

    Attributes:
        message_id: Foreign key to emails table (primary key)

        # Metadata features
        sender_domain: Domain of sender email
        sender_address_hash: Hash of sender address for privacy
        is_newsletter: Whether email appears to be a newsletter
        day_of_week: Day of week (0=Monday, 6=Sunday)
        hour_of_day: Hour of day (0-23)
        subject_length: Length of subject line
        body_length: Length of email body
        has_attachments: Whether email has attachments
        thread_length: Number of emails in thread

        # Historical pattern features
        sender_email_count: Total emails from this sender
        sender_open_rate: Percentage of emails from sender that were opened
        sender_days_since_last: Days since last email from sender
        domain_open_rate: Open rate for sender's domain

        # Topic embeddings (stored as JSON arrays)
        subject_embedding: 384-dim embedding of subject
        body_embedding: 384-dim embedding of body (first 512 tokens)

        # Computed timestamp
        computed_at: When features were computed
    """

    __tablename__ = 'email_features'

    message_id = Column(String, ForeignKey('emails.message_id'), primary_key=True, index=True)

    # Metadata features
    sender_domain = Column(String, index=True)
    sender_address_hash = Column(String, index=True)
    is_newsletter = Column(Boolean, default=False, index=True)
    day_of_week = Column(Integer)  # 0=Monday, 6=Sunday
    hour_of_day = Column(Integer)  # 0-23
    subject_length = Column(Integer)
    body_length = Column(Integer)
    has_attachments = Column(Boolean, default=False)
    thread_length = Column(Integer, default=1)

    # Historical pattern features
    sender_email_count = Column(Integer, default=0)
    sender_open_rate = Column(Float, default=0.0)
    sender_days_since_last = Column(Float, nullable=True)
    domain_open_rate = Column(Float, default=0.0)

    # Topic embeddings (as JSON arrays)
    subject_embedding = Column(JSON, nullable=True)
    body_embedding = Column(JSON, nullable=True)

    # Timestamp
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<EmailFeatures(message_id={self.message_id[:10]}..., "
            f"sender_domain={self.sender_domain})>"
        )
