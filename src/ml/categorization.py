"""Email categorization for automatic labeling."""

from sqlalchemy.orm import Session

from ..database.schema import Email, SenderLabelMapping
from ..utils import get_logger

logger = get_logger(__name__)


class EmailCategorizer:
    """
    Categorize emails for automatic labeling.

    Assigns category labels like "Bot/Newsletter-Tech", "Bot/Promotional",
    etc. based on email content and metadata.

    Example:
        >>> with db.get_session() as session:
        ...     categorizer = EmailCategorizer(session)
        ...     label = categorizer.categorize(email)
        ...     print(f"Category: {label}")
    """

    # Newsletter categories
    NEWSLETTER_TECH = 'Bot/Newsletter-Tech'
    NEWSLETTER_FINANCE = 'Bot/Newsletter-Finance'
    NEWSLETTER_NEWS = 'Bot/Newsletter-News'
    NEWSLETTER_OTHER = 'Bot/Newsletter-Other'

    # Email types
    PROMOTIONAL = 'Bot/Promotional'
    AUTOMATED = 'Bot/Automated'
    PERSONAL = 'Bot/Personal'
    WORK = 'Bot/Work'
    RECEIPTS = 'Bot/Receipts'

    # Confidence marker
    LOW_CONFIDENCE = 'Bot/LowConfidence'
    AUTO_ARCHIVED = 'Bot/AutoArchived'

    # Tech keywords
    TECH_KEYWORDS = [
        'ai', 'ml', 'machine learning', 'artificial intelligence',
        'programming', 'coding', 'developer', 'software', 'tech',
        'python', 'javascript', 'react', 'api', 'github', 'code',
        'startup', 'saas', 'cloud', 'aws', 'docker', 'kubernetes',
        'data science', 'devops', 'engineering'
    ]

    # Finance keywords
    FINANCE_KEYWORDS = [
        'finance', 'investing', 'stock', 'market', 'trading',
        'crypto', 'bitcoin', 'ethereum', 'blockchain',
        'money', 'wealth', 'portfolio', 'dividend', 'etf',
        'banking', 'credit', 'mortgage', 'loan'
    ]

    # News keywords
    NEWS_KEYWORDS = [
        'news', 'daily briefing', 'newsletter', 'digest',
        'politics', 'world', 'breaking', 'update',
        'today', 'this week', 'weekly roundup'
    ]

    # Promotional indicators
    PROMO_INDICATORS = [
        'sale', 'discount', 'offer', 'deal', 'coupon',
        'limited time', 'expires', 'save', 'off',
        'free shipping', 'buy now', 'shop now',
        'special offer', 'exclusive'
    ]

    # Automated/notification indicators
    AUTOMATED_INDICATORS = [
        'notification', 'alert', 'reminder', 'automated',
        'no-reply', 'noreply', 'do not reply',
        '[github]', '[jira]', '[confluence]', '[slack]',
        'pull request', 'issue', 'commit', 'build failed'
    ]

    # Receipt indicators
    RECEIPT_INDICATORS = [
        'receipt', 'order confirmation', 'invoice', 'payment',
        'thank you for your purchase', 'order has been',
        'shipped', 'delivery', 'tracking'
    ]

    def __init__(self, session: Session):
        """
        Initialize categorizer.

        Args:
            session: Database session for learned label lookups
        """
        self.session = session

    def categorize(
        self,
        email: Email,
        is_newsletter: bool = False
    ) -> str:
        """
        Categorize an email and return appropriate label.

        First checks for learned sender-label mappings from user feedback,
        then falls back to keyword-based rules.

        Args:
            email: Email database model
            is_newsletter: Whether email is detected as newsletter

        Returns:
            Category label (e.g., "Bot/Newsletter-Tech")

        Example:
            >>> label = categorizer.categorize(email, is_newsletter=True)
        """
        # Check for learned sender label first
        learned_label = self._get_learned_label(email.from_address)
        if learned_label:
            logger.debug(f"Using learned label for {email.from_address}: {learned_label}")
            return learned_label

        # Combine subject and snippet for analysis
        text = f"{email.subject or ''} {email.snippet or ''}".lower()

        # Check for receipts first (high confidence)
        if self._contains_any(text, self.RECEIPT_INDICATORS):
            return self.RECEIPTS

        # Check for automated notifications
        if self._contains_any(text, self.AUTOMATED_INDICATORS):
            return self.AUTOMATED

        # Check for promotional content
        if self._contains_any(text, self.PROMO_INDICATORS):
            return self.PROMOTIONAL

        # If newsletter, categorize by topic
        if is_newsletter:
            if self._contains_any(text, self.TECH_KEYWORDS):
                return self.NEWSLETTER_TECH
            elif self._contains_any(text, self.FINANCE_KEYWORDS):
                return self.NEWSLETTER_FINANCE
            elif self._contains_any(text, self.NEWS_KEYWORDS):
                return self.NEWSLETTER_NEWS
            else:
                return self.NEWSLETTER_OTHER

        # Check if work-related (common work domains)
        if self._is_work_email(email):
            return self.WORK

        # Default to personal
        return self.PERSONAL

    def _get_learned_label(self, sender_address: str) -> str | None:
        """
        Get learned label for a sender from feedback mappings.

        Args:
            sender_address: Sender email address

        Returns:
            Label if found, None otherwise
        """
        if not sender_address:
            return None

        # Check for exact address match
        mapping = self.session.query(SenderLabelMapping).filter(
            SenderLabelMapping.sender_address == sender_address
        ).first()

        if mapping:
            return mapping.label

        # Check for domain match
        domain = sender_address.split('@')[-1] if '@' in sender_address else None
        if domain:
            mapping = self.session.query(SenderLabelMapping).filter(
                SenderLabelMapping.sender_domain == domain,
                SenderLabelMapping.sender_address.is_(None)
            ).first()
            if mapping:
                return mapping.label

        return None

    def _contains_any(self, text: str, keywords: list[str]) -> bool:
        """
        Check if text contains any of the keywords.

        Args:
            text: Text to search (should be lowercase)
            keywords: List of keywords (should be lowercase)

        Returns:
            True if any keyword found
        """
        for keyword in keywords:
            if keyword in text:
                return True
        return False

    def _is_work_email(self, email: Email) -> bool:
        """
        Check if email appears to be work-related.

        Args:
            email: Email database model

        Returns:
            True if appears to be work email
        """
        # Check for common work domains
        work_domains = [
            '@company.com',  # Replace with actual company domain
            '.edu',  # Educational
        ]

        from_domain = email.from_address.lower()

        for domain in work_domains:
            if domain in from_domain:
                return True

        # Check for work-related keywords in subject
        work_keywords = [
            'meeting', 'calendar', 'invitation', 'project',
            'team', 'review', 'feedback', 'deadline'
        ]

        subject = (email.subject or '').lower()
        for keyword in work_keywords:
            if keyword in subject:
                return True

        return False

    def add_confidence_label(
        self,
        base_label: str,
        confidence: str
    ) -> list[str]:
        """
        Add confidence labels to base category.

        Args:
            base_label: Base category label
            confidence: Confidence level ('high' or 'low')

        Returns:
            List of labels to apply

        Example:
            >>> labels = categorizer.add_confidence_label("Bot/Newsletter-Tech", "low")
            >>> print(labels)  # ["Bot/Newsletter-Tech", "Bot/LowConfidence"]
        """
        labels = [base_label]

        if confidence == 'low':
            labels.append(self.LOW_CONFIDENCE)

        return labels

    def add_action_label(
        self,
        base_labels: list[str],
        action: str
    ) -> list[str]:
        """
        Add action-specific labels.

        Args:
            base_labels: Base category labels
            action: Action taken ('archive' or 'keep')

        Returns:
            Updated list of labels

        Example:
            >>> labels = categorizer.add_action_label(["Bot/Newsletter-Tech"], "archive")
            >>> print(labels)  # ["Bot/Newsletter-Tech", "Bot/AutoArchived"]
        """
        labels = base_labels.copy()

        if action == 'archive':
            labels.append(self.AUTO_ARCHIVED)

        return labels

    @staticmethod
    def get_all_categories(session: Session | None = None) -> list[str]:
        """
        Get list of all possible category labels.

        Includes built-in categories plus any custom labels from user feedback.

        Args:
            session: Database session to fetch custom labels (optional for built-in only)

        Returns:
            List of category labels

        Example:
            >>> categories = EmailCategorizer.get_all_categories(session)
        """
        built_in = [
            EmailCategorizer.NEWSLETTER_TECH,
            EmailCategorizer.NEWSLETTER_FINANCE,
            EmailCategorizer.NEWSLETTER_NEWS,
            EmailCategorizer.NEWSLETTER_OTHER,
            EmailCategorizer.PROMOTIONAL,
            EmailCategorizer.AUTOMATED,
            EmailCategorizer.PERSONAL,
            EmailCategorizer.WORK,
            EmailCategorizer.RECEIPTS,
            EmailCategorizer.LOW_CONFIDENCE,
            EmailCategorizer.AUTO_ARCHIVED,
        ]

        if session:
            # Add custom labels from user feedback
            custom = session.query(SenderLabelMapping.label).distinct().all()
            custom_labels = [c[0] for c in custom if c[0] not in built_in]
            return built_in + custom_labels

        return built_in
