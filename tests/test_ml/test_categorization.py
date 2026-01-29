"""Tests for email categorization."""


from src.database.schema import SenderLabelMapping
from src.ml.categorization import EmailCategorizer


class TestEmailCategorizerBasics:
    """Test basic EmailCategorizer functionality."""

    def test_init(self, temp_db_session):
        """Test categorizer initialization."""
        categorizer = EmailCategorizer(temp_db_session)
        assert categorizer.session is temp_db_session

    def test_category_constants(self):
        """Test that category constants are defined."""
        assert EmailCategorizer.NEWSLETTER_TECH == 'Bot/Newsletter-Tech'
        assert EmailCategorizer.NEWSLETTER_FINANCE == 'Bot/Newsletter-Finance'
        assert EmailCategorizer.PROMOTIONAL == 'Bot/Promotional'
        assert EmailCategorizer.AUTOMATED == 'Bot/Automated'
        assert EmailCategorizer.RECEIPTS == 'Bot/Receipts'


class TestEmailCategorizerCategorize:
    """Test EmailCategorizer.categorize() method."""

    def test_categorize_receipt(self, temp_db_session, email_factory):
        """Test categorizing receipt emails."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="Your order confirmation",
            snippet="Thank you for your purchase"
        )

        result = categorizer.categorize(email)

        assert result == EmailCategorizer.RECEIPTS

    def test_categorize_automated(self, temp_db_session, email_factory):
        """Test categorizing automated notification emails."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="[GitHub] Pull request merged",
            snippet="Your pull request has been merged"
        )

        result = categorizer.categorize(email)

        assert result == EmailCategorizer.AUTOMATED

    def test_categorize_promotional(self, temp_db_session, email_factory):
        """Test categorizing promotional emails."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="50% off sale - limited time!",
            snippet="Save big with our exclusive coupon"
        )

        result = categorizer.categorize(email)

        assert result == EmailCategorizer.PROMOTIONAL

    def test_categorize_tech_newsletter(self, temp_db_session, email_factory):
        """Test categorizing tech newsletters."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="This week in AI and machine learning",
            snippet="Latest developments in artificial intelligence"
        )

        result = categorizer.categorize(email, is_newsletter=True)

        assert result == EmailCategorizer.NEWSLETTER_TECH

    def test_categorize_finance_newsletter(self, temp_db_session, email_factory):
        """Test categorizing finance newsletters."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="Market update: Stock analysis",
            snippet="Your weekly investing digest"
        )

        result = categorizer.categorize(email, is_newsletter=True)

        assert result == EmailCategorizer.NEWSLETTER_FINANCE

    def test_categorize_news_newsletter(self, temp_db_session, email_factory):
        """Test categorizing news newsletters."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="Morning news: Politics and world events",
            snippet="Breaking stories and weekly roundup"
        )

        result = categorizer.categorize(email, is_newsletter=True)

        assert result == EmailCategorizer.NEWSLETTER_NEWS

    def test_categorize_other_newsletter(self, temp_db_session, email_factory):
        """Test categorizing generic newsletters."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            subject="Monthly gardening tips",
            snippet="How to grow tomatoes"
        )

        result = categorizer.categorize(email, is_newsletter=True)

        assert result == EmailCategorizer.NEWSLETTER_OTHER

    def test_categorize_work_email(self, temp_db_session, email_factory):
        """Test categorizing work emails."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            from_address="colleague@company.edu",
            subject="Meeting invitation: Project review",
            snippet="Let's discuss the deadline"
        )

        result = categorizer.categorize(email)

        assert result == EmailCategorizer.WORK

    def test_categorize_personal_default(self, temp_db_session, email_factory):
        """Test default categorization to personal."""
        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            from_address="friend@gmail.com",
            subject="Hey there!",
            snippet="How are you doing?"
        )

        result = categorizer.categorize(email)

        assert result == EmailCategorizer.PERSONAL


class TestLearnedLabels:
    """Test learned label functionality."""

    def test_categorize_uses_learned_label(self, temp_db_session, email_factory):
        """Test that learned sender labels take priority."""
        # Create a sender mapping
        mapping = SenderLabelMapping(
            sender_address="known@example.com",
            label="Bot/MyCustomLabel",
            source="feedback"
        )
        temp_db_session.add(mapping)
        temp_db_session.commit()

        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            from_address="known@example.com",
            subject="50% off sale!",  # Would normally be promotional
            snippet="Exclusive offer"
        )

        result = categorizer.categorize(email)

        # Should use learned label instead of promotional
        assert result == "Bot/MyCustomLabel"

    def test_categorize_uses_domain_label(self, temp_db_session, email_factory):
        """Test that domain-level labels work."""
        # Create a domain mapping
        mapping = SenderLabelMapping(
            sender_domain="company.org",
            sender_address=None,
            label="Bot/CompanyNews",
            source="feedback"
        )
        temp_db_session.add(mapping)
        temp_db_session.commit()

        categorizer = EmailCategorizer(temp_db_session)
        email = email_factory(
            from_address="anyone@company.org",
            subject="Random subject",
            snippet="Some content"
        )

        result = categorizer.categorize(email)

        assert result == "Bot/CompanyNews"


class TestConfidenceLabels:
    """Test confidence and action label methods."""

    def test_add_confidence_label_low(self, temp_db_session):
        """Test adding low confidence label."""
        categorizer = EmailCategorizer(temp_db_session)

        result = categorizer.add_confidence_label("Bot/Newsletter-Tech", "low")

        assert result == ["Bot/Newsletter-Tech", "Bot/LowConfidence"]

    def test_add_confidence_label_high(self, temp_db_session):
        """Test adding high confidence label (no extra label)."""
        categorizer = EmailCategorizer(temp_db_session)

        result = categorizer.add_confidence_label("Bot/Newsletter-Tech", "high")

        assert result == ["Bot/Newsletter-Tech"]

    def test_add_action_label_archive(self, temp_db_session):
        """Test adding archive action label."""
        categorizer = EmailCategorizer(temp_db_session)

        result = categorizer.add_action_label(["Bot/Newsletter-Tech"], "archive")

        assert result == ["Bot/Newsletter-Tech", "Bot/AutoArchived"]

    def test_add_action_label_keep(self, temp_db_session):
        """Test adding keep action label (no extra label)."""
        categorizer = EmailCategorizer(temp_db_session)

        result = categorizer.add_action_label(["Bot/Newsletter-Tech"], "keep")

        assert result == ["Bot/Newsletter-Tech"]


class TestGetAllCategories:
    """Test get_all_categories static method."""

    def test_get_all_categories_without_session(self):
        """Test getting built-in categories without session."""
        categories = EmailCategorizer.get_all_categories()

        assert 'Bot/Newsletter-Tech' in categories
        assert 'Bot/Newsletter-Finance' in categories
        assert 'Bot/Promotional' in categories
        assert 'Bot/Automated' in categories
        assert 'Bot/Personal' in categories
        assert 'Bot/Work' in categories
        assert 'Bot/Receipts' in categories
        assert len(categories) == 11  # All built-in categories

    def test_get_all_categories_with_custom_labels(self, temp_db_session):
        """Test getting categories with custom labels from database."""
        # Add custom labels
        mapping = SenderLabelMapping(
            sender_address="custom@example.com",
            label="Bot/CustomCategory",
            source="feedback"
        )
        temp_db_session.add(mapping)
        temp_db_session.commit()

        categories = EmailCategorizer.get_all_categories(temp_db_session)

        assert 'Bot/Newsletter-Tech' in categories
        assert 'Bot/CustomCategory' in categories
        assert len(categories) == 12  # Built-in + custom

    def test_get_all_categories_no_duplicate_builtin(self, temp_db_session):
        """Test that built-in labels aren't duplicated when in DB."""
        # Add a built-in label as custom
        mapping = SenderLabelMapping(
            sender_address="someone@example.com",
            label="Bot/Newsletter-Tech",  # Built-in label
            source="feedback"
        )
        temp_db_session.add(mapping)
        temp_db_session.commit()

        categories = EmailCategorizer.get_all_categories(temp_db_session)

        # Should not be duplicated
        assert categories.count('Bot/Newsletter-Tech') == 1
        assert len(categories) == 11  # Still just built-in
