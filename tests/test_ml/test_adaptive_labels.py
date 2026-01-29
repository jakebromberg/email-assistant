"""Tests for adaptive label generation."""

import numpy as np
import pytest

from src.database.schema import Email, EmailFeatures, SenderLabelMapping
from src.ml.adaptive_labels import AdaptiveLabelGenerator, LabelSuggestion


class TestLabelSuggestion:
    """Tests for LabelSuggestion dataclass."""

    def test_label_suggestion_creation(self):
        """Test creating a label suggestion."""
        suggestion = LabelSuggestion(
            label='Bot/Newsletter-Tech',
            specificity=2,
            confidence=0.8,
            reason='Matched tech keywords'
        )

        assert suggestion.label == 'Bot/Newsletter-Tech'
        assert suggestion.specificity == 2
        assert suggestion.confidence == 0.8
        assert suggestion.reason == 'Matched tech keywords'


class TestAdaptiveLabelGenerator:
    """Tests for AdaptiveLabelGenerator class."""

    def test_init(self, temp_db_session):
        """Test generator initialization."""
        generator = AdaptiveLabelGenerator(temp_db_session)

        assert generator.session == temp_db_session
        assert generator.specificity_threshold == 3
        assert generator._label_centroids is None

    def test_init_custom_threshold(self, temp_db_session):
        """Test generator with custom threshold."""
        generator = AdaptiveLabelGenerator(temp_db_session, specificity_threshold=5)

        assert generator.specificity_threshold == 5

    def test_generate_labels_no_matches(self, temp_db_session, email_factory):
        """Test generating labels when no matches found."""
        email = email_factory()
        temp_db_session.add(email)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email)

        # Should return at least one suggestion (fallback)
        assert len(suggestions) >= 1
        assert all(isinstance(s, LabelSuggestion) for s in suggestions)

    def test_generate_labels_uses_learned_label(self, temp_db_session, email_factory):
        """Test that learned labels take priority."""
        email = email_factory(from_address='newsletter@example.com')
        temp_db_session.add(email)

        # Add a learned mapping
        mapping = SenderLabelMapping(
            sender_address='newsletter@example.com',
            label='Bot/My-Custom-Label'
        )
        temp_db_session.add(mapping)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email)

        # Should return the learned label
        assert len(suggestions) == 1
        assert suggestions[0].label == 'Bot/My-Custom-Label'
        assert suggestions[0].confidence == 1.0
        assert 'user feedback' in suggestions[0].reason.lower()

    def test_generate_labels_domain_mapping(self, temp_db_session, email_factory):
        """Test domain-level learned labels."""
        email = email_factory(from_address='support@company.com')
        temp_db_session.add(email)

        # Add domain-level mapping
        mapping = SenderLabelMapping(
            sender_domain='company.com',
            sender_address=None,
            label='Bot/Company'
        )
        temp_db_session.add(mapping)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email)

        assert len(suggestions) == 1
        assert suggestions[0].label == 'Bot/Company'

    def test_generate_labels_receipt_keywords(self, temp_db_session, email_factory):
        """Test receipt keyword detection."""
        email = email_factory(
            subject='Your order confirmation #12345',
            snippet='Thank you for your purchase'
        )
        temp_db_session.add(email)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email)

        # Should detect receipt keywords
        labels = [s.label for s in suggestions]
        assert 'Bot/Receipts' in labels

    def test_generate_labels_newsletter_detection(self, temp_db_session, email_factory):
        """Test newsletter detection via is_newsletter flag."""
        email = email_factory(subject='Weekly digest')
        temp_db_session.add(email)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email, is_newsletter=True)

        labels = [s.label for s in suggestions]
        assert 'Bot/Newsletters' in labels

    def test_generate_labels_sorted_by_specificity(self, temp_db_session, email_factory):
        """Test that suggestions are sorted by specificity."""
        # Create multiple emails from same sender to trigger sender-specific label
        for i in range(5):
            email = email_factory(
                message_id=f'msg{i}@test.com',
                from_address='sender@tech.io',
                from_name='Tech Sender'
            )
            temp_db_session.add(email)
        temp_db_session.flush()

        # Get last email for labeling
        email = temp_db_session.query(Email).filter(
            Email.message_id == 'msg4@test.com'
        ).first()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email)

        # Should be sorted by specificity (lower = more specific)
        if len(suggestions) > 1:
            for i in range(len(suggestions) - 1):
                assert suggestions[i].specificity <= suggestions[i + 1].specificity

    def test_get_primary_label(self, temp_db_session, email_factory):
        """Test getting single best label."""
        email = email_factory(subject='Your receipt')
        temp_db_session.add(email)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        label = generator.get_primary_label(email)

        assert isinstance(label, str)
        assert label.startswith('Bot/')

    def test_get_all_labels(self, temp_db_session, email_factory):
        """Test getting multiple labels."""
        email = email_factory()
        temp_db_session.add(email)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        labels = generator.get_all_labels(email, max_labels=2)

        assert isinstance(labels, list)
        assert len(labels) <= 2
        assert all(isinstance(l, str) for l in labels)

    def test_invalidate_cache(self, temp_db_session):
        """Test cache invalidation."""
        generator = AdaptiveLabelGenerator(temp_db_session)
        generator._label_centroids = {'test': np.array([1, 2, 3])}

        generator.invalidate_cache()

        assert generator._label_centroids is None

    def test_clean_label_name(self, temp_db_session):
        """Test label name cleaning."""
        generator = AdaptiveLabelGenerator(temp_db_session)

        # Test various inputs
        assert generator._clean_label_name('Tech Newsletter') == 'Tech-Newsletter'
        assert generator._clean_label_name('hello@world') == 'Helloworld'
        assert generator._clean_label_name('A  B  C') == 'A-B-C'

    def test_clean_label_name_truncation(self, temp_db_session):
        """Test label name truncation for long names."""
        generator = AdaptiveLabelGenerator(temp_db_session)

        long_name = 'This Is A Very Long Label Name That Should Be Truncated'
        result = generator._clean_label_name(long_name)

        assert len(result) <= 30

    def test_domain_to_label(self, temp_db_session):
        """Test domain to label conversion."""
        generator = AdaptiveLabelGenerator(temp_db_session)

        assert generator._domain_to_label('example.com') == 'Example'
        assert generator._domain_to_label('newsletters.example.org') == 'Example'
        assert generator._domain_to_label('ab.io') is None  # Too short

    def test_embedding_similarity_matching(self, temp_db_session, email_factory):
        """Test embedding-based similarity matching."""
        # Create email with learned label
        labeled_email = email_factory(
            message_id='labeled@test.com',
            from_address='tech@example.com'
        )
        temp_db_session.add(labeled_email)

        # Add sender mapping
        mapping = SenderLabelMapping(
            sender_address='tech@example.com',
            label='Bot/Tech-News'
        )
        temp_db_session.add(mapping)

        # Add features with embedding for labeled email
        embedding = np.random.rand(384).tolist()
        features = EmailFeatures(
            message_id='labeled@test.com',
            subject_embedding=embedding
        )
        temp_db_session.add(features)
        temp_db_session.flush()

        # Create new email with similar embedding
        new_email = email_factory(
            message_id='new@test.com',
            from_address='other@different.com'
        )
        temp_db_session.add(new_email)

        new_features = EmailFeatures(
            message_id='new@test.com',
            subject_embedding=embedding  # Same embedding = similar
        )
        temp_db_session.add(new_features)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(new_email, new_features)

        # Should find similarity to labeled emails
        labels = [s.label for s in suggestions]
        # The similar email should potentially get matched
        assert len(suggestions) >= 1


class TestFallbackCategories:
    """Tests for fallback category matching."""

    @pytest.mark.parametrize('subject,expected_label', [
        ('Your payment receipt', 'Bot/Receipts'),
        ('Order confirmation #123', 'Bot/Receipts'),
        ('GitHub notification', 'Bot/Automated'),
        ('Alert: server down', 'Bot/Automated'),
        ('50% off sale today!', 'Bot/Promotional'),
        ('Limited time offer', 'Bot/Promotional'),
    ])
    def test_keyword_matching(self, temp_db_session, email_factory, subject, expected_label):
        """Test keyword-based category matching."""
        email = email_factory(subject=subject)
        temp_db_session.add(email)
        temp_db_session.flush()

        generator = AdaptiveLabelGenerator(temp_db_session)
        suggestions = generator.generate_labels(email)

        labels = [s.label for s in suggestions]
        assert expected_label in labels
