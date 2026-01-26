"""Tests for metadata feature extraction."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.features.metadata import MetadataExtractor
from src.database.schema import Email


class TestMetadataExtractor:
    """Test MetadataExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return MetadataExtractor()

    def _create_mock_email(
        self,
        message_id='msg1',
        from_address='sender@example.com',
        subject='Test Subject',
        body_plain='Test body content',
        date=None,
        labels=None
    ):
        """Create mock Email object."""
        email = Mock(spec=Email)
        email.message_id = message_id
        email.from_address = from_address
        email.subject = subject
        email.body_plain = body_plain
        email.date = date or datetime(2024, 1, 15, 14, 30)
        email.labels = labels or []
        return email

    def test_extract_basic_features(self, extractor):
        """Test basic feature extraction."""
        email = self._create_mock_email(
            from_address='user@example.com',
            subject='Hello World',
            body_plain='This is a test email.'
        )

        features = extractor.extract(email)

        assert features['sender_domain'] == 'example.com'
        assert features['subject_length'] == 11
        assert features['body_length'] == 21
        assert 'sender_address_hash' in features
        assert isinstance(features['is_newsletter'], bool)

    def test_extract_time_features(self, extractor):
        """Test time feature extraction."""
        # Monday, Jan 15, 2024, 2:30 PM
        email = self._create_mock_email(
            date=datetime(2024, 1, 15, 14, 30)
        )

        features = extractor.extract(email)

        assert features['day_of_week'] == 0  # Monday
        assert features['hour_of_day'] == 14

    def test_extract_domain(self, extractor):
        """Test domain extraction."""
        assert extractor._extract_domain('user@example.com') == 'example.com'
        assert extractor._extract_domain('test@sub.example.com') == 'sub.example.com'
        assert extractor._extract_domain('invalid') == 'unknown'
        assert extractor._extract_domain('') == 'unknown'

    def test_hash_address(self, extractor):
        """Test email address hashing."""
        hash1 = extractor._hash_address('user@example.com')
        hash2 = extractor._hash_address('user@example.com')
        hash3 = extractor._hash_address('other@example.com')

        assert hash1 == hash2  # Same email, same hash
        assert hash1 != hash3  # Different email, different hash
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_is_newsletter_via_content(self, extractor):
        """Test newsletter detection via content patterns."""
        # Newsletter indicators in subject
        email1 = self._create_mock_email(
            subject='Weekly Newsletter: Tech Updates'
        )
        assert extractor._is_newsletter(email1) is True

        # Unsubscribe in body
        email2 = self._create_mock_email(
            body_plain='Click here to unsubscribe from this mailing list'
        )
        assert extractor._is_newsletter(email2) is True

        # Regular email
        email3 = self._create_mock_email(
            subject='Meeting tomorrow',
            body_plain='Let me know if you can attend'
        )
        assert extractor._is_newsletter(email3) is False

    def test_is_newsletter_via_labels(self, extractor):
        """Test newsletter detection via Gmail labels."""
        # Email with mailing list labels
        email = self._create_mock_email(
            labels=['INBOX', 'CATEGORY_PROMOTIONS', 'listid']
        )
        # Note: Current implementation checks for 'listunsubscribe', 'listid', 'mailinglist'
        # This test documents the expected behavior
        features = extractor.extract(email)
        # The implementation searches for indicators in the joined labels string

    def test_extract_with_none_values(self, extractor):
        """Test handling of None values."""
        email = self._create_mock_email(
            subject=None,
            body_plain=None
        )

        features = extractor.extract(email)

        assert features['subject_length'] == 0
        assert features['body_length'] == 0
        # Should not raise error

    def test_extract_batch(self, extractor):
        """Test batch extraction."""
        emails = [
            self._create_mock_email(message_id='msg1'),
            self._create_mock_email(message_id='msg2'),
            self._create_mock_email(message_id='msg3')
        ]

        features_list = extractor.extract_batch(emails)

        assert len(features_list) == 3
        assert all(isinstance(f, dict) for f in features_list)
        assert all('sender_domain' in f for f in features_list)

    def test_get_feature_names(self, extractor):
        """Test feature names retrieval."""
        names = MetadataExtractor.get_feature_names()

        assert 'sender_domain' in names
        assert 'sender_address_hash' in names
        assert 'is_newsletter' in names
        assert 'day_of_week' in names
        assert 'hour_of_day' in names
        assert 'subject_length' in names
        assert 'body_length' in names

    def test_extract_works_without_headers_attribute(self, extractor):
        """
        REGRESSION TEST: Ensure extraction works with database Email model
        that doesn't have a 'headers' attribute.

        This test prevents the bug where _is_newsletter() tried to access
        email.headers.items() which doesn't exist on the database model.
        """
        # Create email WITHOUT headers attribute (like database Email)
        email = self._create_mock_email(
            subject='Newsletter Update',
            body_plain='Click to unsubscribe'
        )

        # This should not raise AttributeError
        features = extractor.extract(email)

        # Should successfully detect as newsletter via content
        assert features['is_newsletter'] is True
        assert 'sender_domain' in features
        assert 'subject_length' in features
