"""Tests for feature storage and retrieval."""


from src.features.feature_store import FeatureStore


class TestFeatureStoreBasics:
    """Test basic FeatureStore operations."""

    def test_init(self, temp_db_session):
        """Test FeatureStore initialization."""
        store = FeatureStore(temp_db_session)
        assert store.session is temp_db_session

    def test_get_features_not_found(self, feature_store):
        """Test get_features returns None when not found."""
        result = feature_store.get_features("nonexistent")
        assert result is None

    def test_has_features_false(self, feature_store):
        """Test has_features returns False when not found."""
        result = feature_store.has_features("nonexistent")
        assert result is False

    def test_has_features_true(self, feature_store, email_factory, features_factory):
        """Test has_features returns True when exists."""
        email = email_factory()
        features_factory(message_id=email.message_id)

        result = feature_store.has_features(email.message_id)
        assert result is True


class TestFeatureStoreSave:
    """Test FeatureStore save operations."""

    def test_save_features_new(self, feature_store, email_factory):
        """Test saving new features."""
        email = email_factory()
        features = {
            'sender_domain': 'example.com',
            'is_newsletter': True,
            'sender_open_rate': 0.75,
        }

        result = feature_store.save_features(email.message_id, features)

        assert result.message_id == email.message_id
        assert result.sender_domain == 'example.com'
        assert result.is_newsletter is True
        assert result.sender_open_rate == 0.75

    def test_save_features_update_existing(self, feature_store, email_factory):
        """Test updating existing features."""
        email = email_factory()

        # Save initial features
        initial = feature_store.save_features(email.message_id, {
            'sender_domain': 'old.com',
            'sender_open_rate': 0.5,
        })
        feature_store.session.commit()

        # Update features
        updated = feature_store.save_features(email.message_id, {
            'sender_domain': 'new.com',
            'sender_open_rate': 0.9,
        })

        assert updated.sender_domain == 'new.com'
        assert updated.sender_open_rate == 0.9

    def test_save_features_batch(self, feature_store, email_factory):
        """Test saving features in batch."""
        emails = [email_factory() for _ in range(3)]
        features_list = [
            (emails[0].message_id, {'sender_domain': 'a.com', 'is_newsletter': False}),
            (emails[1].message_id, {'sender_domain': 'b.com', 'is_newsletter': True}),
            (emails[2].message_id, {'sender_domain': 'c.com', 'is_newsletter': False}),
        ]

        result = feature_store.save_features_batch(features_list)

        assert len(result) == 3
        assert result[0].sender_domain == 'a.com'
        assert result[1].sender_domain == 'b.com'
        assert result[2].sender_domain == 'c.com'


class TestFeatureStoreGet:
    """Test FeatureStore get operations."""

    def test_get_features(self, feature_store, email_factory, features_factory):
        """Test getting features by message ID."""
        email = email_factory()
        features_factory(message_id=email.message_id, sender_domain='test.com')

        result = feature_store.get_features(email.message_id)

        assert result is not None
        assert result.sender_domain == 'test.com'

    def test_get_features_batch(self, feature_store, email_factory, features_factory):
        """Test getting features for multiple emails."""
        emails = [email_factory() for _ in range(3)]
        features_factory(message_id=emails[0].message_id, sender_domain='a.com')
        features_factory(message_id=emails[2].message_id, sender_domain='c.com')
        # emails[1] has no features

        result = feature_store.get_features_batch([e.message_id for e in emails])

        assert len(result) == 3
        assert result[0].sender_domain == 'a.com'
        assert result[1] is None  # No features for this email
        assert result[2].sender_domain == 'c.com'

    def test_get_all_features(self, feature_store, email_factory, features_factory):
        """Test getting all features."""
        emails = [email_factory() for _ in range(5)]
        for email in emails:
            features_factory(message_id=email.message_id)

        result = feature_store.get_all_features()
        assert len(result) == 5

    def test_get_all_features_with_limit(self, feature_store, email_factory, features_factory):
        """Test getting all features with limit."""
        emails = [email_factory() for _ in range(5)]
        for email in emails:
            features_factory(message_id=email.message_id)

        result = feature_store.get_all_features(limit=3)
        assert len(result) == 3

    def test_get_all_features_with_offset(self, feature_store, email_factory, features_factory):
        """Test getting all features with offset."""
        emails = [email_factory() for _ in range(5)]
        for email in emails:
            features_factory(message_id=email.message_id)

        result = feature_store.get_all_features(limit=2, offset=2)
        assert len(result) == 2


class TestFeatureStoreCount:
    """Test FeatureStore count operations."""

    def test_count_features_empty(self, feature_store):
        """Test counting features when empty."""
        count = feature_store.count_features()
        assert count == 0

    def test_count_features(self, feature_store, email_factory, features_factory):
        """Test counting features."""
        emails = [email_factory() for _ in range(5)]
        for email in emails:
            features_factory(message_id=email.message_id)

        count = feature_store.count_features()
        assert count == 5


class TestFeatureStoreDelete:
    """Test FeatureStore delete operations."""

    def test_delete_features_success(self, feature_store, email_factory, features_factory, temp_db_session):
        """Test deleting features."""
        email = email_factory()
        features_factory(message_id=email.message_id)

        result = feature_store.delete_features(email.message_id)
        temp_db_session.commit()

        assert result is True
        assert feature_store.get_features(email.message_id) is None

    def test_delete_features_not_found(self, feature_store):
        """Test deleting nonexistent features."""
        result = feature_store.delete_features("nonexistent")
        assert result is False


class TestFeatureStoreEmailsWithoutFeatures:
    """Test getting emails without computed features."""

    def test_get_emails_without_features(self, feature_store, email_factory, features_factory, temp_db_session):
        """Test getting emails that don't have features."""
        # Create 5 emails, only 3 have features
        emails = [email_factory() for _ in range(5)]
        features_factory(message_id=emails[0].message_id)
        features_factory(message_id=emails[2].message_id)
        features_factory(message_id=emails[4].message_id)
        temp_db_session.commit()

        result = feature_store.get_emails_without_features()

        assert len(result) == 2
        message_ids = [e.message_id for e in result]
        assert emails[1].message_id in message_ids
        assert emails[3].message_id in message_ids

    def test_get_emails_without_features_with_limit(self, feature_store, email_factory, temp_db_session):
        """Test getting emails without features with limit."""
        emails = [email_factory() for _ in range(5)]
        temp_db_session.commit()

        result = feature_store.get_emails_without_features(limit=2)

        assert len(result) == 2


class TestFeatureStoreStats:
    """Test FeatureStore statistics."""

    def test_get_feature_stats_empty(self, feature_store, temp_db_session):
        """Test getting stats with no data."""
        stats = feature_store.get_feature_stats()

        assert stats['total_emails'] == 0
        assert stats['features_count'] == 0
        assert stats['coverage_pct'] == 0

    def test_get_feature_stats(self, feature_store, email_factory, features_factory, temp_db_session):
        """Test getting feature statistics."""
        # Create 10 emails, 5 with features
        emails = [email_factory() for _ in range(10)]
        for i, email in enumerate(emails[:5]):
            features_factory(
                message_id=email.message_id,
                sender_open_rate=0.5 + i * 0.1,
                domain_open_rate=0.6,
                is_newsletter=(i < 2),  # 2 newsletters
            )
        temp_db_session.commit()

        stats = feature_store.get_feature_stats()

        assert stats['total_emails'] == 10
        assert stats['features_count'] == 5
        assert stats['coverage_pct'] == 50.0
        assert stats['newsletter_count'] == 2
        assert stats['newsletter_pct'] == 40.0  # 2 out of 5 features
        assert stats['avg_sender_open_rate'] > 0
        assert stats['avg_domain_open_rate'] == 0.6
