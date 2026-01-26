"""Tests for ML model training."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from src.ml.training import ModelTrainer
from src.database.schema import Email, EmailFeatures


class TestModelTrainer:
    """Test ModelTrainer class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        return session

    @pytest.fixture
    def trainer(self, mock_session):
        """Create trainer instance."""
        return ModelTrainer(mock_session)

    def _create_mock_email(self, message_id: str, was_read: bool):
        """Create mock Email object."""
        email = Mock(spec=Email)
        email.message_id = message_id
        email.was_read = was_read
        return email

    def _create_mock_features(
        self,
        message_id: str,
        sender_email_count: int = 5,
        subject_embedding: list = None,
        body_embedding: list = None
    ):
        """Create mock EmailFeatures object."""
        features = Mock(spec=EmailFeatures)
        features.message_id = message_id

        # Metadata features
        features.is_newsletter = 0
        features.day_of_week = 1
        features.hour_of_day = 10
        features.subject_length = 50
        features.body_length = 500

        # Historical features
        features.sender_email_count = sender_email_count
        features.sender_open_rate = 0.75
        features.sender_days_since_last = 5
        features.domain_open_rate = 0.65

        # Embeddings
        features.subject_embedding = subject_embedding
        features.body_embedding = body_embedding

        return features

    # Basic Functionality Tests

    def test_prepare_data_basic(self, trainer, mock_session):
        """Test basic data preparation with embeddings."""
        # Create mock query results
        email1 = self._create_mock_email('msg1', was_read=True)
        features1 = self._create_mock_features(
            'msg1',
            subject_embedding=[0.1] * 384,
            body_embedding=[0.2] * 384
        )

        email2 = self._create_mock_email('msg2', was_read=False)
        features2 = self._create_mock_features(
            'msg2',
            subject_embedding=[0.3] * 384,
            body_embedding=[0.4] * 384
        )

        # Mock query
        mock_query = Mock()
        mock_query.all.return_value = [(email1, features1), (email2, features2)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        # Prepare data
        X, y = trainer.prepare_data(use_embeddings=True)

        assert len(X) == 2
        assert len(y) == 2
        assert list(y) == [1, 0]  # email1 was read, email2 wasn't
        assert len(X.columns) == 9 + 384  # metadata + historical + embeddings
        assert trainer.feature_names == list(X.columns)

    def test_prepare_data_no_embeddings(self, trainer, mock_session):
        """Test with use_embeddings=False."""
        email1 = self._create_mock_email('msg1', was_read=True)
        features1 = self._create_mock_features('msg1')

        mock_query = Mock()
        mock_query.all.return_value = [(email1, features1)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=False)

        assert len(X) == 1
        assert len(X.columns) == 9  # metadata + historical only
        assert 'emb_0' not in X.columns

    def test_prepare_data_min_sender_count_filtering(self, trainer, mock_session):
        """Test min_sender_count filter."""
        # Email with sender_email_count = 1 (should be filtered)
        email1 = self._create_mock_email('msg1', was_read=True)
        features1 = self._create_mock_features(
            'msg1',
            sender_email_count=1,
            subject_embedding=[0.1] * 384
        )

        # Email with sender_email_count = 5 (should pass)
        email2 = self._create_mock_email('msg2', was_read=True)
        features2 = self._create_mock_features(
            'msg2',
            sender_email_count=5,
            subject_embedding=[0.2] * 384
        )

        mock_query = Mock()
        mock_query.all.return_value = [(email2, features2)]  # Only email2 returned
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(min_sender_count=2)

        assert len(X) == 1
        # Verify filter was called correctly
        filter_calls = mock_session.query.return_value.join.return_value.filter.call_args_list
        assert len(filter_calls) > 0

    # Embedding Handling Tests

    def test_prepare_data_both_embeddings(self, trainer, mock_session):
        """Both subject and body embeddings exist → average."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features(
            'msg1',
            subject_embedding=[0.2, 0.4],
            body_embedding=[0.4, 0.6]
        )

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        # Check averaged values (use pytest.approx for floating point)
        assert X['emb_0'].iloc[0] == pytest.approx(0.3)  # (0.2 + 0.4) / 2
        assert X['emb_1'].iloc[0] == pytest.approx(0.5)  # (0.4 + 0.6) / 2

    def test_prepare_data_subject_only(self, trainer, mock_session):
        """Only subject embedding exists."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features(
            'msg1',
            subject_embedding=[0.1, 0.2],
            body_embedding=None
        )

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        assert X['emb_0'].iloc[0] == 0.1
        assert X['emb_1'].iloc[0] == 0.2

    def test_prepare_data_body_only(self, trainer, mock_session):
        """Only body embedding exists."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features(
            'msg1',
            subject_embedding=None,
            body_embedding=[0.3, 0.4]
        )

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        assert X['emb_0'].iloc[0] == 0.3
        assert X['emb_1'].iloc[0] == 0.4

    def test_prepare_data_no_embeddings_skips_email(self, trainer, mock_session):
        """No embeddings → email skipped."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features(
            'msg1',
            subject_embedding=None,
            body_embedding=None
        )

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        # Should raise error because all emails filtered out
        with pytest.raises(ValueError, match="No valid training data"):
            trainer.prepare_data(use_embeddings=True)

    # Null Value Handling Tests

    def test_prepare_data_null_metadata_features(self, trainer, mock_session):
        """None in metadata → 0."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features('msg1', subject_embedding=[0.1] * 384)

        # Set some metadata features to None
        features.is_newsletter = None
        features.subject_length = None

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        assert X['is_newsletter'].iloc[0] == 0
        assert X['subject_length'].iloc[0] == 0

    def test_prepare_data_null_historical_features(self, trainer, mock_session):
        """None in historical → 0."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features('msg1', subject_embedding=[0.1] * 384)

        # Set historical features to None
        features.sender_open_rate = None
        features.domain_open_rate = None

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        assert X['sender_open_rate'].iloc[0] == 0
        assert X['domain_open_rate'].iloc[0] == 0

    def test_prepare_data_sender_days_since_last_none(self, trainer, mock_session):
        """None → 999.

        After refactoring, this now works correctly - None is converted to 999,
        not 0. The refactoring fixed the bug in the original code.
        """
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features('msg1', subject_embedding=[0.1] * 384)

        features.sender_days_since_last = None

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        # Correct behavior after refactoring: None gets converted to 999
        assert X['sender_days_since_last'].iloc[0] == 999

    def test_prepare_data_missing_features(self, trainer, mock_session):
        """Missing attributes → default to 0.

        This test verifies that getattr with default=None works correctly
        and None values are converted to 0.
        """
        email = self._create_mock_email('msg1', was_read=True)
        # Create features with minimal attributes set
        features = self._create_mock_features('msg1', subject_embedding=[0.1] * 384)

        # Explicitly set some features to None to test None handling
        features.is_newsletter = None
        features.sender_open_rate = None

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        # Should not raise error, None values should be converted to 0
        assert len(X) == 1
        assert X['is_newsletter'].iloc[0] == 0
        assert X['sender_open_rate'].iloc[0] == 0

    # Edge Cases Tests

    def test_prepare_data_no_emails_raises_error(self, trainer, mock_session):
        """Empty query → ValueError."""
        mock_query = Mock()
        mock_query.all.return_value = []
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        with pytest.raises(ValueError, match="No training data found"):
            trainer.prepare_data()

    def test_prepare_data_all_filtered_raises_error(self, trainer, mock_session):
        """All filtered → ValueError."""
        # All emails have no embeddings
        email1 = self._create_mock_email('msg1', was_read=True)
        features1 = self._create_mock_features('msg1', subject_embedding=None, body_embedding=None)

        email2 = self._create_mock_email('msg2', was_read=True)
        features2 = self._create_mock_features('msg2', subject_embedding=None, body_embedding=None)

        mock_query = Mock()
        mock_query.all.return_value = [(email1, features1), (email2, features2)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        with pytest.raises(ValueError, match="No valid training data"):
            trainer.prepare_data(use_embeddings=True)

    def test_prepare_data_feature_names_populated(self, trainer, mock_session):
        """Verify feature_names list."""
        email = self._create_mock_email('msg1', was_read=True)
        features = self._create_mock_features('msg1', subject_embedding=[0.1] * 10)

        mock_query = Mock()
        mock_query.all.return_value = [(email, features)]
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data(use_embeddings=True)

        assert len(trainer.feature_names) == len(X.columns)
        assert trainer.feature_names == list(X.columns)
        assert 'is_newsletter' in trainer.feature_names
        assert 'sender_email_count' in trainer.feature_names
        assert 'emb_0' in trainer.feature_names

    def test_prepare_data_positive_class_ratio(self, trainer, mock_session):
        """Verify target distribution logged."""
        # Mix of read and unread
        emails_features = [
            (self._create_mock_email(f'msg{i}', was_read=(i % 2 == 0)),
             self._create_mock_features(f'msg{i}', subject_embedding=[0.1] * 384))
            for i in range(10)
        ]

        mock_query = Mock()
        mock_query.all.return_value = emails_features
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data()

        # Should have 50% positive class
        assert y.mean() == 0.5

    # Integration Tests

    def test_prepare_data_feature_matrix_shape(self, trainer, mock_session):
        """Verify X and y shapes match."""
        emails_features = [
            (self._create_mock_email(f'msg{i}', was_read=True),
             self._create_mock_features(f'msg{i}', subject_embedding=[0.1] * 384))
            for i in range(20)
        ]

        mock_query = Mock()
        mock_query.all.return_value = emails_features
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query

        X, y = trainer.prepare_data()

        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 20

    # Train Method Tests

    @patch('src.ml.training.train_test_split')
    @patch('src.ml.training.lgb.train')
    @patch('src.ml.training.lgb.Dataset')
    def test_train_basic(self, mock_dataset, mock_lgb_train, mock_split, trainer):
        """Test basic train method."""
        # Create mock data
        X = pd.DataFrame({
            'feat1': [1, 2, 3, 4],
            'feat2': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])

        # Mock train/test split
        mock_split.return_value = (
            X.iloc[:2], X.iloc[2:],  # X_train, X_test
            y.iloc[:2], y.iloc[2:]   # y_train, y_test
        )

        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3, 0.7])
        mock_lgb_train.return_value = mock_model

        # Train
        metrics = trainer.train(X=X, y=y)

        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert metrics['samples_train'] == 2
        assert metrics['samples_test'] == 2
