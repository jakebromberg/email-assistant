"""Tests for semantic classifier."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ml.semantic_classifier import SemanticClassifier


class TestSemanticClassifierInit:
    """Test SemanticClassifier initialization."""

    def test_init_creates_empty_classifier(self):
        """Test that init creates an empty classifier."""
        classifier = SemanticClassifier()

        assert classifier.classifier is None
        assert classifier.label_encoder is None
        assert classifier.model_path is None
        assert classifier.metadata == {}

    def test_get_classes_returns_empty_when_not_loaded(self):
        """Test get_classes returns empty list when no model loaded."""
        classifier = SemanticClassifier()

        assert classifier.get_classes() == []


class TestSemanticClassifierPredict:
    """Test SemanticClassifier prediction methods."""

    def test_predict_raises_when_not_loaded(self):
        """Test predict raises error when model not loaded."""
        classifier = SemanticClassifier()
        embedding = np.random.rand(384)

        with pytest.raises(ValueError, match="Model not loaded"):
            classifier.predict(embedding)

    def test_predict_top_k_raises_when_not_loaded(self):
        """Test predict_top_k raises error when model not loaded."""
        classifier = SemanticClassifier()
        embedding = np.random.rand(384)

        with pytest.raises(ValueError, match="Model not loaded"):
            classifier.predict_top_k(embedding, k=3)

    def test_predict_with_mock_classifier(self):
        """Test predict with mocked classifier."""
        classifier = SemanticClassifier()

        # Mock the classifier and label encoder
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])

        mock_encoder = MagicMock()
        mock_encoder.inverse_transform.return_value = ['Newsletter-Tech']

        classifier.classifier = mock_clf
        classifier.label_encoder = mock_encoder

        embedding = np.random.rand(384)
        label, confidence = classifier.predict(embedding)

        assert label == 'Newsletter-Tech'
        assert confidence == 0.7
        mock_clf.predict_proba.assert_called_once()

    def test_predict_top_k_with_mock_classifier(self):
        """Test predict_top_k with mocked classifier."""
        classifier = SemanticClassifier()

        # Mock the classifier and label encoder
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.6, 0.3]])

        mock_encoder = MagicMock()
        mock_encoder.inverse_transform.side_effect = lambda x: [
            ['Newsletter-Tech', 'Job-Alert', 'Security-Alert'][x[0]]
        ]

        classifier.classifier = mock_clf
        classifier.label_encoder = mock_encoder

        embedding = np.random.rand(384)
        results = classifier.predict_top_k(embedding, k=2)

        assert len(results) == 2
        # Top 2 should be index 1 (0.6) and index 2 (0.3)
        assert results[0][1] == 0.6  # Highest confidence
        assert results[1][1] == 0.3  # Second highest


class TestSemanticClassifierSaveLoad:
    """Test SemanticClassifier save/load functionality."""

    def test_save_raises_when_no_model(self):
        """Test save raises error when no model trained."""
        classifier = SemanticClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.pkl"
            with pytest.raises(ValueError, match="No model to save"):
                classifier.save(path)

    def test_save_creates_files(self):
        """Test save creates model and metadata files."""
        classifier = SemanticClassifier()

        # Set up mock classifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        classifier.classifier = LogisticRegression()
        classifier.classifier.classes_ = np.array([0, 1, 2])
        classifier.classifier.coef_ = np.random.rand(3, 10)
        classifier.classifier.intercept_ = np.random.rand(3)

        classifier.label_encoder = LabelEncoder()
        classifier.label_encoder.classes_ = np.array(['A', 'B', 'C'])

        classifier.metadata = {
            'trained_at': '2024-01-01',
            'n_classes': 3,
            'accuracy': 0.85,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.pkl"
            classifier.save(path)

            # Check files exist
            assert Path(path).exists()
            assert Path(path.replace('.pkl', '_metadata.json')).exists()

            # Check metadata content
            with open(path.replace('.pkl', '_metadata.json')) as f:
                saved_metadata = json.load(f)
            assert saved_metadata['accuracy'] == 0.85

    def test_load_restores_model(self):
        """Test load restores model correctly."""
        # First save a model
        classifier = SemanticClassifier()

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        original_encoder = LabelEncoder()
        original_encoder.classes_ = np.array(['Label-A', 'Label-B', 'Label-C'])

        classifier.classifier = LogisticRegression()
        classifier.classifier.classes_ = np.array([0, 1, 2])
        classifier.classifier.coef_ = np.random.rand(3, 10)
        classifier.classifier.intercept_ = np.random.rand(3)
        classifier.label_encoder = original_encoder
        classifier.metadata = {'n_classes': 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.pkl"
            classifier.save(path)

            # Load into new classifier
            new_classifier = SemanticClassifier()
            new_classifier.load(path)

            assert new_classifier.classifier is not None
            assert new_classifier.label_encoder is not None
            assert new_classifier.metadata == {'n_classes': 3}
            assert list(new_classifier.get_classes()) == ['Label-A', 'Label-B', 'Label-C']

    def test_get_classes_after_load(self):
        """Test get_classes returns correct labels after loading."""
        classifier = SemanticClassifier()

        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.classes_ = np.array(['Product-Update', 'Security-Alert', 'Newsletter-Tech'])
        classifier.label_encoder = encoder

        classes = classifier.get_classes()

        assert classes == ['Product-Update', 'Security-Alert', 'Newsletter-Tech']


class TestSemanticClassifierTrain:
    """Test SemanticClassifier training."""

    def test_train_raises_with_no_data(self, temp_db_session):
        """Test train raises error when no training data exists."""
        classifier = SemanticClassifier()

        with pytest.raises(ValueError, match="No training data found"):
            classifier.train(temp_db_session)

    def test_train_raises_with_insufficient_data(self, temp_db_session, email_factory):
        """Test train raises error with insufficient samples."""
        from src.database.schema import EmailFeatures, SemanticLabel

        # Add only a few samples (less than min required)
        for i in range(5):
            msg_id = f"msg{i}@test.com"
            # Create email first (required by foreign key)
            email_factory(message_id=msg_id)

            label = SemanticLabel(
                message_id=msg_id,
                label="TestLabel",
                source="claude"
            )
            features = EmailFeatures(
                message_id=msg_id,
                subject_embedding=np.random.rand(384).tolist()
            )
            temp_db_session.add(label)
            temp_db_session.add(features)
        temp_db_session.flush()

        classifier = SemanticClassifier()

        # Should fail because we need at least 10 samples total
        with pytest.raises(ValueError, match="Not enough training data"):
            classifier.train(temp_db_session, min_samples_per_class=3)

    def test_train_with_sufficient_data(self, temp_db_session, email_factory):
        """Test train succeeds with sufficient data."""
        from src.database.schema import EmailFeatures, SemanticLabel

        # Add enough samples for 3 classes (15 samples each = 45 total)
        labels = ['Label-A', 'Label-B', 'Label-C']
        for label_idx, label_name in enumerate(labels):
            for i in range(15):
                msg_id = f"msg{label_idx}_{i}@test.com"
                # Create email first (required by foreign key)
                email_factory(message_id=msg_id)

                # Create embedding that's somewhat distinct per class
                base_embedding = np.zeros(384)
                base_embedding[label_idx * 100:(label_idx + 1) * 100] = 1.0
                embedding = base_embedding + np.random.rand(384) * 0.1

                label = SemanticLabel(
                    message_id=msg_id,
                    label=label_name,
                    source="claude"
                )
                features = EmailFeatures(
                    message_id=msg_id,
                    subject_embedding=embedding.tolist()
                )
                temp_db_session.add(label)
                temp_db_session.add(features)
        temp_db_session.flush()

        classifier = SemanticClassifier()
        metrics = classifier.train(temp_db_session, min_samples_per_class=3)

        assert 'accuracy' in metrics
        assert 'n_samples' in metrics
        assert 'n_classes' in metrics
        assert metrics['n_classes'] == 3
        assert classifier.classifier is not None
        assert classifier.label_encoder is not None

    def test_train_filters_rare_classes(self, temp_db_session, email_factory):
        """Test that classes with too few samples are filtered out."""
        from src.database.schema import EmailFeatures, SemanticLabel

        # Add many samples for one class, few for another
        for i in range(20):
            msg_id = f"msgA{i}@test.com"
            email_factory(message_id=msg_id)
            label = SemanticLabel(message_id=msg_id, label='Common-Label', source="claude")
            features = EmailFeatures(message_id=msg_id, subject_embedding=np.random.rand(384).tolist())
            temp_db_session.add(label)
            temp_db_session.add(features)

        for i in range(20):
            msg_id = f"msgB{i}@test.com"
            email_factory(message_id=msg_id)
            label = SemanticLabel(message_id=msg_id, label='Also-Common', source="claude")
            features = EmailFeatures(message_id=msg_id, subject_embedding=np.random.rand(384).tolist())
            temp_db_session.add(label)
            temp_db_session.add(features)

        # Add only 2 samples for rare class (below threshold of 3)
        for i in range(2):
            msg_id = f"msgRare{i}@test.com"
            email_factory(message_id=msg_id)
            label = SemanticLabel(message_id=msg_id, label='Rare-Label', source="claude")
            features = EmailFeatures(message_id=msg_id, subject_embedding=np.random.rand(384).tolist())
            temp_db_session.add(label)
            temp_db_session.add(features)

        temp_db_session.flush()

        classifier = SemanticClassifier()
        metrics = classifier.train(temp_db_session, min_samples_per_class=3)

        # Should only have 2 classes (the common ones)
        assert metrics['n_classes'] == 2
        assert 'Rare-Label' not in classifier.get_classes()
