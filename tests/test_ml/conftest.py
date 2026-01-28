"""ML-specific test fixtures."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_training_data():
    """Create sample training data for ML tests."""
    X = pd.DataFrame({
        'sender_email_count': [5, 10, 2, 20],
        'sender_open_rate': [0.8, 0.3, 0.5, 0.1],
        'is_newsletter': [0, 1, 0, 1],
        'subject_length': [20, 50, 15, 60],
        'body_length': [100, 500, 80, 1000]
    })
    y = pd.Series([1, 0, 1, 0])  # 1 = read, 0 = unread

    return X, y


@pytest.fixture
def sample_training_data_with_embeddings():
    """Create sample training data with embedding features."""
    # Basic features
    basic_features = {
        'sender_email_count': [5, 10, 2, 20],
        'sender_open_rate': [0.8, 0.3, 0.5, 0.1],
        'is_newsletter': [0, 1, 0, 1],
        'subject_length': [20, 50, 15, 60],
        'body_length': [100, 500, 80, 1000]
    }

    # Add embedding features (simulating 10-dim embeddings for simplicity)
    embedding_features = {}
    for i in range(10):
        embedding_features[f'emb_{i}'] = [0.1 * (i + 1)] * 4

    # Combine all features
    all_features = {**basic_features, **embedding_features}
    X = pd.DataFrame(all_features)
    y = pd.Series([1, 0, 1, 0])

    return X, y


@pytest.fixture
def trained_model(sample_training_data):
    """Create a trained LightGBM model for testing."""
    import lightgbm as lgb

    X, y = sample_training_data

    model = lgb.LGBMClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)

    return model


@pytest.fixture
def mock_email_scorer():
    """Create a mock email scorer."""
    scorer = Mock()
    scorer.score_email = Mock(return_value=0.5)
    scorer.make_decision = Mock(return_value={
        'action': 'keep',
        'confidence': 'low',
        'score': 0.5,
        'reasoning': 'Test reasoning',
        'labels': []
    })
    scorer.model = Mock()
    scorer.feature_names = ['sender_email_count', 'sender_open_rate', 'is_newsletter']

    return scorer


@pytest.fixture
def mock_email_scorer_high_confidence():
    """Create a mock email scorer with high confidence decisions."""
    scorer = Mock()
    scorer.score_email = Mock(return_value=0.9)
    scorer.make_decision = Mock(return_value={
        'action': 'keep',
        'confidence': 'high',
        'score': 0.9,
        'reasoning': 'High confidence to keep',
        'labels': ['Bot/Personal']
    })
    scorer.model = Mock()
    scorer.feature_names = ['sender_email_count', 'sender_open_rate', 'is_newsletter']

    return scorer


@pytest.fixture
def mock_email_scorer_archive():
    """Create a mock email scorer that recommends archiving."""
    scorer = Mock()
    scorer.score_email = Mock(return_value=0.1)
    scorer.make_decision = Mock(return_value={
        'action': 'archive',
        'confidence': 'high',
        'score': 0.1,
        'reasoning': 'Low importance, archive',
        'labels': ['Bot/Newsletter-Tech', 'Bot/AutoArchived']
    })
    scorer.model = Mock()
    scorer.feature_names = ['sender_email_count', 'sender_open_rate', 'is_newsletter']

    return scorer


@pytest.fixture
def mock_categorizer():
    """Create a mock email categorizer."""
    categorizer = Mock()
    categorizer.categorize = Mock(return_value=['Bot/Newsletter-Tech'])
    categorizer.get_all_categories = Mock(return_value=[
        'Bot/Newsletter-Tech',
        'Bot/Newsletter-Finance',
        'Bot/Newsletter-News',
        'Bot/Personal',
        'Bot/Work',
        'Bot/Promotional'
    ])

    return categorizer


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for evaluation tests."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # Actual labels
    y_pred_proba = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.7, 0.3, 0.15, 0.85, 0.25])  # Predicted probabilities

    return y_true, y_pred_proba


@pytest.fixture
def sample_predictions_perfect():
    """Create perfect prediction data (100% accuracy)."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.95, 0.05, 0.9, 0.85, 0.1])  # Perfect separation

    return y_true, y_pred_proba


@pytest.fixture
def sample_predictions_poor():
    """Create poor prediction data (random guessing)."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # No signal

    return y_true, y_pred_proba


@pytest.fixture
def mock_model_trainer(temp_db_session):
    """Create a mock model trainer."""
    from src.ml.training import ModelTrainer

    trainer = ModelTrainer(temp_db_session)
    trainer.model = Mock()
    trainer.feature_names = ['sender_email_count', 'sender_open_rate', 'is_newsletter']

    return trainer


@pytest.fixture
def mock_model_evaluator():
    """Create a mock model evaluator."""
    from unittest.mock import Mock

    evaluator = Mock()
    evaluator.evaluate = Mock(return_value={
        'auc': 0.85,
        'accuracy': 0.80,
        'precision': 0.82,
        'recall': 0.78,
        'f1': 0.80
    })
    evaluator.evaluate_archive_precision = Mock(return_value={
        'archive_threshold': 0.3,
        'archive_precision': 0.95,
        'n_archived': 20,
        'n_false_positives': 1,
        'false_positive_rate': 0.05
    })

    return evaluator
