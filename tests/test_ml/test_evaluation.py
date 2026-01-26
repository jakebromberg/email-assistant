"""Tests for ML model evaluation."""

import pytest
import numpy as np
from unittest.mock import patch
import json
import tempfile
from pathlib import Path

from src.ml.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return ModelEvaluator()

    # Basic Functionality Tests for evaluate_archive_precision

    def test_evaluate_archive_precision_basic(self, evaluator):
        """Test basic calculation."""
        # 10 emails: 5 read (1), 5 unread (0)
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # Scores: 0.1-0.5 for read, 0.6-1.0 for unread
        y_pred_proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.3)

        # Should archive emails with scores < 0.3: indices 0, 1 (both read=1, so false positives)
        assert metrics['archive_threshold'] == 0.3
        assert metrics['n_archived'] == 2
        assert metrics['n_false_positives'] == 2
        assert metrics['archive_precision'] == 0.0  # All archived were read (wrong)
        assert metrics['false_positive_rate'] == 1.0  # 2/2

    def test_evaluate_archive_precision_threshold_0_3(self, evaluator):
        """Standard threshold."""
        # Mix of correct and incorrect archives
        y_true = [0, 0, 0, 1, 1]  # 3 unread, 2 read
        y_pred_proba = [0.1, 0.2, 0.25, 0.28, 0.35]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.3)

        # Should archive indices 0, 1, 2, 3 (scores < 0.3)
        assert metrics['n_archived'] == 4
        # Correct archives: 0, 1, 2 (unread)
        # False positives: 3 (read)
        assert metrics['n_false_positives'] == 1
        assert metrics['archive_precision'] == pytest.approx(0.75)  # 3/4

    def test_evaluate_archive_precision_perfect(self, evaluator):
        """100% precision case."""
        y_true = [0, 0, 0, 1, 1]  # 3 unread, 2 read
        y_pred_proba = [0.1, 0.15, 0.2, 0.8, 0.9]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.3)

        # Only archive 0, 1, 2 (all unread)
        assert metrics['n_archived'] == 3
        assert metrics['n_false_positives'] == 0
        assert metrics['archive_precision'] == 1.0
        assert metrics['false_positive_rate'] == 0.0

    def test_evaluate_archive_precision_with_false_positives(self, evaluator):
        """FP counting."""
        y_true = [1, 0, 1, 0]  # Alternating read/unread
        y_pred_proba = [0.1, 0.2, 0.4, 0.5]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.3)

        # Archive 0, 1 (scores < 0.3)
        # 0 is read (FP), 1 is unread (correct)
        assert metrics['n_archived'] == 2
        assert metrics['n_false_positives'] == 1
        assert metrics['archive_precision'] == pytest.approx(0.5)  # 1/2
        assert metrics['false_positive_rate'] == pytest.approx(0.5)  # 1/2

    # Edge Cases Tests

    def test_evaluate_archive_precision_no_archives(self, evaluator):
        """No emails below threshold."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.5, 0.6, 0.7, 0.8]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.3)

        assert metrics['n_archived'] == 0
        assert metrics['n_false_positives'] == 0
        assert metrics['archive_precision'] == 0.0
        assert metrics['false_positive_rate'] == 0.0

    def test_evaluate_archive_precision_all_archived(self, evaluator):
        """All emails below threshold."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.1, 0.15, 0.2, 0.25]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.5)

        assert metrics['n_archived'] == 4
        # 2 unread (correct), 2 read (FP)
        assert metrics['n_false_positives'] == 2
        assert metrics['archive_precision'] == pytest.approx(0.5)  # 2/4

    def test_evaluate_archive_precision_empty_arrays(self, evaluator):
        """Empty input arrays."""
        y_true = []
        y_pred_proba = []

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.3)

        assert metrics['n_archived'] == 0
        assert metrics['n_false_positives'] == 0
        assert metrics['archive_precision'] == 0.0
        assert metrics['false_positive_rate'] == 0.0

    def test_evaluate_archive_precision_threshold_boundaries(self, evaluator):
        """Thresholds at 0.0, 0.5, 1.0."""
        y_true = [0, 1, 0, 1]
        y_pred_proba = [0.2, 0.3, 0.7, 0.8]

        # Threshold 0.0: nothing archived
        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.0)
        assert metrics['n_archived'] == 0

        # Threshold 0.5: archive first 2
        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.5)
        assert metrics['n_archived'] == 2

        # Threshold 1.0: archive all 4
        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=1.0)
        assert metrics['n_archived'] == 4

    def test_evaluate_archive_precision_division_by_zero(self, evaluator):
        """Handle zero archives gracefully."""
        y_true = [1, 1, 1]
        y_pred_proba = [0.9, 0.95, 0.99]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.5)

        # No archives, all divisions by zero should be handled
        assert metrics['archive_precision'] == 0.0
        assert metrics['false_positive_rate'] == 0.0

    # Data Validation Tests

    def test_evaluate_archive_precision_array_length_mismatch(self, evaluator):
        """Mismatched lengths should raise error."""
        y_true = [0, 1]
        y_pred_proba = [0.1, 0.2, 0.3]  # Different length

        with pytest.raises((ValueError, IndexError)):
            evaluator.evaluate_archive_precision(y_true, y_pred_proba)

    def test_evaluate_archive_precision_nan_values(self, evaluator):
        """NaN handling."""
        y_true = [0, 1, 0]
        y_pred_proba = [0.1, np.nan, 0.3]

        # Should handle NaN (may raise warning or produce NaN results)
        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba, archive_threshold=0.5)

        # Check that result is returned (may contain NaN)
        assert 'archive_precision' in metrics

    def test_evaluate_archive_precision_correct_metrics_keys(self, evaluator):
        """Verify return dict has correct keys."""
        y_true = [0, 1]
        y_pred_proba = [0.1, 0.9]

        metrics = evaluator.evaluate_archive_precision(y_true, y_pred_proba)

        required_keys = [
            'archive_threshold',
            'archive_precision',
            'n_archived',
            'n_false_positives',
            'false_positive_rate'
        ]

        for key in required_keys:
            assert key in metrics

    # Related Method Tests - evaluate()

    def test_evaluate_basic(self, evaluator):
        """Test main evaluate() method."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.2, 0.3, 0.7, 0.8]

        metrics = evaluator.evaluate(y_true, y_pred_proba, threshold=0.5)

        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert metrics['n_samples'] == 4
        assert metrics['threshold'] == 0.5

    def test_evaluate_confusion_matrix(self, evaluator):
        """Test CM calculation."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.2, 0.7, 0.3, 0.8]
        # Predictions at 0.5: 0, 1, 0, 1

        metrics = evaluator.evaluate(y_true, y_pred_proba, threshold=0.5)

        # TN: [0], FP: [1], FN: [2], TP: [3]
        assert metrics['true_negatives'] == 1
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['true_positives'] == 1

    # Related Method Tests - analyze_thresholds()

    def test_analyze_thresholds(self, evaluator):
        """Test threshold analysis."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.2, 0.3, 0.7, 0.8]

        results = evaluator.analyze_thresholds(
            y_true,
            y_pred_proba,
            thresholds=[0.3, 0.5, 0.7]
        )

        assert len(results) == 3
        assert results[0]['threshold'] == 0.3
        assert results[1]['threshold'] == 0.5
        assert results[2]['threshold'] == 0.7

    # Related Method Tests - find_optimal_threshold()

    def test_find_optimal_threshold(self, evaluator):
        """Test optimal threshold finding."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred_proba = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]

        threshold, metrics = evaluator.find_optimal_threshold(
            y_true,
            y_pred_proba,
            optimize_for='f1'
        )

        assert 0.0 < threshold < 1.0
        assert 'f1' in metrics
        assert metrics['threshold'] == threshold

    def test_find_optimal_threshold_with_min_precision(self, evaluator):
        """Test with constraint."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred_proba = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]

        threshold, metrics = evaluator.find_optimal_threshold(
            y_true,
            y_pred_proba,
            optimize_for='recall',
            min_precision=0.8
        )

        # Should find threshold that meets precision constraint
        assert metrics['precision'] >= 0.8 or threshold == 0.9

    # Additional tests for coverage

    def test_save_and_load_metrics(self, evaluator):
        """Test metrics save/load."""
        metrics = {
            'auc': 0.85,
            'precision': 0.9,
            'recall': 0.8
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'metrics.json'

            # Save
            evaluator.save_metrics(metrics, str(path))
            assert path.exists()

            # Load
            loaded = evaluator.load_metrics(str(path))
            assert loaded == metrics

    def test_get_roc_curve_data(self, evaluator):
        """Test ROC curve data extraction."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.2, 0.3, 0.7, 0.8]

        fpr, tpr, thresholds = evaluator.get_roc_curve_data(y_true, y_pred_proba)

        assert len(fpr) == len(tpr)
        assert len(tpr) == len(thresholds)
        assert all(isinstance(x, (int, float)) for x in fpr)

    def test_get_precision_recall_curve_data(self, evaluator):
        """Test precision-recall curve data."""
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.2, 0.3, 0.7, 0.8]

        precision, recall, thresholds = evaluator.get_precision_recall_curve_data(
            y_true,
            y_pred_proba
        )

        assert len(precision) == len(recall)
        assert len(recall) == len(thresholds) + 1  # PR curve returns n+1 points
        assert all(isinstance(x, (int, float)) for x in precision)
