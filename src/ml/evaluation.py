"""Model evaluation and metrics."""

from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

from ..utils import get_logger


logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance and generate metrics.

    Provides comprehensive evaluation metrics, threshold analysis,
    and visualization data.

    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate(y_true, y_pred_proba)
        >>> print(f"AUC: {metrics['auc']:.3f}")
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

    def evaluate(
        self,
        y_true: List[int],
        y_pred_proba: List[float],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels (1 = read, 0 = not read)
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of evaluation metrics

        Example:
            >>> metrics = evaluator.evaluate(y_true, y_pred_proba)
            >>> print(f"Precision: {metrics['precision']:.3f}")
        """
        y_pred = (np.array(y_pred_proba) >= threshold).astype(int)

        # Basic metrics
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'threshold': threshold,
            'n_samples': len(y_true),
            'positive_rate': np.mean(y_true),
        }

        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
        })

        # False positive rate (for archive decision)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        logger.info(f"Evaluation metrics:")
        logger.info(f"  - AUC: {metrics['auc']:.3f}")
        logger.info(f"  - Precision: {metrics['precision']:.3f}")
        logger.info(f"  - Recall: {metrics['recall']:.3f}")
        logger.info(f"  - F1: {metrics['f1']:.3f}")

        return metrics

    def analyze_thresholds(
        self,
        y_true: List[int],
        y_pred_proba: List[float],
        thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze performance at different thresholds.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to test (default: 0.1 to 0.9)

        Returns:
            List of metrics dictionaries for each threshold

        Example:
            >>> results = evaluator.analyze_thresholds(y_true, y_pred_proba)
            >>> best = max(results, key=lambda x: x['f1'])
            >>> print(f"Best threshold: {best['threshold']:.2f}")
        """
        if thresholds is None:
            thresholds = [i / 10 for i in range(1, 10)]  # 0.1 to 0.9

        results = []

        for threshold in thresholds:
            metrics = self.evaluate(y_true, y_pred_proba, threshold)
            results.append(metrics)

        return results

    def find_optimal_threshold(
        self,
        y_true: List[int],
        y_pred_proba: List[float],
        optimize_for: str = 'f1',
        min_precision: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            optimize_for: Metric to optimize ('f1', 'precision', 'recall')
            min_precision: Minimum precision constraint (optional)

        Returns:
            Tuple of (optimal threshold, metrics)

        Example:
            >>> threshold, metrics = evaluator.find_optimal_threshold(
            ...     y_true, y_pred_proba,
            ...     optimize_for='f1',
            ...     min_precision=0.9
            ... )
        """
        # Test many thresholds
        thresholds = [i / 100 for i in range(1, 100)]
        results = self.analyze_thresholds(y_true, y_pred_proba, thresholds)

        # Filter by minimum precision if specified
        if min_precision:
            results = [r for r in results if r['precision'] >= min_precision]

        if not results:
            logger.warning(f"No thresholds meet min_precision={min_precision}")
            # Return highest threshold
            return 0.9, self.evaluate(y_true, y_pred_proba, 0.9)

        # Find best by optimize_for metric
        best = max(results, key=lambda x: x[optimize_for])

        logger.info(f"Optimal threshold: {best['threshold']:.3f}")
        logger.info(f"  - {optimize_for}: {best[optimize_for]:.3f}")

        return best['threshold'], best

    def evaluate_archive_precision(
        self,
        y_true: List[int],
        y_pred_proba: List[float],
        archive_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Evaluate precision for archive decisions.

        Critical metric: we want very high precision for archives to avoid
        archiving important emails (false positives).

        Args:
            y_true: True labels (1 = read, 0 = not read)
            y_pred_proba: Predicted probabilities
            archive_threshold: Threshold below which to archive

        Returns:
            Dictionary with archive-specific metrics

        Example:
            >>> metrics = evaluator.evaluate_archive_precision(
            ...     y_true, y_pred_proba,
            ...     archive_threshold=0.3
            ... )
            >>> print(f"Archive precision: {metrics['archive_precision']:.3f}")
        """
        # Emails below threshold would be archived
        archive_predictions = np.array(y_pred_proba) < archive_threshold

        # Among archived emails, how many were actually unread (correct)?
        if archive_predictions.sum() > 0:
            archive_precision = (
                (archive_predictions & (np.array(y_true) == 0)).sum() /
                archive_predictions.sum()
            )
        else:
            archive_precision = 0.0

        # False positives: archived but should have been read
        false_positives = (archive_predictions & (np.array(y_true) == 1)).sum()

        metrics = {
            'archive_threshold': archive_threshold,
            'archive_precision': archive_precision,
            'n_archived': int(archive_predictions.sum()),
            'n_false_positives': int(false_positives),
            'false_positive_rate': false_positives / archive_predictions.sum()
                                  if archive_predictions.sum() > 0 else 0,
        }

        logger.info(f"Archive metrics (threshold={archive_threshold}):")
        logger.info(f"  - Would archive: {metrics['n_archived']} emails")
        logger.info(f"  - Precision: {metrics['archive_precision']:.3f}")
        logger.info(f"  - False positives: {metrics['n_false_positives']}")

        return metrics

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            output_path: Path to save metrics

        Example:
            >>> evaluator.save_metrics(metrics, "models/metrics_v20240101.json")
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    def load_metrics(self, metrics_path: str) -> Dict[str, Any]:
        """
        Load metrics from JSON file.

        Args:
            metrics_path: Path to metrics file

        Returns:
            Metrics dictionary

        Example:
            >>> metrics = evaluator.load_metrics("models/metrics_v20240101.json")
        """
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        logger.info(f"Metrics loaded from {metrics_path}")
        return metrics

    def get_roc_curve_data(
        self,
        y_true: List[int],
        y_pred_proba: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Get ROC curve data for plotting.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Tuple of (fpr, tpr, thresholds)

        Example:
            >>> fpr, tpr, thresholds = evaluator.get_roc_curve_data(y_true, y_pred)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        return fpr.tolist(), tpr.tolist(), thresholds.tolist()

    def get_precision_recall_curve_data(
        self,
        y_true: List[int],
        y_pred_proba: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Get precision-recall curve data for plotting.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Tuple of (precision, recall, thresholds)

        Example:
            >>> precision, recall, thresholds = evaluator.get_precision_recall_curve_data(
            ...     y_true, y_pred
            ... )
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        return precision.tolist(), recall.tolist(), thresholds.tolist()
