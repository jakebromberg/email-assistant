"""Email scoring and inference."""

from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb

from ..database.schema import Email, EmailFeatures
from ..utils import get_logger


logger = get_logger(__name__)


class EmailScorer:
    """
    Score emails using trained model to predict importance.

    Loads a trained LightGBM model and predicts likelihood (0.0-1.0)
    that user will read each email.

    Attributes:
        model: Loaded LightGBM model
        feature_names: Expected feature names

    Example:
        >>> scorer = EmailScorer("models/model_v20240101.txt")
        >>> score = scorer.score_email(email, features)
        >>> print(f"Score: {score:.3f}")
    """

    # Decision thresholds
    HIGH_CONFIDENCE_KEEP = 0.7  # Definitely keep in inbox
    HIGH_CONFIDENCE_ARCHIVE = 0.3  # Definitely archive
    # Between 0.3-0.7 is low confidence

    def __init__(self, model_path: str):
        """
        Initialize email scorer with trained model.

        Args:
            model_path: Path to trained model file

        Example:
            >>> scorer = EmailScorer("models/model_v20240101.txt")
        """
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names: List[str] = []
        self._load_feature_config(model_path)

        logger.info(f"Email scorer initialized with model: {model_path}")

    def _load_feature_config(self, model_path: str) -> None:
        """Load feature configuration."""
        import json

        model_path_obj = Path(model_path)
        version = model_path_obj.stem.replace('model_v', '')
        config_path = model_path_obj.parent / f"feature_config_v{version}.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.feature_names = config['feature_names']
        else:
            logger.warning(f"Feature config not found: {config_path}")

    def score_email(
        self,
        email: Email,
        features: EmailFeatures
    ) -> float:
        """
        Score a single email.

        Args:
            email: Email database model
            features: EmailFeatures database model

        Returns:
            Score between 0.0 and 1.0 (likelihood of reading)

        Example:
            >>> score = scorer.score_email(email, features)
            >>> if score >= 0.7:
            ...     print("High confidence keep")
        """
        # Prepare features
        feature_dict = self._extract_features(email, features)

        # Create DataFrame with correct feature order
        X = pd.DataFrame([feature_dict], columns=self.feature_names)

        # Predict
        score = self.model.predict(X)[0]

        return float(score)

    def score_emails_batch(
        self,
        emails_features: List[tuple[Email, EmailFeatures]]
    ) -> List[float]:
        """
        Score multiple emails efficiently.

        Args:
            emails_features: List of (Email, EmailFeatures) tuples

        Returns:
            List of scores

        Example:
            >>> scores = scorer.score_emails_batch(emails_features)
        """
        if not emails_features:
            return []

        # Prepare all features
        feature_dicts = [
            self._extract_features(email, features)
            for email, features in emails_features
        ]

        # Create DataFrame
        X = pd.DataFrame(feature_dicts, columns=self.feature_names)

        # Batch predict
        scores = self.model.predict(X)

        return scores.tolist()

    def _extract_features(
        self,
        email: Email,
        features: EmailFeatures
    ) -> Dict[str, Any]:
        """
        Extract features for prediction.

        Args:
            email: Email database model
            features: EmailFeatures database model

        Returns:
            Dictionary of features
        """
        feature_dict = {}

        # Get all feature values
        for feat_name in self.feature_names:
            if feat_name.startswith('emb_'):
                # Embedding feature
                idx = int(feat_name.split('_')[1])

                # Average of subject and body embeddings
                subject_emb = features.subject_embedding
                body_emb = features.body_embedding

                if subject_emb and body_emb:
                    value = (subject_emb[idx] + body_emb[idx]) / 2
                elif subject_emb:
                    value = subject_emb[idx]
                elif body_emb:
                    value = body_emb[idx]
                else:
                    value = 0.0

                feature_dict[feat_name] = value

            else:
                # Metadata or historical feature
                value = getattr(features, feat_name, None)

                if value is None:
                    if feat_name == 'sender_days_since_last':
                        value = 999  # Large value for no previous email
                    else:
                        value = 0

                feature_dict[feat_name] = value

        return feature_dict

    def make_decision(
        self,
        score: float,
        high_keep_threshold: Optional[float] = None,
        high_archive_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make triage decision based on score.

        Args:
            score: Prediction score (0.0-1.0)
            high_keep_threshold: Threshold for high confidence keep (default: 0.7)
            high_archive_threshold: Threshold for high confidence archive (default: 0.3)

        Returns:
            Dictionary with decision, confidence, and reasoning

        Example:
            >>> decision = scorer.make_decision(score)
            >>> print(f"Action: {decision['action']}")
            >>> print(f"Confidence: {decision['confidence']}")
        """
        if high_keep_threshold is None:
            high_keep_threshold = self.HIGH_CONFIDENCE_KEEP

        if high_archive_threshold is None:
            high_archive_threshold = self.HIGH_CONFIDENCE_ARCHIVE

        if score >= high_keep_threshold:
            return {
                'action': 'keep',
                'confidence': 'high',
                'score': score,
                'reasoning': f'High likelihood of reading (score: {score:.3f})'
            }
        elif score <= high_archive_threshold:
            return {
                'action': 'archive',
                'confidence': 'high',
                'score': score,
                'reasoning': f'Low likelihood of reading (score: {score:.3f})'
            }
        else:
            return {
                'action': 'keep',  # Keep by default for low confidence
                'confidence': 'low',
                'score': score,
                'reasoning': f'Uncertain prediction (score: {score:.3f}), keeping to be safe'
            }

    def get_threshold_metrics(
        self,
        scores: List[float],
        labels: List[int],
        threshold: float
    ) -> Dict[str, float]:
        """
        Calculate metrics at a specific threshold.

        Args:
            scores: Prediction scores
            labels: True labels (1 = read, 0 = not read)
            threshold: Decision threshold

        Returns:
            Dictionary with precision, recall, etc.

        Example:
            >>> metrics = scorer.get_threshold_metrics(scores, labels, 0.5)
            >>> print(f"Precision: {metrics['precision']:.3f}")
        """
        predictions = [1 if s >= threshold else 0 for s in scores]

        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }
