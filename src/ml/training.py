"""Model training for email importance prediction."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

from sqlalchemy.orm import Session

from ..database.schema import Email, EmailFeatures
from ..utils import get_logger


logger = get_logger(__name__)


class ModelTrainer:
    """
    Train LightGBM model to predict email importance.

    Predicts likelihood (0.0-1.0) that user will read an email based on
    historical behavior and features.

    Attributes:
        session: Database session
        model: Trained LightGBM model
        feature_names: List of feature names used

    Example:
        >>> with db.get_session() as session:
        ...     trainer = ModelTrainer(session)
        ...     metrics = trainer.train()
        ...     print(f"AUC: {metrics['auc']:.3f}")
    """

    # Features to use for training
    METADATA_FEATURES = [
        'is_newsletter',
        'day_of_week',
        'hour_of_day',
        'subject_length',
        'body_length',
    ]

    HISTORICAL_FEATURES = [
        'sender_email_count',
        'sender_open_rate',
        'sender_days_since_last',
        'domain_open_rate',
    ]

    # Embedding features handled separately

    def __init__(self, session: Session):
        """
        Initialize model trainer.

        Args:
            session: SQLAlchemy database session

        Example:
            >>> with db.get_session() as session:
            ...     trainer = ModelTrainer(session)
        """
        self.session = session
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def _extract_metadata_row(self, features: EmailFeatures) -> Dict[str, Any]:
        """
        Extract metadata features into row dictionary.

        Args:
            features: EmailFeatures object

        Returns:
            Dictionary with metadata feature values

        Example:
            >>> row = trainer._extract_metadata_row(features)
            >>> print(row['is_newsletter'])
        """
        row = {}
        for feat in self.METADATA_FEATURES:
            value = getattr(features, feat, None)
            row[feat] = value if value is not None else 0
        return row

    def _extract_historical_row(self, features: EmailFeatures) -> Dict[str, Any]:
        """
        Extract historical pattern features into row dictionary.

        Args:
            features: EmailFeatures object

        Returns:
            Dictionary with historical feature values

        Example:
            >>> row = trainer._extract_historical_row(features)
            >>> print(row['sender_open_rate'])
        """
        row = {}
        for feat in self.HISTORICAL_FEATURES:
            value = getattr(features, feat, None)
            # Special case: sender_days_since_last should default to 999, not 0
            if feat == 'sender_days_since_last' and value is None:
                value = 999  # Large value for no previous email
            else:
                value = value if value is not None else 0
            row[feat] = value
        return row

    def _extract_embedding_row(
        self,
        subject_emb: Optional[List[float]],
        body_emb: Optional[List[float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract embedding features into row dictionary.

        Returns None if no embeddings available (email should be skipped).

        Args:
            subject_emb: Subject embedding vector
            body_emb: Body embedding vector

        Returns:
            Dictionary with embedding features, or None if no embeddings

        Example:
            >>> row = trainer._extract_embedding_row(subject_emb, body_emb)
            >>> if row is None:
            ...     continue  # Skip this email
        """
        if not subject_emb and not body_emb:
            return None

        row = {}

        if subject_emb and body_emb:
            # Average the embeddings
            avg_emb = [(s + b) / 2 for s, b in zip(subject_emb, body_emb)]
            for i, val in enumerate(avg_emb):
                row[f'emb_{i}'] = val
        elif subject_emb:
            for i, val in enumerate(subject_emb):
                row[f'emb_{i}'] = val
        elif body_emb:
            for i, val in enumerate(body_emb):
                row[f'emb_{i}'] = val

        return row

    def prepare_data(
        self,
        use_embeddings: bool = True,
        min_sender_count: int = 2
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from database.

        Args:
            use_embeddings: Include embedding features
            min_sender_count: Minimum emails from sender to include

        Returns:
            Tuple of (features DataFrame, target Series)

        Example:
            >>> X, y = trainer.prepare_data(use_embeddings=True)
            >>> print(f"Training samples: {len(X)}")
        """
        logger.info("Preparing training data...")

        # Query emails with features
        query = self.session.query(Email, EmailFeatures).join(
            EmailFeatures,
            Email.message_id == EmailFeatures.message_id
        ).filter(
            EmailFeatures.sender_email_count >= min_sender_count
        )

        results = query.all()

        if not results:
            raise ValueError("No training data found. Run build_features.py first.")

        logger.info(f"Found {len(results)} emails with features")

        # Build feature matrix
        data = []
        targets = []

        for email, features in results:
            row = {}

            # Extract metadata features
            row.update(self._extract_metadata_row(features))

            # Extract historical features
            row.update(self._extract_historical_row(features))

            # Extract embedding features (optional)
            if use_embeddings:
                embedding_row = self._extract_embedding_row(
                    features.subject_embedding,
                    features.body_embedding
                )
                if embedding_row is None:
                    continue  # Skip email with no embeddings
                row.update(embedding_row)

            data.append(row)
            targets.append(1 if email.was_read else 0)

        if not data:
            raise ValueError("No valid training data after filtering")

        X = pd.DataFrame(data)
        y = pd.Series(targets)

        self.feature_names = list(X.columns)

        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Positive class ratio: {y.mean():.1%}")

        return X, y

    def train(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Train LightGBM model.

        Args:
            X: Feature matrix (if None, calls prepare_data)
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            params: LightGBM parameters (optional)

        Returns:
            Dictionary of evaluation metrics

        Example:
            >>> metrics = trainer.train()
            >>> print(f"AUC: {metrics['auc']:.3f}")
        """
        # Prepare data if not provided
        if X is None or y is None:
            X, y = self.prepare_data()

        # Ensure feature_names is set (in case X/y were provided directly)
        if not self.feature_names and X is not None:
            self.feature_names = list(X.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # LightGBM parameters
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': 4,
            }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train model
        logger.info("Training LightGBM model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0)  # Suppress iteration logs
            ]
        )

        # Evaluate
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'samples_train': len(X_train),
            'samples_test': len(X_test),
            'features': len(X.columns),
        }

        logger.info(f"Training complete:")
        logger.info(f"  - AUC: {metrics['auc']:.3f}")
        logger.info(f"  - Precision: {metrics['precision']:.3f}")
        logger.info(f"  - Recall: {metrics['recall']:.3f}")

        return metrics

    def save_model(
        self,
        model_dir: str = "models",
        version: Optional[str] = None
    ) -> Path:
        """
        Save trained model to disk.

        Args:
            model_dir: Directory to save model
            version: Model version (defaults to timestamp)

        Returns:
            Path to saved model file

        Example:
            >>> model_path = trainer.save_model()
            >>> print(f"Model saved to {model_path}")
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = model_dir_path / f"model_v{version}.txt"
        self.model.save_model(str(model_path))

        # Save feature names
        feature_config_path = model_dir_path / f"feature_config_v{version}.json"
        with open(feature_config_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'version': version,
                'created_at': datetime.now().isoformat(),
            }, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Feature config saved to {feature_config_path}")

        return model_path

    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file

        Example:
            >>> trainer.load_model("models/model_v20240101.txt")
        """
        self.model = lgb.Booster(model_file=model_path)

        # Try to load feature config
        model_path_obj = Path(model_path)
        version = model_path_obj.stem.replace('model_v', '')
        feature_config_path = model_path_obj.parent / f"feature_config_v{version}.json"

        if feature_config_path.exists():
            with open(feature_config_path, 'r') as f:
                config = json.load(f)
                self.feature_names = config['feature_names']

        logger.info(f"Model loaded from {model_path}")

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary mapping feature name to importance

        Example:
            >>> importance = trainer.get_feature_importance(top_n=10)
            >>> for feat, imp in importance.items():
            ...     print(f"{feat}: {imp:.3f}")
        """
        if self.model is None:
            raise ValueError("No model loaded")

        importance = self.model.feature_importance()
        feature_importance = dict(zip(self.feature_names, importance))

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return dict(sorted_features[:top_n])
