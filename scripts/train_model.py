#!/usr/bin/env python3
"""
Train ML model for email importance prediction.

Trains a LightGBM model using computed features to predict likelihood
of reading each email.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --no-embeddings
    python scripts/train_model.py --test-size 0.3
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.ml import ModelTrainer, ModelEvaluator
from src.utils import Config, setup_logger


def main():
    """Train email importance model."""
    parser = argparse.ArgumentParser(
        description='Train ML model for email importance prediction'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database file (default: data/emails.db)'
    )
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Train without embedding features (faster, less accurate)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save model (default: models)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('train_model', level='INFO')

    logger.info("=== Email Importance Model Training ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize database
    db_path = args.db_path or f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}")
    db = Database(db_path)

    # Check database has features
    stats = db.get_stats()
    logger.info(f"Database stats:")
    logger.info(f"  - Emails: {stats['emails']}")
    logger.info(f"  - Email features: {stats.get('email_features', 'N/A')}")
    logger.info("")

    if stats['emails'] == 0:
        logger.error("No emails in database. Run scripts/export_history.py first")
        sys.exit(1)

    # Initialize trainer
    logger.info("Initializing model trainer...")
    with db.get_session() as session:
        trainer = ModelTrainer(session)

        # Prepare data
        try:
            logger.info("Preparing training data...")
            X, y = trainer.prepare_data(use_embeddings=not args.no_embeddings)

            logger.info(f"Training data prepared:")
            logger.info(f"  - Samples: {len(X)}")
            logger.info(f"  - Features: {len(X.columns)}")
            logger.info(f"  - Positive rate: {y.mean():.1%}")
            logger.info("")

        except ValueError as e:
            logger.error(f"Failed to prepare data: {e}")
            logger.error("Make sure you've run scripts/build_features.py first")
            sys.exit(1)

    # Train model
    logger.info("Training LightGBM model...")
    logger.info(f"  - Test size: {args.test_size}")
    logger.info(f"  - Using embeddings: {not args.no_embeddings}")
    logger.info("")

    try:
        with db.get_session() as session:
            trainer = ModelTrainer(session)
            metrics = trainer.train(
                X=X,
                y=y,
                test_size=args.test_size
            )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Evaluate archive threshold
    logger.info("\nEvaluating archive precision...")
    evaluator = ModelEvaluator()

    # Get test predictions for threshold analysis
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    y_pred_proba = trainer.model.predict(X_test)

    # Evaluate at different archive thresholds
    for threshold in [0.2, 0.25, 0.3, 0.35]:
        archive_metrics = evaluator.evaluate_archive_precision(
            y_test.tolist(),
            y_pred_proba.tolist(),
            archive_threshold=threshold
        )
        logger.info("")

    # Find optimal thresholds
    logger.info("Finding optimal thresholds...")

    # For keeping emails (optimize recall - don't miss important ones)
    keep_threshold, keep_metrics = evaluator.find_optimal_threshold(
        y_test.tolist(),
        y_pred_proba.tolist(),
        optimize_for='recall',
        min_precision=0.7
    )
    logger.info(f"Suggested keep threshold: {keep_threshold:.3f}")
    logger.info(f"  - Recall: {keep_metrics['recall']:.3f}")
    logger.info(f"  - Precision: {keep_metrics['precision']:.3f}")
    logger.info("")

    # Feature importance
    logger.info("Top 10 most important features:")
    importance = trainer.get_feature_importance(top_n=10)
    for i, (feat, imp) in enumerate(importance.items(), 1):
        logger.info(f"  {i}. {feat}: {imp:.0f}")
    logger.info("")

    # Save model
    logger.info("Saving model...")
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = trainer.save_model(model_dir=args.model_dir, version=version)

    # Save metrics
    all_metrics = {
        'training_metrics': metrics,
        'threshold_analysis': {
            'keep_threshold': keep_threshold,
            'keep_metrics': keep_metrics,
        },
        'feature_importance': importance,
        'training_config': {
            'use_embeddings': not args.no_embeddings,
            'test_size': args.test_size,
            'n_samples': len(X),
            'n_features': len(X.columns),
        }
    }

    metrics_path = Path(args.model_dir) / f"metrics_v{version}.json"
    evaluator.save_metrics(all_metrics, str(metrics_path))

    # Save recommended thresholds
    thresholds = {
        'high_confidence_keep': 0.7,  # Conservative default
        'high_confidence_archive': 0.3,  # Conservative default
        'version': version,
        'created_at': datetime.now().isoformat(),
    }

    thresholds_path = Path(args.model_dir) / f"thresholds_v{version}.json"
    import json
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    logger.info(f"Thresholds saved to {thresholds_path}")

    # Success
    logger.info("\n=== Training Complete ===")
    logger.info(f"Model version: {version}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Metrics path: {metrics_path}")
    logger.info("")
    logger.info("Model performance:")
    logger.info(f"  - AUC: {metrics['auc']:.3f}")
    logger.info(f"  - Precision: {metrics['precision']:.3f}")
    logger.info(f"  - Recall: {metrics['recall']:.3f}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review metrics and feature importance")
    logger.info("  2. Test model with: python scripts/test_model.py")
    logger.info("  3. Proceed to Phase 5: Automation")


if __name__ == '__main__':
    main()
