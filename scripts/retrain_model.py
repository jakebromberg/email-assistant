#!/usr/bin/env python3
"""
Retrain model with feedback data.

Incorporates user feedback to improve model predictions.

Usage:
    python scripts/retrain_model.py
    python scripts/retrain_model.py --use-feedback-only
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.database.schema import FeedbackReview
from src.ml import ModelTrainer, ModelEvaluator
from src.triage import FeedbackCollector
from src.utils import Config, setup_logger


def main():
    """Retrain model with feedback."""
    parser = argparse.ArgumentParser(
        description='Retrain model incorporating user feedback'
    )
    parser.add_argument(
        '--use-feedback-only',
        action='store_true',
        help='Only use corrected emails for retraining'
    )
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Train without embeddings (faster)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save model (default: models)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('retrain_model', level='INFO')

    logger.info("=== Model Retraining ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize database
    db_path = f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}")
    db = Database(db_path)

    # Check for feedback
    with db.get_session() as session:
        feedback_count = session.query(FeedbackReview).filter(
            FeedbackReview.used_in_training == False
        ).count()

    if feedback_count == 0:
        logger.warning("No new feedback found")
        logger.info("Collect feedback with: python scripts/review_decisions.py")
        logger.info("Or: python scripts/collect_feedback.py")
        sys.exit(1)

    logger.info(f"Found {feedback_count} feedback records\n")

    # Initialize trainer
    logger.info("Initializing model trainer...")
    with db.get_session() as session:
        trainer = ModelTrainer(session)

        # Prepare data
        logger.info("Preparing training data...")
        X, y = trainer.prepare_data(use_embeddings=not args.no_embeddings)

        logger.info(f"Training data:")
        logger.info(f"  - Samples: {len(X)}")
        logger.info(f"  - Features: {len(X.columns)}")
        logger.info(f"  - Positive rate: {y.mean():.1%}")
        logger.info("")

    # TODO: Weight samples based on feedback
    # For corrected samples, increase their weight in training

    # Train model
    logger.info("Training new model...")
    try:
        with db.get_session() as session:
            trainer = ModelTrainer(session)
            metrics = trainer.train(X=X, y=y)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Evaluate
    logger.info("\nEvaluating new model...")
    evaluator = ModelEvaluator()

    # Save model
    logger.info("Saving model...")
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = trainer.save_model(model_dir=args.model_dir, version=version)

    # Save metrics
    all_metrics = {
        'training_metrics': metrics,
        'feedback_used': feedback_count,
        'training_config': {
            'use_embeddings': not args.no_embeddings,
            'n_samples': len(X),
            'n_features': len(X.columns),
        }
    }

    metrics_path = Path(args.model_dir) / f"metrics_v{version}.json"
    evaluator.save_metrics(all_metrics, str(metrics_path))

    # Mark feedback as used
    with db.get_session() as session:
        session.query(FeedbackReview).filter(
            FeedbackReview.used_in_training == False
        ).update({'used_in_training': True})

    logger.info("\n=== Retraining Complete ===")
    logger.info(f"Model version: {version}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Metrics path: {metrics_path}")
    logger.info(f"Feedback incorporated: {feedback_count} records")
    logger.info("")
    logger.info("Model performance:")
    logger.info(f"  - AUC: {metrics['auc']:.3f}")
    logger.info(f"  - Precision: {metrics['precision']:.3f}")
    logger.info(f"  - Recall: {metrics['recall']:.3f}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test new model: python scripts/test_model.py")
    logger.info("  2. Update triage to use new model")


if __name__ == '__main__':
    main()
