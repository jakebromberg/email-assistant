#!/usr/bin/env python3
"""
Train local semantic classifier on Claude-generated labels.

Uses embeddings from emails labeled by Claude API to train a fast local
classifier for ongoing inference.

Usage:
    python scripts/train_semantic_classifier.py
    python scripts/train_semantic_classifier.py --min-samples 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Database
from src.ml.semantic_classifier import SemanticClassifier
from src.utils import Config, setup_logger


def main():
    """Train semantic classifier."""
    parser = argparse.ArgumentParser(
        description='Train local semantic classifier on Claude-generated labels'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=3,
        help='Minimum samples per class (default: 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for model (default: models/semantic_classifier.pkl)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger('train_semantic_classifier', level='INFO')

    logger.info("=== Train Semantic Classifier ===\n")

    # Load configuration
    config = Config.load()
    config.validate()

    # Initialize database
    db_path = f"{config.DATA_DIR}/emails.db"
    logger.info(f"Opening database: {db_path}\n")
    db = Database(db_path)

    # Train classifier
    classifier = SemanticClassifier()

    with db.get_session() as session:
        try:
            metrics = classifier.train(
                session,
                min_samples_per_class=args.min_samples
            )
        except ValueError as e:
            logger.error(f"Training failed: {e}")
            logger.error("\nMake sure you've run generate_semantic_labels.py first")
            sys.exit(1)

    # Save model
    output_path = args.output or "models/semantic_classifier.pkl"
    classifier.save(output_path)

    # Summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Samples: {metrics['n_samples']}")
    logger.info(f"Classes: {metrics['n_classes']}")
    logger.info(f"Validation accuracy: {metrics['accuracy']:.1%}")
    logger.info(f"\nModel saved to: {output_path}")

    # Show per-class metrics
    logger.info("\n=== Per-Class Performance ===")
    report = metrics['report']
    for label, scores in sorted(report.items()):
        if isinstance(scores, dict) and 'precision' in scores:
            logger.info(
                f"  {label}: precision={scores['precision']:.2f}, "
                f"recall={scores['recall']:.2f}, "
                f"f1={scores['f1-score']:.2f}, "
                f"support={scores['support']}"
            )

    logger.info("\nClassifier ready for use!")
    logger.info("The triage pipeline will automatically use semantic labels when available.")


if __name__ == '__main__':
    main()
