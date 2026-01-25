"""Machine learning modules for email triage."""

from .training import ModelTrainer
from .inference import EmailScorer
from .evaluation import ModelEvaluator
from .categorization import EmailCategorizer

__all__ = [
    "ModelTrainer",
    "EmailScorer",
    "ModelEvaluator",
    "EmailCategorizer",
]
