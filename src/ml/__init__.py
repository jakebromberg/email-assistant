"""Machine learning modules for email triage."""

from .categorization import EmailCategorizer
from .evaluation import ModelEvaluator
from .inference import EmailScorer
from .training import ModelTrainer

__all__ = [
    "ModelTrainer",
    "EmailScorer",
    "ModelEvaluator",
    "EmailCategorizer",
]
