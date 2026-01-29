"""Machine learning modules for email triage."""

from .adaptive_labels import AdaptiveLabelGenerator, LabelSuggestion
from .categorization import EmailCategorizer
from .evaluation import ModelEvaluator
from .inference import EmailScorer
from .semantic_classifier import SemanticClassifier
from .training import ModelTrainer

__all__ = [
    "ModelTrainer",
    "EmailScorer",
    "ModelEvaluator",
    "EmailCategorizer",
    "AdaptiveLabelGenerator",
    "LabelSuggestion",
    "SemanticClassifier",
]
