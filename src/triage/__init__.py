"""Email triage pipeline and automation."""

from .pipeline import TriagePipeline
from .feedback import FeedbackCollector

__all__ = [
    "TriagePipeline",
    "FeedbackCollector",
]
