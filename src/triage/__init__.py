"""Email triage pipeline and automation."""

from .feedback import FeedbackCollector
from .pipeline import TriagePipeline

__all__ = [
    "TriagePipeline",
    "FeedbackCollector",
]
