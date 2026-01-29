"""Adaptive label generation based on sender patterns and content."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ..database.schema import Email, EmailFeatures, SenderLabelMapping
from ..utils import get_logger

logger = get_logger(__name__)

# Lazy import to avoid circular dependency
_semantic_classifier = None


def _get_semantic_classifier():
    """Lazily load semantic classifier if model exists."""
    global _semantic_classifier

    if _semantic_classifier is not None:
        return _semantic_classifier

    model_path = Path("models/semantic_classifier.pkl")
    if not model_path.exists():
        return None

    try:
        from .semantic_classifier import SemanticClassifier
        _semantic_classifier = SemanticClassifier()
        _semantic_classifier.load(str(model_path))
        logger.info("Loaded semantic classifier for label generation")
        return _semantic_classifier
    except Exception as e:
        logger.warning(f"Failed to load semantic classifier: {e}")
        return None


@dataclass
class LabelSuggestion:
    """A suggested label with metadata."""

    label: str
    specificity: int  # 1 (very specific) to 5 (very generic)
    confidence: float  # 0.0 to 1.0
    reason: str  # Why this label was suggested


class AdaptiveLabelGenerator:
    """
    Generate adaptive labels based on sender patterns, embeddings, and user feedback.

    Priority order:
    1. User feedback (learned sender mappings) - highest confidence
    2. Semantic classifier (trained on Claude-generated labels)
    3. Embedding similarity to existing labeled emails
    4. Sender-specific labels for frequent senders
    5. Hardcoded keyword rules - fallback

    Attributes:
        session: Database session
        specificity_threshold: Emails needed before creating specific label

    Example:
        >>> with db.get_session() as session:
        ...     generator = AdaptiveLabelGenerator(session)
        ...     labels = generator.generate_labels(email, features)
    """

    DEFAULT_SPECIFICITY_THRESHOLD = 3
    SIMILARITY_THRESHOLD = 0.75  # Minimum cosine similarity for cluster match

    # Hardcoded fallback categories
    FALLBACK_CATEGORIES = {
        'receipt': ('Bot/Receipts', ['receipt', 'order confirmation', 'invoice', 'payment', 'shipped']),
        'automated': ('Bot/Automated', ['notification', 'alert', 'no-reply', 'noreply', 'automated']),
        'promotional': ('Bot/Promotional', ['sale', 'discount', 'offer', 'deal', 'coupon', '% off']),
        'newsletter': ('Bot/Newsletters', []),  # Detected via headers, not keywords
    }

    def __init__(
        self,
        session: Session,
        specificity_threshold: int = DEFAULT_SPECIFICITY_THRESHOLD
    ):
        self.session = session
        self.specificity_threshold = specificity_threshold
        self._label_centroids: dict[str, np.ndarray] | None = None

    def generate_labels(
        self,
        email: Email,
        features: EmailFeatures | None = None,
        is_newsletter: bool = False
    ) -> list[LabelSuggestion]:
        """
        Generate adaptive labels for an email.

        Args:
            email: Email to label
            features: Pre-computed features with embeddings
            is_newsletter: Whether email is detected as newsletter

        Returns:
            List of LabelSuggestion objects, ordered by specificity
        """
        suggestions = []

        # 1. Check for user-learned label (highest priority)
        learned = self._get_learned_label(email.from_address)
        if learned:
            suggestions.append(LabelSuggestion(
                label=learned,
                specificity=1,
                confidence=1.0,
                reason='Learned from user feedback'
            ))
            return suggestions

        # 2. Try semantic classifier (trained on Claude-generated labels)
        if features and features.subject_embedding:
            semantic_label = self._get_semantic_label(features)
            if semantic_label:
                suggestions.append(semantic_label)

        # 3. Try embedding-based matching to existing labeled emails
        if features and features.subject_embedding:
            cluster_label = self._find_similar_label(features)
            if cluster_label and (not suggestions or cluster_label.label != suggestions[0].label):
                suggestions.append(cluster_label)

        # 3. Check sender frequency for specific label
        sender_count = self._get_sender_email_count(email.from_address)
        if sender_count >= self.specificity_threshold:
            specific = self._generate_sender_label(email, sender_count)
            if specific and (not suggestions or specific.label != suggestions[0].label):
                suggestions.append(specific)

        # 4. Fall back to hardcoded rules
        fallback = self._get_fallback_category(email, is_newsletter)
        if fallback and (not suggestions or fallback.label not in [s.label for s in suggestions]):
            suggestions.append(fallback)

        # Sort by specificity (most specific first)
        suggestions.sort(key=lambda s: s.specificity)

        return suggestions if suggestions else [LabelSuggestion(
            label='Bot/Uncategorized',
            specificity=5,
            confidence=0.3,
            reason='No matching category found'
        )]

    def get_primary_label(
        self,
        email: Email,
        features: EmailFeatures | None = None,
        is_newsletter: bool = False
    ) -> str:
        """Get the single best label for an email."""
        suggestions = self.generate_labels(email, features, is_newsletter)
        return suggestions[0].label if suggestions else 'Bot/Uncategorized'

    def get_all_labels(
        self,
        email: Email,
        features: EmailFeatures | None = None,
        is_newsletter: bool = False,
        max_labels: int = 2
    ) -> list[str]:
        """Get multiple labels at different specificity levels."""
        suggestions = self.generate_labels(email, features, is_newsletter)
        return [s.label for s in suggestions[:max_labels]]

    def _get_learned_label(self, sender_address: str) -> str | None:
        """Check for user-provided label mapping."""
        mapping = self.session.query(SenderLabelMapping).filter(
            SenderLabelMapping.sender_address == sender_address
        ).first()

        if mapping:
            return mapping.label

        # Check domain-level mapping
        if '@' in sender_address:
            domain = sender_address.split('@')[1]
            domain_mapping = self.session.query(SenderLabelMapping).filter(
                SenderLabelMapping.sender_domain == domain,
                SenderLabelMapping.sender_address.is_(None)
            ).first()
            if domain_mapping:
                return domain_mapping.label

        return None

    def _get_semantic_label(self, features: EmailFeatures) -> LabelSuggestion | None:
        """
        Get label from trained semantic classifier.

        Uses the local classifier trained on Claude-generated labels.
        """
        classifier = _get_semantic_classifier()
        if classifier is None:
            return None

        if not features.subject_embedding:
            return None

        try:
            label, confidence = classifier.predict(features.subject_embedding)

            # Only use if confidence is high enough
            if confidence < 0.4:
                return None

            return LabelSuggestion(
                label=f'Bot/{label}',
                specificity=2,
                confidence=confidence,
                reason=f'Semantic classifier ({confidence:.0%} confidence)'
            )
        except Exception as e:
            logger.warning(f"Semantic classifier error: {e}")
            return None

    def _find_similar_label(self, features: EmailFeatures) -> LabelSuggestion | None:
        """
        Find a label by comparing embeddings to labeled email clusters.

        Uses cached centroids for efficiency.
        """
        if not features.subject_embedding:
            return None

        # Build or use cached centroids
        if self._label_centroids is None:
            self._build_label_centroids()

        if not self._label_centroids:
            return None

        email_emb = np.array(features.subject_embedding)
        email_norm = np.linalg.norm(email_emb)
        if email_norm == 0:
            return None

        best_label = None
        best_similarity = self.SIMILARITY_THRESHOLD

        for label, centroid in self._label_centroids.items():
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                continue

            similarity = np.dot(email_emb, centroid) / (email_norm * centroid_norm)

            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label

        if best_label:
            return LabelSuggestion(
                label=best_label,
                specificity=2,
                confidence=float(best_similarity),
                reason=f'Similar to existing {best_label} emails ({best_similarity:.0%} match)'
            )

        return None

    def _build_label_centroids(self) -> None:
        """Build centroid embeddings for each label from user-labeled emails."""
        self._label_centroids = {}

        # Get all unique labels from sender mappings
        labels = self.session.query(SenderLabelMapping.label).distinct().all()
        labels = [l[0] for l in labels]

        for label in labels:
            # Get sender addresses with this label
            mappings = self.session.query(SenderLabelMapping.sender_address).filter(
                SenderLabelMapping.label == label,
                SenderLabelMapping.sender_address.isnot(None)
            ).all()
            sender_addresses = [m[0] for m in mappings]

            if not sender_addresses:
                continue

            # Get embeddings for emails from these senders
            embeddings = []
            for sender in sender_addresses:
                features_list = self.session.query(EmailFeatures).join(
                    Email, Email.message_id == EmailFeatures.message_id
                ).filter(
                    Email.from_address == sender
                ).limit(10).all()

                for f in features_list:
                    if f.subject_embedding:
                        embeddings.append(np.array(f.subject_embedding))

            if embeddings:
                self._label_centroids[label] = np.mean(embeddings, axis=0)
                logger.debug(f"Built centroid for {label} from {len(embeddings)} emails")

    def invalidate_cache(self) -> None:
        """Clear cached centroids (call after user feedback)."""
        self._label_centroids = None

    def _get_sender_email_count(self, sender_address: str) -> int:
        """Get count of emails from this sender."""
        return self.session.query(Email).filter(
            Email.from_address == sender_address
        ).count()

    def _generate_sender_label(
        self,
        email: Email,
        sender_count: int
    ) -> LabelSuggestion | None:
        """Generate a specific label for a frequent sender based on domain."""
        # Only use domain-based labels, not personal sender names
        if '@' in email.from_address:
            domain = email.from_address.split('@')[1]
            domain_label = self._domain_to_label(domain)
            if domain_label:
                return LabelSuggestion(
                    label=f'Bot/{domain_label}',
                    specificity=3,
                    confidence=0.5,
                    reason=f'Frequent sender from {domain}'
                )

        return None

    def _clean_label_name(self, name: str) -> str:
        """Clean a name for use as a Gmail label."""
        clean = re.sub(r'[^\w\s-]', '', name)
        clean = re.sub(r'\s+', '-', clean.strip())
        clean = '-'.join(word.capitalize() for word in clean.split('-') if word)
        if len(clean) > 30:
            clean = clean[:30].rsplit('-', 1)[0]
        return clean

    def _domain_to_label(self, domain: str) -> str | None:
        """Convert domain to label name."""
        parts = domain.lower()
        for suffix in ['.com', '.org', '.net', '.edu', '.io', '.co']:
            parts = parts.replace(suffix, '')
        for prefix in ['www.', 'mail.', 'email.', 'newsletters.']:
            parts = parts.replace(prefix, '')

        if len(parts) < 3:
            return None

        return self._clean_label_name(parts)

    def _get_fallback_category(
        self,
        email: Email,
        is_newsletter: bool
    ) -> LabelSuggestion | None:
        """Use hardcoded rules as fallback."""
        text = f"{email.subject or ''} {email.snippet or ''}".lower()

        for category_key, (label, keywords) in self.FALLBACK_CATEGORIES.items():
            if category_key == 'newsletter':
                if is_newsletter:
                    return LabelSuggestion(
                        label=label,
                        specificity=4,
                        confidence=0.6,
                        reason='Detected as newsletter (List-Unsubscribe header)'
                    )
            elif any(kw in text for kw in keywords):
                return LabelSuggestion(
                    label=label,
                    specificity=4,
                    confidence=0.5,
                    reason=f'Matched {category_key} keywords'
                )

        # Default fallback
        return LabelSuggestion(
            label='Bot/Personal',
            specificity=5,
            confidence=0.3,
            reason='Default category'
        )
