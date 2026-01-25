"""Topic embedding generation using sentence-transformers."""

from typing import Dict, Any, Optional, List
import numpy as np

from sentence_transformers import SentenceTransformer

from ..database.schema import Email
from ..utils import get_logger


logger = get_logger(__name__)


class EmbeddingExtractor:
    """
    Generate topic embeddings for email subject and body.

    Uses sentence-transformers (all-MiniLM-L6-v2) to create 384-dimensional
    semantic embeddings for similarity calculations.

    Attributes:
        model: SentenceTransformer model
        model_name: Name of the model being used
        max_length: Maximum text length for embeddings

    Example:
        >>> extractor = EmbeddingExtractor()
        >>> features = extractor.extract(email)
        >>> print(f"Subject embedding shape: {len(features['subject_embedding'])}")
    """

    DEFAULT_MODEL = 'all-MiniLM-L6-v2'  # 384-dim, fast, good quality
    MAX_LENGTH = 512  # Tokens

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding extractor.

        Args:
            model_name: Name of sentence-transformer model (defaults to all-MiniLM-L6-v2)

        Example:
            >>> extractor = EmbeddingExtractor()
            >>> # Or use a different model:
            >>> extractor = EmbeddingExtractor('all-mpnet-base-v2')
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully (embedding dim: {self.model.get_sentence_embedding_dimension()})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def extract(self, email: Email) -> Dict[str, Any]:
        """
        Extract embedding features from an email.

        Args:
            email: Email database model

        Returns:
            Dictionary with subject_embedding and body_embedding

        Example:
            >>> features = extractor.extract(email)
            >>> subject_emb = features['subject_embedding']
            >>> print(f"Embedding dimensions: {len(subject_emb)}")
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        features = {}

        # Subject embedding
        subject_text = self._prepare_text(email.subject or '')
        if subject_text:
            features['subject_embedding'] = self._encode(subject_text)
        else:
            features['subject_embedding'] = None

        # Body embedding (first MAX_LENGTH tokens)
        body_text = self._prepare_text(email.body_plain or email.snippet or '')
        if body_text:
            features['body_embedding'] = self._encode(body_text)
        else:
            features['body_embedding'] = None

        logger.debug(f"Extracted embeddings for {email.message_id}")

        return features

    def extract_batch(self, emails: List[Email]) -> List[Dict[str, Any]]:
        """
        Extract embeddings from multiple emails efficiently.

        Uses batch encoding for better performance.

        Args:
            emails: List of Email database models

        Returns:
            List of feature dictionaries

        Example:
            >>> features_list = extractor.extract_batch(emails)
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        # Prepare texts
        subjects = [self._prepare_text(e.subject or '') for e in emails]
        bodies = [self._prepare_text(e.body_plain or e.snippet or '') for e in emails]

        # Batch encode
        logger.debug(f"Batch encoding {len(emails)} emails...")
        subject_embeddings = self._encode_batch(subjects)
        body_embeddings = self._encode_batch(bodies)

        # Combine into feature dicts
        features_list = []
        for i in range(len(emails)):
            features = {
                'subject_embedding': subject_embeddings[i] if subjects[i] else None,
                'body_embedding': body_embeddings[i] if bodies[i] else None,
            }
            features_list.append(features)

        logger.debug(f"Extracted embeddings for {len(emails)} emails")

        return features_list

    def _prepare_text(self, text: str) -> str:
        """
        Prepare text for embedding.

        Args:
            text: Raw text

        Returns:
            Cleaned and truncated text

        Example:
            >>> clean_text = extractor._prepare_text(raw_text)
        """
        if not text:
            return ''

        # Remove excess whitespace
        text = ' '.join(text.split())

        # Truncate to reasonable length (approximate token count)
        # Rough estimate: 1 token ~= 4 characters
        max_chars = self.MAX_LENGTH * 4
        if len(text) > max_chars:
            text = text[:max_chars]

        return text

    def _encode(self, text: str) -> List[float]:
        """
        Encode single text to embedding.

        Args:
            text: Text to encode

        Returns:
            Embedding as list of floats

        Example:
            >>> embedding = extractor._encode("Sample text")
        """
        if not text or not self.model:
            return []

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            List of embeddings

        Example:
            >>> embeddings = extractor._encode_batch(texts)
        """
        if not texts or not self.model:
            return [[] for _ in texts]

        # Filter out empty texts, keeping track of indices
        valid_indices = [i for i, text in enumerate(texts) if text]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return [[] for _ in texts]

        # Encode valid texts
        embeddings = self.model.encode(
            valid_texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Reconstruct full list with empty embeddings for invalid texts
        result = [[] for _ in texts]
        for i, embedding in zip(valid_indices, embeddings):
            result[i] = embedding.tolist()

        return result

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0-1)

        Example:
            >>> similarity = extractor.compute_similarity(emb1, emb2)
            >>> print(f"Similarity: {similarity:.3f}")
        """
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        Get list of feature names produced by this extractor.

        Returns:
            List of feature names

        Example:
            >>> names = EmbeddingExtractor.get_feature_names()
        """
        return [
            'subject_embedding',
            'body_embedding',
        ]

    @property
    def embedding_dim(self) -> int:
        """
        Get embedding dimensionality.

        Returns:
            Number of dimensions in embeddings

        Example:
            >>> dim = extractor.embedding_dim
            >>> print(f"Embedding dimensions: {dim}")
        """
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2
