"""Main email triage pipeline."""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..database import Database, EmailRepository
from ..database.schema import Email
from ..features import (
    EmbeddingExtractor,
    FeatureStore,
    HistoricalPatternExtractor,
    MetadataExtractor,
)
from ..gmail import GmailClient, GmailOperations
from ..ml import AdaptiveLabelGenerator, EmailCategorizer, EmailScorer
from ..utils import get_logger

logger = get_logger(__name__)


class TriagePipeline:
    """
    Main pipeline for automated email triage.

    Orchestrates fetching new emails, scoring them, making decisions,
    and applying labels/archives.

    Attributes:
        gmail_client: Gmail API client
        gmail_ops: Gmail operations
        database: Database instance
        scorer: Email scorer with trained model
        categorizer: Email categorizer

    Example:
        >>> pipeline = TriagePipeline(client, ops, db, "models/model_v20240101.txt")
        >>> results = pipeline.run_triage(dry_run=True)
    """

    def __init__(
        self,
        gmail_client: GmailClient,
        gmail_ops: GmailOperations,
        database: Database,
        model_path: str,
        use_embeddings: bool = True
    ):
        """
        Initialize triage pipeline.

        Args:
            gmail_client: Gmail API client
            gmail_ops: Gmail operations
            database: Database instance
            model_path: Path to trained model
            use_embeddings: Whether to use embeddings (slower but more accurate)

        Example:
            >>> pipeline = TriagePipeline(client, ops, db, "models/model_v20240101.txt")
        """
        self.gmail_client = gmail_client
        self.gmail_ops = gmail_ops
        self.database = database
        self.scorer = EmailScorer(model_path)
        self.use_embeddings = use_embeddings

        # Initialize extractors
        self.metadata_extractor = MetadataExtractor()
        self.embedding_extractor = EmbeddingExtractor() if use_embeddings else None

        logger.info("Triage pipeline initialized")

    def run_triage(
        self,
        days_back: int = 1,
        dry_run: bool = False,
        max_emails: int | None = None
    ) -> dict[str, Any]:
        """
        Run triage on recent emails.

        Args:
            days_back: How many days back to fetch emails
            dry_run: If True, don't actually modify emails
            max_emails: Maximum number of emails to process

        Returns:
            Dictionary with triage results and statistics

        Example:
            >>> results = pipeline.run_triage(days_back=1, dry_run=True)
            >>> print(f"Processed {results['total']} emails")
        """
        logger.info(f"Starting triage (days_back={days_back}, dry_run={dry_run})")

        # Fetch new emails
        start_date = datetime.now() - timedelta(days=days_back)
        logger.info(f"Fetching emails since {start_date.isoformat()}")

        message_ids = self.gmail_client.list_all_messages(
            after_date=start_date,
            max_total=max_emails
        )

        if not message_ids:
            logger.info("No new emails found")
            return {
                'total': 0,
                'high_keep': 0,
                'high_archive': 0,
                'low_confidence': 0,
                'errors': 0,
            }

        logger.info(f"Found {len(message_ids)} emails to process")

        # Fetch email details
        emails = self.gmail_client.get_messages_batch(message_ids)

        # Process each email
        results = {
            'total': len(emails),
            'high_keep': 0,
            'high_archive': 0,
            'low_confidence': 0,
            'errors': 0,
            'decisions': [],
        }

        for email in emails:
            try:
                decision = self._process_email(email, dry_run)
                results['decisions'].append(decision)

                # Update counters
                if decision['confidence'] == 'high':
                    if decision['action'] == 'archive':
                        results['high_archive'] += 1
                    else:
                        results['high_keep'] += 1
                else:
                    results['low_confidence'] += 1

            except Exception as e:
                logger.error(f"Failed to process email {email.message_id}: {e}")
                results['errors'] += 1

        logger.info("Triage complete:")
        logger.info(f"  - Total: {results['total']}")
        logger.info(f"  - High confidence keep: {results['high_keep']}")
        logger.info(f"  - High confidence archive: {results['high_archive']}")
        logger.info(f"  - Low confidence: {results['low_confidence']}")
        logger.info(f"  - Errors: {results['errors']}")

        return results

    def _process_email(
        self,
        gmail_email: Any,
        dry_run: bool
    ) -> dict[str, Any]:
        """
        Process a single email through the pipeline.

        Args:
            gmail_email: Gmail email object
            dry_run: If True, don't actually modify

        Returns:
            Dictionary with decision details
        """
        with self.database.get_session() as session:
            repo = EmailRepository(session)
            store = FeatureStore(session)
            label_generator = AdaptiveLabelGenerator(session)
            categorizer = EmailCategorizer(session)

            # Save email to database
            db_email = repo.save_email(gmail_email)

            # Extract features
            features = self._extract_features(db_email, session)

            # Save features (returns EmailFeatures object)
            email_features = store.save_features(db_email.message_id, features)

            # Score email
            score = self.scorer.score_email(db_email, email_features)

            # Make decision
            decision_info = self.scorer.make_decision(score)

            # Generate adaptive labels (priority: user feedback > embeddings > keywords)
            label_suggestions = label_generator.generate_labels(
                db_email,
                email_features,
                is_newsletter=features.get('is_newsletter', False)
            )

            # Use most specific label as primary category
            category = label_suggestions[0].label if label_suggestions else 'Bot/Uncategorized'

            # Build labels list: primary label + secondary if different specificity
            labels = [category]
            if len(label_suggestions) > 1 and label_suggestions[1].label != category:
                labels.append(label_suggestions[1].label)

            # Add confidence marker if low confidence
            if decision_info['confidence'] == 'low':
                labels.append(categorizer.LOW_CONFIDENCE)

            # Add archived marker if archiving
            if decision_info['action'] == 'archive':
                labels.append(categorizer.AUTO_ARCHIVED)

            # Apply decision
            if not dry_run:
                self._apply_decision(db_email, decision_info, labels, session)
            else:
                logger.debug(
                    f"DRY RUN: Would {decision_info['action']} email "
                    f"{db_email.message_id} with labels {labels}"
                )

            return {
                'message_id': db_email.message_id,
                'subject': db_email.subject,
                'from': db_email.from_address,
                'score': score,
                'action': decision_info['action'],
                'confidence': decision_info['confidence'],
                'category': category,
                'labels': labels,
                'dry_run': dry_run,
            }

    def _extract_features(
        self,
        email: Email,
        session: Session
    ) -> dict[str, Any]:
        """Extract features for an email."""
        features = {}

        # Metadata
        metadata = self.metadata_extractor.extract(email)
        features.update(metadata)

        # Historical patterns
        historical_extractor = HistoricalPatternExtractor(session)
        historical = historical_extractor.extract(email)
        features.update(historical)

        # Embeddings (optional)
        if self.embedding_extractor:
            embeddings = self.embedding_extractor.extract(email)
            features.update(embeddings)

        return features

    def _apply_decision(
        self,
        email: Email,
        decision: dict[str, Any],
        labels: list[str],
        session: Session
    ) -> None:
        """Apply triage decision to email."""
        repo = EmailRepository(session)

        # Record action
        repo.record_action(
            message_id=email.message_id,
            action_type=decision['action'],
            source='bot',
            action_data={
                'score': decision['score'],
                'confidence': decision['confidence'],
                'labels': labels,
            }
        )

        # Apply labels
        if labels:
            self.gmail_ops.add_labels(
                [email.message_id],
                labels,
                dry_run=False
            )
            logger.debug(f"Applied labels to {email.message_id}: {labels}")

        # Archive if decided
        if decision['action'] == 'archive':
            self.gmail_ops.archive(
                [email.message_id],
                dry_run=False
            )
            logger.debug(f"Archived {email.message_id}")
