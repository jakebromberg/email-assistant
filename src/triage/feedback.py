"""Feedback collection from user actions and labels."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from ..gmail import GmailClient
from ..database import EmailRepository
from ..database.schema import Email, EmailAction
from ..utils import get_logger


logger = get_logger(__name__)


class FeedbackCollector:
    """
    Collect implicit feedback from user actions.

    Monitors Gmail for:
    - Emails moved back to inbox (false positives)
    - User-applied/removed labels
    - Opens and replies (engagement)

    Attributes:
        gmail_client: Gmail API client
        database: Database instance

    Example:
        >>> collector = FeedbackCollector(client, db)
        >>> feedback = collector.collect_feedback(days_back=1)
    """

    def __init__(
        self,
        gmail_client: GmailClient,
        database: Any
    ):
        """
        Initialize feedback collector.

        Args:
            gmail_client: Gmail API client
            database: Database instance

        Example:
            >>> collector = FeedbackCollector(client, db)
        """
        self.gmail_client = gmail_client
        self.database = database
        logger.info("Feedback collector initialized")

    def collect_feedback(
        self,
        days_back: int = 1
    ) -> Dict[str, Any]:
        """
        Collect feedback from recent user actions.

        Args:
            days_back: How many days back to check

        Returns:
            Dictionary with feedback statistics

        Example:
            >>> feedback = collector.collect_feedback(days_back=7)
            >>> print(f"False positives: {feedback['false_positives']}")
        """
        logger.info(f"Collecting feedback for last {days_back} days")

        start_date = datetime.now() - timedelta(days=days_back)

        with self.database.get_session() as session:
            repo = EmailRepository(session)

            # Get bot actions from this period
            bot_actions = session.query(EmailAction).filter(
                EmailAction.source == 'bot',
                EmailAction.timestamp >= start_date
            ).all()

            feedback_stats = {
                'bot_actions': len(bot_actions),
                'false_positives': 0,
                'label_changes': 0,
                'verified_correct': 0,
            }

            for action in bot_actions:
                # Check current state of email
                try:
                    current_email = self.gmail_client.get_message(action.message_id)

                    # Check for false positives (bot archived, user moved back)
                    if action.action_type == 'archive':
                        if current_email.is_in_inbox:
                            logger.info(
                                f"False positive detected: {action.message_id} "
                                f"(bot archived, user moved back)"
                            )
                            feedback_stats['false_positives'] += 1

                            # Record feedback
                            repo.save_feedback(
                                message_id=action.message_id,
                                decision_correct=False,
                                correct_decision='keep',
                                user_comment='User moved back to inbox after bot archived'
                            )

                    # Check for label changes
                    original_labels = action.action_data.get('labels', [])
                    current_bot_labels = [
                        l for l in current_email.labels
                        if l.startswith('Bot/')
                    ]

                    if set(original_labels) != set(current_bot_labels):
                        logger.debug(f"Label change detected for {action.message_id}")
                        feedback_stats['label_changes'] += 1

                except Exception as e:
                    logger.warning(f"Failed to check email {action.message_id}: {e}")
                    continue

        logger.info(f"Feedback collected:")
        logger.info(f"  - Bot actions: {feedback_stats['bot_actions']}")
        logger.info(f"  - False positives: {feedback_stats['false_positives']}")
        logger.info(f"  - Label changes: {feedback_stats['label_changes']}")

        return feedback_stats

    def get_feedback_for_retraining(
        self,
        session: Session
    ) -> List[Dict[str, Any]]:
        """
        Get feedback data for model retraining.

        Args:
            session: Database session

        Returns:
            List of feedback records

        Example:
            >>> with db.get_session() as session:
            ...     feedback = collector.get_feedback_for_retraining(session)
        """
        from ..database.schema import FeedbackReview

        # Get unused feedback reviews
        reviews = session.query(FeedbackReview).filter(
            FeedbackReview.used_in_training == False
        ).all()

        feedback_data = []
        for review in reviews:
            feedback_data.append({
                'message_id': review.message_id,
                'decision_correct': review.decision_correct,
                'label_correct': review.label_correct,
                'correct_decision': review.correct_decision,
                'correct_label': review.correct_label,
                'user_comment': review.user_comment,
            })

        logger.info(f"Retrieved {len(feedback_data)} feedback records for retraining")

        return feedback_data
