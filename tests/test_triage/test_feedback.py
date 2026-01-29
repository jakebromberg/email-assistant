"""Tests for feedback collection."""

from datetime import datetime
from unittest.mock import Mock, patch


from src.database.schema import EmailAction, FeedbackReview
from src.triage.feedback import FeedbackCollector


class TestFeedbackCollectorInit:
    """Test FeedbackCollector initialization."""

    def test_init(self, mock_gmail_client, temp_db):
        """Test collector initialization."""
        collector = FeedbackCollector(mock_gmail_client, temp_db)

        assert collector.gmail_client is mock_gmail_client
        assert collector.database is temp_db


class TestFeedbackForRetraining:
    """Test get_feedback_for_retraining method."""

    def test_get_feedback_for_retraining_empty(self, mock_gmail_client, temp_db, temp_db_session):
        """Test getting feedback when none exists."""
        collector = FeedbackCollector(mock_gmail_client, temp_db)

        result = collector.get_feedback_for_retraining(temp_db_session)

        assert result == []

    def test_get_feedback_for_retraining(self, mock_gmail_client, temp_db, temp_db_session, email_factory):
        """Test getting unused feedback for retraining."""
        email = email_factory()

        # Add feedback that hasn't been used
        feedback = FeedbackReview(
            message_id=email.message_id,
            review_date=datetime.now(),
            decision_correct=False,
            correct_decision='keep',
            user_comment='Important email',
            used_in_training=False
        )
        temp_db_session.add(feedback)
        temp_db_session.commit()

        collector = FeedbackCollector(mock_gmail_client, temp_db)
        result = collector.get_feedback_for_retraining(temp_db_session)

        assert len(result) == 1
        assert result[0]['message_id'] == email.message_id
        assert result[0]['decision_correct'] is False
        assert result[0]['correct_decision'] == 'keep'
        assert result[0]['user_comment'] == 'Important email'

    def test_get_feedback_excludes_used(self, mock_gmail_client, temp_db, temp_db_session, email_factory):
        """Test that already-used feedback is excluded."""
        email1 = email_factory()
        email2 = email_factory()

        # Add used feedback
        used_feedback = FeedbackReview(
            message_id=email1.message_id,
            review_date=datetime.now(),
            decision_correct=True,
            used_in_training=True  # Already used
        )
        # Add unused feedback
        unused_feedback = FeedbackReview(
            message_id=email2.message_id,
            review_date=datetime.now(),
            decision_correct=False,
            used_in_training=False
        )
        temp_db_session.add(used_feedback)
        temp_db_session.add(unused_feedback)
        temp_db_session.commit()

        collector = FeedbackCollector(mock_gmail_client, temp_db)
        result = collector.get_feedback_for_retraining(temp_db_session)

        assert len(result) == 1
        assert result[0]['message_id'] == email2.message_id


class TestCollectFeedback:
    """Test collect_feedback method."""

    def test_collect_feedback_no_actions(self, mock_gmail_client, temp_db, temp_db_session):
        """Test collecting feedback when no bot actions exist."""
        # Ensure no actions in database
        collector = FeedbackCollector(mock_gmail_client, temp_db)

        with patch.object(temp_db, 'get_session') as mock_session:
            mock_session.return_value.__enter__ = Mock(return_value=temp_db_session)
            mock_session.return_value.__exit__ = Mock(return_value=None)

            result = collector.collect_feedback(days_back=1)

        assert result['bot_actions'] == 0
        assert result['false_positives'] == 0

    def test_collect_feedback_detects_false_positive(self, mock_gmail_client, temp_db, temp_db_session, email_factory):
        """Test detecting false positives (bot archived, user moved back)."""
        email = email_factory(message_id="test_fp")

        # Record a bot archive action
        action = EmailAction(
            message_id="test_fp",
            action_type='archive',
            source='bot',
            action_data={'labels': ['Bot/Newsletter']},
            timestamp=datetime.now()
        )
        temp_db_session.add(action)
        temp_db_session.commit()

        # Mock Gmail to show email is back in inbox
        mock_email = Mock()
        mock_email.is_in_inbox = True
        mock_email.labels = ['INBOX', 'Bot/Newsletter']
        mock_gmail_client.get_message = Mock(return_value=mock_email)

        collector = FeedbackCollector(mock_gmail_client, temp_db)

        with patch.object(temp_db, 'get_session') as mock_session:
            mock_session.return_value.__enter__ = Mock(return_value=temp_db_session)
            mock_session.return_value.__exit__ = Mock(return_value=None)

            result = collector.collect_feedback(days_back=1)

        assert result['bot_actions'] == 1
        assert result['false_positives'] == 1

    def test_collect_feedback_handles_api_error(self, mock_gmail_client, temp_db, temp_db_session, email_factory):
        """Test handling Gmail API errors gracefully."""
        email = email_factory(message_id="test_error")

        # Record a bot action
        action = EmailAction(
            message_id="test_error",
            action_type='archive',
            source='bot',
            action_data={},
            timestamp=datetime.now()
        )
        temp_db_session.add(action)
        temp_db_session.commit()

        # Mock Gmail to raise an error
        mock_gmail_client.get_message = Mock(side_effect=Exception("API error"))

        collector = FeedbackCollector(mock_gmail_client, temp_db)

        with patch.object(temp_db, 'get_session') as mock_session:
            mock_session.return_value.__enter__ = Mock(return_value=temp_db_session)
            mock_session.return_value.__exit__ = Mock(return_value=None)

            # Should not raise, just skip the problematic email
            result = collector.collect_feedback(days_back=1)

        assert result['bot_actions'] == 1
        assert result['false_positives'] == 0  # Couldn't determine, so not counted
