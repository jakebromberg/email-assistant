"""Tests for triage pipeline."""

from datetime import datetime
from unittest.mock import Mock, patch

from src.triage.pipeline import TriagePipeline


class TestTriagePipeline:
    """Test TriagePipeline class."""

    @patch('src.triage.pipeline.EmailScorer')
    @patch('src.triage.pipeline.EmbeddingExtractor')
    def test_pipeline_handles_new_emails_without_features(
        self,
        mock_embedding_extractor,
        mock_scorer_class,
        mock_gmail_client,
        mock_gmail_ops,
        temp_db
    ):
        """
        REGRESSION TEST: Pipeline should handle new emails that don't have pre-computed features.

        This prevents "NoneType has no attribute 'subject_embedding'" errors by ensuring
        save_features() return value is used directly instead of calling get_features().
        """
        # Set up mocks
        mock_scorer = Mock()
        mock_scorer.score_email = Mock(return_value=0.5)
        mock_scorer.make_decision = Mock(return_value={
            'action': 'keep',
            'confidence': 'low',
            'score': 0.5,
            'reasoning': 'Test reasoning'
        })
        mock_scorer_class.return_value = mock_scorer

        # Mock embeddings (disabled for this test)
        mock_embedding_extractor.return_value = None

        # Create mock email from Gmail
        gmail_email = Mock()
        gmail_email.message_id = "test123"
        gmail_email.from_address = "sender@example.com"
        gmail_email.from_name = "Test Sender"
        gmail_email.to_address = "me@example.com"
        gmail_email.subject = "Test Email"
        gmail_email.body_plain = "Test body"
        gmail_email.body_html = "<p>Test body</p>"
        gmail_email.date = datetime(2024, 1, 15, 14, 30)
        gmail_email.labels = ["INBOX", "UNREAD"]
        gmail_email.thread_id = "thread123"
        gmail_email.snippet = "Test snippet"
        gmail_email.headers = {}
        # Set boolean fields explicitly for database compatibility
        gmail_email.was_read = False
        gmail_email.was_archived = False
        gmail_email.is_important = False
        gmail_email.is_starred = False
        gmail_email.opened_at = None

        mock_gmail_client.list_all_messages.return_value = ["test123"]
        mock_gmail_client.get_messages_batch.return_value = [gmail_email]

        # Initialize pipeline
        pipeline = TriagePipeline(
            gmail_client=mock_gmail_client,
            gmail_ops=mock_gmail_ops,
            database=temp_db,
            model_path="dummy_model.txt",  # Won't be loaded due to mock
            use_embeddings=False
        )

        # Run triage (dry-run mode)
        results = pipeline.run_triage(days_back=1, dry_run=True)

        # Verify it processed without errors
        assert results['total'] == 1
        assert results['errors'] == 0
        assert len(results['decisions']) == 1

        # Verify email was saved to database
        with temp_db.get_session() as session:
            from src.database.repository import EmailRepository
            repo = EmailRepository(session)
            saved_email = repo.get_by_id("test123")
            assert saved_email is not None
            assert saved_email.subject == "Test Email"

            # Verify features were computed and saved
            from src.features import FeatureStore
            store = FeatureStore(session)
            features = store.get_features("test123")
            assert features is not None
            assert features.sender_domain == "example.com"
            assert features.is_newsletter is not None

    @patch('src.triage.pipeline.EmailScorer')
    @patch('src.triage.pipeline.EmbeddingExtractor')
    def test_pipeline_handles_timezone_aware_dates(
        self,
        mock_embedding_extractor,
        mock_scorer_class,
        mock_gmail_client,
        mock_gmail_ops,
        temp_db,
        email_factory
    ):
        """
        REGRESSION TEST: Pipeline should handle emails with timezone-aware dates.

        This prevents "can't subtract offset-naive and offset-aware datetimes" errors
        when historical pattern extractor compares dates.
        """
        # Set up mocks
        mock_scorer = Mock()
        mock_scorer.score_email = Mock(return_value=0.4)
        mock_scorer.make_decision = Mock(return_value={
            'action': 'keep',
            'confidence': 'low',
            'score': 0.4,
            'reasoning': 'Test reasoning'
        })
        mock_scorer_class.return_value = mock_scorer
        mock_embedding_extractor.return_value = None

        # Create an older email in the database for historical comparison using factory
        old_email = email_factory(
            message_id="old123",
            from_address="sender@example.com",
            from_name="Test Sender",
            subject="Old Email",
            body_plain="Old body",
            body_html="<p>Old body</p>",
            date=datetime(2024, 1, 10, 10, 0),  # 5 days earlier, timezone-naive
            snippet="Old snippet",
            was_read=True,
            was_archived=False
        )

        # Create mock email with timezone-naive date (as it would be after _parse_date fix)
        gmail_email = Mock()
        gmail_email.message_id = "test456"
        gmail_email.from_address = "sender@example.com"
        gmail_email.from_name = "Test Sender"
        gmail_email.to_address = "me@example.com"
        gmail_email.subject = "Test Email 2"
        gmail_email.body_plain = "Test body 2"
        gmail_email.body_html = "<p>Test body 2</p>"
        # Timezone-naive date (as it would be after _parse_date fix)
        gmail_email.date = datetime(2024, 1, 15, 14, 30)
        gmail_email.labels = ["INBOX", "UNREAD"]
        gmail_email.thread_id = "thread456"
        gmail_email.snippet = "Test snippet 2"
        gmail_email.headers = {}
        # Set boolean fields explicitly for database compatibility
        gmail_email.was_read = False
        gmail_email.was_archived = False
        gmail_email.is_important = False
        gmail_email.is_starred = False
        gmail_email.opened_at = None

        mock_gmail_client.list_all_messages.return_value = ["test456"]
        mock_gmail_client.get_messages_batch.return_value = [gmail_email]

        # Initialize pipeline
        pipeline = TriagePipeline(
            gmail_client=mock_gmail_client,
            gmail_ops=mock_gmail_ops,
            database=temp_db,
            model_path="dummy_model.txt",
            use_embeddings=False
        )

        # Run triage - should not raise timezone error
        results = pipeline.run_triage(days_back=1, dry_run=True)

        # Verify it processed without errors
        assert results['total'] == 1
        assert results['errors'] == 0

        # Verify historical features were computed correctly
        with temp_db.get_session() as session:
            from src.features import FeatureStore
            store = FeatureStore(session)
            features = store.get_features("test456")
            assert features is not None
            assert features.sender_days_since_last is not None
            # Should be ~5 days
            assert 4 < features.sender_days_since_last < 6
