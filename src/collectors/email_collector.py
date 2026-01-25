"""Historical email collector for database export."""

from datetime import datetime, timedelta
from typing import Optional, List

from ..gmail import GmailClient
from ..database import Database, EmailRepository
from ..utils import get_logger


logger = get_logger(__name__)


class EmailCollector:
    """
    Collector for exporting historical emails to database.

    Fetches emails from Gmail API and stores them in the local database
    with all metadata and behavioral signals.

    Attributes:
        client: GmailClient for API access
        database: Database instance

    Example:
        >>> from src.gmail import GmailAuthenticator, GmailClient
        >>> from src.database import Database
        >>>
        >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
        >>> client = GmailClient(auth)
        >>> db = Database()
        >>>
        >>> collector = EmailCollector(client, db)
        >>> collector.export_historical(months=6)
    """

    def __init__(self, client: GmailClient, database: Database):
        """
        Initialize email collector.

        Args:
            client: Gmail API client
            database: Database instance

        Example:
            >>> collector = EmailCollector(client, db)
        """
        self.client = client
        self.database = database

    def export_historical(
        self,
        months: int = 6,
        max_emails: Optional[int] = None,
        batch_size: int = 100
    ) -> int:
        """
        Export historical emails from Gmail to database.

        Args:
            months: Number of months to go back
            max_emails: Maximum number of emails to export (None for unlimited)
            batch_size: Number of emails to fetch per batch

        Returns:
            Number of emails exported

        Example:
            >>> # Export last 6 months
            >>> count = collector.export_historical(months=6)
            >>> print(f"Exported {count} emails")
        """
        start_date = datetime.now() - timedelta(days=months * 30)

        logger.info(f"Starting historical export from {start_date.date()}")
        logger.info(f"Max emails: {max_emails or 'unlimited'}, batch size: {batch_size}")

        # List all message IDs in date range
        logger.info("Listing message IDs...")
        message_ids = self.client.list_all_messages(
            after_date=start_date,
            max_total=max_emails
        )

        if not message_ids:
            logger.warning("No messages found in date range")
            return 0

        logger.info(f"Found {len(message_ids)} messages to export")

        # Fetch and store in batches
        total_saved = 0
        failed_count = 0

        for i in range(0, len(message_ids), batch_size):
            batch_ids = message_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(message_ids) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} emails)...")

            try:
                # Fetch emails from Gmail
                gmail_emails = self.client.get_messages_batch(batch_ids)

                # Save to database
                with self.database.get_session() as session:
                    repo = EmailRepository(session)
                    saved = repo.save_emails_batch(gmail_emails)
                    total_saved += len(saved)

                logger.info(
                    f"Batch {batch_num}/{total_batches} complete: "
                    f"{len(saved)} emails saved (total: {total_saved})"
                )

            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                failed_count += len(batch_ids)
                continue

        logger.info(
            f"Historical export complete: {total_saved} emails saved, "
            f"{failed_count} failed"
        )

        return total_saved

    def export_recent(
        self,
        days: int = 1,
        batch_size: int = 100
    ) -> int:
        """
        Export recent emails from last N days.

        Args:
            days: Number of days to go back
            batch_size: Number of emails to fetch per batch

        Returns:
            Number of emails exported

        Example:
            >>> # Export last 24 hours
            >>> count = collector.export_recent(days=1)
            >>> print(f"Exported {count} emails")
        """
        start_date = datetime.now() - timedelta(days=days)

        logger.info(f"Exporting recent emails from {start_date.isoformat()}")

        # List message IDs
        message_ids = self.client.list_all_messages(after_date=start_date)

        if not message_ids:
            logger.info("No recent messages found")
            return 0

        logger.info(f"Found {len(message_ids)} recent messages")

        # Fetch and save
        try:
            gmail_emails = self.client.get_messages_batch(message_ids, batch_size=batch_size)

            with self.database.get_session() as session:
                repo = EmailRepository(session)
                saved = repo.save_emails_batch(gmail_emails)

            logger.info(f"Exported {len(saved)} recent emails")
            return len(saved)

        except Exception as e:
            logger.error(f"Failed to export recent emails: {e}")
            raise

    def update_email(self, message_id: str) -> bool:
        """
        Update a single email from Gmail.

        Useful for refreshing an email's current state (labels, read status).

        Args:
            message_id: Gmail message ID

        Returns:
            True if updated successfully

        Example:
            >>> collector.update_email("msg123")
        """
        try:
            logger.debug(f"Updating email: {message_id}")

            # Fetch from Gmail
            gmail_email = self.client.get_message(message_id)

            # Save to database
            with self.database.get_session() as session:
                repo = EmailRepository(session)
                repo.save_email(gmail_email)

            logger.debug(f"Updated email: {message_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update email {message_id}: {e}")
            return False

    def update_emails_batch(self, message_ids: List[str], batch_size: int = 100) -> int:
        """
        Update multiple emails from Gmail.

        Args:
            message_ids: List of Gmail message IDs
            batch_size: Number of emails to fetch per batch

        Returns:
            Number of emails updated

        Example:
            >>> updated = collector.update_emails_batch(["msg1", "msg2", "msg3"])
        """
        if not message_ids:
            return 0

        logger.info(f"Updating {len(message_ids)} emails...")

        total_updated = 0

        for i in range(0, len(message_ids), batch_size):
            batch_ids = message_ids[i:i + batch_size]

            try:
                # Fetch from Gmail
                gmail_emails = self.client.get_messages_batch(batch_ids)

                # Save to database
                with self.database.get_session() as session:
                    repo = EmailRepository(session)
                    saved = repo.save_emails_batch(gmail_emails)
                    total_updated += len(saved)

            except Exception as e:
                logger.error(f"Failed to update batch: {e}")
                continue

        logger.info(f"Updated {total_updated} emails")
        return total_updated

    def get_stats(self) -> dict:
        """
        Get collection statistics.

        Returns:
            Dictionary with database statistics

        Example:
            >>> stats = collector.get_stats()
            >>> print(f"Total emails: {stats['emails']}")
        """
        return self.database.get_stats()
