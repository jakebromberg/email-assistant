"""Gmail API client for fetching emails."""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .auth import GmailAuthenticator
from .models import Email
from ..utils import get_logger


logger = get_logger(__name__)


class GmailClient:
    """
    Main Gmail API wrapper for fetching emails.

    Provides methods to list, fetch, and batch fetch emails with support
    for filters, pagination, and rate limiting.

    Attributes:
        authenticator: GmailAuthenticator instance
        service: Gmail API service object

    Example:
        >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
        >>> client = GmailClient(auth)
        >>> message_ids = client.list_messages(max_results=10)
        >>> emails = client.get_messages_batch(message_ids)
    """

    def __init__(self, authenticator: GmailAuthenticator):
        """
        Initialize Gmail API client.

        Args:
            authenticator: GmailAuthenticator instance for OAuth

        Example:
            >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
            >>> client = GmailClient(auth)
        """
        self.authenticator = authenticator
        self.service = None
        self._build_service()

    def _build_service(self) -> None:
        """
        Build Gmail API service object.

        Raises:
            Exception: If service creation fails
        """
        try:
            creds = self.authenticator.authenticate()
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail API service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to build Gmail service: {e}")
            raise

    def list_messages(
        self,
        query: str = "",
        max_results: int = 100,
        label_ids: Optional[List[str]] = None,
        after_date: Optional[datetime] = None,
        before_date: Optional[datetime] = None,
        page_token: Optional[str] = None
    ) -> tuple[List[str], Optional[str]]:
        """
        List message IDs matching criteria.

        Args:
            query: Gmail search query (e.g., "is:unread", "from:example@gmail.com")
            max_results: Maximum messages to return (up to 500)
            label_ids: Filter by label IDs (e.g., ['INBOX', 'UNREAD'])
            after_date: Only messages after this date
            before_date: Only messages before this date
            page_token: Token for pagination

        Returns:
            Tuple of (list of message IDs, next page token)

        Example:
            >>> # Get unread inbox messages
            >>> ids, next_token = client.list_messages(
            ...     query="is:unread",
            ...     label_ids=["INBOX"],
            ...     max_results=50
            ... )
            >>> print(f"Found {len(ids)} unread messages")
        """
        # Build query with date filters
        query_parts = [query] if query else []

        if after_date:
            # Gmail uses format: YYYY/MM/DD
            date_str = after_date.strftime('%Y/%m/%d')
            query_parts.append(f"after:{date_str}")

        if before_date:
            date_str = before_date.strftime('%Y/%m/%d')
            query_parts.append(f"before:{date_str}")

        full_query = ' '.join(query_parts)

        try:
            logger.debug(f"Listing messages: query='{full_query}', max={max_results}")

            # Build request parameters
            params: Dict[str, Any] = {
                'userId': 'me',
                'maxResults': min(max_results, 500),  # API limit is 500
            }

            if full_query:
                params['q'] = full_query

            if label_ids:
                params['labelIds'] = label_ids

            if page_token:
                params['pageToken'] = page_token

            # Make API request with retry
            response = self._execute_with_retry(
                self.service.users().messages().list(**params)
            )

            message_ids = [msg['id'] for msg in response.get('messages', [])]
            next_page_token = response.get('nextPageToken')

            logger.info(
                f"Listed {len(message_ids)} messages "
                f"(has_more={next_page_token is not None})"
            )

            return message_ids, next_page_token

        except HttpError as e:
            logger.error(f"Failed to list messages: {e}")
            raise

    def list_all_messages(
        self,
        query: str = "",
        label_ids: Optional[List[str]] = None,
        after_date: Optional[datetime] = None,
        before_date: Optional[datetime] = None,
        max_total: Optional[int] = None
    ) -> List[str]:
        """
        List all message IDs matching criteria (handles pagination).

        Args:
            query: Gmail search query
            label_ids: Filter by label IDs
            after_date: Only messages after this date
            before_date: Only messages before this date
            max_total: Maximum total messages to fetch (None for unlimited)

        Returns:
            List of all message IDs

        Example:
            >>> # Get all emails from last 6 months
            >>> from datetime import datetime, timedelta
            >>> six_months_ago = datetime.now() - timedelta(days=180)
            >>> all_ids = client.list_all_messages(after_date=six_months_ago)
            >>> print(f"Found {len(all_ids)} emails in last 6 months")
        """
        all_message_ids = []
        page_token = None

        while True:
            # Determine batch size
            if max_total:
                remaining = max_total - len(all_message_ids)
                if remaining <= 0:
                    break
                batch_size = min(remaining, 500)
            else:
                batch_size = 500

            # Fetch batch
            message_ids, page_token = self.list_messages(
                query=query,
                max_results=batch_size,
                label_ids=label_ids,
                after_date=after_date,
                before_date=before_date,
                page_token=page_token
            )

            all_message_ids.extend(message_ids)

            # Check if done
            if not page_token or not message_ids:
                break

        logger.info(f"Listed total of {len(all_message_ids)} messages")
        return all_message_ids

    def get_message(self, message_id: str) -> Email:
        """
        Fetch a single message by ID.

        Args:
            message_id: Gmail message ID

        Returns:
            Parsed Email object

        Example:
            >>> email = client.get_message("msg123abc")
            >>> print(f"Subject: {email.subject}")
        """
        try:
            logger.debug(f"Fetching message: {message_id}")

            response = self._execute_with_retry(
                self.service.users().messages().get(
                    userId='me',
                    id=message_id,
                    format='full'
                )
            )

            email = Email.from_gmail_message(response)
            logger.debug(f"Fetched message: {email.subject[:50]}")

            return email

        except HttpError as e:
            logger.error(f"Failed to fetch message {message_id}: {e}")
            raise

    def get_messages_batch(
        self,
        message_ids: List[str],
        batch_size: int = 100
    ) -> List[Email]:
        """
        Fetch multiple messages efficiently using batch requests.

        Args:
            message_ids: List of message IDs to fetch
            batch_size: Number of messages per batch (max 100)

        Returns:
            List of Email objects

        Example:
            >>> message_ids = client.list_messages(max_results=50)[0]
            >>> emails = client.get_messages_batch(message_ids)
            >>> print(f"Fetched {len(emails)} emails")
        """
        if not message_ids:
            return []

        emails = []
        batch_size = min(batch_size, 100)  # Gmail API batch limit

        # Process in batches
        for i in range(0, len(message_ids), batch_size):
            batch_ids = message_ids[i:i + batch_size]
            logger.debug(
                f"Fetching batch {i // batch_size + 1}: "
                f"{len(batch_ids)} messages"
            )

            # Create batch request
            batch = self.service.new_batch_http_request()

            # Store results
            batch_results = []

            def create_callback(index):
                """Create callback for batch request."""
                def callback(request_id, response, exception):
                    if exception:
                        logger.warning(
                            f"Failed to fetch message in batch: {exception}"
                        )
                    else:
                        batch_results.append((index, response))
                return callback

            # Add requests to batch
            for idx, msg_id in enumerate(batch_ids):
                batch.add(
                    self.service.users().messages().get(
                        userId='me',
                        id=msg_id,
                        format='full'
                    ),
                    callback=create_callback(idx)
                )

            # Execute batch with retry
            self._execute_with_retry(batch)

            # Parse results in original order
            batch_results.sort(key=lambda x: x[0])
            for _, response in batch_results:
                try:
                    email = Email.from_gmail_message(response)
                    emails.append(email)
                except Exception as e:
                    logger.warning(f"Failed to parse email: {e}")

            logger.info(f"Fetched batch: {len(batch_results)} emails")

        logger.info(f"Fetched total of {len(emails)} emails")
        return emails

    def get_all_labels(self) -> List[Dict[str, str]]:
        """
        Get all Gmail labels.

        Returns:
            List of label dictionaries with 'id' and 'name' keys

        Example:
            >>> labels = client.get_all_labels()
            >>> for label in labels:
            ...     print(f"{label['name']}: {label['id']}")
        """
        try:
            logger.debug("Fetching all labels")

            response = self._execute_with_retry(
                self.service.users().labels().list(userId='me')
            )

            labels = response.get('labels', [])
            logger.info(f"Fetched {len(labels)} labels")

            return labels

        except HttpError as e:
            logger.error(f"Failed to fetch labels: {e}")
            raise

    def _execute_with_retry(
        self,
        request,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ) -> Any:
        """
        Execute API request with exponential backoff retry.

        Args:
            request: Gmail API request object
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds

        Returns:
            API response

        Raises:
            HttpError: If all retries fail
        """
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                return request.execute()

            except HttpError as e:
                # Check if we should retry
                if e.resp.status in [429, 500, 503]:
                    if attempt < max_retries:
                        logger.warning(
                            f"API error {e.resp.status}, "
                            f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        continue

                # Don't retry other errors or if max retries exceeded
                raise

        # Should not reach here
        raise HttpError(resp=None, content=b"Max retries exceeded")
