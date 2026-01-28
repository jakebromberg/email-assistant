"""Gmail API client for fetching emails."""

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..utils import get_logger
from .auth import GmailAuthenticator
from .models import Email
from .rate_limiter import QuotaCosts, RateLimiter

logger = get_logger(__name__)


@dataclass
class _BatchState:
    """Internal state for batch request processing."""
    batch_results: list[tuple[int, Any]]
    failed_messages: list[tuple[int, str]]
    rate_limit_hit: bool = False
    retry_after_seconds: int | None = None


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

    def __init__(
        self,
        authenticator: GmailAuthenticator,
        enable_rate_limiting: bool = True,
        enable_adaptive_sizing: bool = True
    ):
        """
        Initialize Gmail API client.

        Args:
            authenticator: GmailAuthenticator instance for OAuth
            enable_rate_limiting: Enable smart quota-aware rate limiting
            enable_adaptive_sizing: Enable adaptive batch sizing

        Example:
            >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
            >>> client = GmailClient(auth)
        """
        self.authenticator = authenticator
        self.service = None
        self._build_service()

        # Initialize smart rate limiter
        self.rate_limiter = None
        if enable_rate_limiting:
            self.rate_limiter = RateLimiter(
                quota_per_second=250,
                initial_batch_size=50,
                enable_adaptive_sizing=enable_adaptive_sizing
            )
            logger.info("Smart rate limiting enabled")

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
        label_ids: list[str] | None = None,
        after_date: datetime | None = None,
        before_date: datetime | None = None,
        page_token: str | None = None
    ) -> tuple[list[str], str | None]:
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
            params: dict[str, Any] = {
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
        label_ids: list[str] | None = None,
        after_date: datetime | None = None,
        before_date: datetime | None = None,
        max_total: int | None = None
    ) -> list[str]:
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

    def _handle_batch_error(
        self,
        exception: Exception,
        msg_id: str,
        state: _BatchState,
        index: int
    ):
        """
        Handle error from batch request callback.

        Args:
            exception: The exception that occurred
            msg_id: Message ID that failed
            state: Batch state to update
            index: Index of failed message
        """
        status_code = getattr(getattr(exception, 'resp', None), 'status', 'unknown')
        error_msg = str(exception)

        # Check if it's a rate limit error
        if hasattr(exception, 'resp') and exception.resp.status == 429:
            state.rate_limit_hit = True

            # Parse Retry-After header
            if hasattr(exception.resp, 'get'):
                retry_after = exception.resp.get('Retry-After')
                if retry_after:
                    try:
                        state.retry_after_seconds = int(retry_after)
                    except (ValueError, TypeError):
                        pass

            logger.debug(f"Rate limit (429) for message {msg_id[:10]}...")
            state.failed_messages.append((index, msg_id))
        else:
            # Log non-rate-limit errors with full details
            logger.error(
                f"Failed to fetch message {msg_id[:10]}... - "
                f"Status: {status_code}, Error: {error_msg[:200]}"
            )
            state.failed_messages.append((index, msg_id))

    def _create_batch_callback(self, index: int, msg_id: str, state: _BatchState):
        """
        Create callback for batch request.

        Args:
            index: Index of message in batch
            msg_id: Gmail message ID
            state: Batch state to update

        Returns:
            Callback function for batch request
        """
        def callback(request_id, response, exception):
            if exception:
                self._handle_batch_error(exception, msg_id, state, index)
            else:
                state.batch_results.append((index, response))
        return callback

    def _execute_batch_with_retry(
        self,
        batch_ids: list[str],
        message_format: str,
        max_retries: int = 3
    ) -> tuple[list[tuple[int, Any]], bool, int | None]:
        """
        Execute batch request with retry logic.

        Args:
            batch_ids: List of message IDs to fetch
            message_format: Message format (full/metadata/minimal)
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (results, rate_limit_hit, retry_after_seconds)
        """
        state = _BatchState(batch_results=[], failed_messages=[])

        for retry in range(max_retries):
            # Create batch request
            batch = self.service.new_batch_http_request()

            # Determine which messages to request
            if retry == 0:
                requests_to_add = list(enumerate(batch_ids))
            else:
                requests_to_add = state.failed_messages
                if requests_to_add:
                    logger.info(
                        f"Retrying {len(state.failed_messages)} failed messages "
                        f"(attempt {retry + 1}/{max_retries})"
                    )

            state.failed_messages = []  # Reset for this retry

            # Add requests to batch
            for idx, msg_id in requests_to_add:
                batch.add(
                    self.service.users().messages().get(
                        userId='me',
                        id=msg_id,
                        format=message_format
                    ),
                    callback=self._create_batch_callback(idx, msg_id, state)
                )

            # Execute batch
            try:
                batch.execute()
            except Exception as e:
                logger.warning(f"Batch execution failed: {e}")
                if retry < max_retries - 1:
                    time.sleep(2.0 * (retry + 1))
                    continue

            # If no failures, we're done
            if not state.failed_messages:
                break

            # Wait before retrying failed messages
            if retry < max_retries - 1 and state.failed_messages:
                wait_time = 2.0 * (2 ** retry)  # Exponential backoff
                logger.info(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

        return state.batch_results, state.rate_limit_hit, state.retry_after_seconds

    def _parse_batch_results(self, batch_results: list[tuple[int, Any]]) -> list[Email]:
        """
        Parse batch results into Email objects.

        Args:
            batch_results: List of (index, response) tuples

        Returns:
            List of parsed Email objects
        """
        emails = []
        batch_results.sort(key=lambda x: x[0])  # Sort by original index

        for _, response in batch_results:
            try:
                email = Email.from_gmail_message(response)
                emails.append(email)
            except Exception as e:
                logger.warning(f"Failed to parse email: {e}")

        return emails

    def get_messages_batch(
        self,
        message_ids: list[str],
        batch_size: int | None = None,
        message_format: Literal['full', 'metadata', 'minimal'] = 'full'
    ) -> list[Email]:
        """
        Fetch multiple messages efficiently using batch requests with smart rate limiting.

        Args:
            message_ids: List of message IDs to fetch
            batch_size: Number of messages per batch (None = use adaptive sizing)
            message_format: Message format - 'full' (5 quota), 'metadata' (2 quota), 'minimal' (1 quota)

        Returns:
            List of Email objects

        Example:
            >>> message_ids = client.list_messages(max_results=50)[0]
            >>> emails = client.get_messages_batch(message_ids)
            >>> print(f"Fetched {len(emails)} emails")
        """
        if not message_ids:
            return []

        # Determine quota cost per message based on format
        quota_cost_map = {
            'full': QuotaCosts.MESSAGE_GET_FULL,
            'metadata': QuotaCosts.MESSAGE_GET_METADATA,
            'minimal': QuotaCosts.MESSAGE_GET_MINIMAL
        }
        cost_per_message = quota_cost_map[message_format]

        # Use adaptive batch size or specified size
        if batch_size is None and self.rate_limiter:
            current_batch_size = self.rate_limiter.get_recommended_batch_size()
        else:
            current_batch_size = batch_size or 50

        current_batch_size = min(current_batch_size, 100)  # Gmail API batch limit

        emails = []
        total_batches = math.ceil(len(message_ids) / current_batch_size)

        # Process in batches
        i = 0
        batch_num = 0

        while i < len(message_ids):
            batch_num += 1
            batch_ids = message_ids[i:i + current_batch_size]

            logger.info(
                f"Processing batch {batch_num}/{total_batches}: "
                f"{len(batch_ids)} messages (format={message_format}, cost={cost_per_message * len(batch_ids)} quota)"
            )

            # Wait for quota availability
            if self.rate_limiter:
                wait_time = self.rate_limiter.wait_for_batch(len(batch_ids), cost_per_message)
                if wait_time > 0:
                    logger.info(f"Waited {wait_time:.2f}s for quota availability")

            # Execute batch with retry logic
            batch_results, rate_limit_hit, retry_after_seconds = self._execute_batch_with_retry(
                batch_ids, message_format
            )

            # Handle rate limit
            if rate_limit_hit and self.rate_limiter:
                new_batch_size = self.rate_limiter.on_rate_limit(retry_after_seconds)
                if new_batch_size:
                    current_batch_size = new_batch_size
                    logger.info(f"Adjusted batch size to {current_batch_size}")

            # Consume quota for successful requests
            if self.rate_limiter:
                self.rate_limiter.consume_batch(len(batch_results), cost_per_message)

            # Parse results into Email objects
            batch_emails = self._parse_batch_results(batch_results)
            emails.extend(batch_emails)

            success_count = len(batch_results)
            total_requested = len(batch_ids)

            # Record success/failure for adaptive sizing
            if self.rate_limiter and success_count == total_requested:
                new_batch_size = self.rate_limiter.on_batch_success()
                if new_batch_size and new_batch_size != current_batch_size:
                    current_batch_size = min(new_batch_size, 100)
                    logger.info(f"Increased batch size to {current_batch_size}")

            logger.info(
                f"Batch {batch_num}/{total_batches} complete: "
                f"{success_count}/{total_requested} emails fetched "
                f"(total: {len(emails)}/{min(i + current_batch_size, len(message_ids))})"
            )

            # Move to next batch
            i += current_batch_size
            # Recalculate total batches with new batch size
            total_batches = batch_num + math.ceil((len(message_ids) - i) / current_batch_size)

        success_rate = (len(emails) / len(message_ids) * 100) if message_ids else 0
        logger.info(
            f"Fetch complete: {len(emails)}/{len(message_ids)} emails "
            f"({success_rate:.1f}% success rate)"
        )

        # Log rate limiter stats
        if self.rate_limiter:
            self.rate_limiter.log_stats()

        return emails

    def get_all_labels(self) -> list[dict[str, str]]:
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
