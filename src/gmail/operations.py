"""Email operations for modifying Gmail messages."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from googleapiclient.errors import HttpError

from .client import GmailClient
from ..utils import get_logger


logger = get_logger(__name__)


@dataclass
class OperationResult:
    """
    Result of a Gmail operation.

    Attributes:
        success: Whether operation succeeded
        message_ids: List of affected message IDs
        message: Human-readable result message
        details: Additional operation details

    Example:
        >>> result = operations.archive(['msg1', 'msg2'])
        >>> if result.success:
        ...     print(f"Archived {len(result.message_ids)} messages")
    """
    success: bool
    message_ids: List[str]
    message: str
    details: Optional[Dict[str, Any]] = None


class GmailOperations:
    """
    Email modification operations (archive, label, mark read/unread).

    Provides methods to modify emails with support for dry-run mode
    and batch operations.

    Attributes:
        client: GmailClient instance

    Example:
        >>> ops = GmailOperations(client)
        >>> result = ops.archive(['msg1', 'msg2'], dry_run=True)
        >>> print(result.message)
    """

    def __init__(self, client: GmailClient):
        """
        Initialize Gmail operations.

        Args:
            client: GmailClient instance

        Example:
            >>> client = GmailClient(auth)
            >>> ops = GmailOperations(client)
        """
        self.client = client
        self._label_cache: Optional[Dict[str, str]] = None

    def mark_read(
        self,
        message_ids: List[str],
        dry_run: bool = False
    ) -> OperationResult:
        """
        Mark messages as read.

        Args:
            message_ids: List of message IDs to mark as read
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.mark_read(['msg1', 'msg2'])
            >>> print(f"Marked {len(result.message_ids)} as read")
        """
        if dry_run:
            logger.info(f"DRY RUN: Would mark {len(message_ids)} messages as read")
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would mark {len(message_ids)} messages as read"
            )

        return self.batch_modify(
            message_ids=message_ids,
            remove_label_ids=['UNREAD']
        )

    def mark_unread(
        self,
        message_ids: List[str],
        dry_run: bool = False
    ) -> OperationResult:
        """
        Mark messages as unread.

        Args:
            message_ids: List of message IDs to mark as unread
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.mark_unread(['msg1', 'msg2'])
            >>> print(f"Marked {len(result.message_ids)} as unread")
        """
        if dry_run:
            logger.info(f"DRY RUN: Would mark {len(message_ids)} messages as unread")
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would mark {len(message_ids)} messages as unread"
            )

        return self.batch_modify(
            message_ids=message_ids,
            add_label_ids=['UNREAD']
        )

    def archive(
        self,
        message_ids: List[str],
        dry_run: bool = False
    ) -> OperationResult:
        """
        Archive messages (remove from inbox).

        Args:
            message_ids: List of message IDs to archive
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.archive(['msg1', 'msg2'])
            >>> print(f"Archived {len(result.message_ids)} messages")
        """
        if dry_run:
            logger.info(f"DRY RUN: Would archive {len(message_ids)} messages")
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would archive {len(message_ids)} messages"
            )

        return self.batch_modify(
            message_ids=message_ids,
            remove_label_ids=['INBOX']
        )

    def move_to_inbox(
        self,
        message_ids: List[str],
        dry_run: bool = False
    ) -> OperationResult:
        """
        Move messages to inbox (unarchive).

        Args:
            message_ids: List of message IDs to move to inbox
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.move_to_inbox(['msg1', 'msg2'])
            >>> print(f"Moved {len(result.message_ids)} to inbox")
        """
        if dry_run:
            logger.info(f"DRY RUN: Would move {len(message_ids)} messages to inbox")
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would move {len(message_ids)} messages to inbox"
            )

        return self.batch_modify(
            message_ids=message_ids,
            add_label_ids=['INBOX']
        )

    def add_labels(
        self,
        message_ids: List[str],
        label_names: List[str],
        dry_run: bool = False
    ) -> OperationResult:
        """
        Add labels to messages (creates labels if needed).

        Args:
            message_ids: List of message IDs
            label_names: List of label names to add
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.add_labels(
            ...     ['msg1', 'msg2'],
            ...     ['Bot/Newsletter-Tech', 'Bot/AutoArchived']
            ... )
        """
        if dry_run:
            logger.info(
                f"DRY RUN: Would add labels {label_names} "
                f"to {len(message_ids)} messages"
            )
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would add labels to {len(message_ids)} messages"
            )

        # Get or create label IDs
        label_ids = []
        for label_name in label_names:
            label_id = self.get_or_create_label(label_name)
            label_ids.append(label_id)

        return self.batch_modify(
            message_ids=message_ids,
            add_label_ids=label_ids
        )

    def remove_labels(
        self,
        message_ids: List[str],
        label_names: List[str],
        dry_run: bool = False
    ) -> OperationResult:
        """
        Remove labels from messages.

        Args:
            message_ids: List of message IDs
            label_names: List of label names to remove
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.remove_labels(
            ...     ['msg1', 'msg2'],
            ...     ['Bot/LowConfidence']
            ... )
        """
        if dry_run:
            logger.info(
                f"DRY RUN: Would remove labels {label_names} "
                f"from {len(message_ids)} messages"
            )
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would remove labels from {len(message_ids)} messages"
            )

        # Get label IDs (skip if label doesn't exist)
        label_ids = []
        label_map = self._get_label_map()
        for label_name in label_names:
            if label_name in label_map:
                label_ids.append(label_map[label_name])

        if not label_ids:
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message="No labels to remove (labels don't exist)"
            )

        return self.batch_modify(
            message_ids=message_ids,
            remove_label_ids=label_ids
        )

    def batch_modify(
        self,
        message_ids: List[str],
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> OperationResult:
        """
        Batch modify messages (add/remove labels).

        Args:
            message_ids: List of message IDs to modify
            add_label_ids: Label IDs to add
            remove_label_ids: Label IDs to remove
            dry_run: If True, don't actually modify messages

        Returns:
            OperationResult with success status

        Example:
            >>> result = ops.batch_modify(
            ...     message_ids=['msg1', 'msg2'],
            ...     add_label_ids=['Label_123'],
            ...     remove_label_ids=['INBOX']
            ... )
        """
        if not message_ids:
            return OperationResult(
                success=True,
                message_ids=[],
                message="No messages to modify"
            )

        if dry_run:
            logger.info(
                f"DRY RUN: Would batch modify {len(message_ids)} messages"
            )
            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"DRY RUN: Would batch modify {len(message_ids)} messages"
            )

        try:
            # Build request body
            body: Dict[str, Any] = {'ids': message_ids}

            if add_label_ids:
                body['addLabelIds'] = add_label_ids

            if remove_label_ids:
                body['removeLabelIds'] = remove_label_ids

            logger.debug(
                f"Batch modifying {len(message_ids)} messages: "
                f"add={add_label_ids}, remove={remove_label_ids}"
            )

            # Execute batch modify
            self.client.service.users().messages().batchModify(
                userId='me',
                body=body
            ).execute()

            logger.info(f"Successfully modified {len(message_ids)} messages")

            return OperationResult(
                success=True,
                message_ids=message_ids,
                message=f"Successfully modified {len(message_ids)} messages",
                details={
                    'added_labels': add_label_ids,
                    'removed_labels': remove_label_ids
                }
            )

        except HttpError as e:
            logger.error(f"Failed to batch modify messages: {e}")
            return OperationResult(
                success=False,
                message_ids=message_ids,
                message=f"Failed to modify messages: {e}"
            )

    def create_label(self, label_name: str) -> str:
        """
        Create a new Gmail label.

        Args:
            label_name: Name of label to create

        Returns:
            Label ID

        Raises:
            HttpError: If label creation fails

        Example:
            >>> label_id = ops.create_label("Bot/Newsletter-Tech")
            >>> print(f"Created label: {label_id}")
        """
        try:
            logger.debug(f"Creating label: {label_name}")

            label_object = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }

            result = self.client.service.users().labels().create(
                userId='me',
                body=label_object
            ).execute()

            label_id = result['id']
            logger.info(f"Created label '{label_name}': {label_id}")

            # Invalidate cache
            self._label_cache = None

            return label_id

        except HttpError as e:
            logger.error(f"Failed to create label '{label_name}': {e}")
            raise

    def get_or_create_label(self, label_name: str) -> str:
        """
        Get label ID, creating the label if it doesn't exist.

        Args:
            label_name: Name of label

        Returns:
            Label ID

        Example:
            >>> label_id = ops.get_or_create_label("Bot/Newsletter-Tech")
        """
        # Check cache first
        label_map = self._get_label_map()

        if label_name in label_map:
            return label_map[label_name]

        # Create new label
        label_id = self.create_label(label_name)
        return label_id

    def _get_label_map(self) -> Dict[str, str]:
        """
        Get mapping of label names to IDs.

        Returns:
            Dictionary mapping label name to label ID

        Example:
            >>> label_map = ops._get_label_map()
            >>> inbox_id = label_map['INBOX']
        """
        if self._label_cache is None:
            labels = self.client.get_all_labels()
            self._label_cache = {
                label['name']: label['id']
                for label in labels
            }

        return self._label_cache

    def get_label_id(self, label_name: str) -> Optional[str]:
        """
        Get label ID by name (returns None if doesn't exist).

        Args:
            label_name: Name of label

        Returns:
            Label ID or None

        Example:
            >>> label_id = ops.get_label_id("INBOX")
            >>> if label_id:
            ...     print(f"INBOX label ID: {label_id}")
        """
        label_map = self._get_label_map()
        return label_map.get(label_name)
