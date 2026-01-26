"""Email data models for parsing Gmail API responses."""

import base64
import email
from dataclasses import dataclass
from datetime import datetime
from email.header import decode_header
from typing import List, Optional, Dict, Any


@dataclass
class Email:
    """
    Email message parsed from Gmail API response.

    Represents a complete email with metadata, headers, and body content.

    Attributes:
        message_id: Unique Gmail message ID
        thread_id: Gmail thread ID
        from_address: Sender email address
        from_name: Sender display name
        to_address: Recipient email address
        subject: Email subject line
        date: Email date/time
        snippet: Short preview text
        labels: List of Gmail label IDs
        body_plain: Plain text body
        body_html: HTML body
        headers: Full email headers

    Example:
        >>> message = gmail_service.users().messages().get(userId='me', id='123').execute()
        >>> email = Email.from_gmail_message(message)
        >>> print(f"{email.from_name}: {email.subject}")
    """

    message_id: str
    thread_id: str
    from_address: str
    from_name: str
    to_address: str
    subject: str
    date: datetime
    snippet: str
    labels: List[str]
    body_plain: str
    body_html: str
    headers: Dict[str, str]

    @classmethod
    def from_gmail_message(cls, message: Dict[str, Any]) -> 'Email':
        """
        Parse a Gmail API message response into an Email object.

        Args:
            message: Gmail API message resource (from messages().get())

        Returns:
            Parsed Email object

        Example:
            >>> message = gmail_service.users().messages().get(
            ...     userId='me',
            ...     id='msg123',
            ...     format='full'
            ... ).execute()
            >>> email = Email.from_gmail_message(message)
        """
        # Extract headers
        headers = {}
        for header in message.get('payload', {}).get('headers', []):
            headers[header['name'].lower()] = header['value']

        # Parse basic fields
        message_id = message['id']
        thread_id = message['threadId']
        snippet = message.get('snippet', '')
        labels = message.get('labelIds', [])

        # Parse sender
        from_header = headers.get('from', '')
        from_name, from_address = cls._parse_email_address(from_header)

        # Parse recipient
        to_header = headers.get('to', '')
        to_name, to_address = cls._parse_email_address(to_header)

        # Parse subject
        subject = headers.get('subject', '(No Subject)')

        # Parse date
        date_str = headers.get('date', '')
        date = cls._parse_date(date_str)

        # Extract body content
        body_plain, body_html = cls._extract_body(message.get('payload', {}))

        return cls(
            message_id=message_id,
            thread_id=thread_id,
            from_address=from_address,
            from_name=from_name,
            to_address=to_address,
            subject=subject,
            date=date,
            snippet=snippet,
            labels=labels,
            body_plain=body_plain,
            body_html=body_html,
            headers=headers
        )

    @staticmethod
    def _parse_email_address(address_header: str) -> tuple[str, str]:
        """
        Parse email address header into name and address.

        Args:
            address_header: Email header value (e.g., "John Doe <john@example.com>")

        Returns:
            Tuple of (name, address)

        Example:
            >>> Email._parse_email_address("John Doe <john@example.com>")
            ('John Doe', 'john@example.com')
        """
        if not address_header:
            return '', ''

        # Try to parse with email.utils
        from email.utils import parseaddr
        name, address = parseaddr(address_header)

        # Decode name if needed
        if name:
            try:
                decoded_parts = decode_header(name)
                name_parts = []
                for part, encoding in decoded_parts:
                    if isinstance(part, bytes):
                        name_parts.append(
                            part.decode(encoding or 'utf-8', errors='replace')
                        )
                    else:
                        name_parts.append(part)
                name = ''.join(name_parts)
            except Exception:
                pass  # Keep original name if decode fails

        return name, address

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """
        Parse email date string to datetime.

        Args:
            date_str: Date header value

        Returns:
            Parsed datetime object (timezone-naive for database compatibility)

        Example:
            >>> Email._parse_date("Mon, 1 Jan 2024 12:00:00 +0000")
            datetime.datetime(2024, 1, 1, 12, 0, 0)
        """
        if not date_str:
            return datetime.now()

        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            # Strip timezone info for database compatibility
            # All datetimes in database are timezone-naive (implicitly UTC)
            return dt.replace(tzinfo=None)
        except Exception:
            # Fallback to current time if parsing fails
            return datetime.now()

    @staticmethod
    def _extract_body(payload: Dict[str, Any]) -> tuple[str, str]:
        """
        Extract plain text and HTML body from email payload.

        Args:
            payload: Gmail message payload

        Returns:
            Tuple of (plain_text, html)

        Example:
            >>> payload = message['payload']
            >>> plain, html = Email._extract_body(payload)
        """
        def decode_body(data: str) -> str:
            """Decode base64url encoded body."""
            if not data:
                return ''
            try:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
            except Exception:
                return ''

        def extract_parts(part: Dict[str, Any]) -> tuple[str, str]:
            """
            Recursively extract body parts.

            Returns:
                Tuple of (plain_text, html) from this part and its children
            """
            body_plain = ''
            body_html = ''

            mime_type = part.get('mimeType', '')
            body = part.get('body', {})
            data = body.get('data', '')

            # Extract from this part
            if mime_type == 'text/plain' and data and not body_plain:
                body_plain = decode_body(data)
            elif mime_type == 'text/html' and data and not body_html:
                body_html = decode_body(data)

            # Recurse into child parts (only if we haven't found what we need)
            for subpart in part.get('parts', []):
                sub_plain, sub_html = extract_parts(subpart)
                if not body_plain:
                    body_plain = sub_plain
                if not body_html:
                    body_html = sub_html
                # Early exit if we've found both
                if body_plain and body_html:
                    break

            return body_plain, body_html

        # Start extraction
        return extract_parts(payload)

    @property
    def is_unread(self) -> bool:
        """
        Check if email is unread.

        Returns:
            True if email has UNREAD label

        Example:
            >>> if email.is_unread:
            ...     print("This email hasn't been read yet")
        """
        return 'UNREAD' in self.labels

    @property
    def is_in_inbox(self) -> bool:
        """
        Check if email is in inbox.

        Returns:
            True if email has INBOX label

        Example:
            >>> if email.is_in_inbox:
            ...     print("This email is in the inbox")
        """
        return 'INBOX' in self.labels

    @property
    def is_important(self) -> bool:
        """
        Check if email is marked important.

        Returns:
            True if email has IMPORTANT label

        Example:
            >>> if email.is_important:
            ...     print("This email is marked important")
        """
        return 'IMPORTANT' in self.labels

    @property
    def is_starred(self) -> bool:
        """
        Check if email is starred.

        Returns:
            True if email has STARRED label

        Example:
            >>> if email.is_starred:
            ...     print("This email is starred")
        """
        return 'STARRED' in self.labels

    def __repr__(self) -> str:
        """Return string representation of email."""
        return (
            f"Email(id={self.message_id[:10]}..., "
            f"from={self.from_address}, "
            f"subject='{self.subject[:50]}...', "
            f"date={self.date.isoformat()})"
        )
