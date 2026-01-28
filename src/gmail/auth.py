"""Gmail OAuth2 authentication and token management."""

import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


class GmailAuthenticator:
    """
    Handle Gmail OAuth2 authentication flow and token management.

    Manages the OAuth2 authentication process, including initial authentication
    via browser redirect and automatic token refresh.

    Attributes:
        credentials_path: Path to OAuth2 credentials file from Google Cloud Console
        token_path: Path to store access/refresh tokens
        scopes: Gmail API scopes to request

    Example:
        >>> auth = GmailAuthenticator(
        ...     credentials_path="credentials/credentials.json",
        ...     token_path="credentials/token.json",
        ...     scopes=["https://www.googleapis.com/auth/gmail.modify"]
        ... )
        >>> creds = auth.authenticate()
    """

    # Gmail API scopes
    SCOPE_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
    SCOPE_MODIFY = "https://www.googleapis.com/auth/gmail.modify"
    SCOPE_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
    SCOPE_SEND = "https://www.googleapis.com/auth/gmail.send"

    def __init__(
        self,
        credentials_path: str,
        token_path: str,
        scopes: list[str] | None = None
    ):
        """
        Initialize the Gmail authenticator.

        Args:
            credentials_path: Path to OAuth credentials from Google Cloud Console
            token_path: Path to store access tokens
            scopes: List of Gmail API scopes (defaults to gmail.modify)

        Raises:
            FileNotFoundError: If credentials_path doesn't exist
        """
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.scopes = scopes or [self.SCOPE_MODIFY]

        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"Gmail credentials not found at {self.credentials_path}. "
                f"Download OAuth credentials from Google Cloud Console."
            )

        # Ensure token directory exists
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

    def authenticate(self) -> Credentials:
        """
        Authenticate with Gmail API using OAuth2.

        If valid tokens exist, they will be used. Otherwise, initiates
        the OAuth2 flow via browser redirect.

        Returns:
            Valid Google OAuth2 credentials

        Raises:
            Exception: If authentication fails

        Example:
            >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
            >>> creds = auth.authenticate()
            >>> print(f"Authenticated: {creds.valid}")
        """
        creds = None

        # Try to load existing token
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self.token_path),
                self.scopes
            )

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # Refresh expired token
                creds = self.refresh_token(creds)
            else:
                # Run OAuth flow
                creds = self._run_oauth_flow()

            # Save the credentials for future use
            self._save_credentials(creds)

        return creds

    def get_credentials(self) -> Credentials | None:
        """
        Get existing credentials without triggering OAuth flow.

        Returns:
            Credentials if they exist and are valid, None otherwise

        Example:
            >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
            >>> creds = auth.get_credentials()
            >>> if creds is None:
            ...     creds = auth.authenticate()
        """
        if not self.token_path.exists():
            return None

        try:
            creds = Credentials.from_authorized_user_file(
                str(self.token_path),
                self.scopes
            )

            # Refresh if expired
            if creds and creds.expired and creds.refresh_token:
                creds = self.refresh_token(creds)
                self._save_credentials(creds)

            return creds if creds and creds.valid else None

        except Exception:
            return None

    def refresh_token(self, credentials: Credentials) -> Credentials:
        """
        Refresh expired OAuth2 token.

        Args:
            credentials: Expired credentials with refresh token

        Returns:
            Refreshed credentials

        Raises:
            Exception: If token refresh fails

        Example:
            >>> creds = auth.get_credentials()
            >>> if creds.expired:
            ...     creds = auth.refresh_token(creds)
        """
        credentials.refresh(Request())
        return credentials

    def _run_oauth_flow(self) -> Credentials:
        """
        Run the OAuth2 authorization flow.

        Opens browser for user to authorize the application.

        Returns:
            New credentials from OAuth flow
        """
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.credentials_path),
            self.scopes
        )

        # Run local server flow (opens browser)
        creds = flow.run_local_server(
            port=0,
            authorization_prompt_message='Please authorize in your browser...',
            success_message='Authentication successful! You can close this window.',
            open_browser=True
        )

        return creds

    def _save_credentials(self, credentials: Credentials) -> None:
        """
        Save credentials to token file.

        Args:
            credentials: Credentials to save
        """
        with open(self.token_path, 'w') as token_file:
            token_file.write(credentials.to_json())

    def revoke_credentials(self) -> None:
        """
        Revoke and delete stored credentials.

        Useful for re-authentication or changing scopes.

        Example:
            >>> auth = GmailAuthenticator("credentials/credentials.json", "credentials/token.json")
            >>> auth.revoke_credentials()
            >>> # Next authenticate() call will trigger OAuth flow
        """
        if self.token_path.exists():
            os.remove(self.token_path)
