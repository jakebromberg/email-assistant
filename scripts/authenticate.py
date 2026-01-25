#!/usr/bin/env python3
"""
Initial Gmail OAuth2 authentication script.

Run this script to set up OAuth2 credentials for the first time.
It will open your browser to authorize the application.

Usage:
    python scripts/authenticate.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmail.auth import GmailAuthenticator
from src.utils import Config, setup_logger


def main():
    """Run OAuth authentication flow."""
    # Set up logging
    logger = setup_logger('authenticate', level='INFO')

    # Load configuration
    logger.info("Loading configuration...")
    config = Config.load()
    config.validate()

    logger.info(f"Credentials path: {config.GMAIL_CREDENTIALS_PATH}")
    logger.info(f"Token path: {config.GMAIL_TOKEN_PATH}")

    # Check if credentials file exists
    if not Path(config.GMAIL_CREDENTIALS_PATH).exists():
        logger.error(
            f"Credentials file not found: {config.GMAIL_CREDENTIALS_PATH}\n"
            f"\n"
            f"To set up Gmail API credentials:\n"
            f"1. Go to https://console.cloud.google.com\n"
            f"2. Create a new project or select existing project\n"
            f"3. Enable the Gmail API\n"
            f"4. Create OAuth 2.0 credentials (Desktop app type)\n"
            f"5. Download credentials.json\n"
            f"6. Save it to: {config.GMAIL_CREDENTIALS_PATH}\n"
        )
        sys.exit(1)

    # Initialize authenticator
    logger.info("Initializing Gmail authenticator...")
    auth = GmailAuthenticator(
        credentials_path=config.GMAIL_CREDENTIALS_PATH,
        token_path=config.GMAIL_TOKEN_PATH,
        scopes=[GmailAuthenticator.SCOPE_MODIFY]
    )

    # Run authentication
    logger.info("Starting OAuth2 authentication flow...")
    logger.info("Your browser will open for authorization...")

    try:
        creds = auth.authenticate()
        logger.info("Authentication successful!")
        logger.info(f"Token saved to: {config.GMAIL_TOKEN_PATH}")
        logger.info("You can now use the Gmail API client.")

    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
