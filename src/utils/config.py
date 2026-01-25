"""Configuration management using environment variables."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """
    Application configuration loaded from environment variables.

    Environment variables can be set in a .env file in the project root.

    Attributes:
        GMAIL_CREDENTIALS_PATH: Path to Gmail OAuth credentials file
        GMAIL_TOKEN_PATH: Path to stored OAuth token
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_DIR: Directory for log files
        DATA_DIR: Directory for data storage

    Example:
        >>> config = Config.load()
        >>> print(config.GMAIL_CREDENTIALS_PATH)
        credentials/credentials.json
    """

    GMAIL_CREDENTIALS_PATH: str
    GMAIL_TOKEN_PATH: str
    LOG_LEVEL: str
    LOG_DIR: str
    DATA_DIR: str

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.GMAIL_CREDENTIALS_PATH = os.getenv(
            'GMAIL_CREDENTIALS_PATH',
            'credentials/credentials.json'
        )
        self.GMAIL_TOKEN_PATH = os.getenv(
            'GMAIL_TOKEN_PATH',
            'credentials/token.json'
        )
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.LOG_DIR = os.getenv('LOG_DIR', 'logs')
        self.DATA_DIR = os.getenv('DATA_DIR', 'data')

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> 'Config':
        """
        Load configuration from .env file and environment variables.

        Args:
            env_file: Path to .env file (defaults to .env in project root)

        Returns:
            Config instance

        Example:
            >>> config = Config.load()
            >>> config.validate()
        """
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in current directory or parent directories
            load_dotenv()

        config = cls()
        return config

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If required configuration is missing or invalid

        Example:
            >>> config = Config.load()
            >>> config.validate()  # Raises if credentials file doesn't exist
        """
        # Check log level is valid
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {self.LOG_LEVEL}. "
                f"Must be one of {valid_levels}"
            )

        # Create directories if they don't exist
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)

        # Check credentials directory exists
        creds_dir = Path(self.GMAIL_CREDENTIALS_PATH).parent
        creds_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Return string representation of config."""
        return (
            f"Config("
            f"GMAIL_CREDENTIALS_PATH={self.GMAIL_CREDENTIALS_PATH}, "
            f"LOG_LEVEL={self.LOG_LEVEL}, "
            f"LOG_DIR={self.LOG_DIR}, "
            f"DATA_DIR={self.DATA_DIR}"
            ")"
        )
