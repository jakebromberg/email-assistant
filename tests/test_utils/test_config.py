"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.config import Config


class TestConfigInit:
    """Test Config initialization."""

    def test_init_with_defaults(self):
        """Test initialization uses default values when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            assert config.GMAIL_CREDENTIALS_PATH == 'credentials/credentials.json'
            assert config.GMAIL_TOKEN_PATH == 'credentials/token.json'
            assert config.LOG_LEVEL == 'INFO'
            assert config.LOG_DIR == 'logs'
            assert config.DATA_DIR == 'data'

    def test_init_with_env_vars(self):
        """Test initialization reads from environment variables."""
        env_vars = {
            'GMAIL_CREDENTIALS_PATH': '/custom/creds.json',
            'GMAIL_TOKEN_PATH': '/custom/token.json',
            'LOG_LEVEL': 'debug',
            'LOG_DIR': '/custom/logs',
            'DATA_DIR': '/custom/data',
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            assert config.GMAIL_CREDENTIALS_PATH == '/custom/creds.json'
            assert config.GMAIL_TOKEN_PATH == '/custom/token.json'
            assert config.LOG_LEVEL == 'DEBUG'  # Should be uppercased
            assert config.LOG_DIR == '/custom/logs'
            assert config.DATA_DIR == '/custom/data'


class TestConfigLoad:
    """Test Config.load() method."""

    def test_load_returns_config_instance(self):
        """Test load returns Config instance."""
        config = Config.load()
        assert isinstance(config, Config)

    def test_load_with_env_file(self):
        """Test load with specific env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('LOG_LEVEL=WARNING\n')
            f.write('DATA_DIR=/tmp/test_data\n')
            env_file = f.name

        try:
            config = Config.load(env_file)
            # After loading the env file, values should be available
            assert isinstance(config, Config)
        finally:
            os.unlink(env_file)

    def test_load_without_env_file(self):
        """Test load without specifying env file uses default discovery."""
        config = Config.load()
        assert isinstance(config, Config)


class TestConfigValidate:
    """Test Config.validate() method."""

    def test_validate_valid_log_level(self):
        """Test validate passes with valid log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_vars = {
                'LOG_LEVEL': 'INFO',
                'LOG_DIR': f'{tmpdir}/logs',
                'DATA_DIR': f'{tmpdir}/data',
                'GMAIL_CREDENTIALS_PATH': f'{tmpdir}/creds/credentials.json',
            }
            with patch.dict(os.environ, env_vars, clear=True):
                config = Config()
                config.validate()  # Should not raise

    def test_validate_invalid_log_level(self):
        """Test validate raises with invalid log level."""
        config = Config()
        config.LOG_LEVEL = 'INVALID'

        with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
            config.validate()

    def test_validate_creates_directories(self):
        """Test validate creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = f'{tmpdir}/logs'
            data_dir = f'{tmpdir}/data'
            creds_dir = f'{tmpdir}/creds'

            config = Config()
            config.LOG_LEVEL = 'INFO'
            config.LOG_DIR = log_dir
            config.DATA_DIR = data_dir
            config.GMAIL_CREDENTIALS_PATH = f'{creds_dir}/credentials.json'

            config.validate()

            assert Path(log_dir).exists()
            assert Path(data_dir).exists()
            assert Path(creds_dir).exists()

    def test_validate_all_log_levels(self):
        """Test validate accepts all valid log levels."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        with tempfile.TemporaryDirectory() as tmpdir:
            for level in valid_levels:
                config = Config()
                config.LOG_LEVEL = level
                config.LOG_DIR = f'{tmpdir}/logs'
                config.DATA_DIR = f'{tmpdir}/data'
                config.GMAIL_CREDENTIALS_PATH = f'{tmpdir}/creds/credentials.json'

                config.validate()  # Should not raise


class TestConfigRepr:
    """Test Config.__repr__() method."""

    def test_repr(self):
        """Test string representation of config."""
        config = Config()
        repr_str = repr(config)

        assert 'Config(' in repr_str
        assert 'GMAIL_CREDENTIALS_PATH=' in repr_str
        assert 'LOG_LEVEL=' in repr_str
        assert 'LOG_DIR=' in repr_str
        assert 'DATA_DIR=' in repr_str
