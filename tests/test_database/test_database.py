"""Tests for database connection and management."""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from sqlalchemy import text

from src.database.database import Database
from src.database.schema import Base, Email


class TestDatabase:
    """Test Database class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        # Cleanup
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    def test_initialization(self, temp_db_path):
        """Test database initializes correctly."""
        db = Database(temp_db_path)

        assert str(db.db_path) == temp_db_path
        assert db.engine is not None
        assert db.SessionLocal is not None

    def test_create_tables(self, temp_db_path):
        """Test creating database tables."""
        db = Database(temp_db_path)

        db.create_tables()

        # Verify database file exists
        assert os.path.exists(temp_db_path)

        # Verify tables were created by checking if we can query
        with db.get_session() as session:
            # Should not raise an error
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]

            # Check that our tables exist
            assert 'emails' in tables
            assert 'email_labels' in tables
            assert 'email_actions' in tables
            assert 'feedback_reviews' in tables
            assert 'email_features' in tables

    def test_get_session_context_manager(self, temp_db_path):
        """Test session context manager."""
        db = Database(temp_db_path)
        db.create_tables()

        with db.get_session() as session:
            # Session should be active
            assert session is not None
            assert session.is_active

        # Session should be closed after context
        # (Note: SQLAlchemy sessions don't have a simple is_closed check)

    def test_get_session_context_manager_with_error(self, temp_db_path):
        """Test session rolls back on error."""
        db = Database(temp_db_path)
        db.create_tables()

        try:
            with db.get_session() as session:
                # Execute some operation
                session.execute(text("SELECT 1"))
                # Raise an error
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # Database should still be accessible
        with db.get_session() as session:
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_vacuum(self, temp_db_path):
        """Test database vacuum operation."""
        db = Database(temp_db_path)
        db.create_tables()

        # Add some data first
        with db.get_session() as session:
            email = Email(
                message_id="test123",
                thread_id="thread123",
                from_address="sender@example.com",
                subject="Test",
                date=datetime.now(),
                snippet="Test"
            )
            session.add(email)
            session.commit()

        # Get initial file size
        initial_size = os.path.getsize(temp_db_path)

        # Vacuum should not raise an error
        db.vacuum()

        # File should still exist
        assert os.path.exists(temp_db_path)

        # Size might change (could be same, smaller, or slightly larger)
        final_size = os.path.getsize(temp_db_path)
        assert final_size > 0

    def test_vacuum_empty_database(self, temp_db_path):
        """Test vacuum on empty database."""
        db = Database(temp_db_path)
        db.create_tables()

        # Should not raise an error
        db.vacuum()

        assert os.path.exists(temp_db_path)

    def test_wal_mode_enabled(self, temp_db_path):
        """Test that WAL mode is enabled."""
        db = Database(temp_db_path)
        db.create_tables()

        with db.get_session() as session:
            result = session.execute(text("PRAGMA journal_mode")).scalar()
            assert result.lower() == 'wal'

    def test_foreign_keys_enabled(self, temp_db_path):
        """Test that foreign keys are enabled."""
        db = Database(temp_db_path)
        db.create_tables()

        with db.get_session() as session:
            result = session.execute(text("PRAGMA foreign_keys")).scalar()
            assert result == 1

    def test_database_path_creation(self):
        """Test database creates parent directories if needed."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'subdir', 'test.db')

        db = Database(db_path)
        db.create_tables()

        # Parent directory should be created
        assert os.path.exists(os.path.dirname(db_path))
        assert os.path.exists(db_path)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_multiple_sessions(self, temp_db_path):
        """Test multiple sessions can be created."""
        db = Database(temp_db_path)
        db.create_tables()

        # Create multiple sessions
        with db.get_session() as session1:
            with db.get_session() as session2:
                # Both should work
                result1 = session1.execute(text("SELECT 1")).scalar()
                result2 = session2.execute(text("SELECT 1")).scalar()

                assert result1 == 1
                assert result2 == 1

    def test_concurrent_access_wal_mode(self, temp_db_path):
        """Test concurrent read/write with WAL mode."""
        db = Database(temp_db_path)
        db.create_tables()

        # Write in one session
        with db.get_session() as session1:
            email = Email(
                message_id="test1",
                thread_id="thread1",
                from_address="sender@example.com",
                subject="Test",
                date=datetime.now(),
                snippet="Test"
            )
            session1.add(email)
            session1.commit()

            # Read in another session (should work with WAL)
            with db.get_session() as session2:
                count = session2.query(Email).count()
                assert count == 1
