"""Tests for database connection and management."""

import os
import tempfile

from sqlalchemy import text

from src.database.database import Database
from src.database.schema import Email


class TestDatabase:
    """Test Database class."""

    def test_initialization(self, temp_db):
        """Test database initializes correctly."""
        assert temp_db.db_path is not None
        assert temp_db.engine is not None
        assert temp_db.SessionLocal is not None

    def test_create_tables(self, temp_db):
        """Test creating database tables."""
        # Tables already created by temp_db fixture

        # Verify tables were created by checking if we can query
        with temp_db.get_session() as session:
            # Should not raise an error
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]

            # Check that our tables exist
            assert 'emails' in tables
            assert 'email_labels' in tables
            assert 'email_actions' in tables
            assert 'feedback_reviews' in tables
            assert 'email_features' in tables

    def test_get_session_context_manager(self, temp_db):
        """Test session context manager."""
        with temp_db.get_session() as session:
            # Session should be active
            assert session is not None
            assert session.is_active

        # Session should be closed after context
        # (Note: SQLAlchemy sessions don't have a simple is_closed check)

    def test_get_session_context_manager_with_error(self, temp_db):
        """Test session rolls back on error."""
        try:
            with temp_db.get_session() as session:
                # Execute some operation
                session.execute(text("SELECT 1"))
                # Raise an error
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # Database should still be accessible
        with temp_db.get_session() as session:
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_vacuum(self, temp_db, email_factory):
        """Test database vacuum operation."""
        # Add some data first using email_factory
        email_factory(message_id="test123")

        # Get database path for size check
        db_path = str(temp_db.db_path)
        initial_size = os.path.getsize(db_path)

        # Vacuum should not raise an error
        temp_db.vacuum()

        # File should still exist
        assert os.path.exists(db_path)

        # Size might change (could be same, smaller, or slightly larger)
        final_size = os.path.getsize(db_path)
        assert final_size > 0

    def test_vacuum_empty_database(self, temp_db):
        """Test vacuum on empty database."""
        # Should not raise an error
        temp_db.vacuum()

        assert os.path.exists(str(temp_db.db_path))

    def test_wal_mode_enabled(self, temp_db):
        """Test that WAL mode is enabled."""
        with temp_db.get_session() as session:
            result = session.execute(text("PRAGMA journal_mode")).scalar()
            assert result.lower() == 'wal'

    def test_foreign_keys_enabled(self, temp_db):
        """Test that foreign keys are enabled."""
        with temp_db.get_session() as session:
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

    def test_multiple_sessions(self, temp_db):
        """Test multiple sessions can be created."""
        # Create multiple sessions
        with temp_db.get_session() as session1:
            with temp_db.get_session() as session2:
                # Both should work
                result1 = session1.execute(text("SELECT 1")).scalar()
                result2 = session2.execute(text("SELECT 1")).scalar()

                assert result1 == 1
                assert result2 == 1

    def test_concurrent_access_wal_mode(self, temp_db, email_factory):
        """Test concurrent read/write with WAL mode."""
        # Write in one session
        with temp_db.get_session() as session1:
            # Use email_factory (which uses a different session)
            email_factory(message_id="test1")

            # Read in another session (should work with WAL)
            with temp_db.get_session() as session2:
                count = session2.query(Email).count()
                assert count == 1

    def test_drop_tables(self, tmp_path):
        """Test dropping all tables."""
        db_path = tmp_path / "drop_test.db"
        db = Database(str(db_path))
        db.create_tables()

        # Verify tables exist
        with db.get_session() as session:
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            assert 'emails' in tables

        # Drop tables
        db.drop_tables()

        # Verify tables are gone
        with db.get_session() as session:
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            assert 'emails' not in tables

    def test_get_stats(self, temp_db, email_factory, action_factory):
        """Test getting database statistics."""
        # Add some data
        email1 = email_factory(message_id="stats1")
        email2 = email_factory(message_id="stats2")
        action_factory(message_id=email1.message_id)

        stats = temp_db.get_stats()

        assert stats['emails'] >= 2
        assert stats['actions'] >= 1
        assert 'email_labels' in stats
        assert 'feedback_reviews' in stats

    def test_get_stats_empty(self, tmp_path):
        """Test getting stats on empty database."""
        db_path = tmp_path / "empty_stats.db"
        db = Database(str(db_path))
        db.create_tables()

        stats = db.get_stats()

        assert stats['emails'] == 0
        assert stats['actions'] == 0

    def test_get_session_raw(self, temp_db):
        """Test getting raw session."""
        session = temp_db.get_session_raw()
        try:
            assert session is not None
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1
            session.commit()
        finally:
            session.close()

    def test_repr(self, temp_db):
        """Test database string representation."""
        repr_str = repr(temp_db)

        assert 'Database' in repr_str
        assert 'path=' in repr_str
