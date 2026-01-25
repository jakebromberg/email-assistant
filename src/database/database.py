"""Database connection and session management."""

from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from .schema import Base
from ..utils import get_logger


logger = get_logger(__name__)


class Database:
    """
    Database connection manager for SQLite.

    Handles database creation, connection management, and session lifecycle.

    Attributes:
        db_path: Path to SQLite database file
        engine: SQLAlchemy engine
        SessionLocal: Session factory

    Example:
        >>> db = Database("data/emails.db")
        >>> db.create_tables()
        >>> with db.get_session() as session:
        ...     emails = session.query(Email).all()
    """

    def __init__(self, db_path: str = "data/emails.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file

        Example:
            >>> db = Database("data/emails.db")
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with SQLite-specific settings
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # Set to True for SQL debugging
            connect_args={
                "check_same_thread": False,  # Allow multi-threaded access
                "timeout": 30,  # 30 second timeout for locks
            }
        )

        # Enable foreign keys for SQLite
        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging for better concurrency
            cursor.close()

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info(f"Database initialized: {self.db_path}")

    def create_tables(self) -> None:
        """
        Create all tables in the database.

        Creates tables for emails, labels, actions, and feedback if they
        don't already exist.

        Example:
            >>> db = Database()
            >>> db.create_tables()
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def drop_tables(self) -> None:
        """
        Drop all tables from the database.

        WARNING: This will delete all data. Use with caution.

        Example:
            >>> db = Database()
            >>> db.drop_tables()  # Careful! Deletes all data
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session context manager.

        Yields a session that will automatically commit on success or
        rollback on error.

        Yields:
            SQLAlchemy session

        Example:
            >>> db = Database()
            >>> with db.get_session() as session:
            ...     email = session.query(Email).first()
            ...     print(email.subject)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def get_session_raw(self) -> Session:
        """
        Get a raw database session (manual management).

        Caller is responsible for committing/rolling back and closing.

        Returns:
            SQLAlchemy session

        Example:
            >>> db = Database()
            >>> session = db.get_session_raw()
            >>> try:
            ...     email = session.query(Email).first()
            ...     session.commit()
            ... finally:
            ...     session.close()
        """
        return self.SessionLocal()

    def vacuum(self) -> None:
        """
        Vacuum the database to reclaim space and optimize performance.

        Example:
            >>> db = Database()
            >>> db.vacuum()
        """
        try:
            with self.engine.connect() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            raise

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with table row counts

        Example:
            >>> db = Database()
            >>> stats = db.get_stats()
            >>> print(f"Total emails: {stats['emails']}")
        """
        from .schema import Email, EmailLabel, EmailAction, FeedbackReview

        with self.get_session() as session:
            return {
                'emails': session.query(Email).count(),
                'email_labels': session.query(EmailLabel).count(),
                'actions': session.query(EmailAction).count(),
                'feedback_reviews': session.query(FeedbackReview).count(),
            }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Database(path={self.db_path})>"
