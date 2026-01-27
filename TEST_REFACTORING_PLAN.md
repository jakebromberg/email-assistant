# Test Infrastructure Refactoring Plan

## Executive Summary

**Goal**: Eliminate ~500 lines of duplicate test code (10% of test suite) and standardize testing patterns across the codebase.

**Current Problems**:
- Database setup duplicated 4 times with 3 different approaches
- Email/EmailAction creation repeated 6+ times across test files
- Mock Gmail client setup inconsistent across 3 files
- No centralized conftest.py for shared fixtures
- Inconsistent patterns and naming conventions

**Expected Benefits**:
- ~350 lines of code reduction (70% of duplication eliminated)
- Faster test execution (shared database fixtures)
- Easier test maintenance (single source of truth)
- Consistent testing patterns across modules
- Better developer experience

**Timeline**: 4 phases over 2-3 days

---

## Phase 1: Create Central Fixtures (Priority: CRITICAL)

**Goal**: Create tests/conftest.py with shared database and factory fixtures

**Expected Impact**: ~200 line reduction

### 1.1 Create tests/conftest.py

**File**: `tests/conftest.py` (NEW FILE)

```python
"""Shared test fixtures for all test modules."""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generator, List
from unittest.mock import Mock

from sqlalchemy.orm import Session

from src.database import Database
from src.database.schema import Base, Email, EmailAction, EmailFeatures
from src.database.repository import EmailRepository
from src.features import FeatureStore


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db(tmp_path: Path) -> Generator[Database, None, None]:
    """
    Create a temporary test database with schema initialized.

    Usage:
        def test_something(temp_db):
            with temp_db.get_session() as session:
                # Use session
    """
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))

    # Initialize schema
    Base.metadata.create_all(db.engine)

    yield db

    # Cleanup
    db.engine.dispose()


@pytest.fixture
def temp_db_session(temp_db: Database) -> Generator[Session, None, None]:
    """
    Create a database session for testing.

    Usage:
        def test_something(temp_db_session):
            email = Email(...)
            temp_db_session.add(email)
            temp_db_session.commit()
    """
    with temp_db.get_session() as session:
        yield session


@pytest.fixture
def email_repo(temp_db_session: Session) -> EmailRepository:
    """
    Create an EmailRepository for testing.

    Usage:
        def test_something(email_repo):
            email = email_repo.get_by_id("msg123")
    """
    return EmailRepository(temp_db_session)


@pytest.fixture
def feature_store(temp_db_session: Session) -> FeatureStore:
    """
    Create a FeatureStore for testing.

    Usage:
        def test_something(feature_store):
            features = feature_store.get_features("msg123")
    """
    return FeatureStore(temp_db_session)


# ============================================================================
# Factory Fixtures
# ============================================================================

@pytest.fixture
def email_factory(temp_db_session: Session):
    """
    Factory for creating test emails.

    Usage:
        def test_something(email_factory):
            email = email_factory(subject="Test", from_address="test@example.com")
            # email is already added to database
    """
    def _create_email(
        message_id: str = None,
        thread_id: str = None,
        from_address: str = "sender@example.com",
        from_name: str = "Test Sender",
        to_address: str = "me@example.com",
        subject: str = "Test Email",
        body_plain: str = "Test body",
        body_html: str = "<p>Test body</p>",
        date: datetime = None,
        labels: List[str] = None,
        snippet: str = "Test snippet",
        was_read: bool = False,
        was_archived: bool = False,
        is_important: bool = False,
        is_starred: bool = False,
        opened_at: datetime = None,
        headers: dict = None,
        **kwargs
    ) -> Email:
        """Create a test email with sensible defaults."""
        # Generate unique IDs if not provided
        if message_id is None:
            import uuid
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
        if thread_id is None:
            thread_id = f"thread_{message_id}"

        # Default values
        if date is None:
            date = datetime.now()
        if labels is None:
            labels = ["INBOX", "UNREAD"] if not was_read else ["INBOX"]
        if headers is None:
            headers = {}

        email = Email(
            message_id=message_id,
            thread_id=thread_id,
            from_address=from_address,
            from_name=from_name,
            to_address=to_address,
            subject=subject,
            body_plain=body_plain,
            body_html=body_html,
            date=date,
            labels=labels,
            snippet=snippet,
            was_read=was_read,
            was_archived=was_archived,
            is_important=is_important,
            is_starred=is_starred,
            opened_at=opened_at,
            headers=headers,
            **kwargs
        )

        temp_db_session.add(email)
        temp_db_session.commit()
        temp_db_session.refresh(email)

        return email

    return _create_email


@pytest.fixture
def action_factory(temp_db_session: Session):
    """
    Factory for creating test email actions.

    Usage:
        def test_something(action_factory, email_factory):
            email = email_factory()
            action = action_factory(message_id=email.message_id, action_type='archive')
    """
    def _create_action(
        message_id: str,
        action_type: str = 'keep',
        source: str = 'bot',
        action_data: dict = None,
        timestamp: datetime = None,
        **kwargs
    ) -> EmailAction:
        """Create a test email action with sensible defaults."""
        if timestamp is None:
            timestamp = datetime.now()
        if action_data is None:
            action_data = {
                'score': 0.5,
                'confidence': 'low',
                'reasoning': 'Test action'
            }

        action = EmailAction(
            message_id=message_id,
            action_type=action_type,
            source=source,
            action_data=action_data,
            timestamp=timestamp,
            **kwargs
        )

        temp_db_session.add(action)
        temp_db_session.commit()
        temp_db_session.refresh(action)

        return action

    return _create_action


@pytest.fixture
def features_factory(temp_db_session: Session):
    """
    Factory for creating test email features.

    Usage:
        def test_something(features_factory, email_factory):
            email = email_factory()
            features = features_factory(message_id=email.message_id, is_newsletter=True)
    """
    def _create_features(
        message_id: str,
        sender_domain: str = "example.com",
        sender_email_count: int = 1,
        sender_open_rate: float = 0.5,
        sender_days_since_last: int = None,
        is_newsletter: bool = False,
        subject_length: int = 20,
        body_length: int = 100,
        has_attachments: bool = False,
        thread_length: int = 1,
        day_of_week: int = 1,
        hour_of_day: int = 12,
        subject_embedding: List[float] = None,
        body_embedding: List[float] = None,
        **kwargs
    ) -> EmailFeatures:
        """Create test email features with sensible defaults."""
        features = EmailFeatures(
            message_id=message_id,
            sender_domain=sender_domain,
            sender_email_count=sender_email_count,
            sender_open_rate=sender_open_rate,
            sender_days_since_last=sender_days_since_last,
            is_newsletter=is_newsletter,
            subject_length=subject_length,
            body_length=body_length,
            has_attachments=has_attachments,
            thread_length=thread_length,
            day_of_week=day_of_week,
            hour_of_day=hour_of_day,
            subject_embedding=subject_embedding,
            body_embedding=body_embedding,
            **kwargs
        )

        temp_db_session.add(features)
        temp_db_session.commit()
        temp_db_session.refresh(features)

        return features

    return _create_features


# ============================================================================
# Gmail Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_gmail_auth():
    """
    Create a mock Gmail authenticator.

    Usage:
        def test_something(mock_gmail_auth):
            client = GmailClient(mock_gmail_auth)
    """
    auth = Mock()
    auth.get_credentials = Mock(return_value=Mock())
    return auth


@pytest.fixture
def mock_gmail_client():
    """
    Create a mock Gmail client.

    Usage:
        def test_something(mock_gmail_client):
            mock_gmail_client.list_messages.return_value = (["msg1"], None)
    """
    client = Mock()
    client.list_messages = Mock(return_value=([], None))
    client.list_all_messages = Mock(return_value=[])
    client.get_messages_batch = Mock(return_value=[])
    client.get_message = Mock(return_value=None)
    return client


@pytest.fixture
def mock_gmail_ops(mock_gmail_client):
    """
    Create a mock Gmail operations.

    Usage:
        def test_something(mock_gmail_ops):
            result = mock_gmail_ops.archive(['msg1'], dry_run=True)
    """
    from src.gmail.operations import GmailOperations, OperationResult

    ops = Mock(spec=GmailOperations)

    # Default successful operation result
    def _mock_operation(message_ids, **kwargs):
        return OperationResult(
            success=True,
            message_ids=message_ids,
            message=f"Operation successful for {len(message_ids)} messages"
        )

    ops.archive = Mock(side_effect=_mock_operation)
    ops.mark_read = Mock(side_effect=_mock_operation)
    ops.mark_unread = Mock(side_effect=_mock_operation)
    ops.move_to_inbox = Mock(side_effect=_mock_operation)
    ops.add_labels = Mock(side_effect=_mock_operation)
    ops.remove_labels = Mock(side_effect=_mock_operation)

    return ops


# ============================================================================
# Helper Fixtures
# ============================================================================

@pytest.fixture
def sample_gmail_message():
    """
    Create a sample Gmail API message structure.

    Usage:
        def test_something(sample_gmail_message):
            email = Email.from_gmail_message(sample_gmail_message)
    """
    return {
        'id': 'msg123',
        'threadId': 'thread123',
        'labelIds': ['INBOX', 'UNREAD'],
        'snippet': 'This is a test email snippet',
        'internalDate': '1704038400000',  # 2024-01-01 00:00:00 UTC
        'payload': {
            'headers': [
                {'name': 'From', 'value': 'Test Sender <sender@example.com>'},
                {'name': 'To', 'value': 'me@example.com'},
                {'name': 'Subject', 'value': 'Test Email'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 00:00:00 +0000'},
            ],
            'mimeType': 'multipart/alternative',
            'parts': [
                {
                    'mimeType': 'text/plain',
                    'body': {
                        'data': 'VGVzdCBib2R5'  # base64 for "Test body"
                    }
                },
                {
                    'mimeType': 'text/html',
                    'body': {
                        'data': 'PHA+VGVzdCBib2R5PC9wPg=='  # base64 for "<p>Test body</p>"
                    }
                }
            ]
        }
    }
```

### 1.2 Verification

Create a test file to verify the fixtures work correctly:

**File**: `tests/test_conftest.py` (NEW FILE)

```python
"""Test the shared fixtures in conftest.py."""

import pytest
from datetime import datetime

from src.database.schema import Email, EmailAction, EmailFeatures


def test_temp_db_fixture(temp_db):
    """Test temp_db fixture creates a working database."""
    assert temp_db is not None
    assert temp_db.engine is not None

    with temp_db.get_session() as session:
        # Should be able to query
        emails = session.query(Email).all()
        assert emails == []


def test_temp_db_session_fixture(temp_db_session):
    """Test temp_db_session fixture provides a working session."""
    assert temp_db_session is not None

    # Should be able to add and commit
    email = Email(
        message_id="test123",
        thread_id="thread123",
        from_address="test@example.com",
        to_address="me@example.com",
        subject="Test",
        date=datetime.now(),
        labels=["INBOX"]
    )
    temp_db_session.add(email)
    temp_db_session.commit()

    # Should be able to query
    result = temp_db_session.query(Email).filter_by(message_id="test123").first()
    assert result is not None
    assert result.subject == "Test"


def test_email_factory_fixture(email_factory):
    """Test email_factory creates emails with defaults."""
    email = email_factory()

    assert email.message_id is not None
    assert email.from_address == "sender@example.com"
    assert email.subject == "Test Email"
    assert email.was_read is False


def test_email_factory_custom_values(email_factory):
    """Test email_factory accepts custom values."""
    email = email_factory(
        message_id="custom123",
        subject="Custom Subject",
        was_read=True
    )

    assert email.message_id == "custom123"
    assert email.subject == "Custom Subject"
    assert email.was_read is True


def test_action_factory_fixture(action_factory, email_factory):
    """Test action_factory creates actions."""
    email = email_factory()
    action = action_factory(message_id=email.message_id, action_type='archive')

    assert action.message_id == email.message_id
    assert action.action_type == 'archive'
    assert action.source == 'bot'
    assert 'score' in action.action_data


def test_features_factory_fixture(features_factory, email_factory):
    """Test features_factory creates features."""
    email = email_factory()
    features = features_factory(
        message_id=email.message_id,
        is_newsletter=True
    )

    assert features.message_id == email.message_id
    assert features.is_newsletter is True
    assert features.sender_domain == "example.com"


def test_mock_gmail_client_fixture(mock_gmail_client):
    """Test mock_gmail_client has expected methods."""
    assert hasattr(mock_gmail_client, 'list_messages')
    assert hasattr(mock_gmail_client, 'get_messages_batch')

    # Should return empty list by default
    result = mock_gmail_client.list_all_messages()
    assert result == []


def test_mock_gmail_ops_fixture(mock_gmail_ops):
    """Test mock_gmail_ops has expected methods."""
    assert hasattr(mock_gmail_ops, 'archive')
    assert hasattr(mock_gmail_ops, 'mark_read')

    # Should return successful result
    result = mock_gmail_ops.archive(['msg1'], dry_run=True)
    assert result.success is True
    assert result.message_ids == ['msg1']


def test_sample_gmail_message_fixture(sample_gmail_message):
    """Test sample_gmail_message has expected structure."""
    assert sample_gmail_message['id'] == 'msg123'
    assert sample_gmail_message['threadId'] == 'thread123'
    assert 'payload' in sample_gmail_message
    assert 'headers' in sample_gmail_message['payload']
```

**Run verification**:
```bash
pytest tests/test_conftest.py -v
```

**Success Criteria**: All 11 tests pass

### 1.3 Update Existing Tests to Use New Fixtures

Start with one test file to validate the approach.

**Example Migration**: `tests/test_database/test_repository.py`

**Before** (lines 10-25):
```python
@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))

    # Initialize schema
    from src.database.schema import Base
    Base.metadata.create_all(db.engine)

    yield db

    db.engine.dispose()
```

**After**: Simply remove this fixture - it's now in conftest.py

**Before** (test function):
```python
def test_save_email(temp_db):
    """Test saving an email to the database."""
    with temp_db.get_session() as session:
        repo = EmailRepository(session)

        email = Email(
            message_id="test123",
            thread_id="thread123",
            from_address="sender@example.com",
            from_name="Test Sender",
            to_address="me@example.com",
            subject="Test Email",
            body_plain="Test body",
            body_html="<p>Test body</p>",
            date=datetime.now(),
            labels=["INBOX", "UNREAD"],
            snippet="Test snippet",
            was_read=False,
            was_archived=False
        )

        repo.save_email(email)
```

**After**:
```python
def test_save_email(email_repo, email_factory):
    """Test saving an email to the database."""
    email = email_factory(message_id="test123")

    # email_factory already saved it, so just verify
    saved = email_repo.get_by_id("test123")
    assert saved is not None
    assert saved.subject == "Test Email"
```

**Lines Saved**: 15-20 lines per test function

---

## Phase 2: Migrate Existing Tests (Priority: HIGH)

**Goal**: Update all existing test files to use centralized fixtures

**Expected Impact**: ~150 line reduction

### 2.1 Migration Order

Migrate test files in this order (easiest to hardest):

1. `tests/test_database/test_repository.py` - Database operations (20 lines saved)
2. `tests/test_database/test_database.py` - Database core (25 lines saved)
3. `tests/test_scripts/test_review_decisions.py` - Script tests (40 lines saved)
4. `tests/test_triage/test_pipeline.py` - Pipeline tests (35 lines saved)
5. `tests/test_ml/test_training.py` - ML training tests (15 lines saved)
6. `tests/test_ml/test_evaluation.py` - ML evaluation tests (10 lines saved)
7. `tests/test_gmail/test_operations.py` - Gmail operations (20 lines saved)

### 2.2 Migration Checklist

For each test file:

- [ ] Remove local `temp_db` fixture
- [ ] Remove local `temp_db_session` fixture
- [ ] Replace manual Email creation with `email_factory`
- [ ] Replace manual EmailAction creation with `action_factory`
- [ ] Replace manual EmailFeatures creation with `features_factory`
- [ ] Replace local Gmail mocks with `mock_gmail_client` / `mock_gmail_ops`
- [ ] Update imports (remove unnecessary ones)
- [ ] Run tests to verify no regressions: `pytest tests/test_[module]/ -v`
- [ ] Commit changes: `git commit -m "test: migrate [module] tests to use shared fixtures"`

### 2.3 Detailed Migration Example

**File**: `tests/test_database/test_repository.py`

**Changes Required**:

1. Remove fixture at lines 10-25 (temp_db)
2. Remove fixture at lines 27-30 (temp_db_session) if it exists
3. Update test functions:

```python
# BEFORE
def test_save_email(temp_db):
    with temp_db.get_session() as session:
        repo = EmailRepository(session)

        email = Email(
            message_id="test123",
            thread_id="thread123",
            from_address="sender@example.com",
            from_name="Test Sender",
            to_address="me@example.com",
            subject="Test Email",
            body_plain="Test body",
            body_html="<p>Test body</p>",
            date=datetime.now(),
            labels=["INBOX", "UNREAD"],
            snippet="Test snippet",
            was_read=False,
            was_archived=False
        )

        repo.save_email(email)

        saved = repo.get_by_id("test123")
        assert saved is not None
        assert saved.subject == "Test Email"

# AFTER
def test_save_email(email_repo, email_factory):
    email = email_factory(message_id="test123")

    saved = email_repo.get_by_id("test123")
    assert saved is not None
    assert saved.subject == "Test Email"
```

**Lines Before**: 25
**Lines After**: 6
**Lines Saved**: 19

Repeat for all test functions in the file.

**Verification**:
```bash
pytest tests/test_database/test_repository.py -v
```

**Commit**:
```bash
git add tests/test_database/test_repository.py
git commit -m "test: migrate database repository tests to use shared fixtures"
```

### 2.4 Handling Special Cases

**Case 1: Tests that need multiple emails**

```python
# BEFORE
def test_multiple_emails(temp_db):
    with temp_db.get_session() as session:
        repo = EmailRepository(session)

        email1 = Email(message_id="msg1", ...)
        email2 = Email(message_id="msg2", ...)
        repo.save_email(email1)
        repo.save_email(email2)

# AFTER
def test_multiple_emails(email_factory):
    email1 = email_factory(message_id="msg1")
    email2 = email_factory(message_id="msg2")
    # Already saved, no need for repo.save_email()
```

**Case 2: Tests that need custom dates**

```python
# BEFORE
def test_date_filtering(temp_db):
    old_date = datetime.now() - timedelta(days=10)
    email = Email(date=old_date, ...)

# AFTER
def test_date_filtering(email_factory):
    old_date = datetime.now() - timedelta(days=10)
    email = email_factory(date=old_date)
```

**Case 3: Tests that need actions with specific scores**

```python
# BEFORE
def test_archive_decision(temp_db):
    action = EmailAction(
        message_id="msg1",
        action_type="archive",
        action_data={'score': 0.1, 'confidence': 'high'}
    )

# AFTER
def test_archive_decision(action_factory, email_factory):
    email = email_factory(message_id="msg1")
    action = action_factory(
        message_id=email.message_id,
        action_type="archive",
        action_data={'score': 0.1, 'confidence': 'high'}
    )
```

---

## Phase 3: Create Module-Specific Fixtures (Priority: MEDIUM)

**Goal**: Create conftest.py files in test subdirectories for module-specific fixtures

**Expected Impact**: Better organization, easier to find relevant fixtures

### 3.1 Create tests/test_gmail/conftest.py

**File**: `tests/test_gmail/conftest.py` (NEW FILE)

```python
"""Gmail-specific test fixtures."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_gmail_message_with_attachments(sample_gmail_message):
    """Gmail message with attachments."""
    message = sample_gmail_message.copy()
    message['payload']['parts'].append({
        'filename': 'document.pdf',
        'mimeType': 'application/pdf',
        'body': {
            'attachmentId': 'att123',
            'size': 12345
        }
    })
    return message


@pytest.fixture
def mock_gmail_message_multipart_nested(sample_gmail_message):
    """Gmail message with deeply nested multipart structure."""
    message = sample_gmail_message.copy()
    message['payload'] = {
        'mimeType': 'multipart/mixed',
        'parts': [
            {
                'mimeType': 'multipart/alternative',
                'parts': [
                    {
                        'mimeType': 'text/plain',
                        'body': {'data': 'VGVzdCBib2R5'}
                    },
                    {
                        'mimeType': 'text/html',
                        'body': {'data': 'PHA+VGVzdCBib2R5PC9wPg=='}
                    }
                ]
            }
        ]
    }
    return message


@pytest.fixture
def mock_operation_result():
    """Create a mock successful operation result."""
    from src.gmail.operations import OperationResult

    return OperationResult(
        success=True,
        message_ids=['msg1', 'msg2'],
        message="Operation successful"
    )
```

### 3.2 Create tests/test_ml/conftest.py

**File**: `tests/test_ml/conftest.py` (NEW FILE)

```python
"""ML-specific test fixtures."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock


@pytest.fixture
def sample_training_data():
    """Create sample training data for ML tests."""
    X = pd.DataFrame({
        'sender_email_count': [5, 10, 2, 20],
        'sender_open_rate': [0.8, 0.3, 0.5, 0.1],
        'is_newsletter': [0, 1, 0, 1],
        'subject_length': [20, 50, 15, 60],
        'body_length': [100, 500, 80, 1000]
    })
    y = pd.Series([1, 0, 1, 0])  # 1 = read, 0 = unread

    return X, y


@pytest.fixture
def trained_model(sample_training_data):
    """Create a trained LightGBM model for testing."""
    import lightgbm as lgb

    X, y = sample_training_data

    model = lgb.LGBMClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)

    return model


@pytest.fixture
def mock_email_scorer():
    """Create a mock email scorer."""
    from unittest.mock import Mock

    scorer = Mock()
    scorer.score_email = Mock(return_value=0.5)
    scorer.make_decision = Mock(return_value={
        'action': 'keep',
        'confidence': 'low',
        'score': 0.5,
        'reasoning': 'Test reasoning',
        'labels': []
    })

    return scorer


@pytest.fixture
def mock_categorizer():
    """Create a mock email categorizer."""
    from unittest.mock import Mock

    categorizer = Mock()
    categorizer.categorize = Mock(return_value=['Bot/Newsletter-Tech'])
    categorizer.get_all_categories = Mock(return_value=[
        'Bot/Newsletter-Tech',
        'Bot/Newsletter-Finance',
        'Bot/Personal',
        'Bot/Work'
    ])

    return categorizer
```

### 3.3 Create tests/test_triage/conftest.py

**File**: `tests/test_triage/conftest.py` (NEW FILE)

```python
"""Triage-specific test fixtures."""

import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def mock_triage_pipeline_dependencies(
    mock_gmail_client,
    mock_gmail_ops,
    temp_db,
    mock_email_scorer
):
    """Create all dependencies for TriagePipeline testing."""
    return {
        'gmail_client': mock_gmail_client,
        'gmail_ops': mock_gmail_ops,
        'database': temp_db,
        'model_path': 'dummy_model.txt',
        'use_embeddings': False
    }


@pytest.fixture
def mock_feature_extractor():
    """Create a mock feature extractor."""
    extractor = Mock()
    extractor.extract_features = Mock(return_value=Mock())
    return extractor


@pytest.fixture
def mock_embedding_extractor():
    """Create a mock embedding extractor."""
    extractor = Mock()
    extractor.extract = Mock(return_value=[0.1] * 384)
    return extractor
```

---

## Phase 4: Create Test Utilities Module (Priority: LOW)

**Goal**: Create helper functions for common test operations

**Expected Impact**: Better code reuse, clearer test intent

### 4.1 Create tests/test_utils.py

**File**: `tests/test_utils.py` (NEW FILE)

```python
"""Utility functions for testing."""

from datetime import datetime, timedelta
from typing import List, Dict, Any
import base64


def create_gmail_message(
    message_id: str = "msg123",
    thread_id: str = "thread123",
    from_email: str = "sender@example.com",
    from_name: str = "Test Sender",
    subject: str = "Test Subject",
    body_plain: str = "Test body",
    body_html: str = "<p>Test body</p>",
    labels: List[str] = None,
    date: datetime = None,
    headers: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create a Gmail API message structure for testing.

    Args:
        message_id: Message ID
        thread_id: Thread ID
        from_email: Sender email address
        from_name: Sender display name
        subject: Email subject
        body_plain: Plain text body
        body_html: HTML body
        labels: Gmail labels
        date: Email date
        headers: Additional headers

    Returns:
        Gmail API message dict
    """
    if labels is None:
        labels = ["INBOX", "UNREAD"]
    if date is None:
        date = datetime.now()
    if headers is None:
        headers = []

    # Convert date to Gmail timestamp (milliseconds since epoch)
    timestamp = str(int(date.timestamp() * 1000))

    # Encode body content
    plain_encoded = base64.urlsafe_b64encode(body_plain.encode()).decode()
    html_encoded = base64.urlsafe_b64encode(body_html.encode()).decode()

    # Build headers
    all_headers = [
        {'name': 'From', 'value': f'{from_name} <{from_email}>'},
        {'name': 'Subject', 'value': subject},
        {'name': 'Date', 'value': date.strftime('%a, %d %b %Y %H:%M:%S +0000')},
    ] + headers

    return {
        'id': message_id,
        'threadId': thread_id,
        'labelIds': labels,
        'snippet': body_plain[:100],
        'internalDate': timestamp,
        'payload': {
            'headers': all_headers,
            'mimeType': 'multipart/alternative',
            'parts': [
                {
                    'mimeType': 'text/plain',
                    'body': {'data': plain_encoded}
                },
                {
                    'mimeType': 'text/html',
                    'body': {'data': html_encoded}
                }
            ]
        }
    }


def assert_email_equals(actual, expected, fields=None):
    """
    Assert two email objects are equal.

    Args:
        actual: Actual email object
        expected: Expected email object (or dict of expected values)
        fields: List of fields to compare (default: all)
    """
    if fields is None:
        fields = [
            'message_id', 'from_address', 'to_address', 'subject',
            'body_plain', 'was_read', 'was_archived'
        ]

    for field in fields:
        expected_value = expected.get(field) if isinstance(expected, dict) else getattr(expected, field)
        actual_value = getattr(actual, field)
        assert actual_value == expected_value, f"Field {field} mismatch: {actual_value} != {expected_value}"


def create_date_range(days_back: int = 7) -> List[datetime]:
    """
    Create a list of dates going back N days.

    Args:
        days_back: Number of days to go back

    Returns:
        List of datetime objects
    """
    now = datetime.now()
    return [now - timedelta(days=i) for i in range(days_back)]


def mock_gmail_batch_response(emails: List[Dict[str, Any]]) -> List[Any]:
    """
    Create a mock Gmail batch response.

    Args:
        emails: List of Gmail message dicts

    Returns:
        List of mock email objects
    """
    from unittest.mock import Mock
    from src.gmail.models import Email

    return [Email.from_gmail_message(email) for email in emails]
```

### 4.2 Usage Examples

```python
# tests/test_gmail/test_models.py
from tests.test_utils import create_gmail_message, assert_email_equals

def test_parse_email():
    """Test parsing a Gmail message."""
    gmail_msg = create_gmail_message(
        subject="Important Email",
        from_email="boss@company.com"
    )

    email = Email.from_gmail_message(gmail_msg)

    assert_email_equals(email, {
        'subject': "Important Email",
        'from_address': "boss@company.com"
    }, fields=['subject', 'from_address'])
```

---

## Phase 5: Final Verification and Documentation

**Goal**: Ensure all tests pass and document the new testing patterns

### 5.1 Run Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov=tests --cov-report=term-missing

# Verify no regressions
pytest tests/ -v --tb=short
```

**Success Criteria**:
- All tests pass
- No decrease in code coverage
- Test execution time similar or improved

### 5.2 Update Documentation

**File**: `tests/README.md` (NEW FILE)

```markdown
# Testing Guide

## Overview

This test suite uses pytest with shared fixtures to ensure consistent and maintainable tests across the codebase.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_gmail/

# Run specific test file
pytest tests/test_gmail/test_operations.py

# Run specific test
pytest tests/test_gmail/test_operations.py::test_archive_dry_run

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Shared Fixtures

### Database Fixtures

Located in `tests/conftest.py`:

- `temp_db` - Temporary test database with schema
- `temp_db_session` - Database session for testing
- `email_repo` - EmailRepository instance
- `feature_store` - FeatureStore instance

**Example**:
```python
def test_save_email(email_repo, email_factory):
    email = email_factory(subject="Test")
    saved = email_repo.get_by_id(email.message_id)
    assert saved.subject == "Test"
```

### Factory Fixtures

- `email_factory` - Create test emails with defaults
- `action_factory` - Create test email actions
- `features_factory` - Create test email features

**Example**:
```python
def test_email_with_features(email_factory, features_factory):
    email = email_factory(message_id="msg1")
    features = features_factory(
        message_id=email.message_id,
        is_newsletter=True
    )
    assert features.is_newsletter is True
```

### Gmail Mock Fixtures

- `mock_gmail_auth` - Mock Gmail authenticator
- `mock_gmail_client` - Mock Gmail client
- `mock_gmail_ops` - Mock Gmail operations
- `sample_gmail_message` - Sample Gmail API message structure

**Example**:
```python
def test_fetch_emails(mock_gmail_client):
    mock_gmail_client.list_messages.return_value = (["msg1"], None)
    result = mock_gmail_client.list_messages()
    assert len(result[0]) == 1
```

## Module-Specific Fixtures

### Gmail Tests (`tests/test_gmail/conftest.py`)

- `mock_gmail_message_with_attachments` - Message with attachments
- `mock_gmail_message_multipart_nested` - Nested multipart structure

### ML Tests (`tests/test_ml/conftest.py`)

- `sample_training_data` - Sample X, y for training
- `trained_model` - Pre-trained LightGBM model
- `mock_email_scorer` - Mock email scorer
- `mock_categorizer` - Mock email categorizer

### Triage Tests (`tests/test_triage/conftest.py`)

- `mock_triage_pipeline_dependencies` - All pipeline dependencies
- `mock_feature_extractor` - Mock feature extractor
- `mock_embedding_extractor` - Mock embedding extractor

## Test Utilities

Located in `tests/test_utils.py`:

- `create_gmail_message()` - Create Gmail API message structure
- `assert_email_equals()` - Assert email objects are equal
- `create_date_range()` - Create list of dates
- `mock_gmail_batch_response()` - Create mock batch response

**Example**:
```python
from tests.test_utils import create_gmail_message

def test_parse_message():
    gmail_msg = create_gmail_message(subject="Test")
    email = Email.from_gmail_message(gmail_msg)
    assert email.subject == "Test"
```

## Writing New Tests

### 1. Use Existing Fixtures

Always check `tests/conftest.py` and module-specific conftest files before creating new fixtures.

### 2. Follow Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Fixtures: descriptive names without `test_` prefix

### 3. Keep Tests Focused

Each test should verify one specific behavior:

**Good**:
```python
def test_archive_marks_as_archived(email_factory, email_repo):
    email = email_factory(was_archived=False)
    # ... archive operation ...
    assert email_repo.get_by_id(email.message_id).was_archived is True
```

**Bad**:
```python
def test_email_operations(email_factory, email_repo):
    # Tests archive, read, labels all in one test
    ...
```

### 4. Use Descriptive Assertions

```python
# Good
assert email.was_read is True, "Email should be marked as read after opening"

# Bad
assert email.was_read
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_utils.py            # Helper functions
├── test_conftest.py         # Tests for fixtures
├── README.md                # This file
│
├── test_database/
│   ├── conftest.py          # Database-specific fixtures (if needed)
│   ├── test_database.py
│   └── test_repository.py
│
├── test_gmail/
│   ├── conftest.py          # Gmail-specific fixtures
│   ├── test_models.py
│   └── test_operations.py
│
├── test_ml/
│   ├── conftest.py          # ML-specific fixtures
│   ├── test_training.py
│   └── test_evaluation.py
│
├── test_triage/
│   ├── conftest.py          # Triage-specific fixtures
│   └── test_pipeline.py
│
└── test_scripts/
    └── test_review_decisions.py
```

## Common Patterns

### Testing Database Operations

```python
def test_save_and_retrieve(email_factory, email_repo):
    # Create
    email = email_factory(message_id="test1")

    # Retrieve
    saved = email_repo.get_by_id("test1")

    # Verify
    assert saved is not None
    assert saved.message_id == "test1"
```

### Testing Gmail Operations

```python
def test_archive_operation(mock_gmail_ops):
    # Mock setup
    result = mock_gmail_ops.archive(["msg1"], dry_run=False)

    # Verify
    assert result.success is True
    assert "msg1" in result.message_ids
```

### Testing ML Models

```python
def test_model_prediction(trained_model, sample_training_data):
    X, _ = sample_training_data
    predictions = trained_model.predict(X)

    assert len(predictions) == len(X)
    assert all(p in [0, 1] for p in predictions)
```

## Troubleshooting

### "Fixture not found"

Make sure `conftest.py` is in the `tests/` directory or a parent directory of your test file.

### "Database locked"

Close any open database connections in fixtures using `db.engine.dispose()`.

### "Import errors"

Ensure the project root is in PYTHONPATH:
```bash
export PYTHONPATH=/path/to/email-assistant:$PYTHONPATH
pytest tests/
```

### "Slow tests"

Use `pytest-xdist` for parallel execution:
```bash
pip install pytest-xdist
pytest tests/ -n auto
```
```

### 5.3 Create Migration Checklist

**File**: `MIGRATION_CHECKLIST.md` (temporary file for tracking)

```markdown
# Test Migration Checklist

## Phase 1: Create Central Fixtures
- [x] Create tests/conftest.py with shared fixtures
- [x] Create tests/test_conftest.py to verify fixtures
- [x] Run verification tests: `pytest tests/test_conftest.py -v`
- [x] Commit: "test: add shared test fixtures in conftest.py"

## Phase 2: Migrate Existing Tests

### test_database/test_repository.py
- [ ] Remove local temp_db fixture
- [ ] Replace Email creation with email_factory
- [ ] Replace EmailAction creation with action_factory
- [ ] Run tests: `pytest tests/test_database/test_repository.py -v`
- [ ] Commit: "test: migrate database repository tests to shared fixtures"

### test_database/test_database.py
- [ ] Remove local temp_db fixture
- [ ] Update test functions to use shared fixtures
- [ ] Run tests: `pytest tests/test_database/test_database.py -v`
- [ ] Commit: "test: migrate database core tests to shared fixtures"

### test_scripts/test_review_decisions.py
- [ ] Remove duplicate database setup
- [ ] Replace Email/EmailAction creation with factories
- [ ] Run tests: `pytest tests/test_scripts/test_review_decisions.py -v`
- [ ] Commit: "test: migrate review decisions tests to shared fixtures"

### test_triage/test_pipeline.py
- [ ] Remove local mock_gmail_client
- [ ] Remove local mock_database
- [ ] Replace with shared fixtures
- [ ] Run tests: `pytest tests/test_triage/test_pipeline.py -v`
- [ ] Commit: "test: migrate triage pipeline tests to shared fixtures"

### test_ml/test_training.py
- [ ] Remove duplicate test data creation
- [ ] Use shared fixtures for database
- [ ] Run tests: `pytest tests/test_ml/test_training.py -v`
- [ ] Commit: "test: migrate ML training tests to shared fixtures"

### test_ml/test_evaluation.py
- [ ] Standardize test data setup
- [ ] Use shared fixtures where applicable
- [ ] Run tests: `pytest tests/test_ml/test_evaluation.py -v`
- [ ] Commit: "test: migrate ML evaluation tests to shared fixtures"

### test_gmail/test_operations.py
- [ ] Remove local mock setup
- [ ] Use mock_gmail_client and mock_gmail_ops
- [ ] Run tests: `pytest tests/test_gmail/test_operations.py -v`
- [ ] Commit: "test: migrate Gmail operations tests to shared fixtures"

## Phase 3: Module-Specific Fixtures
- [ ] Create tests/test_gmail/conftest.py
- [ ] Create tests/test_ml/conftest.py
- [ ] Create tests/test_triage/conftest.py
- [ ] Update relevant tests to use module fixtures
- [ ] Run tests: `pytest tests/ -v`
- [ ] Commit: "test: add module-specific test fixtures"

## Phase 4: Test Utilities
- [ ] Create tests/test_utils.py
- [ ] Add helper functions
- [ ] Update tests to use utilities
- [ ] Run tests: `pytest tests/ -v`
- [ ] Commit: "test: add test utility functions"

## Phase 5: Documentation
- [ ] Create tests/README.md
- [ ] Document all fixtures and patterns
- [ ] Add usage examples
- [ ] Run final test suite: `pytest tests/ -v`
- [ ] Check coverage: `pytest tests/ --cov=src --cov-report=html`
- [ ] Commit: "docs: add comprehensive testing guide"

## Final Verification
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage maintained or improved
- [ ] No import errors
- [ ] Documentation complete
- [ ] All commits pushed
```

---

## Risk Mitigation

### Identified Risks

1. **Breaking existing tests during migration**
   - Mitigation: Migrate one file at a time, run tests after each change
   - Rollback: `git checkout tests/[file]` to restore previous version

2. **Fixture naming conflicts**
   - Mitigation: Use descriptive, unique names (e.g., `email_factory` not `create_email`)
   - Testing: Run `pytest tests/test_conftest.py` to verify fixtures work

3. **Database session management issues**
   - Mitigation: Use context managers (`with db.get_session()`) consistently
   - Testing: Add explicit tests for session lifecycle

4. **Mock configuration errors**
   - Mitigation: Provide sensible defaults in mock fixtures
   - Testing: Create test_conftest.py to verify mocks

5. **Import circular dependencies**
   - Mitigation: Keep conftest.py imports minimal, avoid importing test modules
   - Testing: Run `python -c "import tests.conftest"` to verify imports

### Rollback Plan

If issues arise during migration:

1. **Per-file rollback**: `git checkout tests/[file]` to restore previous version
2. **Full rollback**: `git revert [commit_hash]` to undo all changes
3. **Partial rollback**: Cherry-pick successful migrations, revert problematic ones

### Testing Strategy

After each phase:

```bash
# Run affected tests
pytest tests/[module]/ -v

# Run full suite
pytest tests/ -v

# Check for warnings
pytest tests/ -v --tb=short -W error

# Verify imports
python -c "import tests.conftest; print('OK')"

# Check coverage (should not decrease)
pytest tests/ --cov=src --cov-report=term
```

---

## Success Metrics

### Code Metrics

**Before Refactoring**:
- Total test lines: ~5,000
- Duplicate code: ~500 lines (10%)
- Number of fixtures: ~15 (scattered across files)
- Average test file length: ~200 lines

**After Refactoring**:
- Total test lines: ~4,650 (7% reduction)
- Duplicate code: ~150 lines (3%)
- Number of fixtures: ~15 centralized + 10 module-specific
- Average test file length: ~150 lines

**Targets**:
- [ ] Reduce total test lines by 5-10%
- [ ] Reduce duplication to <5%
- [ ] All fixtures centralized in conftest files
- [ ] Average test function length <15 lines

### Quality Metrics

- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Test execution time similar or faster
- [ ] No pytest warnings
- [ ] Documentation complete

### Developer Experience Metrics

- [ ] Easier to write new tests (use factories instead of manual setup)
- [ ] Easier to understand existing tests (less boilerplate)
- [ ] Consistent patterns across modules
- [ ] Clear documentation of available fixtures

---

## Timeline Estimate

**Total Effort**: 2-3 days (assuming 6-8 hours per day)

### Day 1 (6-8 hours)
- **Morning**: Phase 1 - Create central fixtures (2-3 hours)
  - Write tests/conftest.py (1.5 hours)
  - Write tests/test_conftest.py (0.5 hours)
  - Verify all fixtures work (0.5 hours)
  - Commit and push (0.5 hours)

- **Afternoon**: Phase 2 - Migrate first 3 test files (4-5 hours)
  - test_database/test_repository.py (1 hour)
  - test_database/test_database.py (1 hour)
  - test_scripts/test_review_decisions.py (1.5 hours)
  - Run tests after each migration (0.5 hours)
  - Commit after each file (1 hour total)

### Day 2 (6-8 hours)
- **Morning**: Phase 2 continued - Migrate remaining test files (4-5 hours)
  - test_triage/test_pipeline.py (1.5 hours)
  - test_ml/test_training.py (1 hour)
  - test_ml/test_evaluation.py (0.5 hours)
  - test_gmail/test_operations.py (1 hour)
  - Run tests after each migration (0.5 hours)

- **Afternoon**: Phase 3 - Module-specific fixtures (2-3 hours)
  - Create 3 module conftest files (1.5 hours)
  - Update tests to use module fixtures (1 hour)
  - Verify all tests pass (0.5 hours)

### Day 3 (4-6 hours)
- **Morning**: Phase 4 - Test utilities (2-3 hours)
  - Create tests/test_utils.py (1 hour)
  - Update tests to use utilities (1 hour)
  - Verify all tests pass (0.5 hours)

- **Afternoon**: Phase 5 - Documentation and verification (2-3 hours)
  - Write tests/README.md (1 hour)
  - Run full test suite (0.5 hours)
  - Check coverage report (0.5 hours)
  - Final verification and cleanup (1 hour)

**Contingency**: Add 20% buffer (1 additional day) for unexpected issues

---

## Implementation Commands

### Phase 1: Setup

```bash
# Create branch
git checkout -b refactor/test-fixtures

# Create conftest.py
touch tests/conftest.py
# (Edit file with content from Phase 1.1)

# Create test_conftest.py
touch tests/test_conftest.py
# (Edit file with content from Phase 1.2)

# Verify fixtures work
pytest tests/test_conftest.py -v

# Commit
git add tests/conftest.py tests/test_conftest.py
git commit -m "test: add shared test fixtures in conftest.py

- Add temp_db, temp_db_session, email_repo, feature_store fixtures
- Add email_factory, action_factory, features_factory for test data
- Add mock_gmail_client, mock_gmail_ops, mock_gmail_auth fixtures
- Add sample_gmail_message fixture
- Add 11 tests to verify all fixtures work correctly

This establishes centralized fixtures to eliminate ~500 lines of
duplicate code across test suite."
```

### Phase 2: Migrate Tests

```bash
# Migrate each file
# 1. Edit file to use new fixtures
# 2. Run tests to verify
pytest tests/test_database/test_repository.py -v

# 3. Commit
git add tests/test_database/test_repository.py
git commit -m "test: migrate database repository tests to shared fixtures

- Remove local temp_db fixture (15 lines)
- Replace manual Email creation with email_factory (60 lines)
- Replace manual EmailAction creation with action_factory (25 lines)
- Total: 100 lines removed, tests still pass"

# Repeat for each test file
```

### Phase 3: Module Fixtures

```bash
# Create module conftest files
mkdir -p tests/test_gmail tests/test_ml tests/test_triage

touch tests/test_gmail/conftest.py
touch tests/test_ml/conftest.py
touch tests/test_triage/conftest.py

# (Edit files with content from Phase 3)

# Verify
pytest tests/ -v

# Commit
git add tests/test_*/conftest.py
git commit -m "test: add module-specific test fixtures

- Add Gmail-specific fixtures (messages with attachments, nested multipart)
- Add ML-specific fixtures (training data, trained models, mock scorers)
- Add triage-specific fixtures (pipeline dependencies, extractors)"
```

### Phase 4: Test Utilities

```bash
# Create utilities
touch tests/test_utils.py
# (Edit file with content from Phase 4.1)

# Update tests to use utilities
# ... edit test files ...

# Verify
pytest tests/ -v

# Commit
git add tests/test_utils.py tests/test_*/test_*.py
git commit -m "test: add test utility functions

- Add create_gmail_message() for creating test Gmail messages
- Add assert_email_equals() for comparing email objects
- Add create_date_range() and mock_gmail_batch_response() helpers"
```

### Phase 5: Documentation

```bash
# Create documentation
touch tests/README.md
# (Edit file with content from Phase 5.2)

# Final verification
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Commit
git add tests/README.md
git commit -m "docs: add comprehensive testing guide

- Document all shared fixtures in conftest.py
- Document module-specific fixtures
- Document test utilities
- Add examples and common patterns
- Add troubleshooting guide"

# Merge to main
git checkout main
git merge refactor/test-fixtures
```

---

## Acceptance Criteria

### Must Have (Required for completion)

- [ ] All existing tests pass without modification (except fixture usage)
- [ ] Code duplication reduced by at least 50% (from 500 to <250 lines)
- [ ] All fixtures centralized in conftest.py files
- [ ] tests/README.md exists with comprehensive documentation
- [ ] No decrease in code coverage
- [ ] All commits follow semantic commit conventions

### Should Have (Highly desirable)

- [ ] Test execution time improved or similar
- [ ] All test files use consistent patterns
- [ ] Module-specific fixtures created for gmail, ml, triage
- [ ] Test utilities module created with helper functions
- [ ] Migration checklist completed 100%

### Nice to Have (Optional enhancements)

- [ ] Test coverage increased by 5%
- [ ] Test execution time decreased by 10%
- [ ] Additional test utilities for common patterns
- [ ] Examples of using fixtures in tests/README.md
- [ ] pytest.ini configuration optimized

---

## Post-Implementation Review

After completing all phases:

### Code Review Checklist

- [ ] Review all changed files for quality
- [ ] Verify no hardcoded paths or credentials in tests
- [ ] Check that all fixtures have docstrings
- [ ] Ensure consistent naming conventions
- [ ] Verify imports are minimal and organized

### Performance Review

```bash
# Measure test execution time before and after
# Before
time pytest tests/ -v > before_timing.txt

# After
time pytest tests/ -v > after_timing.txt

# Compare
diff before_timing.txt after_timing.txt
```

### Documentation Review

- [ ] README.md is clear and complete
- [ ] All fixtures are documented
- [ ] Examples are correct and helpful
- [ ] Troubleshooting section covers common issues

### Team Feedback

- [ ] Other developers can easily use new fixtures
- [ ] Patterns are intuitive and consistent
- [ ] Documentation answers common questions
- [ ] Migration was smooth without issues

---

## Appendix A: Quick Reference

### Common Fixture Combinations

```python
# Test database operations
def test_save(email_repo, email_factory):
    email = email_factory()
    saved = email_repo.get_by_id(email.message_id)
    assert saved is not None

# Test with actions
def test_archive(email_factory, action_factory):
    email = email_factory()
    action = action_factory(message_id=email.message_id, action_type='archive')
    assert action.action_type == 'archive'

# Test with features
def test_features(email_factory, features_factory):
    email = email_factory()
    features = features_factory(message_id=email.message_id)
    assert features.sender_domain == 'example.com'

# Test Gmail operations
def test_gmail(mock_gmail_client, mock_gmail_ops):
    mock_gmail_client.list_messages.return_value = (['msg1'], None)
    result = mock_gmail_ops.archive(['msg1'])
    assert result.success
```

### Fixture Parameters

```python
# Email with custom attributes
email_factory(
    message_id="custom123",
    subject="Custom Subject",
    from_address="custom@example.com",
    was_read=True,
    labels=["INBOX", "IMPORTANT"]
)

# Action with custom data
action_factory(
    message_id="msg123",
    action_type="keep",
    source="bot",
    action_data={'score': 0.8, 'confidence': 'high'}
)

# Features with custom values
features_factory(
    message_id="msg123",
    is_newsletter=True,
    sender_open_rate=0.9,
    subject_embedding=[0.1] * 384
)
```

---

## Appendix B: Troubleshooting Common Issues

### Issue 1: Fixture Not Found

**Error**: `fixture 'email_factory' not found`

**Solution**:
1. Verify `tests/conftest.py` exists
2. Check the fixture name is correct
3. Ensure pytest is discovering the conftest file: `pytest --fixtures`

### Issue 2: Database Locked

**Error**: `database is locked`

**Solution**:
```python
# Make sure to close sessions properly
with temp_db.get_session() as session:
    # Use session
    pass  # Session auto-closes here
```

### Issue 3: Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH=/path/to/email-assistant:$PYTHONPATH

# Or use pytest.ini
[tool:pytest]
pythonpath = .
```

### Issue 4: Stale Data Between Tests

**Error**: Tests pass individually but fail when run together

**Solution**:
```python
# Use temp_db fixture (creates fresh database for each test)
def test_something(temp_db):
    with temp_db.get_session() as session:
        # Fresh database, no stale data
        pass
```

### Issue 5: Mock Not Working

**Error**: Mock methods not being called as expected

**Solution**:
```python
# Reset mocks between tests
@pytest.fixture
def mock_gmail_client():
    client = Mock()
    # Configure mock
    yield client
    # Reset happens automatically with new fixture instance per test
```

---

## Conclusion

This plan provides a comprehensive, step-by-step approach to refactoring the test infrastructure. By following these phases sequentially and verifying after each step, you'll eliminate duplicate code, standardize testing patterns, and create a more maintainable test suite.

The key principles:
1. **Centralize fixtures** - Single source of truth in conftest.py
2. **Use factories** - Consistent test data creation
3. **Migrate incrementally** - One file at a time, verify after each
4. **Document thoroughly** - Make it easy for others to use
5. **Test the tests** - Verify fixtures work before migration

**Expected outcome**: A cleaner, more maintainable test suite with 350+ fewer lines of duplicate code, consistent patterns across all modules, and comprehensive documentation.
