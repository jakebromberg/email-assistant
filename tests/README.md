# Testing Guide

## Overview

This test suite uses pytest with shared fixtures to ensure consistent and maintainable tests across the codebase. The testing infrastructure is organized into three layers:

1. **Shared Fixtures** (`tests/conftest.py`) - Core database, factory, and mock fixtures
2. **Module-Specific Fixtures** (`tests/*/conftest.py`) - Specialized fixtures for each module
3. **Test Utilities** (`tests/test_utils.py`) - Helper functions for common operations

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_gmail/

# Run specific test file
pytest tests/test_gmail/test_operations.py

# Run specific test
pytest tests/test_gmail/test_operations.py::TestGmailOperations::test_archive_dry_run

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "test_archive"
```

### Common Options

```bash
# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Show slowest tests
pytest tests/ --durations=10
```

## Shared Fixtures

### Database Fixtures

Located in `tests/conftest.py`:

#### temp_db
Creates a temporary test database with schema initialized.

```python
def test_something(temp_db):
    with temp_db.get_session() as session:
        # Use session for database operations
        pass
```

#### temp_db_session
Provides a database session directly.

```python
def test_something(temp_db_session):
    email = Email(...)
    temp_db_session.add(email)
    temp_db_session.commit()
```

#### email_repo
Provides an EmailRepository instance.

```python
def test_something(email_repo):
    email = email_repo.get_by_id("msg123")
    assert email is not None
```

#### feature_store
Provides a FeatureStore instance.

```python
def test_something(feature_store):
    features = feature_store.get_features("msg123")
    assert features.sender_domain == "example.com"
```

### Factory Fixtures

#### email_factory
Creates test emails with sensible defaults.

```python
def test_something(email_factory):
    # Create with defaults
    email = email_factory()

    # Create with custom values
    email = email_factory(
        message_id="custom123",
        subject="Custom Subject",
        from_address="custom@example.com",
        was_read=True
    )
```

**Available Parameters:**
- `message_id` (auto-generated if not provided)
- `thread_id` (auto-generated if not provided)
- `from_address`, `from_name`
- `to_address`
- `subject`, `body_plain`, `body_html`
- `date` (defaults to now)
- `labels` (defaults to ["INBOX", "UNREAD"])
- `snippet`
- `was_read`, `was_archived`, `is_important`, `is_starred`
- `opened_at`

**Note:** Email is automatically saved to the database.

#### action_factory
Creates test email actions.

```python
def test_something(action_factory, email_factory):
    email = email_factory(message_id="msg1")

    action = action_factory(
        message_id="msg1",
        action_type="archive",
        action_data={'score': 0.95, 'confidence': 'high'}
    )
```

**Available Parameters:**
- `message_id` (required)
- `action_type` (default: 'keep')
- `source` (default: 'bot')
- `action_data` (default: test data)
- `timestamp` (default: now)

#### features_factory
Creates test email features.

```python
def test_something(features_factory, email_factory):
    email = email_factory(message_id="msg1")

    features = features_factory(
        message_id="msg1",
        is_newsletter=True,
        sender_open_rate=0.9
    )
```

**Available Parameters:**
- `message_id` (required)
- `sender_domain`, `sender_email_count`, `sender_open_rate`
- `sender_days_since_last`
- `is_newsletter`, `subject_length`, `body_length`
- `has_attachments`, `thread_length`
- `day_of_week`, `hour_of_day`
- `subject_embedding`, `body_embedding`

### Gmail Mock Fixtures

#### mock_gmail_auth
Mock Gmail authenticator.

```python
def test_something(mock_gmail_auth):
    client = GmailClient(mock_gmail_auth)
```

#### mock_gmail_client
Mock Gmail client with common methods.

```python
def test_something(mock_gmail_client):
    mock_gmail_client.list_messages.return_value = (["msg1"], None)
    messages, _ = mock_gmail_client.list_messages()
```

**Mocked Methods:**
- `list_messages()` - Returns `([], None)`
- `list_all_messages()` - Returns `[]`
- `get_messages_batch()` - Returns `[]`
- `get_message()` - Returns `None`

#### mock_gmail_ops
Mock Gmail operations with successful results.

```python
def test_something(mock_gmail_ops):
    result = mock_gmail_ops.archive(['msg1'], dry_run=False)
    assert result.success
```

**Mocked Methods:**
- `archive()`, `mark_read()`, `mark_unread()`
- `move_to_inbox()`, `add_labels()`, `remove_labels()`

All return successful `OperationResult` by default.

#### sample_gmail_message
Sample Gmail API message structure.

```python
def test_something(sample_gmail_message):
    email = Email.from_gmail_message(sample_gmail_message)
```

## Module-Specific Fixtures

### Gmail Fixtures (`tests/test_gmail/conftest.py`)

#### Message Variants
- `mock_gmail_message_with_attachments` - Message with PDF attachment
- `mock_gmail_message_multipart_nested` - Nested multipart structure
- `mock_gmail_message_plain_only` - Plain text only (no HTML)
- `mock_gmail_message_html_only` - HTML only (no plain text)
- `mock_gmail_message_with_list_unsubscribe` - Newsletter with unsubscribe

#### Operation Results
- `mock_operation_result` - Successful operation result
- `mock_operation_result_failed` - Failed operation result

### ML Fixtures (`tests/test_ml/conftest.py`)

#### Training Data
- `sample_training_data` - Basic training data (X, y)
- `sample_training_data_with_embeddings` - Training data with embeddings
- `trained_model` - Pre-trained LightGBM model

#### Scorers and Categorizers
- `mock_email_scorer` - Basic email scorer (score=0.5, keep)
- `mock_email_scorer_high_confidence` - High confidence scorer (score=0.9, keep)
- `mock_email_scorer_archive` - Archive scorer (score=0.1, archive)
- `mock_categorizer` - Email categorizer

#### Predictions
- `sample_predictions` - Normal prediction data
- `sample_predictions_perfect` - Perfect predictions (100% accuracy)
- `sample_predictions_poor` - Random predictions (50% accuracy)

#### Trainers and Evaluators
- `mock_model_trainer` - Mock ModelTrainer instance
- `mock_model_evaluator` - Mock ModelEvaluator instance

### Triage Fixtures (`tests/test_triage/conftest.py`)

#### Pipeline Dependencies
- `mock_triage_pipeline_dependencies` - All pipeline dependencies dict
- `mock_triage_pipeline` - Configured TriagePipeline instance

#### Extractors
- `mock_feature_extractor` - Mock metadata extractor
- `mock_historical_extractor` - Mock historical pattern extractor
- `mock_embedding_extractor` - Mock embedding extractor (enabled)
- `mock_embedding_extractor_disabled` - Mock embedding extractor (disabled)

#### Sample Emails
- `sample_gmail_email_for_triage` - Single email for triage
- `sample_gmail_emails_batch` - Batch of 3 emails

#### Triage Results
- `mock_triage_result_keep` - Keep decision
- `mock_triage_result_archive` - Archive decision
- `mock_triage_result_low_confidence` - Low confidence decision

## Test Utilities (`tests/test_utils.py`)

### Gmail Message Creation

#### create_gmail_message()
Create full Gmail API message structure.

```python
from tests.test_utils import create_gmail_message

message = create_gmail_message(
    message_id="test123",
    subject="Test Subject",
    from_email="sender@example.com",
    has_attachments=True
)
```

#### create_gmail_newsletter_message()
Create newsletter with List-Unsubscribe header.

```python
from tests.test_utils import create_gmail_newsletter_message

newsletter = create_gmail_newsletter_message(
    message_id="news123",
    subject="Weekly Tech Newsletter"
)
```

### Assertion Helpers

#### assert_email_equals()
Compare email objects.

```python
from tests.test_utils import assert_email_equals

assert_email_equals(actual_email, expected_email)

# Compare specific fields only
assert_email_equals(
    actual_email,
    {'subject': 'Test', 'from_address': 'test@example.com'},
    fields=['subject', 'from_address']
)
```

#### assert_email_features_complete()
Validate EmailFeatures has all required fields.

```python
from tests.test_utils import assert_email_features_complete

features = feature_store.get_features("msg123")
assert_email_features_complete(features)
```

#### assert_operation_result_success()
Validate successful operation result.

```python
from tests.test_utils import assert_operation_result_success

result = ops.archive(['msg1', 'msg2'])
assert_operation_result_success(result, expected_message_ids=['msg1', 'msg2'])
```

#### assert_triage_decision_valid()
Validate triage decision structure.

```python
from tests.test_utils import assert_triage_decision_valid

decision = scorer.make_decision(email, features)
assert_triage_decision_valid(decision)
```

### Mock Data Creation

#### create_mock_features()
Create feature dictionary.

```python
from tests.test_utils import create_mock_features

features = create_mock_features(
    message_id="msg123",
    is_newsletter=True,
    sender_open_rate=0.9
)
```

#### create_mock_triage_decision()
Create triage decision dictionary.

```python
from tests.test_utils import create_mock_triage_decision

decision = create_mock_triage_decision(
    action='archive',
    confidence='high',
    score=0.15
)
```

#### create_date_range()
Generate sequence of dates.

```python
from tests.test_utils import create_date_range

dates = create_date_range(days_back=7)  # Last 7 days
```

## Writing New Tests

### 1. Use Existing Fixtures

Always check `tests/conftest.py` and module-specific conftest files before creating new fixtures.

```python
# Good: Use existing fixtures
def test_save_email(email_factory, email_repo):
    email = email_factory(subject="Test")
    saved = email_repo.get_by_id(email.message_id)
    assert saved.subject == "Test"

# Bad: Manual setup
def test_save_email(temp_db):
    with temp_db.get_session() as session:
        email = Email(
            message_id="test123",
            from_address="test@example.com",
            # ... 20 more lines of boilerplate ...
        )
        session.add(email)
        session.commit()
```

### 2. Follow Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestGmailOperations`)
- Test functions: `test_*` (e.g., `test_archive_dry_run`)
- Fixtures: descriptive names without `test_` prefix

### 3. Keep Tests Focused

Each test should verify one specific behavior:

```python
# Good: Single behavior
def test_archive_marks_as_archived(email_factory, email_repo):
    email = email_factory(message_id="msg1", was_archived=False)
    # ... archive operation ...
    assert email_repo.get_by_id("msg1").was_archived is True

# Bad: Multiple behaviors
def test_email_operations(email_factory, email_repo):
    # Tests archive, read, labels all in one test
    # ... 50 lines ...
```

### 4. Use Descriptive Assertions

```python
# Good: Clear assertion message
assert email.was_read is True, \
    "Email should be marked as read after opening"

# Bad: No context
assert email.was_read
```

### 5. Use Test Utilities

```python
# Good: Use utility function
from tests.test_utils import assert_email_equals

assert_email_equals(actual, expected)

# Bad: Manual comparison
assert actual.message_id == expected.message_id
assert actual.subject == expected.subject
assert actual.from_address == expected.from_address
# ... 10 more lines ...
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_utils.py            # Helper functions
├── README.md                # This file
│
├── test_database/
│   ├── test_database.py     # Database core tests
│   └── test_repository.py   # Repository tests
│
├── test_gmail/
│   ├── conftest.py          # Gmail-specific fixtures
│   ├── test_client.py       # Gmail client tests
│   ├── test_models.py       # Email model tests
│   ├── test_operations.py   # Gmail operations tests
│   └── test_rate_limiter.py # Rate limiter tests
│
├── test_ml/
│   ├── conftest.py          # ML-specific fixtures
│   ├── test_training.py     # Training tests
│   └── test_evaluation.py   # Evaluation tests
│
├── test_triage/
│   ├── conftest.py          # Triage-specific fixtures
│   └── test_pipeline.py     # Pipeline tests
│
└── test_scripts/
    └── test_review_decisions.py  # Review script tests
```

## Common Testing Patterns

### Testing Database Operations

```python
def test_save_and_retrieve(email_factory, email_repo):
    # Create
    email = email_factory(message_id="test1", subject="Test")

    # Retrieve
    saved = email_repo.get_by_id("test1")

    # Verify
    assert saved is not None
    assert saved.subject == "Test"
```

### Testing Gmail Operations

```python
def test_archive_operation(mock_gmail_ops):
    # Execute
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

### Testing with Multiple Emails

```python
def test_batch_processing(email_factory):
    # Create multiple emails
    emails = [
        email_factory(message_id=f"msg{i}")
        for i in range(3)
    ]

    # Process batch
    # ... your code ...

    # Verify all processed
    assert len(emails) == 3
```

### Testing Error Conditions

```python
def test_handles_missing_email(email_repo):
    # Should return None for missing email
    email = email_repo.get_by_id("nonexistent")
    assert email is None
```

## Troubleshooting

### "Fixture not found"

Make sure `conftest.py` is in the `tests/` directory or a parent directory of your test file.

```bash
# Verify pytest can find fixtures
pytest --fixtures
```

### "Database locked"

Close any open database connections in fixtures:

```python
@pytest.fixture
def temp_db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    yield db
    db.engine.dispose()  # Important: close connections
```

### "Import errors"

Ensure the project root is in PYTHONPATH:

```bash
export PYTHONPATH=/path/to/email-assistant:$PYTHONPATH
pytest tests/
```

Or add to pytest.ini:

```ini
[tool:pytest]
pythonpath = .
```

### "Slow tests"

Use pytest-xdist for parallel execution:

```bash
pip install pytest-xdist
pytest tests/ -n auto
```

### "Flaky tests"

Tests that sometimes pass/fail are often caused by:
- Unordered data (use sorted() for comparisons)
- Timestamp sensitivity (use date ranges or mocking)
- Shared state between tests (ensure proper cleanup)

## Coverage

Check test coverage regularly:

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

**Coverage Goals:**
- Critical paths: >90%
- Overall: >80%
- New code: 100%

## Best Practices

### DO:
✅ Use shared fixtures from conftest.py
✅ Use factory fixtures instead of manual object creation
✅ Write focused tests (one behavior per test)
✅ Use descriptive test names and assertion messages
✅ Test edge cases and error conditions
✅ Run tests before committing
✅ Keep test code DRY with utilities

### DON'T:
❌ Create duplicate fixtures
❌ Write long tests that test multiple things
❌ Use magic numbers without explanation
❌ Skip writing tests for "simple" code
❌ Commit failing tests
❌ Leave commented-out test code

## Performance Optimization

### Slow Tests

If tests are slow, consider:
- Using mocks instead of real database operations
- Reducing the size of test data
- Running tests in parallel
- Optimizing fixture setup

### Memory Usage

For large test suites:
- Use session-scoped fixtures for expensive setup
- Clean up resources properly
- Avoid loading large files in memory

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Mocking with unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## Getting Help

If you encounter issues:
1. Check this README for common patterns
2. Look at existing tests for examples
3. Run `pytest --fixtures` to see available fixtures
4. Check the test utilities in `tests/test_utils.py`
5. Ask the team for help
