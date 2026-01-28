# Gmail Email Triage Bot

An intelligent Gmail triage system that learns from your email behavior to automatically archive newsletters and low-priority emails while keeping important emails in your inbox.

## Features

- **Smart Learning**: Learns from multiple signals including read/unread status, archived vs inbox location, user-applied labels, which emails you open, reply patterns, and time-to-open
- **Auto-Categorization**: Automatically labels emails (e.g., "Bot/Newsletter-Tech", "Bot/Promotional", "Bot/Personal") for easy retrieval
- **Easy Search**: Find archived emails using Gmail search with category labels
- **Interactive Feedback**: Command-line tool to review and correct bot decisions
- **Implicit Learning**: Automatically learns from label changes, moving emails back to inbox, and opening patterns
- **Privacy-First**: Runs locally on your machine, no data sent to external services

## Architecture

```
Gmail API → Features → ML Model → Actions
    ↓           ↓           ↓          ↓
 Emails    Metadata   LightGBM   Archive/
           Patterns   Predictor   Label
           Embeddings
```

**Tech Stack**:
- Gmail API for email access and modification
- SQLite database with vector support for storage
- LightGBM classifier for predictions
- Sentence-transformers for topic embeddings
- Python 3.12 or 3.13

**Performance**: Process ~100 emails/day in <5 seconds

## Project Status

**Status**: All Phases Complete - Production Ready

- [x] Phase 1: Gmail API Foundation
  - [x] Gmail OAuth2 authentication
  - [x] Email fetching and parsing
  - [x] Label and archive operations
  - [x] Batch operations and rate limiting
- [x] Phase 2: Data collection & storage
  - [x] SQLite database with schema
  - [x] Email repository layer
  - [x] Historical email export
  - [x] Action and feedback tracking
- [x] Phase 3: Feature engineering
  - [x] Metadata feature extraction
  - [x] Historical pattern features
  - [x] Topic embeddings (sentence-transformers)
  - [x] Feature storage and retrieval
- [x] Phase 4: ML model training
  - [x] LightGBM classifier training
  - [x] Email scoring and inference
  - [x] Threshold-based decision logic
  - [x] Email categorization system
  - [x] Model evaluation and metrics
- [x] Phase 5: Automation and feedback loop
  - [x] Daily triage pipeline
  - [x] Interactive CLI feedback tool
  - [x] Implicit feedback collection
  - [x] Model retraining with feedback
  - [x] launchd scheduling for macOS

## Prerequisites

1. **Python 3.12 or 3.13** (Python 3.14 not yet recommended due to limited ML package support)
2. **Gmail Account** with API access enabled
3. **Google Cloud Project** with Gmail API enabled

## Setup

### 1. Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project "Email Assistant"
3. Enable the Gmail API:
   - Navigate to "APIs & Services" → "Library"
   - Search for "Gmail API"
   - Click "Enable"
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Choose "Desktop app" as application type
   - Name it "Email Assistant"
   - Download the credentials JSON file
5. Save the downloaded file as `credentials/credentials.json`

### 2. Project Installation

```bash
# Clone or navigate to project directory
cd /Users/jake/Developer/email-assistant

# Verify Python version (should be 3.12 or 3.13)
python3 --version

# If you have Python 3.14, install Python 3.12
# brew install python@3.12

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env if needed (defaults should work)
```

### 3. Authentication

Run the authentication script to set up OAuth2:

```bash
python scripts/authenticate.py
```

This will:
- Open your browser for authorization
- Save access tokens to `credentials/token.json`
- Enable API access for the application

### 4. Test Connection

Verify everything is working:

```bash
python scripts/test_connection.py
```

This will test:
- Authentication
- Listing messages
- Fetching email details
- Batch operations
- Label operations (dry-run only)

## Usage

### Python API

```python
from src.gmail import GmailAuthenticator, GmailClient, GmailOperations
from src.utils import Config

# Load configuration
config = Config.load()

# Initialize
auth = GmailAuthenticator(
    credentials_path=config.GMAIL_CREDENTIALS_PATH,
    token_path=config.GMAIL_TOKEN_PATH
)
client = GmailClient(auth)
ops = GmailOperations(client)

# List recent unread emails
message_ids, _ = client.list_messages(
    query="is:unread",
    max_results=10
)

# Fetch email details
emails = client.get_messages_batch(message_ids)
for email in emails:
    print(f"{email.from_address}: {email.subject}")

# Archive emails (dry-run)
result = ops.archive(message_ids, dry_run=True)
print(result.message)

# Add labels
result = ops.add_labels(
    message_ids,
    ['Bot/Newsletter-Tech'],
    dry_run=False
)
```

### Gmail Client API

**GmailClient** - Fetch emails:
- `list_messages(query, max_results, label_ids, after_date, before_date)` - List message IDs
- `list_all_messages(...)` - List all messages with pagination
- `get_message(message_id)` - Fetch single email
- `get_messages_batch(message_ids, batch_size)` - Batch fetch emails
- `get_all_labels()` - Get all Gmail labels

**GmailOperations** - Modify emails:
- `archive(message_ids, dry_run)` - Archive messages
- `move_to_inbox(message_ids, dry_run)` - Unarchive messages
- `mark_read(message_ids, dry_run)` - Mark as read
- `mark_unread(message_ids, dry_run)` - Mark as unread
- `add_labels(message_ids, label_names, dry_run)` - Add labels
- `remove_labels(message_ids, label_names, dry_run)` - Remove labels
- `create_label(label_name)` - Create new label
- `get_or_create_label(label_name)` - Get or create label

### Database API (Phase 2)

**Database** - Connection management:
```python
from src.database import Database

db = Database("data/emails.db")
db.create_tables()

with db.get_session() as session:
    # Use session for queries
    pass
```

**EmailRepository** - Data access:
```python
from src.database import EmailRepository

with db.get_session() as session:
    repo = EmailRepository(session)

    # Save emails
    repo.save_email(gmail_email)
    repo.save_emails_batch(gmail_emails)

    # Query emails
    email = repo.get_by_id("msg123")
    emails = repo.get_by_sender("sender@example.com")
    unread = repo.get_unread(limit=50)

    # Record actions
    repo.record_action("msg123", "archive", source="bot")

    # Save feedback
    repo.save_feedback("msg123", decision_correct=False,
                      user_comment="Always important")

    # Get statistics
    stats = repo.get_sender_stats("sender@example.com")
```

**EmailCollector** - Historical export:
```python
from src.collectors import EmailCollector

collector = EmailCollector(client, db)

# Export last 6 months
count = collector.export_historical(months=6)

# Export recent emails
count = collector.export_recent(days=1)

# Update specific emails
collector.update_email("msg123")
```

### Command-Line Scripts

**Export historical emails:**
```bash
# Export last 6 months
python scripts/export_history.py --months 6

# Export with limits
python scripts/export_history.py --months 3 --max 5000

# Custom database path
python scripts/export_history.py --db-path data/custom.db
```

**Test database:**
```bash
python scripts/test_database.py
```

**Build features (Phase 3):**
```bash
# Test feature extraction
python scripts/test_features.py

# Build features for all emails
python scripts/build_features.py

# Build features with options
python scripts/build_features.py --batch-size 50 --limit 1000

# Skip embeddings (faster, less accurate)
python scripts/build_features.py --skip-embeddings

# Only compute embeddings for emails with metadata
python scripts/build_features.py --embeddings-only
```

**Train model (Phase 4):**
```bash
# Train LightGBM model
python scripts/train_model.py

# Train without embeddings (faster)
python scripts/train_model.py --no-embeddings

# Custom test split
python scripts/train_model.py --test-size 0.3

# Test trained model
python scripts/test_model.py
python scripts/test_model.py --model models/model_v20240101.txt --limit 20
```

**Run triage (Phase 5):**
```bash
# Daily triage (DRY RUN - no changes)
python scripts/triage_inbox.py --dry-run

# Run triage for real
python scripts/triage_inbox.py

# Custom options
python scripts/triage_inbox.py --days 2 --no-embeddings

# Review bot decisions
python scripts/review_decisions.py
python scripts/review_decisions.py --days 7 --filter archived

# Collect implicit feedback
python scripts/collect_feedback.py --days 7

# Retrain model with feedback
python scripts/retrain_model.py
```

**Schedule automation (macOS):**
```bash
# Copy launchd configs
cp config/launchd/*.plist ~/Library/LaunchAgents/

# Load agents
launchctl load ~/Library/LaunchAgents/com.user.email-triage.plist
launchctl load ~/Library/LaunchAgents/com.user.email-feedback.plist
launchctl load ~/Library/LaunchAgents/com.user.email-retrain.plist

# Check status
launchctl list | grep email

# View logs
tail -f logs/triage.log
```

## Smart Rate Limiting

The Gmail client includes intelligent quota management to avoid hitting API rate limits:

### Quota-Aware Operation

**Gmail API Limits:**
- 250 quota units per user per second
- `messages.get(format='full')` costs 5 quota units per message
- `messages.get(format='metadata')` costs 2 quota units per message

**Smart Features:**
```python
from src.gmail import GmailClient, QuotaCosts

# Initialize with rate limiting enabled (default)
client = GmailClient(auth, enable_rate_limiting=True, enable_adaptive_sizing=True)

# Fetch with different formats to save quota
emails = client.get_messages_batch(
    message_ids,
    message_format='full'      # 5 quota/msg (default)
    # message_format='metadata'  # 2 quota/msg (faster, less data)
    # message_format='minimal'   # 1 quota/msg (fastest, minimal data)
)
```

### Adaptive Batch Sizing

The rate limiter automatically adjusts batch size based on success/failure:
- **Starts at**: 50 messages per batch
- **Increases**: +5 messages after 5 consecutive successful batches (up to 50 max)
- **Decreases**: ÷2 when rate limits hit (down to 10 min)

### Token Bucket Algorithm

Tracks quota usage in real-time:
- Maintains available quota tokens (max 250)
- Refills at 250 tokens/second
- Waits automatically when quota insufficient
- Consumes tokens only for successful requests

### Intelligent Retry

- Parses `Retry-After` header from Google's 429 responses
- Exponential backoff: 2s, 4s, 8s for retries
- Retries only failed messages, not entire batch
- Tracks and reports success rates

### Performance Stats

After each operation, see quota usage:
```
Rate limiter stats - Quota consumed: 2500, Time waited: 8.3s,
Batch size: 45, Success rate: 96.2%
```

## Configuration

Configuration is managed via environment variables in `.env`:

```env
GMAIL_CREDENTIALS_PATH=credentials/credentials.json
GMAIL_TOKEN_PATH=credentials/token.json
LOG_LEVEL=INFO
LOG_DIR=logs
DATA_DIR=data
```

## Project Structure

```
email-assistant/
├── README.md                 # This file
├── CLAUDE.md                # Project overview and roadmap
├── SETUP.md                 # Quick setup guide
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
├── src/                     # Source code
│   ├── gmail/               # Gmail API client
│   │   ├── auth.py          # OAuth2 authentication
│   │   ├── client.py        # API client
│   │   ├── models.py        # Email data models
│   │   ├── operations.py    # Email operations
│   │   └── rate_limiter.py  # Quota management
│   ├── database/            # Database layer
│   │   ├── schema.py        # SQLAlchemy models
│   │   ├── database.py      # Connection management
│   │   └── repository.py    # Data access layer
│   ├── collectors/          # Email collectors
│   │   └── email_collector.py  # Historical export
│   ├── features/            # Feature engineering
│   │   ├── metadata.py      # Metadata extraction
│   │   ├── historical.py    # Historical patterns
│   │   ├── embeddings.py    # Topic embeddings
│   │   └── feature_store.py # Feature storage
│   ├── ml/                  # Machine learning
│   │   ├── trainer.py       # Model training
│   │   ├── scorer.py        # Email scoring
│   │   └── evaluation.py    # Model evaluation
│   ├── triage/              # Triage pipeline
│   │   ├── pipeline.py      # Daily triage orchestration
│   │   ├── decision.py      # Decision logic
│   │   └── feedback.py      # Feedback collection
│   └── utils/               # Utilities
│       ├── config.py        # Configuration
│       └── logger.py        # Logging
├── scripts/                 # Command-line scripts
│   ├── authenticate.py      # OAuth setup
│   ├── test_connection.py   # Test API connection
│   ├── export_history.py    # Export historical emails
│   ├── build_features.py    # Build features for emails
│   ├── train_model.py       # Train ML model
│   ├── test_model.py        # Test model predictions
│   ├── triage_inbox.py      # Daily triage pipeline
│   ├── review_decisions.py  # Interactive feedback CLI
│   ├── collect_feedback.py  # Implicit feedback collection
│   └── retrain_model.py     # Retrain with feedback
├── tests/                   # Unit tests
│   ├── conftest.py          # Shared fixtures
│   ├── test_utils.py        # Test utilities
│   ├── test_gmail/
│   ├── test_database/
│   ├── test_ml/
│   └── test_triage/
├── config/                  # Configuration files
│   └── launchd/             # macOS scheduling configs
├── models/                  # Trained models (gitignored)
├── credentials/             # OAuth credentials (gitignored)
├── logs/                    # Log files (gitignored)
└── data/                    # Data storage (gitignored)
```

## Logging

Logs are written to:
- Console: INFO level and above
- Files: `logs/<module>.log` with daily rotation (30 day retention)

Log levels can be configured via `LOG_LEVEL` in `.env`.

## Development

### Testing

This project has a comprehensive test suite with **202 tests** and **71% code coverage**.

#### Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific module
pytest tests/test_gmail/

# Run specific test
pytest tests/test_gmail/test_operations.py::test_archive_dry_run
```

#### Test Infrastructure

The test suite uses a modern fixture-based architecture:

**Shared Fixtures** (`tests/conftest.py`):
- Database fixtures: `temp_db`, `temp_db_session`, `email_repo`, `feature_store`
- Factory fixtures: `email_factory`, `action_factory`, `features_factory`
- Gmail mocks: `mock_gmail_client`, `mock_gmail_ops`, `mock_gmail_auth`

**Module-Specific Fixtures**:
- `tests/test_gmail/conftest.py` - Gmail message variants, operation results
- `tests/test_ml/conftest.py` - Training data, models, scorers, predictions
- `tests/test_triage/conftest.py` - Pipeline dependencies, extractors, results

**Test Utilities** (`tests/test_utils.py`):
- Gmail message creation helpers
- Assertion helpers for email, features, operations
- Mock data creators

#### Writing Tests

Example using shared fixtures:

```python
def test_save_email(email_factory, email_repo):
    # Create test email with defaults
    email = email_factory(subject="Test Subject")

    # Retrieve and verify
    saved = email_repo.get_by_id(email.message_id)
    assert saved.subject == "Test Subject"
```

For complete testing documentation, see **[tests/README.md](tests/README.md)**.

#### Test Coverage

Current coverage by module:
- Gmail operations: 97%
- Rate limiter: 99%
- Email models: 94%
- Feature extraction: 93-98%
- ML evaluation: 93%
- Overall: **71%**

#### Running in CI/CD

Tests are automatically run on every push via GitHub Actions:

```bash
# What CI runs
pytest tests/ --cov=src --cov-report=xml
```

See `.github/workflows/tests.yml` for the full CI configuration.

### Code Style

- Type hints throughout
- Docstrings for all public functions
- PEP 8 compliant
- Pytest for all tests
- Mock external dependencies (Gmail API)

## Security & Privacy

- **OAuth2 tokens** are stored locally in `credentials/token.json` (gitignored)
- **No data** is sent to external services (except Gmail API)
- **Gmail scope**: `gmail.modify` (read + modify labels/archive, but not send)
- **Credentials** are never committed to version control

## Troubleshooting

### Authentication Issues

If you get authentication errors:
```bash
# Revoke and re-authenticate
rm credentials/token.json
python scripts/authenticate.py
```

### Rate Limiting

The client implements exponential backoff for rate limits. If you hit rate limits:
- Reduce batch sizes
- Add delays between operations
- Check logs for warnings

### Missing Credentials

If you see "Credentials file not found":
1. Ensure you've downloaded credentials from Google Cloud Console
2. Save as `credentials/credentials.json`
3. Run `python scripts/authenticate.py`

### Python Version Issues

If you encounter build errors when installing dependencies:

**Error: "No matching distribution found for torch==2.9.0"** or **"Preparing metadata failed"**
- Likely caused by Python 3.14 being too new or Python 3.11 being too old
- Solution: Use Python 3.12 or 3.13

```bash
# Check your version
python3 --version

# Install Python 3.12 if needed
brew install python@3.12

# Recreate virtualenv
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Error: "No space left on device"** during installation
- scikit-learn and other packages need to compile from source
- Solution: Free up disk space or use Python 3.12/3.13 (which have pre-built wheels)

```bash
# Check disk space
df -h .

# Clean up space
brew cleanup -s
pip cache purge
```

## License

Private project for personal use.

## Contributing

This is a personal project. If you'd like to use it, feel free to fork and adapt for your needs.
