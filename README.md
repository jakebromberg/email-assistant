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
- Python 3.11+

**Performance**: Process ~100 emails/day in <5 seconds

## Project Status

**Current Phase**: Phase 3 - Feature Engineering ✅

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
- [ ] Phase 4: ML model training
- [ ] Phase 5: Automation and feedback loop

## Prerequisites

1. **Python 3.11+**
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

# Create virtual environment
python3 -m venv venv
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
│   │   └── operations.py    # Email operations
│   ├── database/            # Database layer (Phase 2)
│   │   ├── schema.py        # SQLAlchemy models
│   │   ├── database.py      # Connection management
│   │   └── repository.py    # Data access layer
│   ├── collectors/          # Email collectors (Phase 2)
│   │   └── email_collector.py  # Historical export
│   ├── features/            # Feature engineering (Phase 3)
│   │   ├── metadata.py      # Metadata extraction
│   │   ├── historical.py    # Historical patterns
│   │   ├── embeddings.py    # Topic embeddings
│   │   └── feature_store.py # Feature storage
│   └── utils/               # Utilities
│       ├── config.py        # Configuration
│       └── logger.py        # Logging
├── scripts/                 # Command-line scripts
│   ├── authenticate.py      # OAuth setup
│   ├── test_connection.py   # Test API connection
│   ├── test_database.py     # Test database setup (Phase 2)
│   ├── export_history.py    # Export historical emails (Phase 2)
│   ├── test_features.py     # Test feature extraction (Phase 3)
│   └── build_features.py    # Build features for emails (Phase 3)
├── tests/                   # Unit tests
│   ├── test_gmail/
│   └── test_database/
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

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_gmail/test_client.py
```

### Code Style

- Type hints throughout
- Docstrings for all public functions
- PEP 8 compliant

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

## Next Steps

### Phase 2: Data Collection
- Set up SQLite database
- Export 3-6 months of historical emails
- Store metadata and labels

### Phase 3: Feature Engineering
- Extract metadata features
- Compute historical patterns
- Generate topic embeddings

### Phase 4: ML Model
- Train LightGBM classifier
- Tune prediction thresholds
- Implement categorization logic

### Phase 5: Automation
- Daily triage script
- Interactive feedback tool
- Scheduled automation via launchd
- Model retraining pipeline

## License

Private project for personal use.

## Contributing

This is a personal project. If you'd like to use it, feel free to fork and adapt for your needs.
