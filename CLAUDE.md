# Gmail Email Triage Bot - Project Overview

An intelligent Gmail triage system that learns from your email behavior to automatically archive newsletters and low-priority emails while keeping important emails in your inbox.

## Current Status: Phase 3 Complete ✅

**Phase 1: Gmail API Foundation** - Complete ✅
- OAuth2 authentication, email fetching, operations

**Phase 2: Data Collection & Storage** - Complete ✅
- SQLite database with schema
- Email repository and data access layer
- Historical email export functionality
- Action and feedback tracking

**Phase 3: Feature Engineering** - Complete ✅
- Metadata feature extraction (sender, time, structure)
- Historical pattern features (open rates, email frequency)
- Topic embeddings via sentence-transformers (384-dim)
- Feature storage with database integration

## Architecture

```
Gmail API → Feature Engineering → ML Model → Automated Actions
    ↓              ↓                  ↓              ↓
 Emails      Metadata           LightGBM      Archive +
             Patterns          Classifier     Categorize
             Embeddings        (0.0-1.0)      with Labels
```

**Tech Stack:**
- **Gmail API**: Email access and modification
- **SQLite + vector search**: Local data storage with embeddings
- **LightGBM**: Fast gradient boosting classifier
- **Sentence-transformers**: Topic embeddings (all-MiniLM-L6-v2)
- **launchd**: macOS native scheduling

**Performance Target**: Process ~100 emails/day in <5 seconds

## Learning Signals

The bot learns from multiple behavioral signals:
- **Read/unread status**: Which emails you actually read
- **Inbox vs archived**: What you keep vs remove
- **User-applied labels**: Your manual categorization (strong importance signal)
- **Open behavior**: Which emails you click on
- **Reply patterns**: What you respond to
- **Time-to-open**: How quickly you engage
- **Explicit feedback**: Interactive CLI tool and label-based corrections

## Phase 1: Gmail API Foundation ✅ COMPLETE

**Implemented:**
- OAuth2 authentication with automatic token refresh
- Gmail API client with batch operations (up to 100 emails)
- Rate limiting with exponential backoff
- Email data models (parse headers, body, metadata)
- Operations: archive, label, mark read/unread
- Dry-run mode for safe testing
- Comprehensive logging with daily rotation
- Configuration management via `.env`

**API Available:**
```python
from src.gmail import GmailAuthenticator, GmailClient, GmailOperations

client = GmailClient(auth)
message_ids, _ = client.list_messages(query="is:unread", max_results=100)
emails = client.get_messages_batch(message_ids)
ops.archive(message_ids, dry_run=False)
ops.add_labels(message_ids, ['Bot/Newsletter-Tech'])
```

## Phase 2: Data Collection & Storage (Next)

**Objectives:**
- Set up SQLite database with vector support (`sqlite-vss`)
- Export 3-6 months of historical emails (~10k emails)
- Store metadata, labels, and behavioral signals
- Track email actions for learning

**Database Schema:**
- `emails` - Message metadata, body, labels, read status
- `email_labels` - Many-to-many label relationships
- `email_actions` - Audit trail of all actions (user and bot)
- `feedback_reviews` - Interactive feedback from CLI tool

**Implementation:**
```
src/database/
  ├── schema.py          # SQLAlchemy models
  ├── database.py        # Connection management
  └── repository.py      # Data access layer

scripts/
  └── export_history.py  # One-time historical export
```

## Phase 3: Feature Engineering

**Objectives:**
- Extract metadata features from emails
- Compute historical patterns (sender open rates, domain patterns)
- Generate topic embeddings using sentence-transformers
- Store features for ML training

**Feature Types:**

**Metadata Features:**
- Sender domain, address hash
- Newsletter detection (List-Unsubscribe header)
- Time features (day of week, hour of day)
- Structure (subject/body length, attachments, thread length)
- **User-applied labels** (strong signal of importance)

**Historical Pattern Features:**
- Sender open rate (% of emails from this sender you opened)
- Sender email count and recency
- Domain-level patterns
- Time-of-day open rates

**Topic Features (ML-based):**
- Subject embeddings (384-dim from all-MiniLM-L6-v2)
- Body embeddings (first 512 tokens)
- Similarity to previously read emails
- Similarity to previously archived emails

**Implementation:**
```
src/features/
  ├── metadata.py        # Extract metadata features
  ├── historical.py      # Compute historical patterns
  ├── embeddings.py      # Generate topic embeddings
  └── feature_store.py   # Feature storage and retrieval
```

## Phase 4: ML Model

**Objectives:**
- Train LightGBM regression model on historical data
- Predict likelihood (0.0-1.0) of reading each email
- Implement threshold-based decision logic
- Create category labels for organization

**Decision Logic:**
```
score >= 0.7 (high confidence keep):
  → Keep in inbox
  → Apply category label (e.g., "Bot/Newsletter-Tech")

score <= 0.3 (high confidence archive):
  → Auto-archive immediately
  → Apply category label for retrieval

0.3 < score < 0.7 (low confidence):
  → Keep in inbox initially
  → Add "Bot/LowConfidence" label
  → Schedule archive in 3 days if unopened
```

**Category Labels:**

Bot creates and applies semantic labels for organization:
- `Bot/Newsletter-Tech`, `Bot/Newsletter-Finance`, `Bot/Newsletter-News`
- `Bot/Promotional`, `Bot/Automated`, `Bot/Personal`, `Bot/Work`
- `Bot/Receipts`, `Bot/LowConfidence`, `Bot/AutoArchived`

**Benefits:**
- Search: `label:Bot/Newsletter-Tech` to find all tech newsletters
- Filter by category in Gmail UI
- Labels provide feedback (if user removes label → incorrect categorization)
- Easy to see bot's reasoning

**Implementation:**
```
src/ml/
  ├── training.py        # Model training pipeline
  ├── inference.py       # Prediction/scoring
  ├── evaluation.py      # Metrics and validation
  └── thresholds.py      # Threshold tuning

models/
  ├── model_v{N}.pkl           # Trained models
  ├── feature_config_v{N}.json # Feature configuration
  └── thresholds_v{N}.json     # Decision thresholds
```

**Success Criteria:**
- Model AUC > 0.8 on validation set
- Precision for high-confidence archive > 90%
- Recall for important emails > 95%

## Phase 5: Automation & Feedback Loop

**Objectives:**
- Daily triage script (runs every morning)
- Interactive CLI feedback tool with natural language comments
- Implicit feedback collection (label changes, moves, opens)
- Model retraining pipeline
- Scheduled automation via launchd

**Daily Triage Workflow:**
1. Fetch new unread emails from last 24 hours
2. Extract features (including user-applied labels)
3. Predict with current model (score 0.0-1.0)
4. Determine category label based on topic
5. Make decision:
   - High confidence archive → archive + label
   - High confidence keep → inbox + label
   - Low confidence → inbox + "Bot/LowConfidence" label + schedule 3-day review
6. Log all actions
7. Optional summary notification

**Interactive CLI Feedback Tool:**

Primary feedback mechanism via `scripts/review_decisions.py`:

```
Email 5/23: Tech Newsletter Digest
From: newsletter@example.com
Subject: This week in AI: GPT-5 rumors and more
Bot Decision: Archived
Bot Label: Bot/Newsletter-Tech

Decision correct? (y/n/s/c for comment): n
What was wrong? (1) Should keep / (2) Should archive / (3) Wrong label: 3
Available categories:
  1. Bot/Newsletter-Tech
  2. Bot/Newsletter-Finance
  3. Bot/Personal
  4. Bot/Work
  [... more ...]
Select correct label (or type new name): 4
✓ Feedback recorded

Email 6/23: Weekly team update
...
Decision correct? (y/n/s/c): c
Enter feedback: This sender always sends important work updates, never archive
✓ Comment saved
```

**Features:**
- Review today's decisions or last N days
- Filter by decision type (archived, kept, low-confidence)
- Progress indicator
- Natural language comments for complex feedback
- Summary: "23 reviewed, 20 correct (87%), 3 corrected"
- All corrections stored for next retraining

**Implicit Feedback (Automatic):**
- Bot archived, user moved back → False positive
- Bot kept, user opened → True positive
- Bot kept, user archived unread → False negative
- User removes bot category label → Incorrect categorization
- User adds their own labels → Strong signal of importance
- User changes bot label → Feedback on categorization

**Automation via launchd:**

```
~/Library/LaunchAgents/
  ├── com.user.email-triage.plist          # Daily triage (6 AM)
  ├── com.user.email-schedule-archives.plist  # Scheduled archives (7 AM)
  ├── com.user.email-retrain.plist         # Weekly retraining (Sunday 2 AM)
  └── com.user.email-feedback.plist        # Feedback collection (8 PM)
```

**Conservative Rollout:**

**Week 1-3: Dry-run + intensive feedback**
- All operations in dry-run (no actual changes)
- Review 20-30 predictions daily via CLI
- Correct labels, add comments for patterns
- Analyze feedback with `scripts/analyze_feedback.py`
- Tune thresholds and category detection

**Week 4: Ultra-conservative archive**
- Only auto-archive if score < 0.1 (extremely confident)
- Monitor for false positives
- Continue feedback collection

**Week 5+: Gradual threshold increase**
- Lower to 0.2, then 0.25, then 0.3 over weeks
- Only proceed if zero false positives
- Eventually enable low-confidence scheduling

**Implementation:**
```
src/triage/
  ├── pipeline.py        # Main orchestration
  ├── scheduler.py       # Low-confidence scheduling
  └── feedback.py        # Feedback collection

scripts/
  ├── triage_inbox.py           # Main daily script (with --dry-run)
  ├── review_decisions.py       # Interactive CLI feedback
  ├── analyze_feedback.py       # Analyze natural language feedback
  ├── collect_feedback.py       # Process implicit feedback
  ├── retrain_model.py          # Retrain with new data
  └── schedule_archives.py      # Archive low-confidence emails
```

## Key Design Decisions

**Why LightGBM?**
- Fast inference (<10ms per email)
- Handles mixed feature types (categorical + numerical + embeddings)
- Built-in feature importance
- No GPU required

**Why Local SQLite?**
- Complete privacy (no cloud services)
- Fast local access
- Vector search via sqlite-vss extension
- Simple backup strategy

**Why Sentence-Transformers?**
- Fast embedding generation
- Good semantic understanding
- Works offline
- Small model size (90MB)

**Why Conservative Rollout?**
- Email is high-stakes (can't afford false positives)
- Build trust gradually
- Tune thresholds based on actual behavior
- Extensive feedback before full automation

**Why Category Labels?**
- Easy retrieval via Gmail search
- User can see bot's reasoning
- Label changes provide feedback
- Organizes archive naturally

## Performance Targets

For 100 emails/day in <5 seconds:
- Email fetch: <1s (batch API)
- Feature extraction: <2s (20ms per email)
- Prediction: <1s (10ms per email)
- Operations: <1s (batch modify)

## Privacy & Security

- **OAuth tokens** stored locally, never committed
- **No external services** except Gmail API
- **Gmail scope**: `gmail.modify` (no sending, no deletion)
- **All data** stays on your Mac
- **Open source** - you can audit everything

## Development Workflow

Each phase follows:
1. Design and plan
2. Implement core functionality
3. Add logging and error handling
4. Write tests
5. Validate with real data
6. Document
7. Commit and move to next phase

Current commit: `Set up Phase 1: Gmail API Foundation`

## Next Steps

1. **Complete Phase 2**: Set up database and export historical emails
2. **Validate data quality**: Ensure all metadata captured correctly
3. **Phase 3**: Build feature extraction pipeline
4. **Phase 4**: Train initial model and tune thresholds
5. **Phase 5**: Multi-week testing before full automation

## Files to Note

- `README.md` - User-facing documentation
- `SETUP.md` - Quick setup guide for new users
- `requirements.txt` - Python dependencies (updated per phase)
- `.env` - Local configuration (gitignored)
- `scripts/` - Command-line utilities
- `src/` - Core implementation

## Contributing

This is a personal project, but the code is public for learning and inspiration. Feel free to fork and adapt for your own email workflow.
