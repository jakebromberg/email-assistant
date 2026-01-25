# Quick Setup Guide

Follow these steps to get the Gmail Email Triage Bot running.

## Step 0: Verify Python Version

**Required: Python 3.12 or 3.13**

```bash
# Check your Python version
python3 --version

# If you have Python 3.14 (too new) or 3.11 (too old), install Python 3.12
brew install python@3.12
```

**Note**: Python 3.14 is not yet recommended as many ML packages (scikit-learn, numpy) don't have pre-built wheels and require compilation from source.

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Set Up Google Cloud Credentials

### 2.1 Create Google Cloud Project

1. Go to https://console.cloud.google.com
2. Click "Select a project" → "New Project"
3. Name: "Email Assistant"
4. Click "Create"

### 2.2 Enable Gmail API

1. In the Google Cloud Console, go to "APIs & Services" → "Library"
2. Search for "Gmail API"
3. Click on it and press "Enable"

### 2.3 Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: Email Assistant
   - User support email: your email
   - Developer contact: your email
   - Click "Save and Continue" through the scopes and test users
4. Back to "Create OAuth client ID":
   - Application type: "Desktop app"
   - Name: "Email Assistant Desktop"
   - Click "Create"
5. Click "Download JSON" on the credential you just created
6. Save the downloaded file as `credentials/credentials.json` in this project

## Step 3: Authenticate

Run the authentication script:

```bash
python scripts/authenticate.py
```

This will:
- Open your browser
- Ask you to sign in to your Google account
- Request permission to access Gmail
- Save the access token to `credentials/token.json`

**Note**: You may see a warning "This app isn't verified". Click "Advanced" → "Go to Email Assistant (unsafe)" to proceed. This is safe because you created the app yourself.

## Step 4: Test Connection

Verify everything works:

```bash
python scripts/test_connection.py
```

You should see:
- List of your Gmail labels
- Recent emails from the last 7 days
- Details of a sample email
- Successful test results

## Step 5: Start Using the API

You can now use the Gmail client in your Python code:

```python
from src.gmail import GmailAuthenticator, GmailClient, GmailOperations
from src.utils import Config

# Load config
config = Config.load()

# Initialize
auth = GmailAuthenticator(
    credentials_path=config.GMAIL_CREDENTIALS_PATH,
    token_path=config.GMAIL_TOKEN_PATH
)
client = GmailClient(auth)
ops = GmailOperations(client)

# Fetch recent emails
message_ids, _ = client.list_messages(max_results=10)
emails = client.get_messages_batch(message_ids)

for email in emails:
    print(f"{email.from_address}: {email.subject}")
```

## Troubleshooting

### "Credentials file not found"

Make sure you've saved the downloaded JSON file from Google Cloud Console as:
```
credentials/credentials.json
```

### "Access blocked: This app's request is invalid"

Make sure you selected "Desktop app" (not "Web application") when creating OAuth credentials.

### "Authentication failed"

1. Delete the token file: `rm credentials/token.json`
2. Run authentication again: `python scripts/authenticate.py`

### Need to Re-authenticate

If you need to re-authenticate (e.g., to change scopes):

```bash
rm credentials/token.json
python scripts/authenticate.py
```

## Next Steps

Once Phase 1 is working:
1. Phase 2: Set up SQLite database and export historical emails
2. Phase 3: Build feature extraction pipeline
3. Phase 4: Train ML model
4. Phase 5: Set up automation

See README.md for detailed documentation.
