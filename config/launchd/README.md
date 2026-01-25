# launchd Configuration for macOS

These are macOS launchd configuration files for automating email triage.

## Files

- `com.user.email-triage.plist` - Daily triage (6 AM)
- `com.user.email-feedback.plist` - Feedback collection (8 PM daily)
- `com.user.email-retrain.plist` - Weekly retraining (Sunday 2 AM)

## Installation

1. **Edit paths**: Update paths in each `.plist` file to match your setup:
   - Python path: `/Users/jake/Developer/email-assistant/venv/bin/python`
   - Script paths: `/Users/jake/Developer/email-assistant/scripts/...`
   - Log paths: `/Users/jake/Developer/email-assistant/logs/...`

2. **Copy to LaunchAgents**:
   ```bash
   cp config/launchd/*.plist ~/Library/LaunchAgents/
   ```

3. **Load the agents**:
   ```bash
   # Load triage (starts with --dry-run for safety)
   launchctl load ~/Library/LaunchAgents/com.user.email-triage.plist

   # Load feedback collector
   launchctl load ~/Library/LaunchAgents/com.user.email-feedback.plist

   # Load retrainer
   launchctl load ~/Library/LaunchAgents/com.user.email-retrain.plist
   ```

4. **Check status**:
   ```bash
   launchctl list | grep email
   ```

## Conservative Rollout

**IMPORTANT**: Start with dry-run mode!

### Week 1-3: Dry Run
- Triage runs in `--dry-run` mode (no actual changes)
- Review logs daily
- Use `python scripts/review_decisions.py` to provide feedback
- Monitor for false positives

### Week 4+: Enable Gradually
1. Remove `--dry-run` from `com.user.email-triage.plist`
2. Reload: `launchctl unload ~/Library/LaunchAgents/com.user.email-triage.plist && launchctl load ~/Library/LaunchAgents/com.user.email-triage.plist`
3. Monitor closely for first week
4. Adjust thresholds if needed

## Management Commands

**Unload (stop) an agent**:
```bash
launchctl unload ~/Library/LaunchAgents/com.user.email-triage.plist
```

**Reload (restart) an agent**:
```bash
launchctl unload ~/Library/LaunchAgents/com.user.email-triage.plist
launchctl load ~/Library/LaunchAgents/com.user.email-triage.plist
```

**Run immediately (for testing)**:
```bash
launchctl start com.user.email-triage
```

**View logs**:
```bash
tail -f logs/triage.log
tail -f logs/triage-error.log
```

## Troubleshooting

### Agent not running
1. Check if loaded: `launchctl list | grep email`
2. Check logs: `cat logs/triage-error.log`
3. Verify paths in `.plist` file are correct
4. Ensure virtual environment exists and has dependencies

### Permission errors
- Make sure scripts are executable: `chmod +x scripts/*.py`
- Verify Python can access Gmail credentials

### Computer sleep
- launchd jobs won't run if computer is asleep
- Consider using `caffeinate` or adjusting Energy Saver settings
- Or run manually: `python scripts/triage_inbox.py`

## Disable Automation

To stop all automation:
```bash
launchctl unload ~/Library/LaunchAgents/com.user.email-*.plist
```

To remove completely:
```bash
launchctl unload ~/Library/LaunchAgents/com.user.email-*.plist
rm ~/Library/LaunchAgents/com.user.email-*.plist
```
