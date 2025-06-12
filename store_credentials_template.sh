#!/bin/bash
# store_credentials_template.sh
# Template for storing Apple Developer notarization credentials in the keychain.
# 1. Copy this file to store_credentials.sh
# 2. Fill in your credentials below
# 3. Make executable: chmod +x store_credentials.sh
# 4. Run: ./store_credentials.sh

# Your Apple ID (usually your developer email)
APPLE_ID="developer@example.com"

# App-specific password from https://appleid.apple.com
# Generate one specifically for notarization
APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"

# Your Team ID (from Apple Developer Portal)
TEAM_ID="XXXXXXXXXX"

# Profile name in keychain (you can leave this as is)
KEYCHAIN_PROFILE="VideoIndexer-notary"

# Validation
if [ "$APPLE_ID" == "developer@example.com" ] || [ "$APP_SPECIFIC_PASSWORD" == "xxxx-xxxx-xxxx-xxxx" ] || [ "$TEAM_ID" == "XXXXXXXXXX" ]; then
    echo "❌ Please edit this script and fill in your credentials before running."
    exit 1
fi

# Store credentials
echo "▶ Storing notarization credentials in keychain for profile: $KEYCHAIN_PROFILE"
xcrun notarytool store-credentials "$KEYCHAIN_PROFILE" --apple-id "$APPLE_ID" --team-id "$TEAM_ID" --password "$APP_SPECIFIC_PASSWORD"

echo "✅ Credentials stored successfully." 